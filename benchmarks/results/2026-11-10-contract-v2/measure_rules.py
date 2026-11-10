"""Contract v2: measure every candidate element rule on the R2 corpus and apply the pre-declared selection rule.

Selection rule (fixed before measuring, 2026-11-10): among rules whose list length has p90 <= 40 on TRAIN, take
the one with the highest gold coverage on TRAIN; ties -> the shorter TRAIN median. Validation numbers are
reported only.

Gold coverage = share of spatial rows (click / long_press / type_text) where SOME kept element's centre lies
inside the gold target bbox (`target`, the smallest a11y node containing the gold point — what the scorer
credits). Denominator: every spatial row, including the few whose gold target has zero area.

Every count below is computed by the SAME functions training, evaluation and serving call
(`ac_common.element_inputs`, `gold_target_el_v2`); nothing here re-implements a rule.

Usage (CPU only; streams the files):
  python measure_rules.py --train compact_cur15k_v2.jsonl --val compact_validation_v2.jsonl \
      [--tokenizer <d1-2B snapshot dir>] [--out summary.json]
Token lengths need `tokenizers` (state) and, for the exact `target_el` question block, the snapshot's own
`prompt.py` (Python >= 3.10). Without them the script reports characters/4 and says so.
"""
import argparse
import hashlib
import importlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..", "train")))
import ac_common as AC  # noqa: E402

RULES = list(AC.ELEMENT_RULES)
P90_MAX = 40


def choose_rule(train_table):
    """The pre-declared selection rule. `train_table`: {rule: {"len": {"p90", "p50"}, "coverage": {"pct"}}}."""
    ok = [r for r in RULES if train_table[r]["len"]["p90"] <= P90_MAX]
    if not ok:
        return None
    return sorted(ok, key=lambda r: (-train_table[r]["coverage"]["n"], train_table[r]["len"]["p50"], r))[0]


def quantiles(xs):
    xs = sorted(xs)
    if not xs:
        return {}

    def q(p):  # nearest-rank
        return xs[min(len(xs) - 1, max(0, int(round(p * (len(xs) - 1)))))]
    return {"n": len(xs), "p50": q(0.5), "p90": q(0.9), "p99": q(0.99), "max": xs[-1],
            "mean": round(sum(xs) / len(xs), 2)}


def pct(n, d):
    return {"n": n, "of": d, "pct": round(100 * n / d, 2) if d else None}


class Tok:
    """`tokenizers` behind the one method d1's prompt module calls, with the checkpoint's option-code scheme."""

    def __init__(self, snap):
        from tokenizers import Tokenizer
        self.t = Tokenizer.from_file(os.path.join(snap, "tokenizer.json"))
        cfg = json.load(open(os.path.join(snap, "config.json")))
        self.d1_codes = cfg.get("d1_codes", 1)
        self.d1_prompt = cfg.get("d1_prompt", 1)

    def encode(self, text, add_special_tokens=False):
        return self.t.encode(text, add_special_tokens=add_special_tokens).ids

    def count(self, text):
        return len(self.t.encode(text, add_special_tokens=False).ids)


def load_prompt(snap):
    try:
        sys.path.insert(0, os.path.dirname(snap))
        return importlib.import_module(os.path.basename(snap) + ".prompt")
    except Exception as e:  # noqa: BLE001 — Python 3.9 cannot import it (PEP 604 at module level)
        print(f"[tokens] d1 prompt.py not importable ({type(e).__name__}: {e}); question block not measured",
              flush=True)
        return None


def measure(path, tok=None, prompt=None, token_rules=()):
    lens = {r: [] for r in RULES}
    empty = {r: 0 for r in RULES}
    cov = {r: 0 for r in RULES}
    via_idx = {r: 0 for r in RULES}
    via_centre = {r: 0 for r in RULES}
    cases = {r: {"idx": 0, "creditable": 0, "none": 0} for r in RULES}
    unscoreable = {r: 0 for r in RULES}       # supervised element whose centre is NOT inside the gold target
    shown_unsupervised = {r: 0 for r in RULES}          # a creditable element is kept, but the gold rule gives none
    shown_but_other_supervised = {r: 0 for r in RULES}  # a creditable element is kept, a non-creditable one is gold
    # The `target_el` criteria are the label strings (design item 4), so unlabelled elements share the text
    # "(no label)" and repeated labels share theirs: the option text alone does not tell such options apart.
    n_kept_el = {r: 0 for r in RULES}
    n_unlabelled_el = {r: 0 for r in RULES}
    dup_rows = {r: 0 for r in RULES}
    gold_unlabelled = {r: 0 for r in RULES}
    gold_label_ambiguous = {r: 0 for r in RULES}
    by_action = {r: {} for r in RULES}
    state_tokens = {r: [] for r in token_rules}
    state_chars = {r: [] for r in token_rules}
    q_tokens = {r: [] for r in token_rules}
    n_rows = n_spatial = 0
    sha = hashlib.sha256()
    with open(path, "rb") as fh:
        for raw in fh:
            sha.update(raw)
            rec = json.loads(raw)
            n_rows += 1
            W, H = rec["W"], rec["H"]
            at = rec["action"]["action_type"]
            spatial = at in AC.SPATIAL
            n_spatial += spatial
            t = rec.get("target")
            for r in RULES:
                state, kept, keep_idx = AC.element_inputs(rec["goal"], rec.get("history"), rec["elements"], W, H, r)
                lens[r].append(len(kept))
                empty[r] += not kept
                labels = [AC.element_label(e) for e in kept]
                n_kept_el[r] += len(kept)
                n_unlabelled_el[r] += sum(1 for e in kept if not e["label"])
                dup_rows[r] += len(set(labels)) < len(labels)
                if r in state_tokens:
                    state_chars[r].append(len(state))
                    if tok is not None:
                        state_tokens[r].append(tok.count(state))
                        if prompt is not None and kept:
                            q = AC.build_questions("VA", kept, contract="v2")["target_el"]
                            q_tokens[r].append(tok.count(prompt.question_block(tok, prompt.as_question(q))))
                if not spatial:
                    continue
                hit = t is not None and any(AC.centre_in_box(e["bbox"], t) for e in kept)
                cov[r] += hit
                a = by_action[r].setdefault(at, [0, 0])
                a[0] += hit
                a[1] += 1
                pos = set(keep_idx)
                via_idx[r] += rec.get("target_el_idx") is not None and rec["target_el_idx"] in pos
                via_centre[r] += rec.get("target_el_centre_idx") is not None and rec["target_el_centre_idx"] in pos
                gid, case = AC.gold_target_el_v2(rec, keep_idx, kept)
                cases[r][case] += 1
                creditable_gold = gid is not None and t is not None and AC.centre_in_box(kept[int(gid) - 1]["bbox"], t)
                if gid is not None and not creditable_gold:
                    unscoreable[r] += 1
                if gid is not None:
                    g_label = labels[int(gid) - 1]
                    gold_unlabelled[r] += not kept[int(gid) - 1]["label"]
                    gold_label_ambiguous[r] += labels.count(g_label) > 1
                if hit and gid is None:
                    shown_unsupervised[r] += 1
                if hit and gid is not None and not creditable_gold:
                    shown_but_other_supervised[r] += 1
    out = {"rows": n_rows, "spatial_rows": n_spatial, "sha256": sha.hexdigest(), "rules": {}}
    for r in RULES:
        d = {"len": quantiles(lens[r]), "empty_lists": empty[r],
             "coverage": pct(cov[r], n_spatial),
             "coverage_by_action": {k: pct(v[0], v[1]) for k, v in sorted(by_action[r].items())},
             "target_el_idx_kept": pct(via_idx[r], n_spatial),
             "target_el_centre_idx_kept": pct(via_centre[r], n_spatial),
             "gold_cases": {k: pct(v, n_spatial) for k, v in cases[r].items()},
             "supervised_but_centre_outside_gold": pct(unscoreable[r], n_spatial),
             # Diagnostics of the decided gold order (idx -> centre -> none) against coverage; not used by any rule.
             "creditable_kept_but_no_supervision": pct(shown_unsupervised[r], n_spatial),
             "creditable_kept_but_non_creditable_supervised": pct(shown_but_other_supervised[r], n_spatial),
             # Option-text ambiguity of the label criteria.
             "kept_elements_unlabelled": pct(n_unlabelled_el[r], n_kept_el[r]),
             "rows_with_repeated_option_text": pct(dup_rows[r], n_rows),
             "gold_element_unlabelled": pct(gold_unlabelled[r], n_spatial),
             "gold_option_text_not_unique": pct(gold_label_ambiguous[r], n_spatial)}
        if r in state_tokens:
            if tok is not None:
                d["state_tokens"] = quantiles(state_tokens[r])
                if q_tokens[r]:
                    d["target_el_question_tokens"] = quantiles(q_tokens[r])
            else:
                d["state_tokens_estimate_chars_div_4"] = quantiles([c // 4 for c in state_chars[r]])
            d["state_chars"] = quantiles(state_chars[r])
        out["rules"][r] = d
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--tokenizer", default="", help="d1-2B snapshot dir (tokenizer.json, config.json, prompt.py)")
    ap.add_argument("--out", default=os.path.join(HERE, "summary.json"))
    a = ap.parse_args()
    tok = prompt = None
    if a.tokenizer:
        try:
            tok = Tok(a.tokenizer)
            prompt = load_prompt(a.tokenizer)
        except ImportError as e:
            print(f"[tokens] tokenizers missing ({e}); reporting characters/4", flush=True)
    res = {"selection_rule": f"among rules with train p90 list length <= {P90_MAX}, the highest train gold "
                             "coverage; ties -> shorter train median",
           "tokenizer": ("d1-2B tokenizer.json" if tok is not None else "none: characters/4 estimate"),
           "target_el_question_block": ("d1 prompt.question_block (exact)" if prompt is not None
                                        else "not measured")}
    res["train"] = measure(a.train, tok, prompt, token_rules=RULES)
    res["validation"] = measure(a.val, tok, prompt, token_rules=RULES)
    res["chosen_rule"] = choose_rule(res["train"]["rules"])
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"{'rule':26s} {'split':5s} {'p50':>4s} {'p90':>4s} {'p99':>4s} {'max':>5s} {'empty':>5s} "
          f"{'coverage':>15s} {'idx kept':>9s} {'ctr kept':>9s} {'case idx/ctr/none':>20s}")
    for split in ("train", "validation"):
        for r in RULES:
            d = res[split]["rules"][r]
            L, c, g = d["len"], d["coverage"], d["gold_cases"]
            print(f"{r:26s} {split[:5]:5s} {L['p50']:4d} {L['p90']:4d} {L['p99']:4d} {L['max']:5d} "
                  f"{d['empty_lists']:5d} {c['n']:6d} ({c['pct']:5.2f}%) {d['target_el_idx_kept']['pct']:8.2f}% "
                  f"{d['target_el_centre_idx_kept']['pct']:8.2f}% "
                  f"{g['idx']['n']:6d}/{g['creditable']['n']}/{g['none']['n']}")
    print("CHOSEN", res["chosen_rule"])


if __name__ == "__main__":
    main()
