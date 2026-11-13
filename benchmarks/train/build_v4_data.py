#!/usr/bin/env python3
"""Build the contract-v4 training data from the v3 AndroidControl rows (2026-11-12). No emulator, no external model.
Decisions and evidence: docs/research/2026-11-12-V4-WORKLOG-vi.md.

Every transformation keeps the row's screen and its gold unless stated, and is marked in `aug`:
  base        all AndroidControl rows; `is_focused` set False (AndroidWorld never reports it); history recomputed with
              `effect_changed_v4` (no focus flag, no keyboard elements)
  drop        targets that teach a futile repeat: gold repeats an action whose history says "no change" and the repeat
              also changed nothing (the row is removed; later rows keep it in their history)
  nc_field    gold type_text into field F: history + 'click "F" -> no change' (F already focused)
  nc_repeat   gold click on G: history + the same plausible other element X clicked twice, both "-> no change"
  nc_other    one plausible other click "-> no change"
  nc_scroll   right after open_app on a screen with a scrollable element: history + "scroll up -> no change"
  nc_status   gold status: history + one other click "-> no change" (a no-op does not undo a finished task)
  chain       two AndroidControl episodes ending in the same app, the second starting on the home screen: goal "A + B" in
              one of several phrasings, A's final screen -> navigate_home (not status), B continues, status only at the end
  avoid       screen before a save-like click that worked: history + 'click "Save" -> no change', avoid_action status
  answer      the final screen of an episode: goal + a question answered from that screen (key-value text, number of
              switches that are on, is X on), an `answer` row then a `status` row
    python3 build_v4_data.py CODE_DIR IN_AC.jsonl IN_GEN.jsonl OUT_DIR [--seed 0]
"""
import collections
import json
import os
import random
import re
import sys

A = None  # ac_common, imported in main() from CODE_DIR

SAVE_LIKE = re.compile(r"^(save|ok|done|add|create|apply|confirm|set|submit|send|update|finish)\b", re.I)
CHAIN_PHRASINGS = ["{a} Also, {b}", "Do the following: (1) {a} (2) {b}", "Complete these two tasks: {a}; {b}",
                   "{a} When that is done, {b}"]
PROBE_PHRASING = "First, {a} Then, {b}"  # held out of training: the probe family
QUESTION_TEMPLATES = ["{goal} Then answer: what is the {key}?",
                      "{goal} After that, tell me what it says next to '{key}'.",
                      "{goal} Then read the value of '{key}' and answer with it only."]
RATES = {"nc_field": 0.4, "nc_repeat": 0.06, "nc_other": 0.04, "nc_scroll": 0.5, "nc_status": 0.16}
CAPS = {"chain": 300, "avoid": 600, "answer": 250}


def _lc(s):
    s = (s or "").strip()
    return s[:1].lower() + s[1:] if s else s


def unfocus(elements):
    return [dict(e, is_focused=False) for e in elements or []]


def acted(row, rule):
    """The canonical kept element the gold acted on (as add_history_v3.acted_element), under `rule`."""
    if row["action"]["action_type"] not in A.SPATIAL + ("answer",):
        return None
    kept, idx = A.select_elements(row["elements"], row["W"], row["H"], rule)
    el, _ = A.gold_target_el_v2(row, idx, kept)
    if el is not None:
        return kept[int(el) - 1]
    i = row.get("target_el_idx")
    return A.canonical_element(row["elements"][i]) if i is not None else None


def entries_v4(rows, rule):
    """History of every step of one episode under contract v4 (effect_changed_v4)."""
    canon = [[A.canonical_element(e) for e in r["elements"]] for r in rows]
    lines, out = [], []
    for t, r in enumerate(rows):
        out.append(list(lines))
        if t + 1 < len(rows):
            lines.append(A.history_entry_v3(r["action"], acted(r, rule), A.effect_changed_v4(canon[t], canon[t + 1])))
    return out


def _ends(h, word):
    return bool(h) and h[-1].endswith(word)


def is_repeat(prev, row, rule):
    pa, g = prev["action"]["action_type"], row["action"]["action_type"]
    if pa != g:
        return False
    if pa == "scroll":
        return prev["action"].get("direction") == row["action"].get("direction")
    if pa in ("click", "long_press"):
        a, b = acted(prev, rule), acted(row, rule)
        return a is not None and b is not None and A._effect_key_v4(a) == A._effect_key_v4(b)
    return False


def plausible_other(row, rule, rng, avoid_key=None):
    """A clickable kept element that is not the gold one, preferring the gold's class, then editable / View / TextView
    (the classes v3 re-clicked most on AndroidWorld)."""
    kept, _ = A.select_elements(row["elements"], row["W"], row["H"], rule)
    g = acted(row, rule)
    gk = A._effect_key_v4(g) if g is not None else avoid_key
    cand = [e for e in kept if (e["flags"]["clickable"] or e["flags"]["editable"]) and A._effect_key_v4(e) != gk]
    if not cand:
        return None
    pref = [e for e in cand if g is not None and e["class_short"] == g["class_short"]]
    pref = pref or [e for e in cand if e["flags"]["editable"] or e["class_short"] in ("View", "TextView")]
    return rng.choice(pref or cand)


def with_history(row, extra, aug):
    return dict(row, history_v3=list(row["history_v3"]) + list(extra), aug=aug)


def nc_line(action, element):
    return A.history_entry_v3(action, element, False)


def augment_no_change(rows_by_ep, rule, rng):
    out = collections.Counter()
    added = []
    for ep, rows in rows_by_ep.items():
        for t, r in enumerate(rows):
            at = r["action"]["action_type"]
            if at == "type_text" and rng.random() < RATES["nc_field"]:
                f = acted(r, rule)
                if f is not None:
                    added.append(with_history(r, [nc_line({"action_type": "click"}, f)], "nc_field"))
            elif at == "click" and rng.random() < RATES["nc_repeat"]:
                x = plausible_other(r, rule, rng)
                if x is not None:
                    line = nc_line({"action_type": "click"}, x)
                    added.append(with_history(r, [line, line], "nc_repeat"))
            elif at not in ("status",) and rng.random() < RATES["nc_other"]:
                x = plausible_other(r, rule, rng)
                if x is not None:
                    added.append(with_history(r, [nc_line({"action_type": "click"}, x)], "nc_other"))
            if at == "status" and rng.random() < RATES["nc_status"]:
                x = plausible_other(r, rule, rng)
                if x is not None:
                    added.append(with_history(r, [nc_line({"action_type": "click"}, x)], "nc_status"))
            if t > 0 and rows[t - 1]["action"]["action_type"] == "open_app" and rng.random() < RATES["nc_scroll"]:
                if any(A.canonical_element(e)["flags"]["scrollable"] for e in r["elements"]):
                    added.append(with_history(r, [nc_line({"action_type": "scroll", "direction": "up"}, None)], "nc_scroll"))
    for a in added:
        out[a["aug"]] += 1
    return added, out


def save_avoid_rows(rows_by_ep, rule, rng):
    """avoid: on the screen BEFORE a save-like click that changed the screen, pretend that click did nothing."""
    cands = []
    for ep, rows in rows_by_ep.items():
        for t, r in enumerate(rows[:-1]):
            if r["action"]["action_type"] != "click":
                continue
            e = acted(r, rule)
            if e is None or not SAVE_LIKE.match(e["label"] or ""):
                continue
            if not _ends(rows[t + 1]["history_v3"], "→ screen changed"):
                continue
            row = with_history(r, [nc_line({"action_type": "click"}, e)], "avoid")
            row.update(action=None, target=None, target_el_idx=None, avoid_action="status", local_goal=None)
            cands.append(row)
    rng.shuffle(cands)
    return cands[:CAPS["avoid"]]


def _verb(goal):
    m = re.search(r"\b(add|create|delete|remove|set|turn|change|send|save|search|open|book|play|find|check|edit|"
                  r"rename|install|share|download|enable|disable|mark|move|copy)\b", goal or "", re.I)
    return m.group(1).lower() if m else None


def chain_rows(rows_by_ep, rule, rng):
    """Two-part goals: A then B in the same app, B starting on the home screen."""
    final_app, starts_home = {}, {}
    for ep, rows in rows_by_ep.items():
        if rows[0].get("source") == "aw_generator" or rows[-1]["action"]["action_type"] != "status":
            continue
        final_app[ep] = A.current_app([A.canonical_element(e) for e in rows[-1]["elements"]])
        starts_home[ep] = A.current_app([A.canonical_element(e) for e in rows[0]["elements"]]) == "home screen"
    by_app = collections.defaultdict(list)
    for ep, app in final_app.items():
        if app != "home screen":
            by_app[app].append(ep)
    pairs = []
    for app, eps in by_app.items():
        bs = [e for e in eps if starts_home[e]]
        for a in eps:
            for b in bs:
                if a != b:
                    pairs.append((0 if _verb(rows_by_ep[a][0]["goal"]) == _verb(rows_by_ep[b][0]["goal"]) else 1, a, b))
    rng.shuffle(pairs)
    pairs.sort(key=lambda p: p[0])  # same-verb pairs first ("delete X" + "delete Y" ~ AW list goals)
    used, out = set(), []
    for _, a, b in pairs:
        if len(out) >= CAPS["chain"]:
            break
        if a in used or b in used:
            continue
        used.update((a, b))
        ra, rb = rows_by_ep[a], rows_by_ep[b]
        goal = rng.choice(CHAIN_PHRASINGS).format(a=ra[0]["goal"].strip(), b=_lc(rb[0]["goal"]))
        cid = "chain:%s+%s" % (a, b)
        ha, hb = [r["history_v3"] for r in ra], [r["history_v3"] for r in rb]
        rows = []
        for t, r in enumerate(ra[:-1]):
            rows.append(dict(r, ep=cid, step=t, goal=goal, aug="chain", history_v3=ha[t]))
        junction = dict(ra[-1], ep=cid, step=len(ra) - 1, goal=goal, aug="chain", history_v3=ha[-1],
                        action={"action_type": "navigate_home"}, target=None, target_el_idx=None, local_goal=None)
        rows.append(junction)
        bridge = ha[-1] + [A.history_entry_v3({"action_type": "navigate_home"}, None, True)]
        for t, r in enumerate(rb):
            rows.append(dict(r, ep=cid, step=len(ra) + t, goal=goal, aug="chain", history_v3=bridge + hb[t]))
        n = len(rows) - 1
        out.append([dict(r, n_steps=n) for r in rows])
    return out


def key_value_pairs(row, rule):
    """(key, value) labelled elements on the same line (value to the right) or value just below the key."""
    kept, idx = A.select_elements(row["elements"], row["W"], row["H"], rule)
    pairs = []
    for i, k in enumerate(kept):
        kl = k["label"] or ""
        if not k.get("own_text", True):   # a contract-v5 id label ("[id: ...]") is not screen text: never a key
            continue
        if not kl or len(kl) > 30 or k["flags"]["clickable"] or k["flags"]["editable"] or re.fullmatch(r"[\d\W]+", kl):
            continue
        kb = k["bbox"]
        for j, v in enumerate(kept):
            vl = v["label"] or ""
            if j == i or not vl or vl == kl or v["class_short"] in ("Button", "ImageButton") or len(vl) >= A.LABEL_CHARS:
                continue
            vb = v["bbox"]
            same_line = abs((kb["y_min"] + kb["y_max"]) / 2 - (vb["y_min"] + vb["y_max"]) / 2) < 25 and vb["x_min"] >= kb["x_max"]
            below = 0 <= vb["y_min"] - kb["y_max"] < 60 and abs(vb["x_min"] - kb["x_min"]) < 40
            if same_line or below:
                pairs.append((k, v, idx[j]))
                break
    # a key label must be unique on the screen, or the question is ambiguous
    counts = collections.Counter(k["label"] for k, _, _ in pairs)
    return [p for p in pairs if counts[p[0]["label"]] == 1]


def answer_rows(rows_by_ep, rule, rng):
    out = []
    eps = [ep for ep, rows in rows_by_ep.items() if rows[-1]["action"]["action_type"] == "status"
           and rows[0].get("source") != "aw_generator"]
    rng.shuffle(eps)
    for ep in eps:
        if len(out) >= CAPS["answer"]:
            break
        rows = rows_by_ep[ep]
        last = rows[-1]
        canon = [A.canonical_element(e) for e in last["elements"]]
        checkable = [(i, e) for i, e in enumerate(canon) if e["flags"]["checkable"] and e["label"]]
        kind = rng.random()
        if kind < 0.6:
            kv = key_value_pairs(last, rule)
            if not kv:
                continue
            k, v, vi = rng.choice(kv)
            q = rng.choice(QUESTION_TEMPLATES).format(goal=last["goal"].strip(), key=k["label"])
            ans = {"action_type": "answer", "text": v["label"]}
            extra = dict(target=v["bbox"], target_el_idx=vi, answer_value="element")
        elif kind < 0.8:
            sw = [e for _, e in checkable if e["class_short"] in ("Switch", "CheckBox", "ToggleButton", "RadioButton")]
            if not sw:
                continue
            n = sum(e["flags"]["checked"] for e in sw)
            q = "%s Then answer with a number: how many switches or boxes on this screen are turned on?" % last["goal"].strip()
            ans = {"action_type": "answer", "text": str(n)}
            extra = dict(target=None, target_el_idx=None, answer_value=str(min(n, A.ANSWER_COUNT_MAX)))
        else:
            if not checkable:
                continue
            _, e = rng.choice(checkable)
            q = "%s Then answer True or False: is '%s' turned on?" % (last["goal"].strip(), e["label"])
            ans = {"action_type": "answer", "text": str(bool(e["flags"]["checked"]))}
            extra = dict(target=None, target_el_idx=None, answer_value=str(bool(e["flags"]["checked"])))
        cid = "answer:%s" % ep
        n = len(rows)
        ep_rows = [dict(r, ep=cid, goal=q, aug="answer", n_steps=n) for r in rows[:-1]]
        ep_rows.append(dict(last, ep=cid, goal=q, aug="answer", action=ans, local_goal=None, n_steps=n, **extra))
        line = A.history_entry_v3(ans, None, None)
        ep_rows.append(dict(last, ep=cid, step=n, goal=q, aug="answer", history_v3=last["history_v3"] + [line],
                            action={"action_type": "status", "goal_status": "successful"}, target=None,
                            target_el_idx=None, local_goal=None, n_steps=n))
        out.append(ep_rows)
    return out


def build(ac_rows, rule, seed=0):
    rng = random.Random(seed)
    by_ep = collections.OrderedDict()
    for r in ac_rows:
        r = dict(r, elements=unfocus(r["elements"]), aug="base")
        by_ep.setdefault(r["ep"], []).append(r)
    for ep, rows in by_ep.items():
        rows.sort(key=lambda r: r["step"])
        for r, h in zip(rows, entries_v4(rows, rule)):
            r["history_v3"] = h
    stats = collections.Counter()
    base = []
    for ep, rows in by_ep.items():
        for t, r in enumerate(rows):
            if (t > 0 and _ends(r["history_v3"], "→ no change") and is_repeat(rows[t - 1], r, rule)
                    and t + 1 < len(rows) and _ends(rows[t + 1]["history_v3"], "→ no change")):
                stats["drop"] += 1
                continue
            base.append(r)
    nc, c = augment_no_change(by_ep, rule, rng)
    stats.update(c)
    avoid = save_avoid_rows(by_ep, rule, rng)
    stats["avoid"] = len(avoid)
    chains = chain_rows(by_ep, rule, rng)
    stats["chain episodes"] = len(chains)
    answers = answer_rows(by_ep, rule, rng)
    stats["answer episodes"] = len(answers)
    rows = base + nc + avoid + [r for ch in chains for r in ch] + [r for an in answers for r in an]
    stats["base rows"] = len(base)
    stats["total rows"] = len(rows)
    return rows, stats


def main(code_dir, in_ac, in_gen, out_dir, seed=0):
    global A
    sys.path.insert(0, code_dir)
    import ac_common as _A
    A = _A
    rule = A.CONTRACTS["v4"]["element_rule"]
    ac = [json.loads(l) for l in open(in_ac)]
    rows, stats = build(ac, rule, seed)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "compact_awlike_full_v4.jsonl"), "w") as fo:
        for r in rows:
            fo.write(json.dumps(r, separators=(",", ":")) + "\n")
    with open(os.path.join(out_dir, "compact_awgen_v4.jsonl"), "w") as fo:
        for l in open(in_gen):
            r = json.loads(l)
            fo.write(json.dumps(dict(r, elements=unfocus(r.get("elements"))), separators=(",", ":")) + "\n")
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    args = sys.argv[1:]
    seed = int(args[args.index("--seed") + 1]) if "--seed" in args else 0
    main(*[a for a in args if a != "--seed" and a != str(seed)][:4], seed=seed)
