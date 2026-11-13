"""Training rows of v3: candidates per screen with no cap, and token length of state + target_el options with the
40 cap vs with no cap (d1-2B tokenizer, pinned revision)."""
import json, os, sys, glob, statistics as S
sys.path.insert(0, os.path.expanduser("~/d1work/code_v3/benchmarks/train"))
import ac_common as A
from transformers import AutoTokenizer, AutoConfig
snap = os.path.expanduser("~/d1work/runs/v3_s0/ckpt_1824")
tok = AutoTokenizer.from_pretrained(snap)
cfg = json.load(open(os.path.join(snap, "config.json")))
print("config keys with 'position' or 'max':", {k: v for k, v in cfg.items() if "position" in k or "max" in k},
      {k: v for k, v in (cfg.get("text_config") or {}).items() if "position" in k or "max" in k})
def nocap(els, W, H):
    return A._reading(els, [i for i in A.rule_interactive_or_labelled(els, W, H) if not A.is_keyboard(els[i])])
ntoks = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])
cands, t40, tall, over = [], [], [], []
rows = 0
for line in open(os.path.expanduser("~/d1work/acdata/compact_awlike_full_v3.jsonl")):
    r = json.loads(line); rows += 1
    if rows % 4: continue                      # every 4th row: enough for the distribution
    canon = [A.canonical_element(e) for e in r.get("elements") or []]
    full = nocap(canon, r["W"], r["H"]); cands.append(len(full))
    for lst, idx in ((t40, A.ELEMENT_RULES["cap40_reading_noime"](canon, r["W"], r["H"])), (tall, full)):
        kept = [canon[i] for i in idx]
        st = A.build_state_v3(r.get("goal", ""), r.get("history_v3") or [], kept, r["W"], r["H"], A.current_app(canon))
        opts = "\n".join(A.option_texts_v3(kept).values())
        lst.append(ntoks(st) + ntoks(opts))
n = len(cands); q = lambda xs, p: sorted(xs)[min(len(xs) - 1, int(p * len(xs)))]
print("sampled rows", n, "of", rows)
print("candidates (no cap): median", q(cands, .5), "p90", q(cands, .9), "p99", q(cands, .99), "max", max(cands),
      "| rows > 40:", sum(c > 40 for c in cands), "(%.1f%%)" % (100 * sum(c > 40 for c in cands) / n))
for name, xs in (("cap 40", t40), ("no cap", tall)):
    print("text tokens (state + target_el options), %s: median %d p90 %d p99 %d max %d" % (name, q(xs, .5), q(xs, .9), q(xs, .99), max(xs)))
