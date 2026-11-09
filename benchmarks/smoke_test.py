"""C3 smoke test — run ONE decision through the calibration harness using a
real on-device model (LLM2Jev-style extraction from next-token probabilities).

This validates the wiring end-to-end (real model -> probability vector ->
calibration metrics + act/ask/refuse/escalate cascade) on a single row.

Usage: python3 benchmarks/smoke_test.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks import calib_metrics as cm
from benchmarks import cascade as cg

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def extract_option_probs(model, tokenizer, prompt, option_ids):
    """Probability of each option label as the next token after `prompt`.

    Uses the LLM2Jev bracket-numeric trick: prompt ends with 'Best answer: [',
    and we read the model's distribution over the next token for each option
    label (e.g. '1', '2', '3'), then softmax over the option set.
    """
    import torch
    enc = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model(**enc)
    last_logits = out.logits[:, -1, :].squeeze(0)  # [vocab]
    logits = torch.tensor([last_logits[wid] for wid in option_ids])
    probs = torch.softmax(logits, dim=0).tolist()
    return probs


def main():
    try:
        import torch
    except Exception as e:  # pragma: no cover
        print(f"[skip] torch not available: {e}")
        return
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {MODEL} ...")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype="auto")
    model.eval()

    options = ["refund", "escalate", "no_action"]
    labels = [str(i + 1) for i in range(len(options))]
    # map each label to its token id(s)
    option_ids = [tok.encode(lab, add_special_tokens=False)[0] for lab in labels]

    state = ("User message: 'I was charged twice for my subscription this month. "
             "Please issue a refund.'")
    prompt = (
        "You are a customer-support decision model answering gratitude.\n"
        f"State: {state}\n"
        "Which action is correct? Choose one.\n"
        + "\n".join(f"[{i+1}] {o}" for i, o in enumerate(options))
        + "\nBest answer: ["
    )

    probs = extract_option_probs(model, tok, prompt, option_ids)
    # ground-truth = the correct action for this state
    gt = 0  # 'refund'
    print("\n=== 1-ROW SMOKE TEST ===")
    for lab, p, opt in zip(labels, probs, options):
        mark = " <-- GT" if options.index(opt) == gt else ""
        print(f"  [{lab}] {opt:<14} p={p:.4f}{mark}")

    # Feed into the harness
    ece_val, detail = cm.ece([probs], [gt], n_bins=5)
    bs, (rel, res, unc) = cm.brier([probs], [gt], k=len(options))
    conf = cm.max_prob(probs)
    nconf = cm.n_confidence(probs)
    action, meta = cg.gate_utility(probs)

    print("\n=== calibration metrics (this row) ===")
    print(f"  argmax index   : {cm.argmax(probs)} (pred class)")
    print(f"  max_prob conf  : {conf:.4f}")
    print(f"  n_confidence   : {nconf:.4f}")
    print(f"  ECE (1 row)    : {ece_val:.4f}")
    print(f"  Brier          : {bs:.4f}  (rel={rel:.4f}, res={res:.4f}, unc={unc:.4f})")
    print(f"  top prob       : {max(probs):.4f}")
    print(f"  confidence_fn  : max_prob  | n_bins=5")
    print("\n=== cascade (this row) ===")
    print(f"  gate decision  : {action}")
    print(f"  meta           : {meta}")

    # coverage-risk for a single row is trivial; show it returns a valid curve
    curve = cm.coverage_risk([conf], [1], thresholds=[0.2, 0.5, 0.7])
    print("\n=== coverage-risk (single point) ===")
    for pt in curve:
        print(f"  thresh={pt['threshold']} cov={pt['coverage']} acc={pt['accuracy']} risk={pt['risk']}")

    print("\nOK: harness wired to a real on-device model.")


if __name__ == "__main__":
    main()
