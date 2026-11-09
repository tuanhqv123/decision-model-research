# 2026-11-07 — the premature-`status` post-processor is dead (negative result)

**Question.** On AC-v2 the model emits `status` 98 times on 398 steps; 21 of those are premature (gold is a
UI action). Those 21 are 5.3 accuracy points — the entire gap to Mem-W-8B. Can a calibrated gate on
`p(status)` recover them for free?

**Rule (one parameter, fitted on val, evaluated once on test).** If `pred == status` and
`p(status) < τ`, emit the argmax over `{click, long_press, navigate_back, wait}` — the four actions that
need no text or direction readout, so val (no readout dump) and test stay on the same footing. Point =
`box_pt` for spatial types, otherwise none. Scored with the official `score_row`.

**Pre-registered bar** (GOAL.md stage 2): paired Δ ≥ +1.5 pts on the 827-row val split, no gold type
regressing by more than one row.

**Result: no τ helps. The curve is monotone downward.**

| τ | 0.30 | 0.40 | 0.50 | 0.60 | 0.70 | 0.80 | 0.90 |
|---|---|---|---|---|---|---|---|
| val accuracy | **0.628** | 0.628 | 0.625 | 0.623 | 0.615 | 0.602 | 0.590 |

- Base: val **62.76%** (827), test **63.82%** (398, served-path artifact `e2-pure-hist.jsonl`).
- Best τ by val accuracy = 0.30, i.e. **the best rule is "never switch"**; Δ = 0.000 on val and test.
- Every τ ≥ 0.45 loses more correct terminations than it gains premature ones.
- Val `status` accuracy 0.876 → 0.000 at τ=0.30 in the earlier broken run showed the mechanism: the gate
  cannot separate the two batches well enough to pay for the correct terminations it destroys.

**Interpretation.** This is the third zero-GPU attempt to buy the type bucket by post-processing (hand
routing, learned fusion, per-class logit adjustment, now a status gate). All three fail for the same reason:
the score ranks errors but the *decision* needs a signal the model does not have — whether the task is
actually complete. That is a representation problem, so the training arm is the only path, and it must learn
a **progress readout** (`task_complete`), not a threshold.

**Artifacts.** `benchmarks/train/status_gate.py`; `benchmarks/results/2026-11-07-status-gate.json`
(val curve, per-gold-type deltas, McNemar). Test file is the **served-path** pure artifact (254/398), not the
goal-only `e2-pure.jsonl` (132/398).

**Caveat.** The val dump uses the checkpoint's native option names (`terminate`, `write`); the regenerated
test dump uses eval names (`status`, `type_text`). The script normalises before comparing — a mismatch here
silently made every val status row look switchable, which is exactly the class of bug this project keeps
having to fix.

## Post-hoc bound: the premature-`status` rows are worth far less than 5.3 points

After the threshold experiment failed, the obvious follow-up is "what if the signal were perfect?" — i.e. if a
learned progress readout suppressed every premature termination, what could the model actually salvage?

For each of the 21 premature `status` rows on the 398 test steps, take the model's own best expressible
non-status action and its own box point, and score that instead:

| Quantity | Value |
|---|---|
| Premature `status` rows | 21 |
| Rescued by a **perfect** "do not terminate" signal | **5 (24%)** |
| Their gold types | 12 `click`, 9 `scroll` |
| Accuracy if all 21 were rescued | 63.82% → **65.08%** (+1.26 pts) |

So the 21 rows are worth **5.3 points only if the correct action *and* point are supplied from outside**. The
model's own alternatives recover 5 of them. **A perfect progress oracle on this checkpoint is worth at most
+1.3 points**, which is below the pre-registered +1.5 pass condition — the status bucket alone cannot carry it;
the arm must also improve the action or the point.

This bound is why the down-weighting arm's pass condition was set where it was, and it is the number to quote
when someone asks why "fix the premature terminations" is not a free five points.
