# 2026-11-07 — status down-weighting: pre-registered arm, NEGATIVE (and informative)

**Pre-registration:** `docs/research/PRE-REG-2026-11-07-status-downweight.md`, written before the run.
**Pass condition:** paired Δ ≥ **+1.5 pts** on the 827-row val split, **no gold type regressing by more than 1 row**.

## Result — FAILS, on both clauses

| Split | Base (E2) | Arm (status 22% → 5.3%) | Δ | McNemar | 95% cluster CI |
|---|---|---|---|---|---|
| val (827) | 519 (62.76%) | **528 (63.85%)** | **+1.09** | b=49 / c=58, p = 0.44 | [−1.48, +3.85] |
| test (398) | 255 (64.07%) | **242 (60.80%)** | **−3.27** | b=41 / c=28, p = 0.15 | [−7.34, +0.76] |

Δ is positive on val and negative on test — **the two splits disagree in sign**, which on its own would make the
arm unpublishable even if val had cleared +1.5.

**Per gold type (rows, val):**

| type | base | arm | Δ |
|---|---|---|---|
| click | 253/420 | 270/420 | **+17** |
| scroll | 41/86 | 54/86 | **+13** |
| wait | 15/56 | 18/56 | +3 |
| open_app | 38/45 | 40/45 | +2 |
| type_text | 32/53 | 34/53 | +2 |
| navigate_back | 20/29 | 18/29 | −2 |
| **status** | **120/137** | **94/137** | **−26** |

Test shows the same shape harder: click +9, scroll +4, spatial bucket **+2.26 pts**; non-spatial bucket
**−5.53 pts**; status **−25**.

## Mechanism — this is the useful part

The hypothesis was that synthesized completion labels cause premature termination. The intervention moved
premature terminations exactly as predicted and moved true terminations by the same amount:

| | E2 | arm | Δ |
|---|---|---|---|
| premature `status` on non-`status` gold (val) | 51 | **24** | **−27** |
| correct `status` on `status` gold (val) | 120 | **94** | **−26** |

**−27 premature terminations bought −26 correct ones.** The class weight is a single knob on
`p(terminate)`; it cannot separate "this is not finished" from "this is finished" without a signal that encodes
progress. Down-weighting shifts the whole decision boundary, which is why the non-spatial bucket loses 22 rows
on test while grounding *gains* (it is the same model with a different termination prior).

Two consequences worth keeping:

1. **Grounding improved without being trained** (+2.67 pts val, 285/422 → 304/433; +2.26 pts test,
   125/223 → 134/223). Fewer premature terminations mean more steps are spent actually acting, so spatial
   decisions get exercised. This is consistent with the error budget: the 21 premature rows are *not* a
   grounding failure, and removing them exposes the grounding that was there.
2. **The oracle bound predicted this.** A *perfect* progress signal on this checkpoint is worth only
   +1.26 pts, because the model's own next-best action rescues just 5 of the 21 premature rows
   (`benchmarks/results/2026-11-07-status-gate.md`). The arm delivered +1.09 on val — inside that bound, and
   it paid for it with 26 correct terminations. **The status bucket cannot be won by re-weighting.**

## Verdict

**H1 (synthesized `status` over-supervision causes premature termination) is partially supported and
practically insufficient.** The intervention changes the termination boundary as predicted, but a single class
weight trades one error for the other almost exactly, and the cross-split sign flip means no claim survives.

Next arm, if the line is continued: **an explicit progress readout** (`task_complete: noul` as its own slot,
fed to the type decision) rather than a weight on the type label. That is the only variant that gives the
model a signal *other than* the class prior. It is not run here.

## Artifacts

- `benchmarks/results/2026-11-07-arm/eval-arm20-V-val.jsonl` (827), `eval-arm20test-V.jsonl` (398)
- `benchmarks/results/2026-11-07-arm/paired_arm_vs_E2_val.txt`, `paired_arm_vs_E2_test.txt`
- Training log: worker `~/d1work/logs/arm.log` (938 steps, `ARM_DONE_RC=0`, final loss 4.4–4.6)
- Recipe: base `LiquidAI/d1-2B`, `--track V --steps 938 --micro 1 --accum 16 --lr 1e-5 --grad-ckpt
  --freeze-embed --mem-frac 0.38`, data `compact_cur15k_status20` (12,361 rows; status 3,300 → 661)
- **Deviation from the pre-registration, recorded:** the frozen recipe was `micro 2 / accum 8 / mem-frac 0.92`;
  the shared card only had 13.9 GB free, so the run used `micro 1 / accum 16`. Effective batch is identical
  (16), gradient noise is not, so this is a two-variable change. It does not rescue a result that fails both
  clauses and flips sign across splits, but a positive result would have required a matched replication.
