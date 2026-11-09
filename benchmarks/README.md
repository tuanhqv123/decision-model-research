# Calibration-Reliability Benchmark

The core measurement tool for our contribution (C): **does a decision engine / mobile agent know when it's right, and can we trust it when it says it's confident?**

Pure-Python (no numpy) → runs on the Mac, Kaggle, and Colab identically.

## Modules
| File | What it gives |
|---|---|
| `calib_metrics.py` | `max_prob`, `margins`, `n_confidence`, `ece` (ECE + bins), `brier` (score + uncertainty/resolution/reliability decomposition), `coverage_risk` (selective-prediction curve), `calibrated_threshold` (conformal-style FDR threshold) |
| `cascade.py` | `gate_thresholds` (simple act/ask/refuse/escalate rules) and `gate_utility` (expected-utility cascade with reward/penalty/cost model) |

## Key definitions
- **Decision** = a probability vector over K options + the ground-truth option index.
- **Confidence** `max_prob` = P(predicted class); also `margins` and `n_confidence` (Jev-style `(n·p_max − 1)/(n − 1)`).
- **ECE** is only comparable across runs using the **same `n_bins` and `confidence_fn`** (binning is definition-sensitive — always report them).
- **Brier decomposition** is binning-approximate: `Brier ≈ Uncertainty − Resolution + Reliability` (within tolerance).
- **Coverage–risk** = for each confidence threshold, the fraction we act on (coverage) and the error rate among those (risk). This is the honest trust curve.
- **`calibrated_threshold`** = smallest confidence threshold that holds FDR ≤ target on a calibration set (selective prediction without assuming a guarantee in one pass — measure it, don't assert it).

## Run
```bash
python3 tests/test_calib.py        # 13 unit tests
python3 - <<'PY'
from benchmarks import calib_metrics as cm, cascade as cg
# a single decision: probs over 3 options, ground-truth is option 0
probs, labels = [[0.7,0.2,0.1]], [0]
print("ECE:", cm.ece(probs, labels))
print("confidence:", cm.max_prob(probs[0]), cm.n_confidence(probs[0]))
print("gate:", cg.gate_utility(probs[0]))
PY
```

## Status
- ✅ C1: metric + cascade implemented and tested.
- ⬜ C2: frozen task suite (`task_suite.yaml`) + license verification.
- ⬜ C3: run SOTA baselines through this harness → first honest coverage–risk numbers.
