# 2026-11-07 — the cost window: where the gate actually beats both trivial policies

**Question.** The audience evidence says an accuracy tie is only publishable if the gate beats `always-act` and
`always-ask` at a cost ratio a product faces (T2 §A.5, §B.6). Where does that happen?

**Setup.** Normalise `C_ask = 1`, let `R = C_err / C_ask`, and price 1,000 decisions:

```
always-act      = 1000 · (1 − acc) · R
always-ask      = 1000
gate at (c, r)  = 1000 · (c · r · R + (1 − c))
```

Inputs are the saved artifacts, not a new run: `gate_test938.json` (certified δ=0.20 operating point;
test coverage 0.4070, empirical risk 0.1728, CP upper 0.2170, n=162) and accuracy 0.6482 on the same
`E2-938-V` artifact. **Assumption, stated and not measured:** an ask is answered correctly and costs a fixed
`C_ask`. The literature prices the ask only through override/habituation rates (T2 §C.3), never in a product's
own utility.

**Result — cost per 1,000 decisions, in units of `C_ask`.**

| R = C_err/C_ask | always-act | always-ask | gate (empirical) | gate (CP upper) | learned fusion @50% |
|---|---|---|---|---|---|
| 0.5 | **175.9** | 1000 | 628.1 | 637.1 | 547.7 |
| 1 | **351.8** | 1000 | 663.3 | 681.3 | 595.5 |
| 2 | **703.6** | 1000 | 733.7 | 769.6 | 691.0 |
| 3 | 1055.4 | 1000 | **804.0** | 858.0 | **786.4** |
| 4 | 1407.2 | 1000 | **874.4** | 946.3 | **881.9** |
| 6 | 2110.8 | 1000 | 1015.1 | 1123.0 | 1072.9 |
| 8 | 2814.4 | 1000 | 1155.8 | 1299.6 | 1263.8 |
| 16 | 5628.8 | 1000 | 1718.6 | 2006.3 | 2027.6 |

**Break-even ratios.**

| Comparison | Empirical | Using the CP upper bound |
|---|---|---|
| gate beats always-act when | R > **2.11** | R > **2.25** |
| gate beats always-ask when | R < **5.79** | R < **4.61** |
| learned fusion beats always-act when | R > **1.95** | — |
| learned fusion beats always-ask when | R < **5.24** | — |

**Reading.** The gate dominates both trivial policies over roughly a **2.1–5.8×** cost window (2.25–4.61 using
the conservative bound on the risk estimate). Below 2×, acting is cheap enough that the interruption is not
worth paying for; above ~6×, asking every time is better because the gate still waves through 40.7% of
decisions at 17.3% risk.

That is the decision-level statement an accuracy tie cannot make on its own. It also sets the honest bar for
the next checkpoint: if the window collapses on `E2@705` (where the same threshold gave 23.9% risk), the cost
argument is checkpoint-local and the paper must say so.

**Second checkpoint (does the window survive?).** Repeating the arithmetic on `E2@705` (accuracy 61.31%, gate
at 38.9% coverage, empirical risk 23.9%, CP upper 28.8%) gives **2.08–4.19** empirically and **2.22–3.47** with
the bound. The window survives the checkpoint change, narrower — the cost argument is not checkpoint-local,
though the certificate is (the 705 threshold is `held=False`).

**What this does not establish.** The ask is assumed perfect and fungible. Real asks are dismissed (49–96%
override rates in clinical settings), habituate within days, and are not always available. The next
measurement that would matter is the *observed* ask acceptance rate for a GUI gate, which no paper reports
(T1 §6).

Artifacts: `benchmarks/train/cost_window.py`, `benchmarks/results/2026-11-07-cost-window.json`.
