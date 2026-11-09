# Grounding-precision research — 2026-11-02 (Feynman alpha reads + our measured error decomposition)


> **Correction (2026-11-07).** The headline in this note predates the internal audit: **64.32% (256/398) exists
> in no artifact** — the pure numbers are 64.07% (255/398) in-process and 63.82% (254/398) served, and the
> 56.50% grounding is the step-705 checkpoint (938 pure is 56.05%). See
> `benchmarks/results/2026-11-03-acv2-error-budget-and-routed-gate.md` and paper §5.1.

> Chủ đề: d1-2B hit a ceiling on AC-v2 (~64.3–64.8 across 3 SFT recipes). Remaining gap is **grounding**,
> not type. This note records (1) our measured error decomposition, (2) the 3 papers that target it, and
> (3) the decision: close grounding via **training-time** fixes (V2P + ConClick), NOT RL. Sources in
> `references/ledger.md` (bibkeys: v2p2026, conclick2026, seg2click2026, magmar1dense2026).

---

## 1. Our current state (measured, official scorer `eval_rl.py`)

| Run | Data | Step acc | Type | Grounding | Notes |
|---|---|---|---|---|---|
| E2 (720px) | cur15k V-only | **64.07** (pure, 255/398) | 84.17 | **56.05** (125/223) | 3rd SFT recipe; CI [59.2, 68.6]. *Was written 64.32/56.50 before the audit; the 56.50 belongs to step 705.* |
| groundB (858px + hit loss + aux grounding) | cur15k | **64.82** | 82.16 | **60.09** (134/223) | result file `benchmarks/results/2026-11-02-d1-acv2-groundB.md` |
| E2 + app-name hint | cur15k | 64.82 | 84.17 | 57.40 (128/223) | extra input, NOT pure protocol; NOTE on-disk `E2-938-V.jsonl` IS this run |

- groundB improved click grounding (+8 click rows; wrong-place misses 71→58) and nudged overall step acc 64.07 (pure) → 64.82 (hinted) (255→258, +3; the gain is confounded with the hint), but **action-TYPE accuracy dropped** (84.17→82.16, −8 type rows) — the grounding gain is partially offset by the type loss. 3 recipes all stop at ~64.3–64.8. **We have hit the ceiling of this SFT framing.**
- **Comparison to leaderboard:** GUI-Libra-8B 64.3 (NeurIPS 2026, rank-A) · Mem-W-8B 68.59 (preprint) ·
  Mem-W-4B 63.07 · UI-Venus-1.5-8B 61.06. We are **tied #2**, not beating.

## 2. Measured error decomposition (E2-938-V.jsonl, 223 spatial rows)

| Error type | Share | Detail |
|---|---|---|
| Type correct but grounding wrong | 68 rows | model knows the action but clicks the wrong place (65 `click` + 3 `type_text`, **0 `open_app`**) |
| **Wrong element (far)** | **86.8% of misses** | median **437px** from correct element (NOT precision) — mean 663px; ~7.4% ≤100px |
| Near-miss (≤50px) | 4.4% | click close but outside target |
| Vertical bias | median +0.5px | NOT a systematic bias — **dispersion, not bias** |

- Gold y spread: 39% top third / 26% mid / 35% bottom — no concentrated region.
- **Conclusion: the problem is element SELECTION (semantic) + attention dispersion, not pixel precision.**
  This exactly matches ConClick's cross-model finding (74–88% of incorrect clicks hit a non-target element)
  → our error is a known, well-studied failure.

## 3. The 3 papers that target our failure (all training-time)

### V2P (v2p2026) — attention calibration
- Two causes: **background distraction** + **center-edge confusion** (uniform target modeling).
- Fix: (a) **Suppression Attention** — penalize attention on non-target patches; (b) **Fitts-Gaussian Peak
  Modeling** — KL to 2D Gaussian centered on target, **σ ∝ element width/height**.
- Result: ScreenSpot-Pro 44.6→52.5 (V2P-7B); ablation FGPM −5.0, SA −3.2; small element **+10.0**.
- **Relevance:** our box loss uses fixed σ=4, no suppression. Should scale σ + add suppression.

### ConClick (conclick2026) — element-wise contrastive
- **Systematically confirms** 74–88% of incorrect clicks hit non-target elements; same distractor across models.
- Fix: build target–distractor pairs (from GUI-Libra/GTA1 mis-clicks), add **contrastive** listwise+pairwise
  margin on region-selection scores.
- Result: Qwen3-VL-8B ScreenSpot-Pro 54.71→**66.03** (+11.3); +RECLICK →72.36.
- **Relevance:** error profile = exactly ours. Operation is on attention scores → need to check d1 readout.

### Seg2Click (seg2click2026) — pixel-level action mask
- **Single-point coordinate supervision is flawed** (many valid click points; token footprint spans target+neighbor+background).
- Fix: box-supervised **sub-token action-field mask** → centroid of largest connected component = click.
- Result: Seg2Click-9B ScreenSpot/v2/Pro = 92.06/94.10/**55.22**; 0.8B beats coords by +24.
- **Relevance:** d1's box slot already supervises a region — Seg2Click argues to supervise the whole
  executable region, not collapse to center-point.

## 4. Why NOT RL (evidence)

- Galaxy of the RL literature (MobileRL, Magma-R1, UI-AGILE, Adaptive Milestone Reward) confirms RL needs
  careful reward engineering and **risks inflating confidence** (sparse reward → reward hacking).
- **RL is a worse match for our error:** our error is element *selection* (semantic) + attention drift, not
  click-precision around a known target. Dense 2D-Gaussian reward (Magma-R1) targets precision, not selection.
- **Calibration is our differentiator** — RL post-training is the easiest thing to break it (HyperClick exists
  precisely to re-add a confidence-alignment reward). Training-time attention/contrastive fixes preserve it.

## 5. Decision

**Close grounding with training-time fixes on the existing SFT path, before touching RL:**
1. **V2P-ify the box loss**: σ ∝ element width/height (replaces fixed σ=4) + background-suppression term.
2. **ConClick-style contrastive**: build target–distractor pairs from **our own d1-2B E2 mis-clicks** (the 68
   wrong-grounding rows → wrong element = distractor), add listwise/pairwise margin pushing target logit
   above distractor logit.
3. **Keep calibration readout + temperature refit untouched** (calibration = our differentiator).

---

## ⚠️ Calibration check (2026-11-02 deep-verify)
Reproducible from saved eval: p(action) ECE **0.231** (E2@705) / **0.2038** (E2@938); AUROC 0.626/0.631.
The earlier-drafted §5.2 claimed ECE 0.146→**0.036** isotonic, AUROC 0.723, risk-gate cov 55.0%/36.7% —
**these do NOT reproduce from the jsonl** (recompute gives p×box ECE ~0.29–0.36; isotonic fit-on-val→test
collapses to τ=1.0). Treat 0.146/0.036/55% as **reported-in-run / pending GPU re-fit**, NOT proven.
4. **Measure** on the official 223-step set → target grounding 56.5 → ~66+, which would push total toward ~69.

**Blocking question (must read before planning):** d1-2B reads **logits at a slot** (the `box` slot), NOT an
attention heatmap over patches. V2P/ConClick/Seg2Click operate on attention/region scores. Need to read
`benchmarks/train`'s `slots.py`/`pointer.py`/`box.py` to confirm whether d1 exposes a **spatial attention map**
I can supervise. If not, the **ConClick (logit-level contrastive)** and **Seg2Click (box-region mask)** paths are
more directly feasible than V2P's attention-suppression path.

## 6. Action log
- 2026-11-02: led v2p2026 / conclick2026 / seg2click2026 / magmar1dense2026 → `references/ledger.md` + `bib.bib`.
- 2026-11-02: this note written. groundB result file now exists (`benchmarks/results/2026-11-02-d1-acv2-groundB.md`).
