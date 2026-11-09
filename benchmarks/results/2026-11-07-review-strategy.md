# Strategy review — `PLAN-2026-11-07-better.md`

*Adversarial review, angle STRATEGY: is this the right work at all? Written 2026-11-07. Sources numbered in
§Sources; internal artifacts are cited by path and marked `[local]`. Every external claim has a URL.*

**Verdict up front.** The plan is well-disciplined and its guard rails are the best thing in the programme. It
is also, at the level of strategy, aimed at a target that has three defects the plan does not name: the
benchmark number it chases is protocol-unstable by ~5 points, the effect size it can produce (~+2 pts) is
below the detection floor of its own test split (≥3.0 pts at α=.05), and the one genuinely new idea in the
list (work #3) is a two-pass method grafted onto a programme whose entire differentiation is *one* pass. Ranked
by expected publishable value, four of the five work items are worth less than the paper the programme already
has sitting un-submitted.

Nine findings follow. Four are FATAL, four MAJOR, one MINOR.

---

## Finding 1 — [FATAL] The plan's opening premise is a claim this programme already retracted and lists as do-not-cite.

**Claim under attack.** PLAN §0, fact 1: *"Type is nearly solved; placement is not. 84% of the time the model
knows what to do and 56% of the time it knows where… Perfect placement alone is worth +17.84 pts; perfect type is
worth +15.83."*

**Evidence.**
1. The insight is used to justify the plan's ordering, and it is the first thing a future session reads.
2. `notes/GOAL.md` §8 decision log, 2026-11-03: *"**'The remaining gap is grounding, not type' corrected to
   'both, comparable'** — 63 vs 68 (test), 161 vs 137 (val), z≈1.4 — not a ranking."* `[local]`
3. `notes/GOAL.md` §9 (Do-not-cite list): *"'The remaining gap is grounding, not type' — both, comparable."
   `[local]`
4. The plan's own two oracle numbers already concede this: +17.84 (placement) vs +15.83 (type) is a **2.0-point
   difference on a 398-row split**, i.e. 8 rows, which is inside the noise band this document computes below
   (Finding 3: a net of 8 rows gives p = 0.18). The plan cites two figures that differ by exactly the amount
   they cannot resolve, and reads a ranking off them.

**Why it matters.** Every downstream argument about where to spend GPU inherits the direction. Work #5 (grounding
levers, 6 GPU-h) and work #3 (a *point* verifier, which verifies placement, not the verb) are both placed
downstream of a premise the programme's own north-star file has already struck.

**Recommendation.** Rewrite PLAN §0 fact 1 to the GOAL.md wording — "both, comparable (63 vs 68 test, 161 vs 137
val, z≈1.4)" — and re-derive the ordering. If the ordering survives, fine; but it must survive on the corrected
premise, not on a retracted one.

---

## Finding 2 — [FATAL] AC-v2 is still alive in 2026 but it is not a *board*, it is a family of mutually
incompatible protocols; the "bar to beat" the plan uses is not a stable target.

**Answer to "does anyone care about AC-v2 in 2026?".** Bluntly: **yes, but as a table row, not as a claim** —
and the numbers attached to the name are not comparable across papers. "Stale" would be the wrong word.
"Unrankable" is the right one.

**Evidence that it is still used (so the plan is not chasing a dead benchmark).**
- `Mem-W` (arXiv:2605.09317, May 2026) reports **AndroidControl-v2 High, 398 cleaned samples** — the exact
  protocol this programme uses — with Pass@1 **68.59** / Type **81.16** / Grounding **71.30** for Mem-W-8B [2].
- `ECHOES` (arXiv:2610.22408, **14 Oct 2026**) reports AC-v2-High as one of four benchmarks in its main table,
  with a 13-system comparison [1].
- `GUI-Libra` (arXiv:2602.22190, 25 Feb 2026) introduced AC-v2 as *"a cleaned version of AndroidControl with
  corrected action labels"* and is the origin of the 398-row convention [3][18].
- Aggregator boards still carry the "AndroidControl High v2" track, last updated with May/June-2026 entries [10].

**Evidence that it cannot serve as a ranking target.**
1. **Same backbone, same nominal benchmark, 5.2 points apart.**
   - Qwen3-VL-8B on AC-v2-High: **49.62** in ECHOES (Oct 2026) [1]; **54.80** in the Mem-W Table-2 aggregate
     [10]. Qwen3-VL-32B: **52.53** [1] vs **59.05** [10]. A 5-point protocol gap on an *identical* backbone.
2. **The number the programme is 4.5 points short of is an outlier, not a frontier.** In ECHOES's Table 1 the
   entire field sits at 46.9–53.6 on AC-v2-High — GPT-5.5 **46.92**, Gemini 3.1 Pro **50.82**, MAI-UI-8B
   **51.68**, Qwen3-VL-8B **49.62**, best-in-table **53.60** [1]. Mem-W's 68.59 and GUI-Libra's 64.30 are 11–15
   points above every other number reported on the same benchmark. **The same benchmark name yields a spread
   twice as large as the entire deficit the programme is trying to close.**
3. **The name collides with a different, easier board.** `paperswithcode`'s AndroidControl leaderboard — "joint
   step success (exact match)… element coordinates match when they fall inside the ground-truth target box" on
   the official test split — tops out at **86.1** (UItron-72B), with OmegaUse 79.1, GUI-Owl-32B 76.6 [9]. A
   reviewer who types "AndroidControl SOTA" finds 86.1, not 68.59. The programme's "we tie the rank-A 8B
   baseline" reads to an outsider as 22 points below SOTA.
4. **The benchmark's own validity is contested in print.** `AndroidControl-Curated` (arXiv:2510.18488) reports
   "notable shortcomings in AndroidControl, including ambiguities and factual errors, which systematically
   underrates agent capabilities", and finds +15% after purification [8]. The task-level evaluation literature
   says the same thing from the other side: `A3` (Findings of ACL 2026) opens by naming "static frame
   assessments such as AndroidControl" as a paradigm that "fail[s] to capture agent performance" [5]; and
   AndroidWorld itself is now described as saturated at >90% SR, with MobileWorld built to replace it [4].

**Consequence for the programme.** The benchmark number cannot be the entry ticket the plan treats it as,
because there is no single number to beat. This cuts two ways and both are bad for the plan: (a) a real +2 pts
is unverifiable against the literature, and (b) the programme cannot claim a win either, because the comparison
set it uses (Mem-W 68.59) is protocol-mismatched *in its own favour* by ~5 points relative to what an
independent group gets from the same 8B backbone.

**Recommendation.**
- Stop quoting cross-paper AC-v2 numbers in the plan. Any accuracy sentence in the paper must be
  **protocol-matched and self-run**: put Qwen3-VL-2B and Qwen3-VL-8B through *our* scorer on *our* 398 rows
  (~1 GPU-h) and compare to those, not to Mem-W's Table 2.
- Add one sentence to the paper stating the measured protocol spread (49.62 vs 54.80 for one backbone) with both
  citations. It costs nothing and pre-empts the reviewer who finds 86.1.

---

## Finding 3 — [FATAL] The ensemble cannot produce a significant result on 398 rows. This is arithmetic, not opinion, and it is the single most important thing in this review.

**Claim under attack.** The plan's implicit thesis: "the headroom is real and large (+16.08), every combiner
converts almost none of it, so a better combiner will convert more of it, and that will be the result."

**The arithmetic.** I recomputed McNemar's exact test from the artifacts' own discordant counts. My model
reproduces every published p-value in the programme to two decimals — p = 0.1849 for the ±8 px majority vote
(reported 0.18), 0.4244 for the 2-checkpoint fusion (reported 0.42), 0.4394 for the status arm on val (reported
0.44) — so the model is calibrated to their procedure, not to mine.

| switched rows (b+c) | rows needed net (b−c) for p<.05 | precision needed on switched rows | delivered Δ on n=398 |
|---|---|---|---|
| 20 | 9 | 72.5% | +2.26 pts |
| 28 (the measured case) | **12** | **71.4%** | **+3.02 pts** |
| 40 | 14 | 67.5% | +3.52 pts |
| 60 | 18 | 65.0% | +4.52 pts |
| 80 | 20 | 62.5% | +5.03 pts |
| 100 | 22 | 61.0% | +5.53 pts |

**What the programme has actually achieved, measured against that bar:**

| combiner | net rows | rows switched | precision | Δ | p | verdict |
|---|---|---|---|---|---|---|
| majority vote ±8 px | +8 | 28 | 64.3% | +2.01 | 0.185 | needs 71.4% |
| 2-checkpoint fusion, val-fitted τ | +5 | 25 | 60.0% | +1.26 | 0.424 | needs 72.0% |
| learned 20-feature arbiter | +0 | 28 | 50.0% | +0.00 | 1.000 | coin flip |
| status down-weight arm (val) | +13 | 69 | 59.4% | +2.26 val | 0.439 | val-split sign flip on test |

*Recomputed from `benchmarks/results/2026-11-07-fusion-headroom.md` ("gained 15 / lost 10"; "gained 18/lost 10"),
`benchmarks/results/2026-11-07-fusion-arbiter.json` (14/14), and `2026-11-07-status-downweight-arm.md`
(b=49/c=58) `[local]`. This also resolves an inconsistency inside PLAN §1c: the same +2.01 is listed with
p=0.077 and with p=0.18. Exact McNemar on 18/10 gives **p=0.185**. The 0.077 entry is wrong and should be
deleted.*

**The corollary that should reorder the plan.** A *better* combiner does not fix this, because a better combiner
switches **more** rows and the precision bar then moves down only slowly (65% at 60 rows). The right way to read
the table is not "we need a bigger Δ", it is:

> **A combiner is publishable when it is right on ≥65% of the rows it changes, on ≥60 changed rows. It is
> currently right on 60–64% of 25–28 rows.**

The 10-checkpoint union supplies 64 recoverable rows (322 union vs 258 best single, both on 398) `[local,
PLAN §1b]`. At 70% precision on 60 of those, net = 24 rows = **+6.0 pts, p<0.01**. So the target is reachable in
principle — and the plan's own stop rule for work #2 ("if CV accuracy ≤ best single, ensembling is dead") does not
measure the thing that decides it. CV accuracy is dominated by the ~80% of rows where every checkpoint already
agrees; it will sit near the base rate no matter how good the arbiter is on the rows that matter.

**Recommendation.**
- Replace work #2's pass condition with: **precision ≥65% on switched rows AND ≥60 switched rows on val**, with
  the test scored once. Report gained/lost/precision, never bare accuracy.
- Add the multiplicity correction. The same 398 rows have now carried at least six tests (±8 px, ±32 px, ±64 px,
  2-ckpt fusion, learned arbiter, status arm, plus the status gate). Bonferroni at 6 tests sets α=.008, i.e.
  **net ≥ 15 rows (+3.77 pts)**. Any future "significant" ensemble number at p=.02 is a re-look, not a result.
- Pre-commit now: **if the pooled arbiter does not clear +3.8 pts net on test, the ensemble line is closed and
  is written up as a negative result, not re-tuned.**

---

## Finding 4 — [FATAL] Work #5 re-introduces the exact plan the programme's own files mark INVALIDATED and "do NOT spend GPU on".

**Claim under attack.** PLAN §2, work #5: *"grounding-specific levers (ConClick hard negatives from our 68 wrong
elements; element-size-scaled box σ)… ~6 h GPU."*

**Evidence.**
- `docs/HANDOFF.md` §3.3 is titled **"The grounding V2P-σ plan is INVALIDATED (do NOT spend GPU on it)"** and
  states: *"only V2P element-size Gaussian + ConClick logit-level were feasible — now moot."* `[local]`
- `notes/GOAL.md` §9, Do-not-cite list, final bullet: *"The V2P-σ + ConClick grounding plan as a recommended
  next step — INVALIDATED (attention suppression is not implementable on d1-2B)."* `[local]`
- The *reason* for invalidation is the killer, and it is a measurement the plan itself cites in another
  section: the failure is **element selection**, not precision (`HANDOFF` §2: median miss **437 px ≈ 9
  button-widths**, 86.8% of misses > 150 px, and **80.6% of misses are genuinely different elements**). Hard
  negatives from 68 wrong elements teach the model to avoid *those 68*; a logit-level contrastive term shapes
  the same pointer distribution that already clicks precisely (median 0.13 from the intended centre). Neither
  addresses selection among distractors.
- The plan presents #5 with no reference to the invalidation. A future session reading PLAN will re-run it.

**Recommendation.** Delete work #5. If grounding is to be re-attacked, the entry condition must be a *new*
measurement, not a re-run: e.g. "does a marked-click verifier separate the 71 element-wrong rows from the 152
element-right rows at AUROC ≥ 0.70?" (that is Finding 5's question), and only then a training lever. Add a
one-line cross-reference from PLAN §2 to `HANDOFF §3.3` so the invalidation cannot be re-lost.

---

## Finding 5 — [MAJOR] Work #3 does not break the *zero-output-token* thesis — it breaks the *single-pass* thesis, and it breaks it exactly where the programme's identity lives.

The plan says #3 "is the only approach in the literature that converts disagreement into accuracy rather than
into a threshold", tags it "the only genuinely new idea", and asks for one new readout + GPU. Here is what the
paper actually reports.

**What AdaGround actually is** (arXiv:2610.36874) `[12]`: a training-free zoom portfolio. Two zoom chains run on
every sample; one marked-click Yes/No verifier decides (i) whether to accept a zoom, (ii) which chain to trust,
(iii) whether to re-ground with the rejected point crossed out, and (iv) with an existence query, whether to
abstain. The verifier is
`v(p) = P(Yes)/(P(Yes)+P(No))` over **first-token probabilities** on a window of side
`clip(√(WH/8), 512, 1024)` rendered at **4× area with a red circle drawn at p** [12, §3.3].

**Consequence 1 — the readout is compatible; the pass count is not.** Reading P(Yes) at the first token is
architecturally the *same* thing d1-2B does (a slot readout, no decode). So "zero output tokens" survives. What
does not survive is "single forward pass": each verification is an additional image-conditioned forward pass on
a *newly rendered* image. AdaGround's own cost table quantifies the scale of the pattern — Qwen3-VL-8B on
OSWorld-G: single pass **64.7% at 0.9 s mean / 1.3 s p95, 1.7k visual tokens, 1.0 call**; full pipeline
**66.2% at 24.9 s / 45.8 s p95, 31.9k tokens, 12.5 calls**; and on Qwen3-VL-32B **12.2–13.7 calls per sample,
6.2–6.6 of them sequential**, **3.4–4.2× the latency of the fixed crop**, **6–12× the visual tokens of the
single pass** [12, App. A.4, Table 6].

**Consequence 2 — the honest budget for *one* verifier call on this model.** At 103 ms/pass, one verification
of one candidate ≈ +1 pass ≈ +103 ms *plus* a render of a 4×-area window. If the mark verifier is applied only
where the two checkpoints disagree (25–40 of 398 rows, per Finding 3), the mean latency moves by
`0.08 × 206 ms ≈ +16 ms` — invisible. But p50 for *those rows* doubles and p95 for the step triples. The paper's
latency claim is currently "~103 ms"; after #3 it becomes "~103 ms typical, ~310 ms when the model is unsure".
That is a defensible claim. **"Single pass, 103 ms" is not.** The plan cannot keep both.

**Consequence 3 — the gain is smaller than the headline implies, and the paper says where it goes to zero.**
AdaGround's own ablation sets the ceiling on verifier arbitration: it recovers **38.8% / 35.9%** of the headroom
between the better chain and the oracle on ScreenSpot-Pro [12, §4.3]. Applied to this programme's measured
2-checkpoint grounding headroom of **+5.83 pts** (`fusion-headroom.md` `[local]`), 38% is **≈ +2.2 grounding
pts ≈ +1.2 step pts** — statistically indistinguishable from the **+1.26** the existing free signal
(`box_conf × found`) already delivers on 2 checkpoints. The verifier has to beat 76%-on-21-rows to earn its
extra pass, and the plan's stop rule (**AUROC > 0.65**) is set *below* the incumbent's apparent performance.

**Consequence 4 — the regime mismatch is explicit in the paper.** *"The extra compute pays off where targets are
small. On ScreenSpot-v2, whose targets are large, AdaGround matches the fixed crop at a higher cost"* [12].
AndroidControl is a phone-screenshot benchmark; its eval screenshots are ~1080×2400 with widget-sized targets,
much closer to ScreenSpot-v2's large-target regime than to ScreenSpot-Pro's 4K professional UIs. The programme's
own numbers say the same from the error side: clicks land precisely and *on the wrong element*, median miss 437
px `[local, HANDOFF §2]`. Zooming and verifying helps when a target is too few tokens to resolve. It does not
help when the model has resolved a different, nearby, plausible element — and the paper warns about exactly
that: *"a verifier that prefers a plausible but wrong element passes that preference to every decision it
controls"* [12, §5].

**Consequence 5 — the score saturates.** In AdaGround, *"both scores are at least 0.999 in 45.2–76.9% of TNZ
acceptance decisions, and the two highest scores are [within a small margin] in 10.3–47.2% of arbitrations"*
[12, App. B.1]. A verifier that returns 0.999 for everything has a fine AUROC and no decision value. This is the
failure mode the plan's stop rule (AUROC > 0.65) cannot detect. **A verifier that clears AUROC 0.65 and is
saturated at 0.999 would pass the stop rule and add nothing.** The stop rule must be a *decision* metric, not a
ranking metric — the same lesson `C1` already taught this programme (`GOAL.md` §4: "an AUROC is not a
certificate") `[local]`.

**Consequence 6 — it is not a new idea for mobile.** `V-Droid` (MobiCom 2026, arXiv:2503.15937) is a mobile GUI
agent that uses an LLM as a **verifier over candidate actions before acting**, with a prefilling-only workflow,
reporting **4.3 s/step** and **59.5% on AndroidWorld** [13]. A verifier-in-the-loop for mobile action selection
is published. Work #3 can be a *component* of this programme's paper; it cannot be its novelty claim.

**Recommendation.** Keep #3, but re-scope it to one measurement and change its pass bar:
- **Scope:** one marked-click verifier call on the ~64 rows where the 10 checkpoints disagree. No zoom portfolio,
  no re-grounding, no existence query (AndroidControl has no absent-target class; half of AdaGround's VLR cannot
  transfer).
- **Pass bar (replace "AUROC > 0.65"):** *precision ≥65% on switched rows at ≥60 switched rows* (Finding 3), **and**
  report the score distribution — if >50% of decisions are within 0.01 of 1.0, the signal is saturated and the
  item is dead regardless of AUROC.
- **Report both latencies**: p50 and p95 of passes-per-step, before and after. Retire the phrase "single
  forward pass" from the abstract the moment a verifier is in the loop; "single forward pass, plus one
  verification pass on 16% of steps" is still a good story and it is true.

---

## Finding 6 — [MAJOR] "+2.01 pts on a correlated ensemble, p=0.18" is not publishable as a positive claim at *any* venue — but that is not the reason the plan thinks.

**What TMLR actually requires.** TMLR's acceptance criteria are *"whether the claims made in the submission are
supported by accurate, convincing and clear evidence and whether some individuals in TMLR's audience would be
interested"*, with the first named as the most important [16]; TMLR's own FAQ describes its evaluation as
focusing *"on technical correctness of claims rather than more subjective, editorial measures"* and states that
replications are welcome [17]. Novelty and SOTA are not criteria. ICML separates soundness from impact
(quoted with citations in `outputs/calibrated-decision-value.md` §3 `[local]`).

So the test is not "is +2.01 impressive". The test is **"does the evidence support the claim"**. It does not:

| what a reviewer checks | what the artifact says |
|---|---|
| is the effect distinguishable from zero? | p = 0.18, CI [−1.23, +3.73] (`fusion-headroom.md`) `[local]` |
| was the tolerance fitted honestly? | **No** for the headline: τ was fitted on a **3-model val** and applied to a **10-model test** (`PLAN §2` work #1 concedes this) `[local]` |
| is the comparison matched? | the base is "best single checkpoint" (64.82 groundB) while the ensemble is over 10 — different resource budgets |
| is the claim falsifiable? | at n=398 it is not: MDES at 80% power is **3.0–3.8 pts** (Finding 3) for the observed discordance |
| is it novel? | checkpoint ensembling from one pretrain basin is old, and the diversity literature already says correlated members buy little [19][20] |

The last row is why the "reduce the claim" escape hatch (TMLR tells authors they may reduce claims rather than
add experiments [16]) does not rescue this either: reduced far enough, the claim becomes *"ten checkpoints of one
1.92B model, ensembled, gave +2 pts (n.s.)"* — a single-case study on one benchmark with one model family. The
TMLR guide's negative-result shape requires *"boundary conditions and a constructive lesson"* that generalise;
the programme has the lesson (**the arbiter's data, not the headroom, is the binding constraint**) but only one
setting.

**What WOULD be publishable, in descending order of reach:**
1. **The cost window + certificate boundary** (Finding 9). Decision-level, has the baselines, has two
   checkpoints, has a documented failure. Novel *as a priced decision*, not as a certificate.
2. **The mis-specified score mechanism** — `p(action) × box_conf` is at chance (AUROC **0.504** test / 0.577 val)
   on the 44% of steps that need no coordinates, and type-routing lifts 0.649 → 0.729 free (`GOAL.md` §3 C1)
   `[local]`. That is a mechanism with a predicted magnitude, which is what a reliability paper needs.
3. **A properly powered ensemble null** — "correlated checkpoints do not ensemble on this task: 10 checkpoints,
   80.90% union, every combiner ≤64% precision on ≤28 switched rows". Worth a TMLR submission *only if* it is
   extended past AC-v2 and past d1-2B. As scoped today it is a workshop paper.

Note the ordering: the programme's strongest asset is #1, and the plan does not contain it as a work item at all.

---

## Finding 7 — [MAJOR] Work #2 is a 30-minute wiring job on code that already exists, not "hours" of new work — and the plan's diagnosis of it ("a plain coding bug") is wrong.

**Evidence.** `benchmarks/train/fusion_arbiter.py` already contains `build_pooled()`, whose docstring is verbatim
the plan's proposal: *"The correct scheme for >2 checkpoints: one example per (row × run). The earlier version
asked a mutually-exclusive question ('which single run is right'), which is undefined whenever two or more runs
are right — i.e. on 57% of rows, which were then discarded. Per-run correctness is a well-posed binary label and
uses every row."* `[local]` The function **has no caller**: `main()` uses only the pairwise `build()` path, which
is why the published artifact records `fit_rows: 107` (`2026-11-07-fusion-arbiter.json`) `[local]`.

So: the scheme was not "a bug that discarded rows by accident"; the pairwise path was a deliberate
disagreement-only fit, the pooled path was written and left unwired, and it needs ≥3 checkpoints' **val** dumps
to be meaningful — i.e. it is **blocked on work #1**, not on debugging.

**Two further problems with the plan's framing.**
- *"That is 3980 examples, not 107"* is a sample-size illusion for *evaluation* purposes. The pooled label set has
  3,980 examples, but they are 1,225 rows (827 val + 398 test) × N checkpoints, **mutually dependent**, and the
  decision evaluated is one argmax per row. The effective n stays 398/827, and the McNemar bar of Finding 3 does
  not move. What the pooled scheme genuinely buys is *more usable fit rows on val* (all 827 instead of the ~107
  disagreement rows) — that is real and worth doing, but it is a fit-side fix, not a power fix.
- The plan's stop rule ("if CV accuracy ≤ best single, ensembling is dead") measures base-rate-dominated
  accuracy. Replace with precision on switched rows (Finding 3).

**Recommendation.** Merge work #2 into work #1 as its second step. Sequence: (1) dump val for the missing 7
checkpoints; (2) wire `build_pooled` and fit on all 827 val rows × N; (3) evaluate precision/recall of switching
on test once. Cost: ~20 min GPU + ~1 h CPU.

---

## Finding 8 — [MAJOR] The programme's "only defensible claim" is no longer unclaimed: CORA (Apr 2026) publishes conformal risk-controlled execute/abstain for mobile GUI agents, with coverage, harm and interruption numbers.

**Evidence.** `CORA: Conformal Risk-Controlled Agents for Safeguarded Mobile GUI Automation` (arXiv:2604.09155,
10 Apr 2026) is *"a post-policy, pre-action safeguarding framework that provides statistical guarantees on
harmful executed actions"*: a Guardian model scores action-conditional risk, **Conformal Risk Control calibrates
an execute/abstain boundary against a user-specified risk budget**, and rejected actions route to a Diagnostician
with `{Abort, Confirm, Reflect}` interventions; evaluated on a new **Phone-Harm** benchmark (Harm-150 + Normal-150)
and on public benchmarks [15]. Reported: at δ = 0.05 (n=1075), τ̂ = 0.952, **coverage 89.95%**, executed-harm
**2.42%**, Diagnostician invocation **10.05%**; GAR 79.80%, IF1 85.29%, OIR 19.23% on Harm-150; AndroidWorld
**40.0%** vs AutoGLM 30.0% [15].

This directly overlaps the three sentences the plan calls its strongest: "certified selective risk", "act/ask/
refuse", "on mobile GUI". `SafeGround` (arXiv:2602.02419, Feb 2026) overlaps the certificate half from the
grounding side: spatial uncertainty + Learn-Then-Test FDR calibration, **+5.38 pp over Gemini-only inference** on
ScreenSpot-Pro [14].

**What survives — and it is narrow but real.** I read CORA's own framing for a cost model: it reports the
*interruption rate* (OIR) as a consequence of the budget, and gives no cost ratio, no C_err/C_ask arithmetic, and
no comparison against *always-ask* as a policy. Its guarantee is on harm, not on step correctness. So the
surviving delta for this programme is precisely the thing that already exists in
`benchmarks/results/2026-11-07-cost-window.md`: **a priced ask.** The value brief's claim that "no GUI paper
prices the ask" survives contact with CORA — but only just, and only if the paper states the delta explicitly
and cites CORA in the first two pages. **The gate is not the contribution any more. The price is.**

Also note: CORA is an arXiv preprint (v1, Apr 2026); venue unstated on the abstract page [15]. That does not
make it ignorable — it will be cited against this paper.

**Recommendation.** Rewrite the paper's positioning paragraph to name CORA and SafeGround and to concede the
certificate. Then re-state the contribution as the cost window + the certificate's *checkpoint-locality* (17.3%
→ 23.9% under the same threshold `[local, GOAL.md §4]`), which neither paper measures.

---

## Finding 9 — [MINOR/MAJOR] Internal claim hygiene: two numbers in the plan and the boilerplate would not survive a reviewer, and one is checkable against a leaderboard in five minutes.

1. **"Type accuracy 84.17% — best reported on the board"** (`PLAN §0`, `HANDOFF §2`) `[local]`. True against
   Mem-W's Table 2 (81.16) but *not* against the wider board: the aggregated AndroidControl-High table lists
   **SE-GA 84.4** and **OS-Atlas 85.2** action-type accuracy [11]. Protocol differs, which is the point —
   the claim needs the qualifier "on the AC-v2 398-row protocol".
2. **The same +2.01 appears with p=0.077 and p=0.18** in the same table (`PLAN §1c`). Exact McNemar on 18/10 is
   **p=0.185**. Delete the 0.077 row.
3. **A 2B-class comparison is missing and cheap.** Mem-W's aggregate lists **Qwen3-VL-2B at 45.48 Pass@1 / 64.07
   Type / 59.64 Grounding** on AC-v2-High [10]. If that row is right, (a) this programme's +18.6 step accuracy
   over a same-size general VLM is a *far* better headline than "tie with an 8B", and (b) its grounding (56.05)
   is 3.6 points *below* an untuned 2B general model — which is a finding about what the fine-tune traded away.
   **I read that row only from an aggregator, not from a primary table — verify before citing.** If it holds,
   it changes the paper's framing for the better and makes Finding 5's zoom argument live again (those are small
   Android targets).

---

## Question 4 — Rank the five work items by expected value per GPU-hour

**Method.** GPU-hours are the plan's own estimates. "EV" is *expected change in a publishable quantity*, not
"knowledge gained". Where a work item's ceiling is bounded by an artifact, the bound is stated. I also flag where
the ratio is degenerate, because two of the five items cost ~0 GPU and would sort infinitely otherwise.

| rank | item | GPU-h (plan) | measured/derivable ceiling | EV/GPU-h | notes |
|---|---|---|---|---|---|
| 1 | **#1 val dumps for 10 checkpoints** | 0.33 | no accuracy gain; *unblocks* #2 and makes the only honest ensemble number interpretable | **highest** — the only item whose absence invalidates a number already quoted | also the cheapest way to kill the ensemble line (§2 guard rail 3) |
| 2 | **#2 pooled arbiter (wired, not written)** | ~0 (needs #1) | +16.08 oracle; realistic +6.0 if precision ≥70% on ~60 switched rows | **very high, degenerate ratio** | code exists (`build_pooled`, no caller); stop rule must change to precision |
| 3 | **#3 marked-click verifier** | ~0.2 for eval (398×2 passes ≈ 2 min) + days of engineering | 38% of a +5.83 grounding headroom ≈ **+1.2 step pts** [12] | **high on GPU-hours, misleading** — cost is engineer-days and the single-pass claim | only item that adds a new *signal* (an abstain score) rather than a combiner |
| 4 | **#4 decorrelated checkpoint** | 1.5 | raises the disagreement surface — the *actual* constraint (only 29 rows have exactly 1-of-3 right today) | **medium** | it is the enabler for #2's ceiling; plan puts it *after* #2, which is backwards |
| 5 | **#5 grounding levers (ConClick / box σ)** | 6.0 | invalidated (`HANDOFF §3.3`, `GOAL.md §9`); addresses precision, measured failure is selection | **lowest** | 18× the cost of #1 for a lever the programme already retired |

**I would delete #5 outright** (Finding 4) — 6 GPU-hours to re-run a plan the programme's own entry-point document
stamps "do NOT spend GPU on it".

**I would also, if one more cut is needed, reverse the order of #2 and #4.** The plan's own §1d is the argument:
*"only 29 rows have exactly one of the three right"*. The arbiter's precision is computed over the rows where
checkpoints disagree, and that set is 25–29 rows today. Work #2 cannot reach 60 switched rows without more
disagreement, and disagreement is what #4 manufactures. As ordered, #2 is being asked to score on a surface too
small to pass Finding 3's bar.

---

## Question 5 — What is the strongest paper this programme can actually publish in three months?

Not an ensemble paper. Not an accuracy paper. **The honest answer is "pivot the claim, not the benchmark"** —
and the pivot is to the thing the plan does not list as a work item.

**Title:** *Priced Interruptions: Cost-Anchored Risk Control for a Single-Pass GUI Decision Model, and the
Boundary of Its Certificate*

**One-sentence claim:** A 1.92B single-pass decision model that ties an 8B baseline on step accuracy nonetheless
carries a usable calibrated signal — its type-routed score certifies a 20% selective-risk budget at 40.7% test
coverage, and gating on it beats both always-act and always-ask for error-to-interruption cost ratios in
**2.11–5.79×**, a window that survives a checkpoint change (2.08–4.19×) while the *certificate* does not
(17.3% → 23.9% risk at the same threshold).

**Why this and not the ensemble.** It is the only claim in the programme that (a) does not depend on beating an
unrankable benchmark (Finding 2), (b) is decision-level rather than a 2-point step delta (Finding 3), (c) has a
documented failure with a mechanism (conformal transfer), and (d) has a surviving delta against CORA/SafeGround —
namely that **none of them prices the ask** (Finding 8). TMLR's criteria fit it exactly: claims supported by
evidence, novelty not required, modest contributions explicitly acceptable [16][17].

**Minimum evidence (what must exist on submission day):**

| # | evidence | status |
|---|---|---|
| 1 | risk-at-coverage curve + Clopper–Pearson bound on ≥2 checkpoints, with the val-side/test-side distinction stated | **have it** (`cost-window.md`, `GOAL.md` §4) `[local]` |
| 2 | cost window on both checkpoints, with the C_ask assumption stated as an assumption | **have it** (2.11–5.79 / 2.08–4.19) `[local]` |
| 3 | the certificate failure: same threshold, 17.3% → 23.9% under checkpoint change | **have it** `[local]` |
| 4 | baselines at matched coverage: always-act, always-ask, `p(action)` product (the mis-specified score), random-confidence | **mostly have**; the product score is the mechanism and is at chance (0.504) `[local]` |
| 5 | **a measured ask-acceptance rate.** The whole cost argument prices an interruption nobody has measured for a GUI gate. | **missing — this is the 3-month critical path** |
| 6 | protocol-matched accuracy baselines run by us (Qwen3-VL-2B and -8B through our scorer, ~1 GPU-h) | **missing, cheap, removes the cross-paper comparison entirely** |
| 7 | the negative results as the boundary (status gate, per-class adjustment, learned fusion, status arm) | **have it** — 4 measured negatives, all with paired tests `[local]` |
| 8 | p50/p95 latency per step, measured on a pinned GPU | partially (103–130 ms batched, 4090) `[local]` |

**Item 5 is the paper.** If the 3 months buy exactly one new measurement, buy that one: even an n=20 internal
study of "would you accept this suggestion" under the gate's operating point, plus a sensitivity sweep over the
literature's 49–96% override range (cited in `outputs/calibrated-decision-value.md` §3 `[local]`). Without it, the
cost window is arithmetic over an assumed constant, and a reviewer will say so — the programme's own value brief
already says so (§3, "What a builder needs").

**Venue.** TMLR, not ICML. ICML 2027 with a January abstract deadline is 2.5 months from now and this paper needs
a human-subjects measurement it does not have; the accuracy tie guarantees a soundness objection at a conference
that rewards impact. TMLR's rolling review (≈9 weeks target [17]) tolerates a modest, honest, well-instrumented
result — which is what this is.

**If the answer has to be "none, pivot", the pivot is:** stop writing a GUI-agent paper and write the measurement
paper that the evidence already supports — *"what a 398-row GUI benchmark can and cannot certify"* — combining
the protocol-spread measurement (Finding 2: 49.62 vs 54.80 for one backbone), the power analysis (Finding 3:
MDES 3.0–3.8 pts at n=398), and the four measured negatives. That paper needs **no GPU at all**, is defensible
from artifacts already on disk, and is more useful to the field than +2 points on a benchmark whose own numbers
disagree by five. It is also the least likely to be wrong.

---

## Coverage Status

**Checked directly (read, not inferred):**
- `PLAN-2026-11-07-better.md`, `docs/HANDOFF.md`, `notes/GOAL.md`, `docs/research/PRE-REG-2026-11-07-status-downweight.md`,
  `benchmarks/results/2026-11-07-fusion-headroom.md`, `2026-11-07-fusion-arbiter.json`, `2026-11-07-status-downweight-arm.md`,
  `2026-11-07-what-the-arm-bought.md`, `2026-11-07-cost-window.md`, `2026-11-07-status-gate.json`,
  `benchmarks/train/fusion_arbiter.py`, `outputs/calibrated-decision-value.md` §3–§4. All `[local]`.
- AdaGround full text (arXiv:2610.36874): components, verifier definition, Table 1/2/3, Table 6 cost rows,
  App. A.4, App. B.1/B.5, App. D.2.
- CORA abstract + HTML extract (arXiv:2604.09155): design, δ=0.05 numbers, GAR/OIR/HR, AndroidWorld numbers,
  absence of a cost ratio.
- SafeGround abstract (arXiv:2602.02419).
- V-Droid abstract (arXiv:2503.15937): verifier-driven mobile agent, 4.3 s/step, 59.5% AndroidWorld.
- Mem-W: AC-v2 evaluation set = **398 cleaned samples, filtered from 500**; table values for Qwen3-VL-4B/8B
  (extracted via targeted Q&A over the HTML, not a full manual read of Table 2).
- ECHOES full Table 1 (AC-v2-High column) at arXiv:2610.22408.
- AndroidControl-Curated abstract (arXiv:2510.18488).
- A3 ACL Findings 2026 abstract.
- TMLR official FAQ (jmlr.org/tmlr/faq.html) and TMLR blog post on acceptance criteria.
- GUI-Libra arXiv metadata (v1 25 Feb 2026) and its AndroidControl eval README (500-step sample provenance).
- Google Research `android_control` README (dataset fields, action space).
- Power/McNemar computations: my own, in Python (scipy), calibrated by reproducing the programme's published
  p-values to two decimals.

**Uncertain / needs follow-up:**
- The Qwen3-VL-2B row (45.48 / 64.07 / 59.64) comes from an **aggregator** (sota2), not a primary table. Do not
  cite until verified (Finding 9.3).
- CORA's venue is unstated on its arXiv page; it may be under review somewhere. Treat as a live preprint, not a
  published priority claim.
- I did not read Mem-W's Table 2 directly (extracted via targeted Q&A). The 68.59 / 81.16 / 71.30 triple matches
  two independent aggregators and `HANDOFF §2`, so I treat it as reliable; the per-baseline rows are less certain.
- I did not verify whether any 2026 paper reports a *checkpoint-ensemble* result on a GUI benchmark specifically.
  My negative claim about novelty (Finding 6) is about ensembling generally, supported by the diversity
  literature, not by an exhaustive GUI-specific search.
- `AdaGround` is a preprint (Oct 2026); no venue. Its ScreenSpot-v2 "no gain at higher cost" result is the one
  I lean on hardest for Finding 5 and it is a single-paper claim, not replicated.

**Not done (out of scope for this angle):** internal-consistency review of the training recipe (brief 1), the
statistical scorer (brief 2), the write-up (brief 4).

---

## Sources

1. Guo et al., *ECHOES: Seeing Beyond the Current Image via Recursive History Compression and Privileged
   On-Policy Distillation*, arXiv:2610.22408 (14 Oct 2026) — https://arxiv.org/html/2610.22408
2. Zhang et al., *Mem-W: Latent Memory-Native GUI Agents*, arXiv:2605.09317 (May 2026) —
   https://arxiv.org/abs/2605.09317
3. Yang et al., *GUI-Libra: Training Native GUI Agents to Reason and Act…*, arXiv:2602.22190 (v1 25 Feb 2026) —
   https://arxiv.org/abs/2602.22190
4. *MobileWorld: Benchmarking Autonomous Mobile Agents…*, arXiv:2512.19432 —
   https://arxiv.org/html/2512.19432v3
5. Chai et al., *A3: Android Agent Arena for Mobile GUI Agents with Essential-State Procedural Evaluation*,
   Findings of ACL 2026, pp. 3774–3789 — https://aclanthology.org/2026.findings-acl.184/
6. *AndroidFlux: Evaluating and Reward Modeling Failure Recovery Capabilities in Mobile Use Multimodal Agents*,
   arXiv:2610.34384 — https://arxiv.org/abs/2610.34384
7. *PhoneWorld: Scaling Phone-Use Agent Environments*, arXiv:2605.29486 —
   https://arxiv.org/html/2605.29486v1
8. Leung, Xi & Zuo, *AndroidControl-Curated: Revealing the True Potential of GUI Agents through Benchmark
   Purification*, arXiv:2510.18488 — https://arxiv.org/abs/2510.18488
9. Papers with Code, *AndroidControl leaderboard* (AndroidControl-High, official test split) —
   https://paperswithcode.co/benchmark/androidcontrol
10. SOTA² Research, *Action Prediction on AndroidControl High v2 leaderboard* —
    https://www.sota2.com/research/sota/action-prediction-on-androidcontrol-high-v2
11. SOTA² Research, *GUI planning on AndroidControl High leaderboard* —
    https://www.sota2.com/research/sota/gui-planning-on-androidcontrol-high
12. Wang et al., *AdaGround: Token-Normalized Adaptive Zooming with Verifier-in-the-Loop Refinement for GUI
    Grounding*, arXiv:2610.36874 — https://arxiv.org/html/2610.36874
13. *V-Droid: Advancing Mobile GUI Agent Through Generative Verifiers*, arXiv:2503.15937 (MobiCom 2026) —
    https://arxiv.org/abs/2503.15937
14. *SafeGround: Know When to Trust GUI Grounding Models via Uncertainty Calibration*, arXiv:2602.02419 —
    https://arxiv.org/abs/2602.02419
15. Feng et al., *CORA: Conformal Risk-Controlled Agents for Safeguarded Mobile GUI Automation*, arXiv:2604.09155
    (10 Apr 2026) — https://arxiv.org/abs/2604.09155
16. TMLR Blog, *Increasing Emphasis on Clear Writing in Acceptance Criteria* (2026) —
    https://blog.tmlr.org/2026/increasing-emphasis-on-clear-writing-in-acceptance-criteria/
17. TMLR, *Frequently Asked Questions* — https://jmlr.org/tmlr/faq.html
18. GUI-Libra repo, `evaluation/offline_evaluation/AndroidControl/README.md` (500-step sample provenance) —
    https://github.com/GUI-Libra/GUI-Libra/blob/main/evaluation/offline_evaluation/AndroidControl/README.md
19. Abe et al., *Pathologies of Predictive Diversity in Deep Ensembles*, arXiv:2302.00704 —
    https://arxiv.org/html/2302.00704v3
20. *Deep Ensembles Work, But Are They Necessary?*, arXiv:2202.06985 — https://arxiv.org/abs/2202.06985
21. Google Research, *AndroidControl* dataset README (fields, action space, splits) —
    https://github.com/google-research/google-research/blob/master/android_control/README.md

Local artifacts (no URL): `docs/research/PLAN-2026-11-07-better.md`, `docs/HANDOFF.md`, `notes/GOAL.md`,
`docs/research/PRE-REG-2026-11-07-status-downweight.md`, `benchmarks/results/2026-11-07-fusion-headroom.md`,
`benchmarks/results/2026-11-07-fusion-arbiter.json`, `benchmarks/results/2026-11-07-status-downweight-arm.md`,
`benchmarks/results/2026-11-07-what-the-arm-bought.md`, `benchmarks/results/2026-11-07-cost-window.md`,
`benchmarks/results/2026-11-07-status-gate.json`, `benchmarks/train/fusion_arbiter.py`,
`outputs/calibrated-decision-value.md`.

---

## Evidence table

| # | Source | URL | Key claim used here | Type | Confidence |
|---|---|---|---|---|---|
| 1 | ECHOES, arXiv:2610.22408 (Oct 2026) | https://arxiv.org/html/2610.22408 | AC-v2-High still reported; field spread 46.9–53.6; Qwen3-VL-8B = 49.62 | primary | high |
| 2 | Mem-W, arXiv:2605.09317 | https://arxiv.org/abs/2605.09317 | AC-v2-High = 398 cleaned samples; Mem-W-8B 68.59/81.16/71.30 | primary | high |
| 3 | GUI-Libra, arXiv:2602.22190 | https://arxiv.org/abs/2602.22190 | AC-v2 = cleaned AndroidControl; origin of the protocol | primary | high |
| 4 | MobileWorld, arXiv:2512.19432 | https://arxiv.org/html/2512.19432v3 | AndroidWorld saturated >90%; field moving online | primary | medium |
| 5 | A3, Findings of ACL 2026 | https://aclanthology.org/2026.findings-acl.184/ | static-frame benchmarks like AndroidControl "fail to capture" real performance | primary (peer-reviewed) | high |
| 6 | AndroidFlux, arXiv:2610.34384 | https://arxiv.org/abs/2610.34384 | 2026 evaluation effort concentrates on online/AndroidWorld | primary | medium |
| 7 | PhoneWorld, arXiv:2605.29486 | https://arxiv.org/html/2605.29486v1 | environment-scale benchmarks are where 2026 effort goes | primary | medium |
| 8 | AndroidControl-Curated, arXiv:2510.18488 | https://arxiv.org/abs/2510.18488 | AC has "ambiguities and factual errors" that underrate agents; +15% after curation | primary | medium |
| 9 | paperswithcode AndroidControl board | https://paperswithcode.co/benchmark/androidcontrol | AndroidControl-High tops at 86.1 (UItron-72B), 79.1 (OmegaUse) | secondary (aggregator) | medium |
| 10 | SOTA² AndroidControl High v2 board | https://www.sota2.com/research/sota/action-prediction-on-androidcontrol-high-v2 | Mem-W Table-2 aggregates incl. Qwen3-VL-8B 54.80, Qwen3-VL-2B 45.48 | secondary (aggregator) | medium-low for the 2B row |
| 11 | SOTA² GUI planning AC-High board | https://www.sota2.com/research/sota/gui-planning-on-androidcontrol-high | SE-GA type acc 84.4, OS-Atlas 85.2 — above our "best on board" 84.17 | secondary (aggregator) | medium |
| 12 | AdaGround, arXiv:2610.36874 | https://arxiv.org/html/2610.36874 | verifier = first-token Yes/No on 4×-area marked window; 12.2–13.7 calls/sample; 3.4–4.2× latency; 6–12× tokens; no gain on large-target benchmarks; verifier scores saturate ≥0.999; arbitration recovers 38.8%/35.9% of headroom | primary | high |
| 13 | V-Droid, arXiv:2503.15937 | https://arxiv.org/abs/2503.15937 | verifier-over-candidates already published for mobile GUI: 4.3 s/step, 59.5% AndroidWorld | primary | high |
| 14 | SafeGround, arXiv:2602.02419 | https://arxiv.org/abs/2602.02419 | FDR-calibrated execute/abstain for GUI grounding, +5.38 pp on ScreenSpot-Pro | primary | high |
| 15 | CORA, arXiv:2604.09155 | https://arxiv.org/abs/2604.09155 | conformal risk-controlled execute/abstain for mobile GUI; δ=.05 → coverage 89.95%, harm 2.42%, OIR 10.05%; **no cost ratio** | primary (preprint) | high |
| 16 | TMLR blog, acceptance criteria | https://blog.tmlr.org/2026/increasing-emphasis-on-clear-writing-in-acceptance-criteria/ | criteria = claims supported by accurate/convincing/clear evidence + audience interest | primary (venue) | high |
| 17 | TMLR FAQ | https://jmlr.org/tmlr/faq.html | "technical correctness of claims rather than more subjective, editorial measures"; replications welcome; ~9-week decision | primary (venue) | high |
| 18 | GUI-Libra eval README | https://github.com/GUI-Libra/GUI-Libra/blob/main/evaluation/offline_evaluation/AndroidControl/README.md | 398 rows derive from UGround's `500_steps` sample | primary (code) | high |
| 19 | Pathologies of Predictive Diversity, arXiv:2302.00704 | https://arxiv.org/html/2302.00704v3 | diversity interventions help small ensembles, harm large ones | primary | medium |
| 20 | Deep Ensembles Work, But Are They Necessary?, arXiv:2202.06985 | https://arxiv.org/abs/2202.06985 | ensemble gains track single-model improvement more than diversity | primary | medium |
| 21 | Google Research AndroidControl README | https://github.com/google-research/google-research/blob/master/android_control/README.md | dataset fields, action space, licence | primary (code) | high |
| 22 | Local artifacts (see list above) | `benchmarks/results/*`, `notes/GOAL.md`, `docs/HANDOFF.md` | all programme numbers, discordant counts, GPU costs, retractions | primary (internal) | high |

---

```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "Wrote benchmarks/results/2026-11-07-review-strategy.md (9 findings, 4 FATAL, severity-tagged, each with evidence + recommendation), answering all five assigned strategy questions: (1) AC-v2 still reported in Oct-2026 work but protocol-unstable by 5.2 pts on an identical backbone (ECHOES 49.62 vs Mem-W aggregate 54.80 for Qwen3-VL-8B) so it cannot serve as a ranking target; (2) +2.01 pts at p=0.18 is not publishable and the binding constraint is statistical (MDES 3.0-3.8 pts at n=398), with a table of what would be; (3) AdaGround verifier judged against its own cost table (12.2-13.7 calls/sample, 3.4-4.2x latency, 6-12x visual tokens, no gain on large-target benchmarks) - breaks single-pass but not zero-output-token, expected gain ~+1.2 step pts; (4) five work items ranked with GPU-hours, #5 deleted and the #2/#4 order reversed; (5) concrete title + one-sentence claim + 8-row minimum-evidence table, with a no-GPU fallback paper."
    }
  ],
  "changedFiles": [
    "/Users/macbookpro/.pi/agent/sessions/--Users-macbookpro-Workspace-research--/subagent-artifacts/outputs/d7e2bc21-145a-41c3-90cd-0d56b24d0bee/benchmarks/results/2026-11-07-review-strategy.md"
  ],
  "testsAddedOrUpdated": [],
  "commandsRun": [
    {
      "command": "python3 (scipy) exact McNemar recalibration vs published p-values",
      "result": "passed",
      "summary": "Reproduced the programme's own p-values from b/c counts to 2 dp: 18/10 -> 0.1849 (reported 0.18), 15/10 -> 0.4244 (0.42), 49/58 -> 0.4394 (0.44). Model validated before being used for power claims."
    },
    {
      "command": "python3 (scipy) power / MDES scan at n=398 and n=827",
      "result": "passed",
      "summary": "At n=398, observed discordance 28 gives 80%-power MDES 3.8 pts and power 0.28 for a true +2.01; p<.05 needs net >= 12 rows (71% precision). At n=827 MDES is 2.6 pts."
    },
    {
      "command": "grep -n 'build_pooled\\|build(' benchmarks/train/fusion_arbiter.py",
      "result": "passed",
      "summary": "Confirmed build_pooled() is defined at line 73 and has no caller (main() uses build() at 172/176), verifying that plan work #2 is an unwired function, not new code."
    }
  ],
  "validationOutput": [
    "McNemar recalibration: b=18 c=10 nd=28 net=+8 delta=+2.01 exact p=0.1849 (matches artifact 0.18); b=15 c=10 p=0.4244 (0.42); b=41 c=28 p=0.1480; b=49 c=58 p=0.4394 (0.44).",
    "Precision table: nd=28 needs 12 net rows and 71.4% precision for p<.05; nd=60 needs 18 net and 65.0%; nd=100 needs 22 net and 61.0%.",
    "Measured combiners on switched rows: maj-vote +-8px 64.3% precision, 2-ckpt fusion 60.0%, learned arbiter 50.0% -- all under the bar.",
    "AdaGround cost rows read from Table 6 / App. A.4: OSWorld-G 8B single pass 64.7% @ 0.9s mean / 1.3s p95, 1.7k tokens, 1 call; full pipeline 66.2% @ 24.9s / 45.8s, 31.9k tokens, 12.5 calls; 32B 12.2-13.7 calls/sample with 6.2-6.6 sequential.",
    "grepped fusion_arbiter.py: build_pooled defined line 73, never called."
  ],
  "residualRisks": [
    "The Qwen3-VL-2B AC-v2 row (45.48/64.07/59.64) comes from an aggregator (sota2), not a primary table; it is flagged in the document as cite-only-after-verification. If wrong, Finding 9.3 must be dropped (it does not affect Findings 1-8).",
    "Mem-W's per-baseline table values were extracted by targeted Q&A over the HTML rather than a full manual read of Table 2; the headline 68.59/81.16/71.30 triple is corroborated by two independent aggregators and by docs/HANDOFF.md Section 2, so it is treated as reliable.",
    "CORA (arXiv:2604.09155) is a preprint with no stated venue. If it is published at a venue with a real review record, the scoop risk in Finding 8 is stronger than stated, not weaker.",
    "The power analysis assumes independent rows. Rows cluster into 350 episodes, so the effective n is slightly below 398 and the true MDES is marginally worse than computed - which only strengthens Findings 3 and 4, but the exact figure is not episode-clustered.",
    "I did not run an exhaustive GUI-specific search for checkpoint-ensembling results; the novelty claim in Finding 6 rests on the general deep-ensemble diversity literature plus the absence of such work in the sources checked.",
    "I did not verify whether GUI-Libra or any other paper reports a distinct per-paper AC-v2 test size beyond the 398 subset; GUI-Libra's own arXiv abstract page did not state it, so the '398 is a shared convention among GUI-Libra/Mem-W' statement rests on Mem-W's explicit description plus the GUI-Libra eval README."
  ],
  "noStagedFiles": true,
  "diffSummary": "One new file: a 9-finding adversarial strategy review of PLAN-2026-11-07-better.md, with evidence table (22 numbered sources, 21 with URLs), a McNemar-based power/MDES analysis recomputed from the programme's own artifacts, a 5-item EV-per-GPU-hour ranking with a delete recommendation, and a concrete 3-month paper proposal with a minimum-evidence table and a no-GPU fallback.",
  "reviewFindings": [
    "blocker: PLAN Section 0 fact 1 ('type is nearly solved, placement is not') is a retracted claim listed in notes/GOAL.md Section 9 do-not-cite; the plan's ordering rests on it.",
    "blocker: the ensemble cannot reach significance at n=398 - observed discordance 28 gives 80%-power MDES 3.8 pts, and p<.05 requires 71% precision on switched rows versus 60-64% achieved.",
    "blocker: PLAN work #5 re-introduces the V2P-sigma/ConClick grounding plan that docs/HANDOFF.md Section 3.3 marks 'INVALIDATED (do NOT spend GPU on it)'.",
    "blocker: the 'gated decision' contribution is no longer unclaimed - CORA (arXiv:2604.09155, Apr 2026) publishes conformal risk-controlled execute/abstain for mobile GUI with coverage/harm numbers; only the priced-ask framing survives.",
    "major: work #3 breaks the single-pass latency claim (AdaGround: 12.2-13.7 calls/sample, 3.4-4.2x latency) though not the zero-output-token claim, and its expected gain (~+1.2 step pts) is no better than the free box_conf x found signal already measured.",
    "major: work #2's pass condition ('CV accuracy <= best single') measures base-rate-dominated accuracy rather than precision on the rows the arbiter switches; and the pooled label scheme it proposes is already implemented as dead code (build_pooled, no caller) and blocked on work #1.",
    "minor: PLAN Section 1c lists the same +2.01 with p=0.077 and p=0.18; exact McNemar on 18/10 is 0.185, so the 0.077 row is wrong. Also '84.17 Type is best on the board' is unsafe without the 'AC-v2 398-row protocol' qualifier (SE-GA 84.4 / OS-Atlas 85.2 appear on other AndroidControl tables)."
  ],
  "manualNotes": "The plan's guard rails (never gate on gold; fit on val, score test once; report the denominator and the losses) are genuinely excellent and should survive any re-write - my criticism is of what the plan aims at, not of how carefully it has been written. The single most actionable output is the precision/switch-count bar in Finding 3: it converts an unfalsifiable '+2 points is not significant' complaint into a concrete engineering target (>=65% precision on >=60 switched rows), and it explains why the learned arbiter scored exactly +0.00 on test despite +2.90 on val. Second most actionable: work #5 must be deleted, and the plan should cross-reference HANDOFF Section 3.3 so the invalidation is not lost again."
}
```
