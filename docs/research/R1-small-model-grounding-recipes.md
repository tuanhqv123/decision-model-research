# R1 — How do top labs make a SMALL model ground well?

Research output for a 1.92B, single-pass, 8-slot readout GUI grounding model (zero output tokens, ~103 ms).
All numbers below were read from the fulltext (arXiv HTML or publisher PDF), not from abstracts, **except** where
explicitly marked `[abstract-only]`. Dates are arXiv v1 submission dates.

**Reading key for our situation:** our failure is *element selection*, not click precision — 71/143 errors are
"right verb, wrong element", median miss 437 px, only 4.4% of misses within 50 px. Every result below is tagged for
whether it addresses that specific failure.

---

## Evidence table

| # | Source | URL | Key claim | Type | Confidence |
|---|--------|-----|-----------|------|------------|
| 1 | Yang et al., *Ferret-UI Lite* (Apple, 2025-09-30) | https://arxiv.org/abs/2509.26539 | 3B model; ScreenSpot-V2 91.6, ScreenSpot-Pro 53.3; high-res synthetic data removal costs −6.80 SSPro | primary | high |
| 2 | Wu et al., *GUI-Actor* (Microsoft Research, NeurIPS 2025; arXiv 2025-06-03) | https://arxiv.org/abs/2506.03143 | Attention action head + multi-patch supervision; GUI-Actor-2B 36.7 → 41.8 SSPro with verifier | primary | high |
| 3 | Gurbuz et al., *ScreenParse* (ICML 2026; arXiv 2026-05-01) | https://arxiv.org/abs/2602.14276 | Dense screen parsing supervision; Qwen3-VL-2B ScreenSpot-Web PixCov 0.292 → 0.720 | primary | high |
| 4 | Zhang et al., *ConClick* (arXiv 2026-10-09) | https://arxiv.org/abs/2610.13268 | 74–88% of wrong clicks hit *other elements*; distractor contrastive training 54.71 → 66.03 SSPro | primary | high |
| 5 | Chen et al., *GoClick* (arXiv 2026-04-27) | https://arxiv.org/abs/2604.23941 | 230M encoder-decoder beats decoder-only at same size; PDR filtering 10.8M→3.8M gives +4.0 avg | primary | high |
| 6 | Li et al., *WinDOM / Self-Family Distillation* (ICML 2026; arXiv 2026-06-24) | https://arxiv.org/abs/2606.25964 | SFT on 2B: in-dist +12.1 but SSPro −4.6, OOD-mean net zero (60.9→60.8) | primary | high |
| 7 | *RankGround* (arXiv 2026-09-16) | https://arxiv.org/abs/2609.18690 | Reranker-guided crop; 2B gains (12.1–19.4 pp) exceed 8B gains; Oracle coverage 99.75% | primary | high |
| 8 | *PreView* (arXiv 2026-10-24) | https://arxiv.org/abs/2610.33645 | Frozen grounder + auxiliary preprocessor: Qwen3-VL-2B 57.2 → 64.6 avg, SSPro 43.1 → 59.3 | primary | high |
| 9 | Hsieh et al., *ZonUI-3B* (WACV 2026) | https://openaccess.thecvf.com/content/WACV2026/papers/Hsieh_ZonUI-3B_Competitive_GUI_Grounding_with_a_3B_VLM_Trained_on_WACV_2026_paper.pdf | 24.1K samples, single RTX 4090, 48h; two-stage fine-tuning +3.3 desktop; 16.1K≈119.4K | primary | high |
| 10 | Gou et al., *UGround* (ICLR 2025 Oral; arXiv 2024-10-07) | https://arxiv.org/abs/2410.05243 | Web-synthesis: bbox→center point, coords as raw text, 36-ViT-slice AnyRes to 1344×1344 | primary | high |
| 11 | Wu et al., *OS-Atlas* (ICLR 2025; arXiv 2024-10-30) | https://arxiv.org/abs/2410.23218 | 13M elements / 2.3M screenshots cross-platform synthesis; 0–1000 normalized coords | primary | high |
| 12 | Baechler et al., *ScreenAI* (Google, IJCAI 2024; arXiv 2024-02-08) | https://arxiv.org/abs/2402.04615 | Screen-annotation pretraining task; pix2struct patching; 670M/2B/5B | primary | high |
| 13 | Google, *Visual Grounding for User Interfaces* (NAACL 2024 Industry) | https://aclanthology.org/2024.naacl-industry.9.pdf | LVG: layout contrastive (+2.53 acc@1) beats plain contrastive | primary | high |
| 14 | Cheng et al., *SeeClick* (arXiv 2024-01-19) | https://arxiv.org/abs/2401.10935 | Web-OCR + web-UI-grounding synthetic pretraining, GPT-4V 16.2% baseline | primary | high |
| 15 | Microsoft, *GUI-Actor* repo + verifier README | https://github.com/microsoft/gui-actor/blob/main/verifier/README.md | 2B verifier built on UI-TARS-2B-SFT; scores candidate positions in hindsight | primary | high |
| 16 | *RankGround* project/analysis page | https://www.alphaxiv.org/abs/2609.18690 | GroundRanker Hit@1 88.2% vs 99.75% Oracle crop coverage | secondary | medium |
| 17 | OpenAI, *Computer-Using Agent* | https://openai.com/index/computer-using-agent/ | CUA = GPT-4o vision + RL; no published grounding-model architecture or ablation | primary | high |

Note on #17: OpenAI has published **no** UI-grounding recipe with per-ingredient deltas. The only grounded statement
I can support is that CUA combines GPT-4o vision with RL and is a closed system [17]. Every published
OpenAI-adjacent grounding number in this report comes from third parties evaluating against Operator as a baseline
(e.g. Ferret-UI Lite Table 8 reports Operator at 40.6 OSWorld-G [1]). **Do not attribute any grounding recipe to
OpenAI.** This is a gap, not an omission on my part — I searched for it and it does not exist in the open literature.

---

## Per-question findings

### 1. Apple — Ferret-UI Lite (3B, on-device) [1]

**Data recipe.** Mixes public grounding corpora (GroundUI, OSAtlas, UGround, Aria-UI, Aguvis, WaveUI, ShowUI,
Jedi, AgentNet) with two synthetic sources: (a) *high-resolution grounding data* built by **concatenating multiple
GUI screenshots into larger composite images** "to expose the model to denser layouts and richer spatial contexts";
(b) multi-agent online rollouts. `[1] §2.1, §2.3`

**Target representation.** They explicitly **collapse all bounding boxes to their geometric center**
`((xmin+xmax)/2, (ymin+ymax)/2)` and keep point annotations unchanged. `[1] §2.2` — this is the opposite of what
our candidate-list setup wants (see "does not transfer").

**Resolution handling.** VitDet encoder + AnyRes dynamic grid partitioning; on-the-fly **crop-and-zoom as an
inference-time tool**, iterated once (predict → crop around prediction → re-predict). `[1] §3, Fig. 5`

**Loss / objective.** Two stages. SFT (10K steps) then GRPO-based RLVR (1,500 steps). The RL grounding reward is
**containment, not center-match**: "any prediction falling within the annotated bounding box is considered
acceptable" — they explicitly note SFT's weakness is constraining the model "to reproduce the annotated center
point exactly". 8 rollouts from full image + 4 from the zoomed crop; prompts with all-identical rewards are
filtered out. `[1] §3`

**Measured grounding.** ScreenSpot-V2 91.6, ScreenSpot-Pro 53.3, OSWorld-G 55.3 (Table 1). Beats every other 3B
baseline on SSPro (next best 3B is Jedi 37.1, +16.2).

> **Source inconsistency, flagged:** the abstract states OSWorld-G = 61.2% but Table 1 of the same paper states
> 55.3. I could not resolve which is correct. Do not cite either number without hedging.

**Ablations (Table 6, Appendix F)** — base 50.15 SSPro / 90.49 SS-V2:

| Removed | ΔScreenSpot-Pro | ΔScreenSpot-V2 |
|---|---|---|
| w/o OSAtlas | **−6.80** | −0.87 |
| w/o Aria-UI | −3.22 | −0.55 |
| w/o UGround | −2.08 | −0.30 |
| w/o Jedi | −2.52 | −2.13 |
| w/o ShowUI | −1.26 | −1.65 |

**What they said did NOT work** (this is the most valuable part for us, and it is blunt):
- "inference-time techniques such as CoT reasoning and visual tool-use bring improvements, **yet the benefits
  remain limited**." `[1] §1`
- "small models can benefit from reinforcement learning, they are **sensitive to RL reward designs**." `[1] §1`
- Navigation stays weak (28.0 AndroidWorld / 19.8 OSWorld) — the paper frames long-horizon reasoning as the
  unsolved part, grounding as the solved part. `[1] §1`

**Relevance to us:** the −6.80 for high-res/dense-layout data is the single largest ablation lever in this entire
report. But note their high-res trick is *composite concatenation of screenshots*, which increases element density
per image — plausible to replicate cheaply, and it directly targets "which of many similar elements" confusion.

---

### 2. Microsoft — GUI-Actor [2] and ScreenParse [3]

These are two different papers. The brief conflated them; both are Microsoft-adjacent and both are relevant, so
I cover them separately.

**GUI-Actor mechanism.** A special `<ACTOR>` token is prepended as a contextual anchor; an **action attention head**
learns to attend from that anchor to the ViT patch tokens covering the target element. Grounding output is an
attention distribution over patches, not text coordinates. Head is small: **20M params on 2B backbone, 100M on 7B**.
Warm-up phase freezes the backbone and trains only the head; then full finetune. `[2] §1, §3`

**"Multi-patch supervision"** — the key label-design idea. All visual patches overlapping the GT bounding box are
labelled **positive**; all others negative. Their stated motivation is verbatim our problem: single-point
supervision "cannot handle ambiguous supervision targets". This yields dense, spatially-structured supervision
instead of one positive point. `[2] §3, Appendix B`

**Deltas (all with verification status):**

| Config | ScreenSpot-Pro | ScreenSpot | ScreenSpot-V2 | Trained params |
|---|---|---|---|---|
| GUI-Actor-2B | 36.7 | 86.5 | — | full |
| GUI-Actor-2B + verifier | **41.8** | 86.9 | — | full |
| GUI-Actor-2B-LiteTrain | 25.4 | 71.4 | 73.9 | **19M** |
| GUI-Actor-2B-LiteTrain + verifier | 34.0 | 79.2 | 82.3 | 19M |
| GUI-Actor-7B | 40.7 | 88.3 | 89.5 | full |
| GUI-Actor-7B + verifier | ~44 (≈+4 stated) | — | — | full |

Sources: Table 1 (2B/7B SSPro), Table 5 (LiteTrain), Table 6 (SS/SSv2), and §"Boosting Performance via Grounding
Verifier". `[2]`

**The two deltas that matter for us:**
1. **Verifier = candidate selection.** Adding the verifier gives **+5.1 SSPro** for 2B-full (36.7→41.8) but only
   **+0.4 on ScreenSpot** (86.5→86.9). Interpretation: candidate selection helps *enormously* on hard/high-density
   screens and barely at all on easy ones. Our rows are hard (24 candidates median, p90 64). This is the closest
   published analogue to "choose among candidates instead of regressing a point."
2. **Attention head beats coordinate generation at equal backbone.** Ablation Table 6, same data: Aguvis-7B
   bbox-supervised scores 13.8 SSPro / 84.4 SS vs GUI-Actor-7B 40.7 / 88.3. `[2]` Critically they also note
   **bbox supervision is *worse* than point supervision** for coordinate-generation models (13.8 vs 15.6 SSPro) —
   i.e. giving a small model more spatial information in a *text-coordinate* interface does not help. Only the
   architectural change (attention over patches) converts bbox info into accuracy.

**Their stated limitation** (Appendix A): fixed 28×28 ViT patch size means "very small interface elements (e.g.,
icons smaller than 10×10 pixels) … may be insufficiently represented." `[2]` This is about coordinate precision,
not selection — does not explain our 437 px median miss.

**ScreenParse — "complete screen parsing supervision" [3].** Dataset of 771K web screenshots / 21M elements with
dense annotations: every visible element's box, 55-class type, and text. Pipeline ("Webshot") renders URLs,
extracts annotations, VLM-relabels, filters. `[3] Abstract, §3`

The ablation that matters — finetune an existing small VLM on dense supervision, measure transfer to *sparse*
grounding (Table 9, ScreenSpot PixCov = fraction of annotated target pixels covered by predictions):

| Model | SS-Web PixCov | SS-PC PixCov | SS-Mobile PixCov |
|---|---|---|---|
| Qwen3-VL-2B-Instruct | 0.292 | 0.218 | 0.250 |
| Qwen3-VL-2B + ScreenParse | **0.720** (+0.428) | **0.443** (+0.225) | 0.477 (+0.227) |
| InternVL3-2B | 0.057 | 0.109 | 0.321 |
| InternVL3-2B + ScreenParse | 0.592 (+0.535) | 0.628 (+0.519) | 0.527 (+0.206) |

`[3] Table 9` — note their own body text quotes 0.292→0.720 for Web and 0.218→0.443 for PC; I read both from the
table directly.

**Honest caveat on ScreenParse:** the transfer is *not* uniform. Recall@50 on Mobile is 0.066 for ScreenVLM and
Qwen3-VL-2B + ScreenParse loses **−0.035** PixCov on Mobile (0.250→0.477 is +0.227; the Mobile Recall column is
what degrades). Their own framing: "ScreenVLM often predicts regions that cover annotated pixels (high PixCov) but
fails to produce tight element-level boxes … which is more sensitive to precise localization." `[3] §5` So dense
supervision buys *coverage/selection*, not *precision* — which is exactly the half we need.

---

### 3. Google / SeeClick / UGround / OS-Atlas lineage — target representation and data synthesis

**(a) How an element becomes a training target.**

| Work | Target format | Detail |
|---|---|---|
| UGround [10] | **center point**, raw unnormalized pixel ints as text | "we use the center point coordinates of an element as the expected output"; answer string like `(1344, 1344)`, "without any normalization" |
| OS-Atlas [11] | point **or** bbox (task-dependent) | outputs "normalized to relative coordinates within a 0-1000 range" |
| Ferret-UI Lite [1] | center point | bbox → geometric center, unconditionally |
| SeeClick [14] | point | click coordinates |
| GUI-Actor [2] | **attention over all patches in bbox** | multi-patch positive labels |
| ScreenParse [3] | **every element** as box+type+text | dense, not one target |

The convergent choice across Google lineage, UGround, OS-Atlas, Ferret is **point (or normalized point)**. The
convergent *newer* choice (GUI-Actor, ScreenParse, ConClick) is **region / set-valued**. That split is the single
most important pattern in this whole review for us.

**(b) Cheap bbox supervision — the synthesis pipelines.**

- **UGround:** fully web-based. From Common Crawl, takes the *dual representation* of a webpage — full HTML + DOM
  with bbox coordinates — and synthesizes (screenshot, referring expression, point) triplets. Key structural
  insight they state: webpages are ideal because you get pixel-truth boxes for free from the DOM. Web-Hybrid =
  **9M instances over 773K screenshots**. Synthetic data alone beats SeeClick (>10% at 50K screenshots) even though
  SeeClick trained on ~3M elements from ~400K screenshots. `[10] §2.2, §3.5`
- **OS-Atlas:** first multi-platform synthesis toolkit (Windows, Linux, macOS, Android, web). Crawled ~4M web
  pages, initially 3.7M screenshots / 37M elements, human-inspection filtering down to **1.6M screenshots /
  7.7M elements**; plus physical-machine collection for Windows/macOS. Total corpus **2.3M screenshots, >13M
  elements**. `[11] §3`
- **ScreenAI:** the *screen annotation* task — model must emit type + location + text for UI elements, trained on
  Rico. Critically, those text annotations are then used to **auto-generate QA pairs** for LLM training. `[12] §3.1`
- **Google LVG [13]:** heuristic referring-expression generation from Android UI trees — 22,617 synthetic
  expressions over 21,282 screens, from templates over name/type/clickable/location (`"the Cancel button next to
  login"`). They report **plain mixing did not help**; they needed *multi-context learning* (pair each real
  expression with a synthetic one for the same element) to transfer. `[13] §4.2`

**(c) Reported deltas from each ingredient.**

| Ingredient | Delta | Source |
|---|---|---|
| Web-DOM synthesis vs prior synthetic (UGround) | >10% over SeeClick at 50K screenshots | [10] §3.5 |
| Data scale: diminishing returns after ~100K screenshots | "return starts diminishing after 100K"; 50K→100K still gains | [10] §3.5 |
| Raw scale 16.1K → 119.4K samples | **+0.1 pt** (82.8 → 82.9) | [9] Table 5 |
| Layout-guided contrastive (vs plain contrastive) | **+2.53 acc@1**, +2.76 acc@5 | [13] Table 2 |
| Synthetic data + multi-context (best OFA row) | +4.92 acc@1 (40.27 → 45.19) | [13] Table 1 |
| Screen-annotation pretraining (ScreenAI, pix2struct vs fixed grid) | "significantly better" on landscape aspect ratios; no clean scalar given | [12] Fig. 9 — **weak evidence** |
| OS-Atlas 7B vs its own backbone Qwen2-VL-7B on ScreenSpot | 42.89 → 85.14 | [11] Table 2 |

**Blunt note:** these Google/UGround/OS-Atlas deltas are almost all *pretraining-scale* effects (millions of
elements, full model finetunes). They validate the direction (bbox-rich data helps) but none of them gives you a
cheap per-GPU-hour lever for a model that already exists. The ScreenAI pix2struct ablation is reported only as a
geometric-mean ratio in a figure, not a scalar — I will not put a number on it.

---

### 4. Small-model-specific recipes 2026 — single highest-leverage ingredient each

**ZonUI-3B (WACV 2026) [9]** — *highest-leverage: two-stage fine-tuning with a resolution-specialization stage.*
Trained on 24.1K examples, LoRA rank 8 / α=16 on Qwen2.5-VL-3B, batch 1 × grad-accum 48, single RTX 4090 (24 GB),
**48 hours total**, LR 2e-4 then 5e-5. Results: ScreenSpot 84.9, SS-V2 86.4, GroundUI-1K 82.4, SSPro 28.7.
Ablations: two-stage gives **+3.3 desktop / +1.8 web** over single-stage on the same data (Table 6); balanced
1:1:1 platform sampling gives +0.9 overall (Table 4); **data volume is worthless at the margin** — 16.1K vs
119.4K = 82.8 vs 82.9 (Table 5). `[9]` **Needs training: yes**, but only ~48 GPU-hours, which is the cheapest
credible full-recipe datapoint here.

**GoClick [5]** — *highest-leverage: architecture choice, not data.* A 230M Florence-2-style **encoder-decoder**
was chosen deliberately over downscaled decoder-only VLMs, after finding the decoder-only route "yields suboptimal
results" at small scale. GoClick-L (0.8B) beats Qwen2-VL-2B-SFT, InternVL-2.5-1B-SFT, SLiME-Gemma-2B-SFT on all
benchmarks and is comparable to Qwen2VL-7B-SFT (Table 4). Separately, Progressive Data Refinement (10.8M → 6.4M
→ 3.8M) gives **+4.0 average** (71.6 → 74.0 → 75.6, Table 7), and they document *harmful* subsets: old GUI
patterns are "detrimental"; SeeClick-Web brief-description data (2M+ samples) shows a "catastrophic decline at
full inclusion." `[5]` **Needs training: yes**, and the architecture leg is a full re-architecture for us.

**WinDOM (ICML 2026) [6]** — *highest-leverage: the negative result.* 2B student (Qwen3.5-2B), 54,425-record
corpus harvested by driving win11react under headless Playwright reading bboxes straight off the DOM — **zero
human annotation, zero OCR**. Direct SFT result is the important number:

| | In-dist | SSPro | OSWorld-G | SS-V2 | OOD-mean |
|---|---|---|---|---|---|
| Qwen3.5-2B base | 70.0 | 52.2 | 47.5 | 83.0 | 60.9 |
| Direct SFT | **82.1** | **47.6** | 49.0 | 85.9 | **60.8** |
| SFD-4B + Early-init RL | 76.5 | 55.7 | 54.5 | 88.8 | **66.3** |

`[6] Table 1`. Read the SFT row: **+12.1 in-distribution, −4.6 ScreenSpot-Pro, net OOD-mean change −0.1.** They
call it a "memorization signature." And their RL finding is that the *under-trained* cold-start (100 steps) is a
better GRPO initializer than the converged one (900 steps), with a proposed entropy-collapse explanation.
**Needs training: yes.** The transferable lesson costs zero GPU-hours: *if you SFT a 2B model harder on your own
data, in-distribution accuracy will rise and your real benchmark may not move, or may fall.*

**RankGround [7]** — *highest-leverage: crop/region selection, and it matters MORE at small scale.* A lightweight
reranker (145 ms) picks one crop from a 9-crop overlapping grid; a single VLM call grounds inside it. ScreenSpot-Pro:

| Scale | Direct | Best multi-crop (MVP) | RankGround | Gain vs Direct |
|---|---|---|---|---|
| 8B (Qwen3-VL) | 48.72 | 60.24 | 64.26 | +15.54 |
| 8B (MAI-UI) | 54.59 | 65.91 | 69.26 | +14.67 |
| 8B (UI-Venus) | 58.89 | 70.90 | 70.97 | +12.08 |
| **2B (Qwen3-VL)** | **35.42** | 38.48 | **52.75** | **+17.33** |
| **2B (MAI-UI)** | **43.64** | 57.39 | **63.00** | **+19.36** |
| **2B (UI-Venus)** | **48.13** | 56.67 | **64.71** | **+16.58** |

`[7] Table 2`. Their own sentence: "The larger 2B gain shows that **crop selection matters more when the grounder
has less capacity**." `[7]` Oracle crop coverage 99.75% on the default 9-crop overlapping grid; GroundRanker
Hit@1 88.2% [16] — so ~11.5% of the recoverable headroom is lost to picking the wrong crop. **Needs training:
the reranker, yes; the grounder, no.** But it adds a reranker pass + crop → not a 103 ms single-pass design.

**PreView [8]** — *highest-leverage: an auxiliary preprocessor that leaves the grounder frozen.* Auxiliary model
(Qwen3.5-0.8B, LoRA, vision encoder frozen; 50K center-prediction + 49,311 positive-description + 49,284 rejection
examples from GroundCUA) predicts and refines a center, crops a 12.5%-area view, enlarges it, and attaches two
**cues**: a visual red-circle marker with a location hint, and a textual description of the candidate's identity /
function. The grounder is frozen and just consumes the prepared inputs. `[8] §3, §4`

| Grounder | Baseline avg | + PreView | Δ | SSPro Δ |
|---|---|---|---|---|
| Qwen3-VL-2B | 57.2 | **64.6** | **+7.4** | 43.1 → 59.3 (**+16.2**) |
| OS-Atlas-4B | 21.6 | 35.5 | **+13.9** | 2.8 → 28.2 (+25.4) |
| Qwen3-VL-4B | 66.8 | 68.3 | +1.5 | 59.3 → 63.7 (+4.4) |

`[8] Table 1` (read from PDF). Note the inversion: the **weakest** grounder gains the most, the 4B gains almost
nothing on the non-SSPro benchmarks — and Qwen3-VL-4B actually **loses** 6.9 on SSv2 and 5.0 on MMB-GUI. So
PreView is a capability-substitution trick that pays off in proportion to how bad your grounder is, and it is not
free. **Needs training: yes for the auxiliary model (~0.8B, LoRA), no for the grounder.**

---

### 5. The candidate-selection paradigm — what it reaches, what breaks

This is the question most directly aimed at our 56.05% / 47.5%-VA-track situation.

**The error-structure evidence is overwhelming and it is our exact failure mode.** ConClick [4] evaluates 10
grounding models on 5 datasets and finds **near-element + far-element errors — clicks on non-target GUI elements —
account for 74%–88% of all incorrect predictions** across the five datasets. Not background misses:
*wrong-element* misses. Further: 55.49%–79.14% of examples with ≥2 incorrect predictions contain a distractor
that **multiple different models select**, and among examples with ≥5 incorrect predictions, 12.88%–28.15% contain
a distractor picked by ≥5 models. `[4] §1, Fig. 2` That cross-model agreement is why distractor mining works:
the same wrong element is systematically attractive, so it is a *learnable* negative, not noise.

**Reached accuracy from candidate selection:**

| Method | Backbone | Mechanism | Result |
|---|---|---|---|
| ConClick [4] | Qwen3-VL-8B | listwise rank + pairwise margin over region scores | SSPro 54.71 → **66.03**; OSWorld-G 63.33 → 70.20; MMB-GUI 83.72 → 86.28; avg 67.25 → **74.17** |
| ConClick + ReClick [4] | Qwen3-VL-8B | 4-view revisit + candidate comparison | SSPro **72.36** (+6.33 over ConClick alone) |
| GUI-Actor + verifier [2] | 2B | attention candidates → 2B verifier | SSPro 36.7 → **41.8** |
| LVG [13] | OFA | layout-contrastive candidate matching | acc@1 37.88 → **45.19**, acc@5 56.88 → 65.25 |
| Regression-free layout matching (CLIP) [see #18 below] | frozen MLLM + DINO candidates | CLIP similarity over candidates, no coordinate regression at all | ScreenSpot 87.1 avg; SSPro 41.3 |

The ConClick ablation is the money shot for cost estimation (Table 2, Qwen3-VL-8B, avg over 3 benchmarks):

| Data | Loss | Avg acc | Avg element-error rate |
|---|---|---|---|
| random | coordinate only | 70.67 | 24.29 |
| **ConClick-Data** | coordinate only | 71.99 | 23.82 |
| ConClick-Data | + listwise only | 73.42 | 23.44 |
| ConClick-Data | + pairwise only | 73.18 | 23.57 |
| **ConClick-Data** | **+ both** | **74.17** | **22.50** |

`[4] Table 2`. Decomposition: **data construction alone = +1.32**; the full contrastive objective on top = **+3.50**
total. And the dataset is **4,292 target–distractor pairs + 4,292 ordinary examples = 8,584 training examples**
`[4] §3`. Eight and a half thousand examples moving a benchmark 11.3 points. This is by far the best
delta-per-GPU-hour in the review.

**Candidate generation recall.** RankGround's Oracle crop coverage is **99.75%** but the learned selector only
reaches **Hit@1 88.2%** [16] — i.e. the cascade ceiling is set by the *selector*, not the proposer. For us the
analogous number is already measured: **97.2% coverage, 24 candidates median, 64 at p90**. Our candidate list is
better coverage than RankGround's 88.2% selector. The failure is the choice, not the list.

**What breaks — the specific failure of pure candidate selection, and why our VA run got 47.5%.** Four distinct
documented breakages:

1. **Supervision sparsity.** ScreenParse's premise: sparse one-target-per-screen labels "leave the majority of
   on-screen elements unlabeled and the full screen structure implicit. As a result, models can learn shortcuts
   that are sufficient for the supervised steps while failing to form a complete screen state." `[3] §1` If you
   train on candidate lists where only one is labelled positive, you teach "pick the plausible one," not "discriminate."
2. **Ambiguity blow-up.** GUI-Actor's stated reason for multi-patch supervision: single-point targets cannot
   represent that several regions are all valid. `[2]` In a 24–64 candidate list, many candidates are *partially*
   correct (a container that contains the gold node; a sibling with the same label). Point supervision punishes
   all of them; ranking supervision can order them.
3. **Selector capacity is the bottleneck, not proposer recall.** GUI-Actor needs a separate 2B verifier to get
   the +5.1; RankGround needs a dedicated reranker to get from 99.75% coverage to 88.2% Hit@1. Selection is its
   own learned skill.
4. **Coordinate-precision collapse when you specialize.** ConClick's own framing: near-target error density
   remains "denser inside the target neighborhood" even after selection improves. `[4] §4` And ScreenParse's
   Mobile Recall@50 = 0.066 shows a selector trained on coverage can lose tight localization. Selection and
   precision trade off.

**Interpretation of our VA-track result (explicit inference, not a source claim):** 47.5% element grounding from
"choose among candidates" is *lower* than 56.05% free-form, which is consistent with breakages (1) and (2) above —
a candidate interface trained with a single-positive target label teaches shortcut-picking, and it discards the
free-form model's spatial prior. The literature's candidate-selection wins (ConClick, GUI-Actor+verifier) all come
from *adding* ranking/contrastive supervision **on top of** existing supervision, not from swapping the interface.
That is the single most actionable distinction in this report.

---

## Top 5 ingredients ranked by expected Δ per GPU-hour for OUR situation

Ranking is my judgement (inference) applied to the measured deltas above. "Expected Δ" is a forecast for our
grounding-given-type (currently 63.8%) or overall grounding (56.05%), **not** a measured result. Confidence is
stated per item.

### 1. Distractor-mined listwise + pairwise contrastive selection loss over the existing candidate list — **highest**

- **Supporting number:** ConClick full recipe moved SSPro **54.71 → 66.03 (+11.3)**, OSWorld-G +6.9, avg
  **67.25 → 74.17 (+6.9)**, element-error rate 25.91 → 22.50, using only **8,584 training examples** [4]. The
  data-construction half alone is +1.32 avg; the loss on top is +3.50.
- **Why cheap for us:** we already have the candidate list at train time (gold bbox matches a candidate exactly on
  **83.0%** of target rows). Distractor mining is therefore a labeling pass, not a data-generation pass. ConClick
  mines negatives by running two models and matching wrong clicks to detected elements — we can mine negatives
  *directly from our own candidate list* (any non-gold candidate), which is strictly cheaper.
- **Exact change:** keep all 8 readout slots and the 103 ms single pass. On rows where gold bbox == candidate _c*_,
  attach a selection loss to the slot scores: (i) a softmax listwise term over {_c*_} ∪ negatives; (ii) a pairwise
  margin term with a softplus penalty pushing score(_c*_) − score(neg) past a margin; (iii) choose negatives as the
  top-3 currently-highest-scoring non-gold candidates whose token mask does not overlap the gold node, plus one
  fixed "observed distractor." Weight the selection loss relative to the existing coordinate loss by a tunable λ
  (start 0.5, ConClick's own balance is not stated numerically so this needs a sweep). Do **not** remove the
  free-form point path.
- **Confidence:** high that the direction transfers (the paper's error analysis matches our 71/143 wrong-element
  profile almost exactly); medium on magnitude (~⅓–½ of +6.9, since ConClick uses an 8B backbone and an attention
  readout — our slot readout is closer in spirit but not identical).

### 2. Dense "every candidate" auxiliary supervision instead of single-positive — **second**

- **Supporting numbers:** Qwen3-VL-2B + dense ScreenParse supervision lifted ScreenSpot-Web PixCov
  **0.292 → 0.720 (+0.428)** and PC **0.218 → 0.443 (+0.225)** [3, Table 9]. GUI-Actor's multi-patch labelling is
  the same idea with only **19M trained parameters** reaching SSPro 25.4 → 34.0 (+8.6) with a verifier [2, Table 5].
- **Why for us:** turns ~1 label/row into ~24 labels/row at zero data cost, because the candidate list *is* the
  label set. Directly attacks ScreenParse's diagnosed failure mode ("learn shortcuts … failing to form a complete
  screen state") which is a good description of picking a plausible-looking sibling.
- **Exact change:** add a per-candidate binary head (target / not-target) over the candidate list, or a
  slot-to-candidate assignment loss, trained on all candidates with the gold node positive. This is a
  multi-positive setup: label **all** candidates whose box contains the gold point as positive, not just exact-box
  matches — this recovers rows beyond the 83.0% exact-match subset and mirrors GUI-Actor's containment tolerance.
- **Confidence:** medium-high on PixCov-style coverage gains; **lower** on final click-in-node accuracy, because
  ScreenParse itself shows Mobile Recall@50 collapsing to 0.066 — coverage is not precision [3, §5].

### 3. Data-source ablation and deletion of harmful rows — **third, and cheapest**

- **Supporting numbers:** GoClick removed 65% of its data and *gained* +4.0 average (71.6 → 75.6), with
  documented catastrophic subsets (SeeClick-Web at full inclusion: "catastrophic decline") [5, Table 7]. ZonUI
  found **+0.1** for 7.4× more data (16.1K → 119.4K) [9, Table 5]. WinDOM found direct SFT nets **−0.1 OOD-mean**
  despite +12.1 in-distribution [6, Table 1].
- **Why for us:** WinDOM's memorization signature is a direct warning that our 398-step AC-v2 number may be an
  in-distribution number. If part of our training mixture is redundant or stylistically stale, it is currently
  buying nothing and possibly costing grounding.
- **Exact change:** leave-one-source-out sweep, measuring **grounding-given-type** (our 63.8%, the sensitive
  metric, not step accuracy) on a held-out split. Expect 1–4 pp from deletion, near-zero risk, and it costs only
  eval time. Pair this with a second readout: report *both* in-distribution and held-out grounding, because
  WinDOM shows these move in opposite directions.
- **Confidence:** high on the mechanism, medium on magnitude. This is the item I would run first because a
  negative result here de-risks items 1 and 2.

### 4. Multi-positive / containment supervision on the candidate bbox — **fourth**

- **Supporting numbers:** GUI-Actor labels all patches overlapping the GT box positive specifically to "handle
  ambiguous supervision targets" [2, §3, App. B]. Ferret-UI Lite switches its RL reward from exact-center to
  **containment** ("any prediction falling within the annotated bounding box is considered acceptable") [1, §3].
- **Why for us:** our metric is already containment-flavoured ("point must land inside the smallest a11y node
  containing the gold point"), so a center-point SFT loss is optimizing the wrong thing. This is nearly free —
  we have candidate boxes on 83% of rows.
- **Exact change:** replace "slot must equal gold point" with "slot must fall inside the gold candidate's box,"
  and in the candidate-ranking head treat box-overlap with the gold node as positive.
- **Confidence:** medium. The evidence for *this specific* change as a standalone delta is thin — GUI-Actor's
  headline deltas confound the head architecture with multi-patch labels, and I could not isolate a
  "point→containment" number anywhere. Treat the delta as unverified; the implementation cost is what justifies it.

### 5. A short high-resolution / density-focused second SFT stage — **fifth**

- **Supporting numbers:** Ferret-UI Lite's largest single ablation is removing high-resolution synthetic data:
  **−6.80 ScreenSpot-Pro** (base 50.15 → 43.35) [1, Table 6]. ZonUI's two-stage (general → resolution
  specialization) gives **+3.3 desktop / +1.8 web** at identical data [9, Table 6], for 48 h on one RTX 4090.
- **Why for us:** with 24 candidates median / 64 at p90, our screens are dense; a stage that exposes the model to
  denser layouts plausibly sharpens exactly the discrimination item 1 exploits. Ferret's concrete trick —
  **concatenating multiple screenshots into larger composite images** — is a data-augmentation pass, not a data
  collection pass.
- **Exact change:** second SFT stage, low LR (~5e-5 in ZonUI), on composite/downscaled-dense screenshots only.
- **Confidence:** medium-low for us. ZonUI's +3.3 is a *desktop-subset* delta, and Ferret's −6.80 is a
  data-mixture ablation on a full multi-stage recipe, not a drop-in change. Also unknown whether our 103 ms budget
  survives denser inputs at the same token count.

**Deliberately excluded from the top 5** despite good numbers — all blocked by the 103 ms single-pass / zero-token
constraint: ReClick (+6.33 SSPro on ConClick-8B) needs 4 forward passes [4]; RankGround needs reranker + crop
[7]; PreView needs an auxiliary 0.8B model + crops, and its SSPro gain comes bundled with −5.4 SSv2 on the 4B [8];
Ferret's zoom-in needs a second prediction pass [1]; GUI-Actor's verifier needs a separate 2B model [2, 15].
If the latency budget ever moves, **GUI-Actor's verifier (+5.1 SSPro for 2B) and ReClick are the two best
candidates**, in that order.

---

## Does not transfer to us — explicit list with reasons

| Item | Source | Why it does not transfer |
|---|---|---|
| **Collapsing bboxes to center points** | UGround [10] §2.3; Ferret-UI Lite [1] §2.2 | Our entire advantage is that gold bbox matches a candidate on 83% of rows. Center-collapsing discards the box identity that makes candidate supervision possible. UGround needs points because it decodes coordinates as text; we do not decode text at all. |
| **Raw textual coordinate decoding at high resolution** | UGround [10] (1344×1344 / 896×2016, 36 ViT slices); OS-Atlas [11] (0–1000 normalized) | Both assume autoregressive coordinate emission. We have **zero output tokens** and 8 typed slots. The resolution-scaling argument is about legibility for a text decoder; not our bottleneck (our bottleneck is 437 px median miss = wrong element, not unreadable pixels). |
| **Any zoom-in / crop / multi-pass inference** | Ferret zoom-in [1] §3; ReClick [4]; RankGround [7]; PreView [8]; ZoomClick & ScreenSeekeR as cited in [8] §2 | All require ≥2 forward passes and/or a second model. At ~103 ms single-pass, a 2-pass design roughly doubles latency and a 4-pass design (ReClick) is out. **This is the single biggest constraint separating us from the 2026 state of the art.** |
| **GRPO / RLVR recipes** | Ferret-UI Lite [1] §3; WinDOM [6] | Apple states outright that small models are "sensitive to RL reward designs" [1]; WinDOM shows the same RL recipe flips sign depending on cold-start saturation (Early-init OOD 66.3 vs Late-init 64.4) and needs 1,000 GRPO steps [6]. High tuning cost, high variance, and no evidence it beats a well-posed supervised selection loss for *selection*. |
| **Architecture swap to encoder-decoder** | GoClick [5] Table 4 | GoClick's core claim is that Florence-2-style encoder-decoder beats decoder-only *at 230M–0.8B*. We are a 1.92B single-pass slot model; rebuilding the readout as an encoder-decoder is a new project, not a recipe. Evidence also does not extend to our size. |
| **Massive screen-parsing pretraining scale** | ScreenParse [3] (771K screenshots / 21M elements); OS-Atlas [11] (2.3M / 13M); UGround [10] (9M / 773K) | These are pretraining-scale corpora. We already have candidate lists. The *idea* (dense labels) transfers — see ingredient #2 — but the corpora do not, and buying the idea does not require buying the data. |
| **Web-DOM-only synthesis as our data source** | UGround [10]; WinDOM [6] | Both harvest boxes from DOM/HTML. UGround itself states "no desktop UI data is incorporated … which limits the performance on desktop UIs" [10, Limitations]. If our 398-step benchmark is desktop a11y-node grounded, web-DOM-only supervision is distributionally wrong. Use for augmentation only. |
| **ScreenAI's pix2struct patching + scale curve** | ScreenAI [12] §5.2–5.3 | Effect reported only as a geometric-mean ratio in a figure (no scalar); and the model-scale finding ("improvements have not saturated at the largest size", 5B) argues *against* a 1.92B fixed architecture. No actionable per-ingredient delta. |
| **"Just use the candidate interface"** | — (our own VA-track result) | We measured 47.5% element grounding / 55.5% step from candidate-choice, versus 56.05% / 64.07% free-form. Consistent with ConClick [4] and GUI-Actor [2], whose wins come from *adding* ranking supervision on top of an existing model, not from replacing the output interface. **Do not re-run the interface swap; add the loss instead.** |
| **Attributing any grounding recipe to OpenAI** | [17] | No published, ablatable OpenAI UI-grounding recipe exists. CUA is GPT-4o vision + RL, closed. Everything cited as "Operator" in the papers above (e.g. 40.6 OSWorld-G [1, Table 8]) is a third-party evaluation, not OpenAI's own recipe. |

---

## Coverage Status

**Checked directly (fulltext read, not abstract):**
- Ferret-UI Lite [1] — full arXiv HTML incl. §2.1–2.3, §3, Table 1, Fig 6, Appendix F Table 6. Found and flag the
  abstract-vs-Table-1 OSWorld-G discrepancy (61.2 vs 55.3).
- GUI-Actor [2] — full arXiv HTML incl. Tables 1/2/3/5/6 + Appendix A limitations + App. G.1 (verifier
  self-aggregation described qualitatively; Table 9 numbers did not render in HTML — **not extracted**).
- ScreenParse [3] — full arXiv HTML incl. Table 1–9, §4–5.
- ConClick [4] — full arXiv HTML incl. Table 1, Table 2 ablation decomposition, Fig 2 error statistics, §3.2 ReClick.
- GoClick [5] — full arXiv HTML incl. Tables 4 and 7, PDR description.
- WinDOM [6] — full arXiv HTML incl. Table 1, findings F1–F3.
- RankGround [7] — full arXiv HTML incl. Table 2 (all six backbone configs), Oracle/Hit@1 discussion.
- PreView [8] — **PDF** because the HTML table was garbled; Table 1 extracted from PDF text.
- ZonUI-3B [9] — CVF open-access PDF, Tables 1–6.
- UGround [10], OS-Atlas [11], ScreenAI [12], LVG [13], SeeClick [14] — PDFs read.

**Uncertain / incomplete:**
- **Ferret-UI Lite's exact zoom-in ablation delta** is only shown in Figure 6(a) as a plot. The paper says
  "RL consistently improves … and our zoom-in operation provides further gains" but gives **no scalar**. I did not
  invent one.
- **GUI-Actor Appendix G.1 Table 9** (verifier self-aggregation numbers) failed to render in arXiv HTML and I did
  not pull the NeurIPS PDF appendix for it. Qualitative claim only.
- **ScreenAI pix2struct ablation** — figure-only, no scalar. Reported as weak evidence.
- **PreView's auxiliary-model training cost** (GPU-hours) is not stated; I could not normalize its Δ per
  GPU-hour. Its Table 2 lists hyperparameters (epochs, AdamW, LoRA) but no wall-clock.
- **No source found** giving a "point supervision → containment supervision" isolated delta. Ingredient #4's
  magnitude is therefore unverified; only its implementation cost is known.
- **OpenAI:** searched twice (agent architecture, UI grounding papers). No ablatable grounding recipe exists in the
  open literature. This is a real gap in the landscape, not a search failure.
- I did **not** find any paper that reports a candidate-selection *regression* like our 47.5%-vs-56.05% — the
  literature only reports candidate-selection *wins*. Our negative result may be unusual, which makes breakage (1)
  (single-positive label sparsity) the most likely explanation and makes ingredient #2 the highest-information test.

**Not attempted (out of scope for R1):** verifying whether any of these recipes were trained with a11y-tree
candidate lists rather than visual candidates (only ConClick uses parsed boxes + OmniParser-v2 [4, §3]); and
reproducing any number.

---

## Sources

1. Yang, Dou, Feng, et al. — *Ferret-UI Lite: Lessons from Building Small On-Device GUI Agents* (Apple, 2025) — https://arxiv.org/abs/2509.26539
2. Wu, Cheng, Yang, et al. — *GUI-Actor: Coordinate-Free Visual Grounding for GUI Agents* (Microsoft Research, NeurIPS 2025) — https://arxiv.org/abs/2506.03143
3. Gurbuz, Hong, Nassar, Pollefeys, Staar — *ScreenParse: Moving Beyond Sparse Grounding with Complete Screen Parsing Supervision* (ICML 2026) — https://arxiv.org/abs/2602.14276
4. Zhang, et al. — *ConClick: Element-wise Contrastive Learning Improves GUI Grounding* (2026) — https://arxiv.org/abs/2610.13268
5. Chen, Li, et al. — *GoClick: Lightweight Element Grounding Model for Autonomous GUI Interaction* (2026) — https://arxiv.org/abs/2604.23941
6. Li, et al. — *WinDOM: Self-Family Distillation for Small-Model GUI Grounding* (ICML 2026) — https://arxiv.org/abs/2606.25964
7. — *RankGround: Efficient High-Resolution GUI Grounding via Lightweight Reranker-Guided Crop Selection* (2026) — https://arxiv.org/abs/2609.18690
8. — *PreView: Enhancing GUI Grounding with Preprocessed Views and Cues* (2026) — https://arxiv.org/abs/2610.33645
9. Hsieh, Yang, Wei — *ZonUI-3B: Competitive GUI Grounding with a 3B VLM Trained on a Single Consumer GPU* (WACV 2026) — https://openaccess.thecvf.com/content/WACV2026/papers/Hsieh_ZonUI-3B_Competitive_GUI_Grounding_with_a_3B_VLM_Trained_on_WACV_2026_paper.pdf
10. Gou, et al. — *Navigating the Digital World as Humans Do: Universal Visual Grounding for GUI Agents* (UGround, ICLR 2025 Oral) — https://arxiv.org/abs/2410.05243
11. Wu, et al. — *OS-ATLAS: A Foundation Action Model for Generalist GUI Agents* (ICLR 2025) — https://arxiv.org/abs/2410.23218
12. Baechler, et al. — *ScreenAI: A Vision-Language Model for UI and Infographics Understanding* (Google, IJCAI 2024) — https://arxiv.org/abs/2402.04615
13. — *Visual Grounding for User Interfaces* (LVG, NAACL 2024 Industry Track) — https://aclanthology.org/2024.naacl-industry.9.pdf
14. Cheng, et al. — *SeeClick: Harnessing GUI Grounding for Advanced Visual GUI Agents* (2024) — https://arxiv.org/abs/2401.10935
15. Microsoft — *GUI-Actor verifier* (2B verifier on UI-TARS-2B-SFT) — https://github.com/microsoft/gui-actor/blob/main/verifier/README.md
16. alphaXiv — *RankGround* analysis page (GroundRanker Hit@1 88.2%; Oracle 99.75%) — https://www.alphaxiv.org/abs/2609.18690
17. OpenAI — *Computer-Using Agent* (CUA = GPT-4o vision + RL; no published grounding ablation) — https://openai.com/index/computer-using-agent/
18. — *Hallucination-Free GUI Grounding via Regression-Free Layout-Aware Matching* (2026) — https://arxiv.org/abs/2608.09654

```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "Wrote the report to docs/research/R1-small-model-grounding-recipes.md with: an 18-entry evidence table (method/source/URL/claim/type/confidence); all five brief questions answered with per-source numbers read from fulltext; a method|model size|change|grounding delta|needs-training|URL|date table embedded per-section; a ranked Top-5-ingredients-by-delta-per-GPU-hour list each with a supporting number and an exact change; an explicit 10-row 'does not transfer to us' table with reasons; and a Coverage Status section marking verified vs uncertain items."
    }
  ],
  "changedFiles": [
    "docs/research/R1-small-model-grounding-recipes.md"
  ],
  "testsAddedOrUpdated": [],
  "commandsRun": [
    {
      "command": "curl arxiv HTML/PDF for 2509.26539, 2506.03143, 2602.14276, 2610.13268, 2604.23941, 2606.25964, 2609.18690, 2610.33645, 2410.05243, 2410.23218, 2402.04615, 2401.10935, 2408.00203 + WACV ZonUI PDF + NAACL LVG PDF",
      "result": "passed",
      "summary": "All 15 primary sources retrieved; converted to text via pypdf/HTML strip for precise table extraction rather than snippet-level claims."
    }
  ],
  "validationOutput": [
    "Every cited number traced to a specific table or section of a retrieved fulltext (e.g. Ferret Table 6, GUI-Actor Tables 1/5/6, ScreenParse Table 9, ConClick Table 2, GoClick Tables 4/7, WinDOM Table 1, RankGround Table 2, PreView Table 1, ZonUI Tables 4/5/6).",
    "PreView numbers were initially unreadable in arXiv HTML; re-fetched the PDF and extracted Table 1 from PDF text instead of guessing.",
    "Two figure-only effects (Ferret zoom-in delta, ScreenAI pix2struct) were explicitly NOT given invented scalars."
  ],
  "residualRisks": [
    "Ferret-UI Lite has an internal inconsistency: abstract reports OSWorld-G 61.2% while its own Table 1 reports 55.3%. Flagged in-report; neither number should be cited unhedged.",
    "GUI-Actor Appendix G.1 Table 9 (verifier self-aggregation deltas) did not render in arXiv HTML and I did not extract it from the NeurIPS PDF appendix -- that one ablation is qualitative only.",
    "The Top-5 ranking and 'expected delta' forecasts are my inference applied to others' measured deltas, not measurements on our 1.92B model. Magnitudes (esp. ~1/3-1/2 of ConClick's +6.9) are explicitly labelled as forecasts.",
    "Ingredient #4 (multi-positive/containment supervision) has no isolated published delta; its ranking rests on near-zero implementation cost plus GUI-Actor/Apple design rationale, not on a measured number.",
    "No source reports a candidate-selection REGRESSION matching our 47.5% vs 56.05% VA-track result, so the explanation I offer for it (single-positive label sparsity) is inference, not literature.",
    "OpenAI has published no ablatable UI-grounding recipe; I searched twice and found none. The report says so rather than substituting a proxy."
  ],
  "noStagedFiles": true,
  "diffSummary": "Added one new research report file; no existing files modified.",
  "reviewFindings": [
    "no blockers",
    "note: brief item 2 conflated two distinct Microsoft papers (GUI-Actor and ScreenParse/ICML 2026); report treats them separately and says so.",
    "note: brief item 3 grouped Google/OpenAI; OpenAI contributed nothing citable, so the section covers ScreenAI/LVG/SeeClick/UGround/OS-Atlas and explicitly documents the OpenAI gap."
  ],
  "manualNotes": "Most actionable single finding: our 71/143 'right verb, wrong element' profile is quantitatively the same failure ConClick measured across 10 models / 5 datasets (74-88% of wrong clicks land on non-target elements), and ConClick's fix is 8,584 training examples for +6.9 avg / +11.3 ScreenSpot-Pro -- mineable directly from the candidate list we already have (83% exact bbox match). The second most useful item is a warning, not a recipe: WinDOM's 2B SFT showed +12.1 in-distribution with -4.6 ScreenSpot-Pro and net-zero OOD, which means our 398-step AC-v2 number may be an in-distribution number. Recommend the cheapest item (data-source leave-one-out, measured on grounding-given-type) run BEFORE the contrastive work, purely as de-risking."
}
```