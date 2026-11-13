# Literature note: learning when to stop, long-horizon compounding, termination data (2026-11-11)

Labels: VERIFIED = number read in the paper's HTML text this session (via fetch tool; full text, not a PDF).
UNVERIFIED = seen only in a search snippet or a secondary summary. INFERRED = my reading, not stated by the source.
Rule: numbers from different harnesses/benchmarks are NOT comparable and are never put in one column.

## 0. Headline negative: no source found that trains an in-policy completion/stop head and reports an AndroidWorld effect
I found no primary source that (a) adds a stop/progress readout to a single policy and (b) reports its effect on AndroidWorld.
Everything that reports a termination effect uses an extra model call or prompt (VLAA-GUI judge, Latent-State prompt),
inference-time activation steering (ConflictGuard), or RL reward shaping (MobileGUI-RL). The in-policy evidence that exists
is on AndroidControl (VeriGUI) with a recovery metric, not task success. (VERIFIED absence within the sources I opened;
UNVERIFIED beyond them.)

## 1. Termination / completion detection

### 1.1 False "done" is the dominant failure of agents that have a done action
- VLAA-GUI, arXiv 2604.21375, Sec. 4.3 "Completeness-Aware Planner Mitigates False Completion Ratio" (100-step budget, OSWorld, 361 tasks). VERIFIED.
  - Sonnet 4.6: False DONE count 105 -> 91; FDF (false DONE among failed tasks) 95.5% -> 91.9%; FDA (across all 361) 29.1% -> 25.2%.
  - Gemini 3 Flash: 102 -> 100; FDF 91.9% -> 86.2%; FDA 28.3% -> 27.7%.
  - Paper's own conclusion: the verifier only partially fixes it (FDF stays >86%).
  - Mechanism: CAP = Completion Gate (criteria derived from instruction, checked each step by the Manager) + a separate "Completeness Model Judge" MLLM call (Table 7: same backbone, temperature 0.2). So the fix is NOT one-model.
  - Ablation (Table 2, OSWorld): removing CAP changes Gemini 3 Flash 66.00 (w/o CAP) vs 63.14 (full) at 50 steps, and 67.34 vs 68.77 at 100 steps; Sonnet 4.6 68.53 vs 71.11 (50) and 68.81 vs 71.67 (100). WAA (Table 3): full 60.4/61.0, w/o CAP 51.3/51.3. Note the sign flip for Gemini at 50 steps: the effect is not uniform.
- SPA-Bench, arXiv 2410.15164, Sec. 5.1 / Table 3 (single-app English tasks, 150 tasks). VERIFIED.
  - Premature termination signal = agent self-reports completion while checker says fail; overdue = hits step cap while task is already done.
  - M3A: success 0.640, premature 0.244, overdue 0; T3A: 0.487 / 0.368 / 0.136; MobileAgentV2: 0.433 / 0.333 / 0.111; AutoDroid 0.327 / 0.494 / 0.078; SeeAct 0.393 / 0.100 / 0.276.
  - Sec. 6.2: lower premature rate tends to come with higher overdue rate ("trade-off in the agent's internal success detector"). The ideal detector minimises both. So a stop threshold is a precision/recall trade, not a free lunch.
- Latent State Estimation Helps UI Agents to Reason, arXiv 2405.11120 (benchmarks PixelHelp, AitW, own Android-50; NOT AndroidWorld). VERIFIED.
  - Completion is judged by a SEPARATE LLM call ("have you completed all required steps?", App. E.5) after the next action is proposed; variants without it predict "done" as an action.
  - Table 2: completion accuracy 97.3 total vs 92.8 for always-"not done"; hard completion cases 81.0%.
  - Table 3 zero-shot premature stops: 16.3% (with state) vs 68.9% (without); task success 41.5% vs 25.9%. CoT-SC 24.4% vs 54.1%; ReAct 14.1% vs 11.1% (slightly worse). The paper does not isolate the completion component from progress/mistake estimates.

### 1.2 In-policy methods (same model, no second call)
- VeriGUI ("Don't Act Blindly"), arXiv 2604.05477. VERIFIED (text, Tables 2, 4, 5, 6, 12).
  - One policy emits Thinking, Verification, Action and an <expected_effect> in one output; next step it labels the screen SUCCESS or NO_CHANGE and recovers. Training: AndroidControl-High (~15k train trajectories), Stage 1 Robust SFT mixing 70% success samples with 30% synthetic failure-recovery samples (GPT-4o writes the CoT; failure types: 30% coordinate offset, 25% action type error, 20% target misidentification, 15% timing, 10% null click), Stage 2 GRPO with action + verification + effect rewards.
  - Table 5 (negative ratio), metric RSR (recovery success rate) on their own AndroidControl-based robustness benchmark: 29.7 with no negatives (100:0), 45.5 at 70:30, 45.4 at 50:50. Table 4 (3B): Base 35.0, standard SFT 29.7, Robust SFT 45.5, +GRPO 51.1; LR (loop rate?, lower is better; paper's abbreviation, definition not confirmed by me) 30.0 / 30.4 / 26.5 / 24.3. Table 6: RSR 44.9 (action reward only) -> 48.1 (+verification) -> 51.1 (+effect).
  - Caveats: the metric is recovery on a synthetic benchmark the authors built, not AndroidWorld task success; negatives are synthetic and GPT-4o-annotated; it generates tokens (CoT), which our model does not. Idea transfers only as a readout, which is INFERRED.
  - The 72.3% claim "execution timeouts caused by repeated ineffective actions = 72.3% of failures across 1,265 task executions" is quoted from VeriGUI, which cites MemGUI-Bench (arXiv 2602.06075). I could NOT find it in the first 100k characters of MemGUI-Bench's text; UNVERIFIED at source. MemGUI-Bench does confirm 1,265 executions and that memory hallucination is 58.9% of non-timeout failures.
- ConflictGuard, arXiv 2609.03438. VERIFIED (Table 1, ablation). Opposite problem (agents fail to terminate on infeasible tasks): vanilla conflict-SR 1.4-21.5% (e.g. UI-Venus-1.5-8B 1.40, Claude Sonnet 4.6 21.48); with an inference-time feasibility prompt + activation steering gated by a conflict detector, 44-78%. Important lesson for us: REMOVING the conditional gate caused indiscriminate termination: feasible SR 70.80 -> 46.20 (Qwen3-VL-8B) and 76.80 -> 29.20 (UI-TARS-1.5-7B). Pushing a model toward "stop" without a discriminating signal destroys normal performance. No training; uses 300+300 feasible/conflict pairs for calibration of steering directions.
- Task-State Representation, arXiv 2607.00502. VERIFIED. State tracker is a SECOND prompted LLM call (not in-policy). AndroidWorld effect: Qwen3.5-plus 61.21 -> 57.76 (-3.45), Kimi-k2.5 51.72 -> 55.17 (+3.45); MobileWorld +12.00 / +9.00. No premature-finish metric, no repetition rate, no variance. Treat as weak/noisy evidence.
- Hidden-state probes for stopping/correctness (arXiv 2504.05419 math reasoning; 2609.00823 late-stage pressure in tool-use agents; 2607.18553 AUROC 0.745). UNVERIFIED (search snippets only), and none is a GUI agent. Cited only as pointers that a linear readout on the policy's hidden state can carry completion/correctness signal.

### 1.3 Training-signal approaches (RL; need online rollouts)
- MobileGUI-RL, arXiv 2507.05720, Sec. 3.4.2 and Table 3 (AndroidWorld). VERIFIED. Failed trajectories get R = -beta_max * (1 - |tau|/T_max): earlier exit is punished more (Eq. 3-4). Motivation: sparse-reward agents "give up" early. The ablation "w/o Decaying Reward" replaces the WHOLE multi-component reward with binary and gives 30.0 -> 23.5 (7B) and 44.8 -> 35.5 (32B); it does NOT isolate the premature penalty and the paper gives no early-exit rate. So the effect of the termination penalty alone is unmeasured in the source.
- LACL-GUI, arXiv 2608.21830. VERIFIED. Contrastive RLVR; negatives are ranked by how many erroneous steps follow divergence from a reference success (SSIM of screenshots, theta=0.85). OSWorld 222-task subset: 8B 50.0 vs REAL 47.3 / GRPO 47.2 (about 6 tasks, no seeds, best checkpoint selected on eval). Explicitly does NOT target premature stop: an early stop gets gentler suppression (a gap in the method).

## 2. Error compounding over long horizons, and cheap fixes without a second model
- Success vs task length, one policy, AndroidWorld 116 tasks: LLaDA-UI, arXiv 2609.13287, Sec. 5.2.5 Table 7 (intermediate checkpoint, a diffusion decoder, not single-pass): optimal actions 1-5: 34/54 (62.96%), 6-10: 17/35 (48.57%), 11-20: 8/20 (40.00%), 20+: 1/7 (14.29%). VERIFIED. Exact repetition 5.41% of steps in successful vs 26.70% in failed trajectories (Table 6 / text). Valid-format rates 98.6-100%, so failures are not syntactic. Same qualitative shape as ours (monotone fall with length); different harness and model, so no numeric comparison.
- GraceBench, arXiv 2610.19900, Sec. 4.2-4.3. VERIFIED. 663/1,294 successful trajectories (51.24%) contain at least one deviation or loop; loop incidence 0.65 in failures vs 0.21-0.27 in successes. Evaluation only, prescribes no fix; does not measure premature termination.
- Idempotent failures: VeriGUI intro says repeating an action that leaves the screen unchanged "is not a corner case". VERIFIED (statement), figure UNVERIFIED (see 1.2).
- Rule-based loop breaking: VLAA-GUI Loop Breaker, Sec. 3 / 4.3. VERIFIED. Tiers 1-2 are deterministic counters (action repeated with no visible screen change n>=2 -> force a different interaction mode; same screen recurring n>=3 -> force strategy change); Tier 3 is a model judge that blacklists the repeated action. Measured: Gemini 3 Flash wasted steps 4.9% -> 2.8%, loop incidence on failed tasks 20.7% -> 16.2%; OSWorld 50-step Gemini full 63.14 vs 58.90 w/o Loop Breaker; Sonnet 4.6 at 100 steps +0.04 (negligible). So effect is large only when step-limited and for a loop-prone model.
- Recovery training data: RoTS, arXiv 2605.29447 (VERIFIED): 800k synthesized recovery samples, SFT; GUI-RobustEval Table 2: RoTS-32B 49.7 at error depth 0 vs 33.2 at depth 5; best open baseline at depth 5 is Qwen3-VL-8B 25.0. Built with critics/reflector in data generation only, so the deployed model is one model. Needs large online rollouts. SGCD, arXiv 2606.18890: abstract only, "low-30%" to "over 50%" on OSWorld-Verified (UNVERIFIED beyond the abstract).
- Pure compounding statements ("one local error moves the UI into a wrong state", Qwen-UI-Agent 2607.28227) are UNVERIFIED (snippets).

## 3. Training data that teaches termination
- Hard negatives for premature done: the only direct recipes found are (a) VeriGUI's 30% synthetic failure-recovery samples (Table 5, 29.7 -> 45.5 RSR, in-policy; targets wrong/no-effect actions rather than premature done), (b) ConflictGuard's forced contrastive suffix pairs (terminate vs over-compliant action on the same prompt) used for steering-vector calibration, not finetuning, (c) MobileGUI-RL's exit penalty (RL). I found no paper that labels not-yet-complete intermediate screens as "not done" for SFT and reports an effect. UNVERIFIED absence.
- Our own corpus fact (from the repo record, not the literature): AndroidControl `status` is the final row of all episodes, so "is the task done" = "is this the last step" (137/137 on val; 0 positives among 687 non-status rows). Premature-done negatives therefore do not exist in the data; they must be constructed (states from our own rollouts that look done but fail the checker are the natural source). INFERRED.

## 4. Ranked interventions satisfying the one-model constraint

1. **Action-effect (no-change) awareness trained in-policy with synthetic and rollout-derived negatives.**
   Source: VeriGUI 2604.05477, Tables 4-6 (RSR 29.7 -> 45.5 with 30% failure samples; 51.1 with GRPO). Claimed effect: recovery/ no-op handling up ~16-21 RSR points on their AndroidControl robustness benchmark; NOT a task-success effect. Our adaptation (readout "did last action change the screen / expected effect", used to suppress repeats) is INFERRED.
   Cheapest offline test: from the existing B0'/R1/Q11 trajectories, label each step by screen change (pixel/a11y hash between consecutive frames); (i) measure repeat-on-unchanged rate per task (we have 41-66%); (ii) fit a logistic probe on the logged readouts (top-1, margin, entropy, step index, previous action equality) to predict "no effect", episode-clustered CV, report AUROC with CI. Existing online AUROC(confidence -> effect) is 0.59 [0.48, 0.69]; the new probe must clear it clearly.
   Drop if: held-out clustered AUROC for predicting no-effect < ~0.70, or if a free rule (screen-hash unchanged) already gives all the signal (then it belongs to item 2, not a trained head).

2. **Deterministic loop-breaker on the same forward pass (no training).**
   Source: VLAA-GUI Loop Breaker Tiers 1-2 (n_action>=2, n_screen>=3), 2604.21375 Sec. 4.3 / Fig. 3. Claimed effect: wasted steps 4.9% -> 2.8% (Gemini), 50-step success +4.2 pts, Sonnet at 100 steps +0.04. Tier 3 (model judge) is excluded. Whether masking the repeated action and taking the next-best head output counts as "one model" is a policy call for the operator (INFERRED).
   Cheapest offline test: (a) on AC-v2 398 rows plus training episodes, count rows where gold action == previous action; (b) replay logged readouts of the repeat-on-unchanged steps and check whether the runner-up action differs in type/target and is valid; (c) on the logged online trajectories, count how many of the 90 declared-done / failed tasks contained >=2 identical no-effect repeats (upper bound on what could change).
   Drop if: gold repeats the previous action in more than a few percent of AC rows (masking would then cost teacher-forced accuracy), or if fewer than ~5 failed tasks have a loop that could have been broken before the budget ended.

3. **Done head with constructed hard negatives and a selective threshold (accept/abstain on "done").**
   Source for the need: VLAA-GUI false-DONE 86-95% of failures; SPA-Bench premature/overdue trade-off; ConflictGuard gate ablation (an ungated push toward stopping cost 24-48 points of feasible SR). No source reports a trained in-policy head with an AndroidWorld effect, so the effect size is UNKNOWN; the closest in-policy evidence is item 1's negative-ratio ablation.
   Cheapest offline test: among the 61 (new model) / 90 declared-done episodes, test whether cheap, already-logged features predict "declared-done and success" (p(status), margin, step index, screen-changed-since-start, number of distinct screens): AUROC with episode-clustered CI, plus risk at the observed coverage vs the 14/61 base. Include the 7 tasks whose end state already holds as a known-positive stratum (a "no change since start" done is right on exactly those).
   Drop if: AUROC CI includes 0.5 or risk-at-coverage does not beat 23% precision (14/61) by a margin larger than its binomial CI. Note the repo already recorded p(status) thresholding as monotone-harmful on AC-v2 (M12): this test must be run online-trajectory-side, not on AC-v2.

4. **Premature-exit penalty in an RL/rollout-weighted objective.**
   Source: MobileGUI-RL Eq. 3-4, Sec. 3.4.2 (ablation lumps the decaying reward with the other terms: 30.0 -> 23.5 at 7B, 44.8 -> 35.5 at 32B; the termination penalty alone is unmeasured). Single policy, but needs online rollouts and GPU.
   Cheapest offline test: reward replay on existing rollouts: compute the penalty-weighted return per failed episode and check that premature-done episodes (median 7 steps) are separable from timeout episodes by the weights; count how many failed episodes terminated before step ceil(0.5*budget).
   Drop if: premature episodes are mostly at step 0-1 (the penalty then equals a constant and teaches nothing), or if the offline count shows fewer than ~20 episodes with a usable signal.

## 5. What not to cite
- The VeriGUI 72.3% figure as an established fact (source unverified).
- Any cross-paper comparison of AndroidWorld/OSWorld/AndroidControl numbers above; harnesses differ (e.g. LLaDA-UI 51.72% is an intermediate checkpoint of a diffusion decoder).
- Search-snippet-only items marked UNVERIFIED above.
