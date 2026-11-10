# AndroidWorld / AndroidLab plan (2026-10-29)


> **Correction (2026-11-07).** The headline in this note predates the internal audit: **64.32% (256/398) exists
> in no artifact** — the pure numbers are 64.07% (255/398) in-process and 63.82% (254/398) served, and the
> 56.50% grounding is the step-705 checkpoint (938 pure is 56.05%). See
> `benchmarks/results/2026-11-03-acv2-error-budget-and-routed-gate.md` and paper §5.1.

Decision (user): AndroidWorld + AndroidLab are the reference benchmarks; keep d1-2B; write our own AGENT HARNESS, but score with the
benchmarks' own environments, tasks and checkers so numbers stay comparable.

## Where AC-v2 ended (official eval_rl.py, pure protocol)
E2 64.07 (Type 84.17, Ground 56.05) | grounding run 64.82 (Type 82.16, Ground 60.09). Validation 63.48. Plateau 64.1-64.8. *(Corrected 2026-11-07; before the audit this line read 64.32 / 56.50.)*
Errors (new run, 140 wrong): 71 wrong type (34 involve status = "done yet?"), 58 clicked the wrong place (66% are a WRONG ELEMENT
>150px away, median 332 px), 11 wrong detail. Scroll predicted on only 21 of 37 gold scrolls.

## What the repos show (verified = read in this session)
| repo | license | finding |
|---|---|---|
| google-research/android_world | Apache-2.0 | verified: `EnvironmentInteractingAgent.step(goal)` -> `AgentInteractionResult(done, data)`; `env.get_state()` -> pixels + forest + ui_elements; JSONAction types click/double_tap/input_text/keyboard_enter/navigate_home/navigate_back/open_app/scroll/swipe/wait/status/answer/long_press; emulator needs `-grpc 8554`, API 33, datetime frozen to Oct 2023; run.py flags tasks/n_task_combinations/fixed_task_seed/checkpoint_dir; transition_pause default 1.0 s |
| X-PLUG/MobileAgent `Mobile-Agent-v3.5/android_world_v3.5` | MIT | verified: full AndroidWorld fork running GUI-Owl-1.5 (`agents/gui_owl.py`, `run_guiowl15.sh`): talks to a vLLM server over an OpenAI-style API, keeps the last 5 screenshots, resizes screenshots to a model-specific size (`coordinate_resize.py`), builds history from the model's OWN "Action:" text (self-generated, not human-written), has a Chrome first-run init hack (click "Use without an account"), `hide_automation_ui()` |
| THUDM/Android-Lab | MIT | verified from README only: 138 tasks / 9 apps; metrics SR, Sub-SR, RRR, ROR; Docker on Linux x86_64 or AVD on Mac arm64; ~6 GB RAM + 9 GB disk per session; XML and SoM modes |
| Tongyi-MAI/MAI-UI | none stated | verified: no AndroidWorld harness released (only ScreenSpot-Pro grounding eval) |
| inclusionAI/UI-Venus | none stated | no AndroidWorld/AndroidLab paths in the repo tree |
| minitap-ai/mobile-use (3.3k stars), droidrun (9.9k), Open-AutoGLM (26.5k), Core-Mate/OpenGUI | Apache/MIT/Apache/? | frameworks, no AndroidWorld eval code (not read in detail) |
Real engineering issues known from the field (my background knowledge, NOT re-verified now): non-ASCII/space text needs an ADB keyboard
(`adb input text` breaks), app launch needs name->package mapping, popups/permission dialogs, soft-keyboard overlap, screen not yet
stable after an action, loops/stuck detection, scroll-direction conventions.

## Harness design (ours)
- Env/tasks/checkers: AndroidWorld's own (Apache). Agent = `D1Agent(EnvironmentInteractingAgent)`.
- Observation: screenshot (858 px wide) + numbered a11y element list (track VA) + goal + OUR OWN history (templated from our actions,
  e.g. "clicked [12] 'Search'"), never human-written step text (AC-v2 history is human-written; not available online).
- Questions per step (one pass, 0 tokens): action (incl. answer / infeasible / navigate_home / keyboard_enter / swipe), target element
  (choice over the list) + box, scroll direction, text to type / answer (pointer read over the state's a11y text).
- Safety layer: calibrated confidence (p(action) x box confidence, isotonic on validation) -> act / retry / ask / declare infeasible;
  screen-changed check after each action; loop detector (same action + same screen hash 3x -> alternative action).
- Logging: every step's latency (model ms, emulator ms), tokens (=0), confidence, screenshot hash; resumable.

## Gates (stop if one fails)
G0 harness reproduces a published open baseline (GUI-Owl-1.5-2B self-reports AW 67.9) within +-5 pts, run through THEIR agent code.
G1 D1Agent completes all 116 tasks with no crash; baseline number (expected low). G2 retrained d1 beats G1 on HELD-OUT task templates.
G3 gate keeps its risk budget on held-out tasks. Hold-out: train rollouts from ~60 AW templates, report on the other ~56 (+ full suite disclosed).

## Latency / "lighter" claim: what must be MEASURED, not assumed
Measured so far: d1-2B 108 ms/step (batched 8, RTX 4090); single-request latency NOT measured. d1-2B = 1.92B params, MAI-UI-2B 2.13B,
GUI-Owl-1.5-2B 2.13B -> NOT fewer parameters; the claim is no decoding (0 tokens, no thinking, no growing KV cache). End-to-end a step also
pays the emulator and the 1 s settle pause, so the whole-task speedup will be smaller than the model-only speedup. Plan: run GUI-Owl-1.5-2B
and d1 on the SAME machine; report p50/p95 model ms, ms per step end to end, wall time per task, tokens per task, peak GPU memory.

## Open items
- Hardware for emulators: servers 5090 and GPU_HOST_A have /dev/kvm (GPU_HOST_B does not); user has a PC with 32 GB RAM and GTX 1070 (OS/CPU/virtualisation unknown); installing emulators on the 5090 prod host needs the user's OK.
- d1 on Pascal (1070): Triton kernel / bf16 support unverified.
