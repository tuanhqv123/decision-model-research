# Why R1 scores 12–15% on AndroidWorld — diagnosis from the validation set and the online episodes (2026-11-11)

Post-hoc, not pre-registered. Every number below comes from a file in `benchmarks/results/2026-11-10-stage1/`
(`val_errors.py/.txt`, `bench_failures.py/.txt`, `appresolve.json`, `posthoc2.txt`) or from the raw episodes on the
emulator host (command at the end). R1 = contract-v1 model, seeds 0 and 1 online (seed 2 still running);
B0′ = the previous model (E2). Validation = 827 AndroidControl rows / 137 episodes, 3 seeds, teacher-forced.

## 0. One-paragraph answer

The model fails mostly **before** any fine motor problem: it often cannot open the right app (the app name is copied
out of the goal text, so a goal that does not name the app yields an invalid name), it clicks by guessing pixels from a
720-px screenshot instead of choosing among the accessibility-tree elements the benchmark provides, it cannot answer
the 25 information-retrieval tasks at all, and it says "done" one step too early. Under the training recipe (one epoch
of 15k single-step AndroidControl demonstrations, imitation only, gold history) none of these can be learned well, so
per-step accuracy in the middle of an episode is 55%, and even with the TRUE history only 15–17 of 137 validation
episodes are right at every step — the same order as the online 12–15%.

## 1. Where the points go (online, 116 tasks)

| Bucket | Tasks | R1 successes | Cause (measured) |
|---|---|---|---|
| Information-retrieval (`answer` only) | 25 | 0 | contract v1 has no `answer` action; scored only on the answer text (paper App. C.2, D.3) |
| Task-completion, no `open_app` ever resolved | 24 (s0) / 25 (s1) | 8 / 8 (≈ the 7 "state already holds" tasks) | 29% / 36% of R1's `open_app` names are not app names AndroidWorld can launch: `''`, `wifi`, `bluetooth`, `brightness`, `timer`, `contact`, `video`, `Downloads in the file manager`, … (`appresolve.json`, checked with `adb_utils.get_adb_activity` inside the container) |
| Other task-completion | ~65 | 6 / 9 | below |

Failure taxonomy of the failed task-completion episodes (`bench_failures.txt`; first matching rule; app map and
"needs typing" are heuristics):

| | R1 s0 (76 failed) | R1 s1 (73) | B0′ (82) |
|---|---|---|---|
| wrong or unnamed app | 23 | 23 | 30 |
| said done early (≤ half the step budget) | 21 | 26 | 5 |
| said done later, still wrong | 12 | 4 | 3 |
| ran out of steps in a no-effect loop | 10 | 12 | 25 |
| task needs text, never typed | 6 | 7 | 9 |
| ran out of steps while moving | 4 | 1 | 10 |

Clicks (raw episodes, `click_on_interactive` × `ui_changed`):

| | R1 s0 (920 clicks) | R1 s1 (753) | B0′ (1,406) |
|---|---|---|---|
| hit a non-interactive element, nothing changed | 29% | 21% | 19% |
| hit a non-interactive element, screen changed | 8% | 9% | 5% |
| hit an interactive element, nothing changed | 32% | 36% | 46% |
| hit an interactive element, screen changed | 31% | 34% | 31% |

So ~30–37% of R1's clicks land on something that is not clickable, and another third land on a clickable element
that does nothing (re-clicking the current tab or item). The model is sent **no element list** (`elements_sent = 0`;
track V): it predicts a box over a 0–999 grid from the screenshot only.

Action mix (R1 s1): click 66%, open_app 12%, wait 9%, status 8%, type_text 3.4%, scroll 1.8%, navigate_back 0.1%.
Of 20 tasks that need typed text, R1 typed in 13–14 and solved 0.

## 2. Where the points go (offline validation, 3 seeds)

| gold type | rows | step acc | type acc | grounding |
|---|---|---|---|---|
| click | 420 | 57.7% | 89.2% | 64.7% |
| status | 137 | 86.9% | 86.9% | – |
| scroll | 86 | 53.9% | 55.0% | – |
| wait | 56 | 30.4% | 30.4% | – |
| type_text | 53 | 61.0% | 83.6% | 72.9% |
| open_app | 45 | 83.0% | 94.8% | – |
| navigate_back | 29 | 60.9% | 60.9% | – |

* Wrong rows split 52% wrong action type / 48% right type but wrong element or value (397 of 945 are clicks on the
  wrong element). Wrong clicks are far from the target (median 464 px, 58% > 400 px, 9% < 100 px): they are
  **choices of a different element**, not near misses.
* Position: first step 63.7%, middle steps 55.3%, last step 86.9% (the last step is always `status`).
* Premature `status`: 56–59 rows per seed; 103 of the 171 (3 seeds) are exactly **one step before the end** (the final
  save/confirm/send click), 31 two steps before. Gold `wait` is predicted `status` 61/168 times.
* Gold `scroll` is predicted `click` 20% and `status` 16% of the time; direction is almost never the problem (3 rows).
* Confidence: on predicted clicks, `box_conf` separates right from wrong with AUROC 0.81 (usable for a gate);
  on `type_text`, confidence carries no signal (0.48–0.51).
* Seeds: rows right in 0/1/2/3 seeds = 238 / 76 / 79 / 434. A 3-seed majority-vote ensemble gives 63.5% vs 61.9%
  single (+1.6); the any-seed oracle is 71.2%. Ensembling is not the lever.
* Fully-right episodes (true history): 15 / 17 / 15 of 137; by length 1–2: 4–5/7, 3–4: 5–10/43, 5–8: 3–5/66, 9+: 0/21.

## 3. Root causes, mapped to the recipe

| # | Symptom | Root cause in what we built | Evidence |
|---|---|---|---|
| C1 | invalid app names; 25 tasks never start | `app_name` is a **pointer copy from the goal**; AndroidWorld goals often do not name the app ("Turn wifi off", "Take one photo") | §1 table; `appresolve.json` |
| C2 | 30–37% clicks on non-clickable targets; far wrong clicks | clicks are **pixel boxes from a 720-px screenshot**; the a11y element list AndroidWorld provides (and M3A uses) is never given to the model; the box loss is flat for misses beyond ~200 bins (2026-11-07 mechanism) | §1 click table; val distances |
| C3 | 25 IR tasks = 0 | no `answer` action and no way to produce answer text | paper App. C.2/D.3 |
| C4 | "done" one step early; wait → done | AndroidControl ends **every** episode with `status` (13,603/13,603), so "done" is learned as "screen looks finished", never as "goal verified"; no negatives | audit D7; val premature rows |
| C5 | loops on unchanged screens | history renders clicks as bare `click` (no target, no effect), no previous screenshot; training history is always the gold, correct one (PRE-REG A4) — the model never saw its own mistakes (exposure bias) | 21–36% no-effect clicks; loop bucket |
| C6 | long tasks ≈ 0 | imitation of single steps only; no trajectory-level signal, no reward from task success, no recovery data | 9+ step episodes 0/21 even teacher-forced; budget 41+ 0/11 online |
| C7 | weak transfer to AW apps | training = 15k of the 88k AndroidControl train decisions, one epoch (938 × 16 = 15,008 samples); 13/20 AndroidWorld apps have zero training goals | `run_v1_seeds.sh`; audit |
| C8 | scroll/typing under-used | class imbalance in the 15k set; typing chosen in 2–3% of decisions | action mix |

## 4. What we are not doing (ML/DL), and which cause each addresses

Each line: the cause it targets → what it costs → how it fits a single-pass decision model. Literature numbers:
`outputs/2026-11-11-beyond-sft-gui-agents.md` (separate survey).

1. **App choice head** (C1): classify over the fixed list of launchable apps instead of copying a span. A pure
   decision-model fit (a `choice` over ~25 names). Needs goal→app supervision for goals that do not name the app.
2. **Element choice over the a11y tree** (C2): pick an element id from the AndroidWorld element list (what M3A does),
   with the box head as fallback. Our own matched-weights test (R4, 2026-11-07): candidate interface +6.0 step-acc
   points, +10.2 grounding at step 300; it collapsed when the list was long (> 40 candidates), so the pre-registered
   two-stage shortlist (median 6 candidates; gold inside the shortlist on 78.5% of AC-v2 spatial rows, vs E2's 56.05% grounding there) is the version to run. The
   AW-format corpus (`compact_*_v2`, gold inside the list 93.8%) already exists.
3. **Answer head** (C3): copy a span from on-screen element text (titles, places) + a small count/number head for
   "how many / total" questions. Needs IR-style training data that is NOT AndroidWorld's own tasks.
4. **Termination learned from verification, not position** (C4): negatives where the screen looks finished but is
   one step short (exactly the 103 val rows), and a done readout conditioned on goal vs screen.
5. **History that carries effect and target** (C5): render "click 'Save' → screen changed / no change", add the
   previous screenshot or a screen-diff flag; mask a repeated no-effect action at inference (deterministic, same model).
6. **On-policy data — DAgger / self-training on own rollouts** (C5, C6): run the model on an emulator on tasks we
   generate (not the 116), label or filter, retrain — fixes exposure bias.
7. **Reinforcement learning from task success** (C6): a decision model is a softmax policy, so policy-gradient
   (GRPO/REINFORCE over the action and element heads) applies directly — no text generation needed. Needs an
   emulator task pool with checkers, which AndroidWorld's task framework can generate for apps and templates that are
   not the evaluation instances (contamination rule to be set by the operator).
8. **More and broader data** (C7): all 88k AndroidControl decisions, several epochs, trajectory corpora
   (GUI-Odyssey, AMEX for element grounding), class re-balancing for scroll/type.
9. **Resolution** (C2): 720 px width downsamples 1080-px screens; small text is lost. Cheap to test (the 858 px
   setting exists).

## 5. How the raw click numbers were produced

On the emulator host, over `~/awruns/aw116_{r1s0,r1s1,b0prime}/episodes/*.jsonl` (abandoned files excluded): for every
step whose `action_executed.action_type == "click"`, count `(click_on_interactive, ui_changed)`.

## 6. What the literature measures (survey `outputs/2026-11-11-beyond-sft-gui-agents.md`, 2026-11-11)

Spot-checked by hand: MobileRL Fig. 3a / Table 7 (arXiv 2509.18119v2) — AndroidWorld 27.6 → **50.2 after
reasoning-free SFT** (Qwen2.5-VL-7B; 9B: 7.7 → 48.1) on 97.9k steps (21.3k AndroidControl-Low + 14.3k
AndroidControl-High + 62.2k human-annotated), → 56.8 reasoning SFT, → 72.0 AdaGRPO. Other numbers are the survey's,
each with its table, not re-verified here. Reading for us (INFERRED): the largest small-model gains come from
**multi-step trajectory data in the AndroidWorld apps** (OpenMobile, UI-Voyager's rollout fine-tuning), which needs
operator decision 1 of the R3 spec; online GRPO at small scale gains little; V-Droid (candidate scoring, closest to
our head) reaches 59.5 with a pairwise loss over candidate actions and depends on its memory module.
