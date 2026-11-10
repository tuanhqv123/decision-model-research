# AUDIT 2026-11-10 — AndroidWorld contract vs what we built

*Every line below was checked by reading the primary artifact myself: upstream `android_world`
@ `55ddaf5` (2026-10-23), our code, our corpora, and the 104 episodes of the `aw116_appint` run.
Prior notes (HANDOFF, A5, GOAL) were treated as hypotheses, not evidence. Two of them were wrong.*

---

## 1. What AndroidWorld actually requires (verified in source)

| Item | Contract | Source |
|---|---|---|
| Input | `State.pixels` (screenshot) **and** `State.ui_elements` (a11y list). Both are official inputs, not privileged. | `env/interface.py:42-56` |
| Element list | `forest_to_ui_elements`: keeps a node only if it is a **leaf**, has `content_description`, or is scrollable | `env/representation_utils.py:136-161` |
| Element fields | text, content_description, hint_text, class_name, bbox_pixels, **is_checked, is_selected**, is_clickable, is_editable, is_focused, is_scrollable, … | `representation_utils.py:53-77` |
| Output | one `JSONAction` of 14 types: click, double_tap, long_press, input_text, keyboard_enter, navigate_home, navigate_back, scroll, swipe, open_app, wait, **status**, **answer**, unknown | `env/json_action.py:24-54` |
| Grounding | `index` **or** x,y. With `index`, the env taps the exact bbox centre of `ui_elements[index]` | `env/actuation.py:42-72` |
| `input_text` | clicks the field itself, (optional `clear_text`), types, **always presses Enter** | `actuation.py:74-105` |
| Scoring | `agent_successful = task_successful if interaction_results.done else 0.0` — **no `status` ⇒ 0**, even if the device state is correct | `suite_utils.py:261` |
| Answer tasks | **25/116** are scored only on the `answer` text (`interaction_cache`); empty ⇒ 0 | `information_retrieval.py:109-118`, counted from `task_metadata.json` × IR textproto |
| Infeasible | no task reads `goal_status`; `complete` and `infeasible` both just end the episode | grep of the repo |
| Budget | `int(10 * task.complexity)` per task (median 12, max 120) | `suite_utils.py:491-502` |
| Protocol | seed 30, `n_task_combinations=1`, report per task and the average | `run.py` defaults |
| Official best agent | M3A: element list with state flags + SoM + **index actions** + a one-line summary of each past step as history | `agents/m3a.py:40-260` |

## 2. Where we deviate — each one checked

| # | Deviation | Evidence | Effect |
|---|---|---|---|
| D1 | **No `answer` action** | `adapter.EMITTABLE` has 9 types | 25/116 tasks are unreachable. **Hard ceiling 91/116 = 78.4%** |
| D2 | **The element list is extracted differently in training and in AndroidWorld** | train: `preprocess_ac.flat_nodes` keeps **every** node (incl. clickable parent containers); AW: leaves/content-desc/scrollable only | Same screen ⇒ a different option set at train vs serve |
| D3 | **The element list carries no state** | `pick_candidates` keeps only `label` (60 chars) + position; drops `is_checked`, `is_selected`, `is_editable`, `hint_text` | The model cannot see that wifi is ON or that a field is filled — exactly what "is it done?" needs |
| D4 | **History says nothing about what was acted on** | train history is `click` (bare) in 99,536 click lines; serve mirrors it | "click, click, click" carries no state; consistent with the observed loops (CameraTakePhoto: 7 identical clicks) |
| D5 | **Three different history distributions** | E2/cur15k: human-written instructions ("Open the YouTube Music app"); episodic-s0: bare action strings; AW: self-generated | The AC-v2 score is teacher-forced; AndroidWorld can never provide that input |
| D6 | **Domain: 13 of 20 AW apps never appear in training** | 0 goals mentioning Markor, Simple Calendar, Pro Expense, Joplin, OsmAnd, Retro Music, VLC, Audio Recorder, Simple SMS, Gallery, Draw, Open Tracks, Chrome (4,551 unique cur15k goals) | The model has never seen most of the benchmark's apps |
| D7 | **Termination has no training signal** | AndroidControl appends an artificial terminate as the last step of every episode; `is_done` ≡ `action==status` (137/137 val) | Learns "the last step is status", not "the goal state is visible" |
| D8 | **Grounding by pixel box although `index` is allowed and exact** | adapter emits x,y from the box head | A whole error class (437 px median misses) that the contract does not require us to have |
| D9 | **The served checkpoint is `episodic-s0`**, which failed its own pre-registration (−8.29 on AC-v2) | HANDOFF-2026-11-10 §2 | 2026-11-10 benchmarked a checkpoint already known to be worse |
| D10 | App memory line: trained on 100% of E2/episodic rows, never served; 64.07 was measured `--no-memory` | `train_d1.py:467` default on; `2026-10-29-d1-acv2-sft.md:76` | Another train ≠ serve input |

## 3. Claims in earlier notes that are wrong

- **HANDOFF-2026-11-10 P4 ("eval_d1 drops history; 64.07 is a lower bound") is FALSE.** Test rows carry
  `previous_actions` on 315/398; `val_row` maps `history → previous_actions`. 64.07 stands as measured.
- **"−8.29 is is_done consuming capacity"** ignores the obvious confound: `episodic-s0` was trained on
  bare-action history and evaluated on human-instruction history (D5).
- **"15 successes" in the aw116 run is 12 under the official metric.** SimpleCalendarDeleteOneEvent,
  SystemBrightnessMaxVerify, SystemWifiTurnOffVerify never declared `status` (`suite_utils.py:261` ⇒ 0).
- **"The model cannot type" is wrong.** In 24/24 `type_text` decisions the pointer read produced the right
  string ('Emily Mohammed', '965-688-5552', 'Lentil Soup', …). The trace recorded only the `click`; the old
  adapter sent click + input_text, and input_text clicks again (the double tap fixed 2026-11-10). What is
  actually rare is *choosing* to type: 24 of ~1,400 actions.

## 4. Where the run succeeds and fails (aw116_appint, official done gate)

| optimal_steps | ran | success | of which answer-only (unreachable) |
|---|---|---|---|
| 1–2 | 16 | 6 | 7 |
| 3–5 | 37 | 4 | 12 |
| 6–10 | 32 | 2 | 4 |
| 11+ | 18 | 0 | 1 |

Suite median `optimal_steps` = 6. Of the 9 *feasible* 1–2-step tasks, 6 succeed. Everything longer
collapses: no domain data (D6), no state in the element list or history (D3, D4), no termination signal (D7).

## 5. Data audit (2026-11-10, second pass) — what is in the data that we had not looked at

Scripts: scratchpad `audit/a1.py`, `audit/a2.py`. Corpus = full `compact_train.jsonl` (88,288 rows,
13,603 episodes), `compact_validation.jsonl`, AC-v2 `500_steps_filtered.json`, 104 aw116 episodes.

### 5.1 Training corpus (AndroidControl train)

| # | Finding | Count | Why it matters |
|---|---|---|---|
| T1 | `status` is **always the last row, always `successful`** | 13,603 / 13,603; 0 mid-episode; 0 infeasible | Termination is learned as "this is the final screen", never as "this screen satisfies the goal". There are no negatives (a screen that *looks* done but is not). |
| T2 | **History is human-written instructions, not actions; one instruction can span several actions** | last two history entries identical on **13,053** rows | History count ≠ action count semantics; the text often states the *plan* ("Swipe up to Favorites option"). Not reproducible at serve time. |
| T3 | **Human scroll wording contradicts itself**; the gold `direction` field is consistent | gold `down` ↔ "swipe up" 4,624, "scroll up" 1,524, **"scroll down" 1,206** | The gold field matches AW (`down` = reveal content below). The instruction text in history is noise for direction. ~265 rows (≈3%) have gold direction contradicting both phrasings. |
| T4 | **Gold target covers >50% of the screen** | 747 spatial rows (1.4%) | "Smallest node containing the click" falls back to a full-screen container when the widget has no a11y node (WebView, canvas). The box head is taught "box = whole screen". |
| T5 | Zero-area gold target / point off-screen / spatial row with no target | 51 / 13 / 54 | Unscoreable supervision; should be dropped. |
| T6 | Gold element absent from the candidate list | 1,135 / 51,870 (2.2%); exact bbox match 82.1%, centre-in 15.7% | Acceptable, but measured on **our** extraction (all nodes), not AW's (leaves only) — see D2. Must be re-measured after switching extractors. |
| T7 | `type_text` string **not** verbatim in the goal | 871 / 5,406 (16.1%) | Mix of legit derivations ("Knoxville." from "from Knoxville", emails from contacts) and **annotation errors** (goal "open the JioMart app" → types "egg tray"). Pointer read cannot supervise these; they are labelled `escalate`. |
| T8 | `open_app` name not verbatim in goal | 695 / 5,044 (13.8%) | Aliases ("google files" → "Files by Google"), unicode junk (`Contacts﻿+`), and world knowledge ("build number" → Settings). |
| T9 | Goals that are really a single-step command | 58 episodes | Goal-annotation errors ("Click on the View Cart button at the bottom."). |
| T10 | Episode length | median 5–6 steps, p90 11, max 92; 1,022 episodes ≤ 2 steps | AW median `optimal_steps` is 6 but budgets reach 120; the long tail is under-represented. |
| T11 | Screens | 1080×2400 on 96.2% of rows (same as AW's Pixel 6) | Not a problem. |
| T12 | No `navigate_home` rows, 157 `long_press` | — | Two actions AW tasks use are almost unsupervised. |

### 5.2 Evaluation sets

| # | Finding | Count |
|---|---|---|
| E1 | Episode overlap train∩val∩test | **0** (clean) |
| E2 | AC-v2 test goals that appear verbatim in train (different episodes) | **14 / 398** — mild template leakage, disclose |
| E3 | AC-v2 spatial rows whose smallest gold node is >50% of the screen | 7 / 223 — grounding there is near-free |
| E4 | AC-v2 rows with ≤5 a11y nodes (WebView/media apps) | 6 / 398 |

### 5.3 AndroidWorld runtime data (aw116 traces + upstream source)

| # | Finding | Evidence |
|---|---|---|
| R1 | Element lists at runtime are **healthy** in this run | median ~23 elements/step; 2 / 1,443 steps empty |
| R2 | `execute_action` resolves `index` against a **fresh** `get_state(wait_to_stabilize=False)`, not the state the agent saw | `interface.py:306-310` → if the UI moved, `index` hits a different element. **Execute with the x,y centre of the element we chose** (what the adapter does), not the raw index. |
| R3 | `scroll` with `index=0` is treated as "whole screen" | `actuation.py:141` `if action.index:` — avoid index-0 scrolls. |
| R4 | Element list = leaves / content-desc / scrollable, **invisible excluded** | `android_world_controller.py:241-246` (`exclude_invisible_elements=True`) — training must use the same rule. |
| R5 | The a11y forwarder degrades after hours of uptime | measured 2026-11-09 (12 setup errors); not visible in R1 because it shows as setup failure, not empty lists |
| R6 | 4.9% of steps did not change the screen | 70 / 1,443 — loops are a smaller share than the 25-step timeouts suggest; most failing episodes move, just in the wrong direction |

## 6. What data AndroidWorld needs that we do not have — research (2026-11-10)

Decision taken by the operator 2026-11-10: **AndroidWorld is the single primary benchmark**; AC-v2 is kept
as a past result only.

### 6.1 What the contract needs vs what each source provides (files parsed, not cards)

| Need (from §1–§5) | AndroidControl (have) | OS-Genesis AW (public) | OpenMobile (public) | Own collection in the AW emulator |
|---|---|---|---|---|
| AW's 20 apps | 7/20 apps barely; 13 absent | **yes** (AW emulator) | **yes** (20 apps, AW task inits) | yes |
| Element list in AW format (`forest_to_ui_elements`, bbox + state flags) | raw forest, re-extractable | **no** — `{label: centre}` dict only, no bbox, no flags | **no** — screenshots only | **yes** (`State.ui_elements`) |
| History reproducible at serve time | no (human instructions) | model-written "step N: thought + action" | model-written step summaries | yes (we render it) |
| Termination supervision | last row only, always `successful` | `status` 1,688 rows | `terminate` ends 289/517 trajectories | **per-step checker label** (`task.is_successful`) |
| `answer` supervision | none | 210 rows | 70/517 trajectories end in `answer` | yes (IR templates) |
| Size | 88k steps | 27,190 steps / 7,329 instructions | ~34k steps / 2.8k instructions (split4: 3,855 / 517) | as much as we collect |
| Action format | AC JSON | **AW JSON** (`click x,y`, `open_app`, `status`, `answer`, `keyboard_enter`) | Qwen `mobile_use` (999×999 grid) + target bbox | AW JSON |
| Overlap with the 116 test templates (token Jaccard vs `task_template`) | — | ≥0.5: **34 / 7,329**, ≥0.7: **0** | ≥0.5: **1 / 517**, ≥0.7: 0 | must exclude seed-30 instances |
| Quality issues seen in the file | §5.1 | malformed synthesized params (`+14587371174*,6`, `Here 15 (Sul)`) | 125/517 trajectories end on a non-terminal step (expert-only steps kept) | teacher-dependent |
| License | Apache-2.0 | **MIT** (HF card) | code Apache-2.0; **data license not stated** | ours |
| Published value | — | AW 9.82 → 17.41 (paper's own number) | Qwen2.5-VL-7B **51.7%**, Qwen3-VL-8B 64.7% (paper's own numbers) | MobileRL builds 2,000 train tasks from AW templates with new params, removing exact test overlaps (App. D.4.1) |

### 6.2 Judgment

1. **No public dataset ships AW's observation format.** OS-Genesis and OpenMobile give in-domain *behaviour*
   (which apps, which screens, when to stop, when to answer) but not the element list with state flags that
   the contract offers and that "is it done?" depends on (D3).
2. **Only the emulator can give a real completion label.** `task.is_successful(env)` evaluated after every
   step of a rollout on a *training* instance gives positives and negatives for "the goal is satisfied on this
   screen" — the signal AndroidControl structurally lacks (T1). This is the MobileRL recipe (2,000 template
   instances, exact test overlaps removed). It must be disclosed as *same templates, different parameters*.
3. **Cheapest route to AW-format observations for public trajectories: replay them in the emulator.**
   OS-Genesis actions are AW JSON at 1080×2400 from the AW snapshot; replaying them and recording
   `State.ui_elements` at each step would convert behaviour data into contract-format data. Unmeasured —
   needs a 20-trajectory spike measuring how often the replayed screen matches the recorded one.
4. **AndroidControl stays, re-extracted** with `forest_to_ui_elements` + state flags + element-aware
   history, minus the broken rows of §5.1 — it is the only large source of generic Android UI behaviour.

## 7. Self-review — which of the above is evidence and which is my guess (2026-11-10, slow pass)

Rule applied: a claim is **FACT** only if a file or source line shows it; **EFFECT** claims (it causes
failures / fixing it raises SR) need a measurement on our model, and none of them has one yet.

| Claim made earlier today | Status | Why |
|---|---|---|
| 25/116 tasks need `answer`; ceiling 78.4% | **FACT** | source + metadata. But at ~12/116 the ceiling is not binding; it is not the first thing to fix. |
| No `status` ⇒ 0; official count is 12 not 15 | **FACT** | `suite_utils.py:261`; 3 episodes without `declared_done` |
| Train extracts all nodes; AW extracts leaves/content-desc/scrollable | **FACT** (mechanism) / effect **UNMEASURED** | How often the gold element is a parent container that AW drops needs the raw forest (on the training host) — measure before acting. |
| `status` is always the final AC row, never a negative | **FACT** | 13,603/13,603 |
| The served checkpoint failed its pre-registration | **FACT** | |
| "The model cannot see that wifi is ON" | **WITHDRAWN** | The screenshot shows the switch. Missing `is_checked` text is a redundancy gap, not blindness. |
| "Domain (13 unseen apps) is the dominant cause" | **WITHDRAWN as stated** | Cross-tab: seen-app long tasks **0/6**, unseen-app long **3/66**, seen-app short **8/19**, unseen-app short **1/12**. Length and familiarity are confounded at n≈100; long tasks fail even in familiar apps. |
| "Bare `click` history causes the loops" | **HYPOTHESIS** | Only 4.9% of steps leave the screen unchanged; causation not shown. |
| "Typing failed because of the harness double tap" | **HYPOTHESIS** | The pointer read produced the right strings 24/24 (FACT); whether the text landed is not in the trace. |
| "Index actions remove the grounding error class" | **mechanism FACT, benefit for our model MIXED** | On matched weights VA beat V (+6.03, p=1.5e-3, 2026-11-07); the 938-step VA arm then lost −7.29 with a 14-candidate cap that dropped 58.6% of candidates. Not settled. |
| "Replay OS-Genesis in the emulator to get AW-format observations" | **IDEA, no evidence** | Do not plan on it. |
| "OpenMobile shows in-domain data works" | **WEAK** | 51.7% is a 7B with CoT and a different base; it does not isolate the data. OS-Genesis' own ablation (9.8 → 17.4) is the only in-paper evidence that AW-domain synthetic data helps. |

**Consequence for the plan.** Nothing should be built on a HYPOTHESIS row. The first step is not new data or
a new head — it is to *measure what we already have under the official contract*, then size each hypothesis
with an offline count before spending GPU on it.

## 8. Harness code review against `suite_utils` / `episode_runner` (line by line)

| # | Our code | Upstream | Effect | Severity |
|---|---|---|---|---|
| H1 | `runner.py:236` `task_successful = bool(task.is_successful(env))` | returns a **float**; composite tasks return 0.5; SR = mean of floats (`suite_utils.py:261, 684-716`) | a half-credit episode counts as a full success | wrong number |
| H2 | `run_bench.py:253` `task_class.generate_random_params()` with **no seed** ("takes no seed in this version") | `_instantiate_task`: `random.seed(sha256(f"{seed}_{name}_{i}") % 2**32)` **then** `generate_random_params()` (`suite_utils.py:97-103, 149-167`) | every run evaluates a **different random set of instances**, never the official seed-30 set; runs are not comparable to each other or to any paper | wrong benchmark |
| H3 | `runner.py:188-209` `env.reset(go_home=True)` **before** `initialize_task` | `initialize_task` first, then `agent.reset(go_home=task.start_on_home_screen)` — `True` for every non-MiniWoB task (`episode_runner.py:79`, `task_eval.py:40`) | if initialisation leaves an app open, our agent starts there instead of on the home screen | wrong start state |
| H4 | `dump_step_budgets.py` reads `complexity` from the **class** | IR instances set `self.complexity = self.task.complexity` from the proto in `__init__` (`information_retrieval.py:70`), overriding the class value | 2 IR tasks get 12/22 steps instead of 10/20 | small |
| H5 | `awenv_bridge.screenshot()` and `.elements()` each call `get_state()` (stabilised vs not) | an agent reads one `State` — pixels and `ui_elements` of the same moment | the screenshot and the element list can describe two different screens | wrong observation |
| H6 | Denominator: tasks with setup/checker exceptions excluded | `_create_failed_result` → `NaN`; pandas `mean()` skips it | **matches upstream** (A5 §1 said otherwise; A5 was wrong) | OK |
| H7 | Done gate `success = task_successful and declared` | `agent_successful = task_successful if done else 0.0` | matches (except H1's float) | OK |
| H8 | Per-task budget from the table | `int(10 * task.complexity)` | matches except H4 | OK |
| H9 | `answer` not emittable; `interaction_cache` never written | 25 tasks scored only on it | 0 on 25/116 by construction (§1) | known ceiling |

**Status 2026-11-10 (branch `aw-official-harness`):** H1 fixed (`6197e44e`), H3 fixed (`6197e44e`),
H2 + H4 fixed (`398dda1a`), H5 fixed (`3ea7d49e`); official SR added to the report (`40740947`; on the
2026-11-08 episodes it reproduces the hand count 12/80 = 15.0%); `typed_landed` recorded per typing step
(`65dff45d`). Every fix is pinned by a test citing the upstream line. Open before the baseline run:
which of the image's two `android_world` copies executes the tasks.

## 9. Infrastructure note

`git` was hanging because `core.fsmonitor=true` and the daemon stalls on a 55 GB tree. Set to `false`
locally on 2026-11-10 (reversible: `git config core.fsmonitor true`). With it off, `git status` takes 0.06 s.
Stray file at repo root: `ok:", has_arrow)\nPY` — a heredoc accident, safe to delete.
