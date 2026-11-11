# R3 data recipe (2026-11-11) — from the three source reports

Sources: `outputs/2026-11-11-data-recipes-A-models.md` (how strong models were trained),
`…-B-synthesis.md` (how trajectories are made), `…-C-datasets.md` (what the released files actually contain; [F] =
verified from file). Operator decisions: no external model at any time; typing copies from the goal; 2 past
screenshots after the first model; less AndroidControl, more diversity.

## 1. Correction to our own record

Raw AndroidControl has **no `status` action** ([F]: 83,848 actions in all 342 mirror parquets; click 51,918 / scroll
11,130 / input_text 6,091 / wait 5,782 / open_app 5,697 / navigate_back 3,034 / long_press 167 / navigate_home 29).
The terminal `status` row of every episode is **added by our `preprocess_ac.py:126-128`** (the screen after the last
action, labelled `status / successful`). "AndroidControl always ends in status" (AGENTS 2026-11-10 (c), audit D7)
describes our construction, not the dataset. The label is reasonable (the demo is complete there) but it is ours.

## 2. What exists that fits element choice (needs a tree or element boxes per step)

| Source | Per step | Steps usable | AW apps | Licence | Use in R3 |
|---|---|---|---|---|---|
| AndroidControl | full a11y forest, gold element by point-in-node [F] | ~52k click/long-press, 6k typing, 11k scroll | system apps only (Clock 203, Settings 128, Contacts 123, Files 89 train episodes) | Apache-2.0 (card) | **reduced**: keep every episode in AW-like apps and every typing/scroll/back/open_app row; subsample the remaining click-heavy episodes |
| AndroidLab XML (AndroidInstruct) | text a11y tree, `Tap` bbox = a node's bounds 4,183/4,183 [F] | ~6.2k steps / ~726 episodes, ends with `finish` | partial (Clock, Contacts, Calendar, maps …) | check (Google Drive zip) | **all**, converted to the AW element list; screenshots to be confirmed in the zip |
| AMEX | element boxes + text, **no class, no editable flag**; tap in a box 98% (sample) [F] | 38,709 steps / 3,046 episodes, ends TASK_COMPLETE / TASK_IMPOSSIBLE | few | CC-BY-4.0 | **all completed episodes** (impossible ones dropped); missing class/flags rendered as `-` (a train/serve gap to measure) |
| OS-Atlas `aw_mobile` | AW-emulator screens, element boxes + text + class, no actions/goals [F] | 15,905 screens / 89,860 elements, 17/20 AW apps | **yes (screens)** | check | auxiliary element-choice rows with goals written by **our own template** ("Tap on '<label>'") — no external model; screens only, no tasks |
| Our own rollouts | AW emulator, full tree, checker-verified | target 1–3k trajectories | yes | ours | only if decision D1 (other seeds of the 116 templates) = yes; else own templates |
| MobileViews | best trees, **no goals** | ~1,614 states in 9 AW apps | partly | MIT | not in the first model (no goals; no external model to write them) |

Screenshot-only (unusable for element choice without a detector): GUI-Odyssey (8,334 ep / 127,893 steps), AITW,
AITZ, GUIAct (= AITW episodes), OpenMobile v1/v2 (also collected on AndroidWorld task classes, no licence).

## 3. What the strong models did (for the mix decision)

- AndroidControl is a minority wherever shares are given: MobileRL SFT 36% (35.6k of 97.9k steps, rest human
  demonstrations); UI-Venus v1 84k of a 1.0M pool sampled to 350k.
- No paper ablates removing AndroidControl; no paper trains without AW apps and tests on AW. The data that moves AW is
  step-labelled, checker-verified, in the target apps (OpenMobile 0.5k trajectories: 25.5 → 41.7; MobileGen +14.7 from 500).
- The best verifier is AndroidWorld's own checker (AndroidGen: no filter 16.2, LLM critic 25.2, AW checker 27.0).
- For a choice head, a pairwise loss beats index cross-entropy on the same data (V-Droid: 47.4 vs 35.8).
- Recovery rows help but collapse the policy into "back" above ~2.5% of the data (V-Droid).

## 4. Initial mix for the first R3 model (tuned by plan Task 9, then frozen in the pre-registration)

| Part | Rows (approx.) | Share |
|---|---|---|
| AndroidControl, reduced | ~40k | ~45% |
| AMEX completed episodes | ~35k | ~38% |
| AndroidLab XML | ~6k | ~7% |
| OS-Atlas `aw_mobile` template rows | ~9k (10% sample) | ~10% |
| Own rollouts | added in the self-training rounds (Task 10) | — |

Task 9 compares, one seed each on the AW-format validation set: AC-only (full) vs this mix vs this mix + rollouts.

## 5. Checked by hand (2026-11-11, on the samples in the session scratchpad `dsprobe/`)

- **AMEX** (`instruction_anno.zip`, all 3,046 episodes): 38,709 steps; actions TAP 24,815 / SWIPE 7,628 / TASK_COMPLETE
  2,828 / TYPE 2,419 / PRESS_ENTER 651 / TASK_IMPOSSIBLE 220 / PRESS_BACK 135 / PRESS_HOME 13; last step TASK_COMPLETE
  2,828, TASK_IMPOSSIBLE 218. 76 packages; **AndroidWorld apps ≈ 0** (`com.google.android.apps.tasks` 1,176 steps is
  Google Tasks, not AW's Tasks; Files 40, Chrome 74). Conversion rules needed: SWIPE is stored as finger
  touch→lift (e.g. y 2831→470 = finger up = AW `scroll down`); PRESS_ENTER follows TYPE (AW's `input_text` presses
  Enter itself → merge); some TYPE rows have touch (0,0) = no field → no element supervision. Element files:
  `clickable_elements` / `scrollable_elements` with bbox + `xml_desc` text only (no class, no flags) + `page_caption`.
- **AndroidLab**: `androidlab-xml-train.json` = 6,208 rows of **text only** (XML tree in the prompt, no image path);
  actions Tap 4,053 / finish 729 / Type 496 / Swipe 417 / Launch 354 / Back 29; some UI labels are Chinese. The
  `som-train` set (6,053) has screenshots **with set-of-mark boxes drawn on them**. Clean screenshots aligned with the
  XML rows would have to come from the 569 MB release zip (not opened). Until that is checked, AndroidLab is not usable
  for a model that needs a clean screenshot.
- **OS-Atlas `aw_mobile.json`**: 15,905 screens; each element = `instruction` (its text), normalised bbox, `data_type`
  (class); no clickable/editable flags; images are a separate download (not checked).
- **AndroidControl**: first MB of two shards: 125 actions (click 71, scroll 15, navigate_back 12, open_app 11, wait 10,
  input_text 6), no `status` — consistent with the full-scan claim in report C (that full count not re-run here).

## 6. Revised size (operator 2026-11-11: "do not feed rows that change nothing")

* Two different kinds of data were being mixed up. **General phone data** (MobileRL 97.9k steps, UI-Venus 350k) is
  used to train a model that knows phones; **target-app trajectories** (OpenMobile 500 trajectories ≈ 6k steps →
  +16.2 points) are added *on top of* a model that already has that general skill (Qwen2.5-VL with GUI pretraining).
* Our own record already shows saturation for AndroidControl-only data: 15k → 24k rows and +8k continued training gave
  no gain on AC-v2 (2026-10-29, one seed).
* **Revised first-model mix: ~20–25k rows** — AndroidControl ~12k (episodes in AndroidWorld-like apps + all
  non-click rows) + AMEX ~8–10k (completed episodes, converted) + OS-Atlas AW screens ~2k; then **500–1,000
  checker-verified AndroidWorld-app trajectories (~6–12k steps)** from our rollouts (decision D1), and programmatic
  recovery rows ≤ 2.5%. ≈ 4–5 h per training run at 858 px.
* A source or a size step is kept only if it moves the AndroidWorld-format validation set by more than seed noise.

## 7. Opening the right app when the goal does not name it (measured)

* AndroidWorld: 91/116 templates declare `app_names`; for **21** of them the seed-30 goal does not name the app
  (Camera ×2, Clock ×3, Contacts ×2, Settings ×13, Clipper ×1). AndroidControl cur15k: 117 of 1,042 `open_app`
  rows do not name the app in the goal under a strict match (several are spelling variants such as "Easy note" →
  "Easy Notes"; true intent-only rows are fewer).
* `open_app` in AndroidWorld launches from any screen — recovering from a wrong app does not need "home" first; the
  right action is `open_app(<right app>)` again.
* Labels without any external model: on rollouts of template instances (D1), the right app of every step is the
  template's `app_names`; a step whose foreground app is not in it gets the label `open_app(<template app>)`
  (recovery row, capped at 2.5% of the mix — V-Droid saw collapse into "back" above that). The first step of an
  intent-only goal gets the same label. Without D1 the only source is AndroidControl's few intent rows.

### 7.1 Read row by row (2026-11-11; list in `benchmarks/results/2026-11-10-stage1/ac_open_app_unnamed.txt`)

The 117 AndroidControl `open_app` rows that failed the strict match were read one by one:
* ~75 name the app with a different spelling ("Easy note" → Easy Notes, "All trails" → AllTrails, "FNP" → Ferns N Petals);
* ~5 name a **different** app than the label (goal "OpenTable", label TheFork; "Trovit" → Homesnap; "Kitchen Story" →
  Tasty; "Microsoft word" → Docs; "Local weather live" → Daily Weather) — label noise;
* **~37 really do not name the app** ("set an alarm for 7:15 a.m." → Clock; "find my phone's build number" → Settings ×3;
  "search for a flight from Knoxville to Hawaii" → Adani One / ixigo; "view the photos" → Gallery ×3). None is a
  Wi-Fi / Bluetooth / brightness toggle.
So the 15k subset has ~37 intent-only app choices (~3.6% of its `open_app` rows). The 21 AndroidWorld goals without
an app name are short commands with a hint word ("Take one photo." → camera, "Run the stopwatch." → clock, "Create a
new contact for …" → contacts, "Turn wifi on." → settings).

## 8. Final source decision (2026-11-11, after full-file checks — `outputs/2026-11-11-dataset-unknowns-resolved.md`)

* **AndroidLab — dropped.** The release zip (568.7 MB) has no clean screenshots: its 6,053 images are all
  `*-before_labeled.jpg` with set-of-mark tags, rescaled to 1440x3120; the XML rows carry no image reference (83.1% can
  be matched by content only); screens are vendor phones with Chinese system UI (31.3% of rows contain CJK text); its
  "calendar" is a vendor lunar calendar, not Simple Calendar; data licence unknown.
* **OS-Atlas `aw_mobile` — dropped.** The screens are AndroidWorld emulator screens (1080x2400, 600/600 sampled), but
  (viewed by hand) the pointer-location developer overlay is baked into the pixels — the official protocol hides it
  (H10; on our own runs it changed 80/94 trajectories) — the UI locale is randomised (a Simple Calendar screen in
  Azerbaijani), and a screen has a median of 3 annotated elements, so it cannot reproduce our ≤ 40-element list.
* **AMEX — kept** (CC-BY-4.0 per card). Over all 38,709 trajectory steps: element file 100%, screenshot 100%,
  TAP inside a clickable box 96.55% (23,960/24,815), median 15 clickable elements per screen, screens 1440 px wide.
  Screenshots are a 93.5 GB split zip whose paths start with `../` → unpack with path sanitising on the emulator host
  (1.4 TB free), keep only the steps used, resized to 858 px.
* **AndroidControl — kept, reduced** (full counts confirmed: 83,848 actions, no terminal action).
* **Answer (spec D3) — decided 2026-11-11: allowed** — the `answer` span may be copied from on-screen element text.

**First-model data:** AndroidWorld practice rollouts (main, checker-verified) + AndroidControl ~12k + AMEX ~8–10k.

## 9. Operator decision 2026-11-11 (final): AndroidControl + AndroidWorld only

AMEX is dropped too: the operator judges it redundant and time-consuming once the model learns the AndroidWorld data
pattern from practice instances. **Training data = AndroidControl (reduced, ~12k rows, all with the AW-style element
list) + AndroidWorld practice rollouts (checker-verified, the main part).** Diversity beyond these two is not pursued
in R3; if the unseen-template number (one third of templates never practised) is far below the practised one, that is
the signal to revisit.

## 10. Correction (2026-11-11): AndroidControl has many tasks of AndroidWorld's KINDS (full train split, 13,603 episodes)

Counted on all train episodes (HF mirror `smolagents/android-control`, ids equal to `splits.json`; keyword heuristics on
goals, so categories overlap and are approximate): files/folders 849, tasks/to-do/reminders 601, clock
(alarm/timer/stopwatch) 459, maps 431, recipes 352, calendar/events 332, photos/gallery/camera 327, audio recording 282,
music/playlists 270, contacts 259, notes 245, video 207, drawing 89, Wi-Fi/Bluetooth/brightness 78, expenses 48, SMS 8.
**4,416 episodes (32%), 24,264 actions, are in an AndroidWorld-like category.** Earlier statements from the 15k subset
("~22%", "no Wi-Fi/Bluetooth/brightness rows") understated this. What is true: the AndroidWorld **apps themselves** are
absent — `open_app` launches of the 14 F-Droid apps: 0; system apps: Clock 49, Settings 32, Files 27, Contacts 26,
Tasks 13 — so the task kinds are present, the specific app UIs are not. 86 goals are question-like ("find out the phone
build number"), but AndroidControl has no `answer` action.

**Step-A selection:** every AndroidWorld-like-category episode (~24k actions) + a random sample of the rest to ~30k rows.
