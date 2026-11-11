# Contract v3 — model input/output and benchmark input/output (frozen 2026-11-11)

**Principle (operator, 2026-11-11):** imitation learning only, **no text generation**. One forward pass per step
returns choices (and copied spans); every output maps deterministically to exactly one AndroidWorld `JSONAction`
that AndroidWorld's own code accepts and executes. Nothing the model emits can be "almost right" in format.

Upstream references are android_world @55ddaf5 as installed in the emulator container (`/app/android_world`).

## 1. Benchmark side — what AndroidWorld gives and what it expects

| | Value | Source |
|---|---|---|
| Goal | `task.goal`, a string | `suite_utils` |
| Screenshot | `state.pixels`, 1080×2400 RGB | `env/interface.py` |
| Element list | `state.ui_elements = forest_to_ui_elements(forest, exclude_invisible_elements=True)`, from the **same** `get_state` call as the screenshot (H5 fixed) | `android_world_controller.py:241-246`; vendored verbatim in `benchmarks/train/aw_extract.py` |
| Action | `JSONAction(action_type, index \| x,y, text, direction, app_name, goal_status, clear_text, keycode)`; `action_type ∈ {click, double_tap, scroll, swipe, input_text, navigate_home, navigate_back, keyboard_enter, open_app, status, wait, long_press, answer, unknown}`; `direction ∈ {left, right, down, up}`; index XOR (x, y) | `env/json_action.py` |
| End of episode | the agent declares done (`status`), or the per-instance step budget runs out | `suite_utils.py` |
| Success | `task.is_successful(env) > 0.5` **and** done was declared (`agent_successful = task_successful if done else 0.0`) | `suite_utils.py:261` |
| Information-retrieval tasks (25/116) | scored on the `answer` text only | `task_evals/information_retrieval` |

Execution semantics that matter (`env/actuation.py:execute_adb_action`):

| JSONAction | What the device does |
|---|---|
| `click` / `long_press` with `index` | tap / long-press at `int(center)` of `screen_elements[index].bbox_pixels` |
| `click` / `long_press` with `x, y` | tap / long-press at `(int(x), int(y))` |
| `input_text` with `x, y` (or index) and `text` | tap the field, sleep 1 s, if `clear_text`: Ctrl+A then Delete, type `text`, **press Enter** |
| `scroll` + `direction`, no index | swipe from screen centre: `down` → to the top edge (content moves up, reveals what is below); `up` → to the bottom edge; `right` → to the left edge; `left` → to the right edge |
| `open_app` + `app_name` | `launch_app`: `_DEFAULT_URIS` or the first `_PATTERN_TO_ACTIVITY` regex that `re.match`es the lower-cased name; otherwise `monkey -p <name>` (fails for a non-package) |
| `navigate_back` / `navigate_home` / `wait` | back key / home key / sleep 1 s |

## 2. Model side — input

| Part | Exact form | One function (train = eval = serve) |
|---|---|---|
| Image | the screenshot resized to width **720** (bicubic, aspect kept; §2.2 — matches the stored training images) | `train_d1.load_image` (`D1_IMG_W`) |
| Goal line | `Goal: <task.goal>` | `_goal_history_lines` |
| History | one line per executed step, oldest first (format §2.1) | `render_history_line_v3` |
| Element list | header + `[i] <label> \| <class_short> \| <flags> \| x:a-b% y:c-d%`, i = 1..n, the elements kept by rule `cap40_reading` (≤ 40, reading order) from the AndroidWorld element list | `element_inputs` / `render_elements` |
| App options | the launchable apps of the device (`aw_apps.json`: every installed package that `_PATTERN_TO_ACTIVITY` resolves, by its first alias) | `app_options` |

### 2.1 History line (v3)

| Executed action | Line |
|---|---|
| click / long_press on element e | `click "<label(e)>" (changed)` or `(no change)` |
| type_text s into element e | `type_text "<s>" into "<label(e)>" (changed)` |
| scroll d | `scroll <d> (changed)` / `(no change)` |
| open_app a | `open_app <a>` |
| navigate_back / navigate_home | `navigate_back (changed)` … |
| wait | `wait` |

`changed` = training: `effect_app` of the corpus row; serving: `ui_changed` of the executed step (same definition:
the element multiset, system-UI elements ignored). Never coordinates.

### 2.2 Image size (measured 2026-11-11)

* The vision tower is native-resolution (LFM3-VL: 16-px patches merged 2×2, so one image token covers 32×32 px of
  the resized picture; `processor_config.json`), with d1's own cap `d1_vision_max_pixels = 1,638,400`
  (`config.json`, applied by `media.cap_pixels`). Any size is accepted; size changes the token count and how many
  pixels a glyph or icon gets.
* Screens: AndroidControl train rows are 96.3% 1080×2400 (2.8% 1440×3120, 0.9% 1080×2160), validation 94.6% /
  5.4%; AndroidWorld is 1080×2400. Same geometry, so one width serves both.
* Token cost for a 1080×2400 screen: width 540 → 646 tokens; **720 → 1,100** (704×1600 after rounding to 32);
  **858 → 1,534** (832×1888 — the size d1's own pipeline uses for this screen); 1080 uncapped → 2,550 (above
  d1's cap, never seen by d1).
* At 720 a 14-sp label (~37 px tall at full size on this 420-dpi screen) becomes ~24 px, under one token row; at 858 ~29 px.
* Evidence so far: one 858-px arm (2026-11-07) gave +2.79 grounding points, withdrawn because seed sd is 2.1 — the
  effect of 858 vs 720 is **not measured**. No AndroidWorld resolution ablation exists in the literature surveyed.
* In v3 the element **labels arrive as text** from the a11y tree, so reading text no longer depends on pixels;
  resolution matters for icon-only elements and visual state (toggles, selection).
* Minor skew: training screenshots pass JPEG q90 twice (stored jpg, then resize + q90), serving once (PNG → resize
  + q90).

**Decision for v3 (revised 2026-11-11):** train and serve at **720**. The stored AndroidControl training images are
**720x1600** (`preprocess_ac.py:27`, `IMG_W = 720`; measured on cur15k_v2/validation_v2: 720x1600 for 1080x2400 screens,
720x1560 for 1440x3120), so 858 would upsample blurred training pictures while serving downsamples sharp 1080 px
screenshots — a train/serve skew. 858 is possible only after re-extracting the images at 858 from the raw records
(`extract_images.py --width 858`). A first v2 run started at 858 was stopped after ~10 steps for this reason.

## 3. Model side — output (one forward pass, zero generated tokens)

| Question | Type | Options | Read when |
|---|---|---|---|
| `action` | choice | `click, long_press, type_text, scroll, wait, navigate_back, navigate_home, open_app, status` | always |
| `target_el` | choice | `1..n` (the rendered element ids) | action ∈ {click, long_press, type_text} |
| `scroll_dir` | choice | `up, down, left, right` | action = scroll |
| `app_name` | choice | the app options of §2 | action = open_app |
| `type_text` | span copy | a contiguous span of the **goal** (position copy, as in v1/v2; operator 2026-11-11) | action = type_text |

If the element list is empty, the spatial actions are masked out of `action` before the argmax (there is nothing to
choose). No `answer`, `keyboard_enter`, `double_tap`, `swipe` in v3 (Enter is pressed by `input_text` itself).

## 4. Output → benchmark action (the adapter; deterministic)

| Model output | JSONAction sent |
|---|---|
| `click`, el i | `{"action_type": "click", "x": int(cx_i), "y": int(cy_i)}`, (cx_i, cy_i) = centre of kept element i — the same point AndroidWorld's own `index` path computes |
| `long_press`, el i | same with `long_press` |
| `type_text`, el i, span s | `{"action_type": "input_text", "x": int(cx_i), "y": int(cy_i), "text": s, "clear_text": true}` |
| `scroll`, d | `{"action_type": "scroll", "direction": d}` |
| `open_app`, a | `{"action_type": "open_app", "app_name": a}` (a resolves by construction) |
| `navigate_back` / `navigate_home` / `wait` | `{"action_type": <same>}` |
| `status` | no device action; the agent returns done (`goal_status = "complete"`) |

## 5. Training labels — AndroidControl → v3 (same meanings, measured)

| AndroidControl gold | v3 gold |
|---|---|
| click / long_press at (x, y) | `target_el` = the kept element whose centre lies in the gold target (`gold_target_el_v2`); none → action supervised, element not |
| input_text | `type_text` + its field element + the text span in the goal (no supervision when the text is not in the goal) |
| scroll direction | identical meaning — **measured 2026-11-11**: on 158 AC `down` rows the shared elements move **up** by a median 1,092 px in the next screen (95% of rows), 13 `right` rows move left (85%), 10 `up` rows move down (90%): AC `down` = reveal content below = AW `down` |
| open_app name | `app_name` choice; options = gold app + 24 distractors (training), the device list (serving) |
| navigate_back / navigate_home / wait / status | same |

## 6. Conformance checks — must all pass before any online run

1. Every output of §3 maps (§4) to a dict that `JSONAction(**d)` accepts — run inside the container for each action
   type, including the empty-element-list case.
2. Every app option resolves (`get_adb_activity(name)` not None or name in `_DEFAULT_URIS`) → unresolvable
   `open_app` = 0 by construction (R1 today: 29–36%).
3. On the device (Gate A extension): for a chosen element, the element under the executed tap is that element;
   `input_text` replaces the field content; `scroll down` reveals content below.
4. History parity: the same (action, element, effect) renders the same line from a training row and from an
   executed step.
5. Preflight refuses a run unless the served checkpoint's `contract.json` says `v3` with `element_rule =
   cap40_reading` and the agent sends both `elements[]` and `apps[]`.
6. Live element parity: on the device, the element list the agent sends equals `aw_extract.forest_to_ui_elements`
   of the same forest (and AndroidWorld's own `state.ui_elements`), element for element.
7. History cap: at most the last 12 lines, preceded by `(<n> earlier actions)`, identical in training and serving.

## 7. Known limits of v3 (not format errors — tasks it cannot express)

- 25 information-retrieval tasks need `answer` (not in v3; operator decision).
- Canvas/map screens with no element at the target (BrowserDraw, OsmAnd map taps) cannot be clicked precisely.
- Text that is not in the goal cannot be typed (clipboard, transcription, address read from a message).

## 8. Measured 2026-11-11 — element cap, context, cost

* **Context is not the limit:** d1-2B text config `max_position_embeddings = 128,000` (24 layers, sliding window 256
  plus global layers). A v2-format state (≤ 40 elements + goal + short history) measured **~2,300 tokens per sample
  at 720 px and ~2,780 at 858 px** (trainer `tokens` stat, 20-step probe on `cur15k_v2`), i.e. ~1,200 text tokens,
  ≈ 25 tokens per element line. 80 elements would add ~1,000 tokens — far inside the context.
* **Cost:** probe at micro 1 / accum 16 / grad-ckpt on the 4090: 720 px 2.2–2.5 samples/s, 11.0 GB peak;
  858 px 1.8 samples/s, 11.2 GB peak. One epoch of 60k rows ≈ 9 h at 858 (≈ 7 h at 720).
* **Coverage on real AndroidWorld screens** (4,532 online steps of R1 s0/s1 and B0′; elements = the AW list recorded
  per step): candidates after the base filter p50 22 / p90 62 / max 103; 32% of screens have > 40. The clicked element
  survives `cap40_reading` on 81.0% of all clicks but on **127/130 clicks of successful episodes**; `cap60` 129/130,
  `cap80` 129/130, interactive-only (no cap) 76/130. The misses of the 81% are mostly clicks on unlabelled,
  non-interactive elements (R1's wrong clicks).
* **Decision:** keep `cap40_reading`. The cap is set by coverage of correct targets (98%), not by the context; raising
  it to 60 buys ~1.5% coverage for ~500 tokens, and long candidate lists previously cost accuracy (AGENTS: > 40-candidate
  collapse, measured under the old candidate interface).
