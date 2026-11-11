# Data recipes C — what public mobile GUI datasets actually contain (read from the released files)

*Measured 2026-11-11. Rule applied: "never characterise a dataset from its card; open the file, parse it, count."*

**Method.** File trees and sizes came from the Hugging Face Hub API (`/api/datasets/<id>`, `/tree/main`), the public
GCS JSON listing (`storage.googleapis.com/storage/v1/b/gresearch/o`), GitHub raw READMEs, and HTTP range requests.
Large archives were **not** downloaded. Instead:
(a) remote **parquet** files were read column by column over HTTP range, so no image columns were fetched;
(b) remote **zip** central directories were listed over range, and single members were extracted the same way;
(c) gzip **TFRecord** shards were read from the first 3–50 MB and decoded with a minimal protobuf reader;
(d) small files under ~60 MB were downloaded whole.
All parsing used `python3 -I` in a scratch directory. Scripts are in the session scratchpad (`scripts/`), not the repo.

**Labels.** **[F]** = VERIFIED-FROM-FILE (I parsed it). **[F-s]** = verified on a sample, with the sample size given.
**[C]** = FROM-CARD/README only. **[U]** = UNKNOWN.

**AndroidWorld (AW) app matching.** Package ids where the file has them (e.g. `net.gsantner.markor`,
`com.simplemobiletools.smsmessenger`, `org.tasks`). Otherwise app-name regexes over goals or app fields. The generic
names Contacts / Clock / Camera / Files / Chrome / Settings are reported separately from the 14 F-Droid apps. A name
match is not proof of the same app: Google Tasks ≠ `org.tasks`, Files by Google ≠ AW's DocumentsUI, and "broccoli"
can be the vegetable.

**Our target format.** goal + current screenshot + text history + an AW-style element list (text, content_desc, class,
bbox, clickable / editable / scrollable flags) + gold element + action.

---

## 1. AndroidControl (Google)

- **Where:** `gs://gresearch/android_control/`, 20 gzip TFRecord shards of 2.35–2.56 GB each, **49.9 GB** total, plus
  `splits.json` and `test_subsplits.json` **[F]**. License: Apache-2.0 per the README's License section; that section
  does not say separately whether it covers the data **[C]**. HF mirror `smolagents/android-control` (67.4 GB parquet)
  **drops the a11y trees**. Its columns are `episode_id, goal, screenshots_b64, actions, step_instructions` **[F]**.
- **Counts [F]:** official `splits.json` has train **13,603** / validation **137** / test **1,543** = **15,283**
  episodes. `test_subsplits`: IDD 721, category_unseen 700, app_unseen 631, task_unseen 803.
  A column-only scan of all 342 mirror parquet files gives **15,283 episodes and 83,848 actions**. The mirror uses its
  own train/test split (12,232 / 3,051), not the official one.
- **Per step [F-s, 25 episodes decoded from 2 shards]:** screenshot ✓; **a11y forest ✓** (`AndroidAccessibilityForest`
  proto). Its nodes carry class_name, text, content_description, hint, package, view-id, `bounds_in_screen` and
  boolean flags; one tree held 189 nodes. Per-step `step_instructions` ✓ (low-level). Episode `goal` ✓.
  **screens = actions + 1** (125 actions, 150 screens/trees over 25 episodes). Resolution 1080×2400 (24/25), 1080×2160 (1/25).
- **Action vocabulary over all 83,848 actions [F]:** `click` 51,918 · `scroll` 11,130 · `input_text` 6,091 · `wait`
  5,782 · `open_app` 5,697 · `navigate_back` 3,034 · `long_press` 167 · `navigate_home` 29. Clicks are `x, y` pixels,
  with **no element id**.
- **Terminate convention [F]:** **there is no `status` action anywhere in the 83,848 raw actions.** The last action of
  each episode is click 10,534 / wait 2,312 / scroll 2,102 / input_text 189 / navigate_back 86 / open_app 46 / other 5.
  The `status` row that every episode in our corpus ends with is **synthesised by our own `preprocess_ac.py`** (line 127)
  from the final screenshot. That is a modelling choice to state in the paper, not a property of the data.
- **AW overlap [F]:** episodes whose goal names an AW app (train / test). Clock 203 / 53 · Settings 128 / 26 ·
  Contacts 123 / 35 · Files 89 / 18 · Camera 30 / 9 · Broccoli 6 / 2 (word match, unverified) · Chrome 2 / 2.
  **None of the other 13 F-Droid apps appears.** This agrees with the repo audit D6 for cur15k: 13/20 AW apps have zero goals.
- **Conversion:** **easy; already done** (`benchmarks/train/preprocess_ac.py`). The gold element comes from
  point-in-node. `input_text` has no coordinate, so the field is the preceding click or the focused node.

## 2. AMEX (Android Multi-annotation EXpo)

- **Where:** `Yuxiang007/AMEX`, **93.6 GB**: screenshots in 9 split-zip parts (~86 GB), `element_anno.zip` 116 MB,
  `instruction_anno.zip` 3 MB **[F]**. License tag CC-BY-4.0 **[C]**.
- **Trajectories [F, all 3,046 instruction files]:** **3,046 episodes, 38,709 steps**, median 11 steps, max 55.
  Step keys: `step_id, action, touch_coord, lift_coord, device_dim, package_name, type_text, need_human_check,
  interest_region, image_path`. Episode `instruction` ✓. There is **no per-step low-level instruction**.
- **Actions [F]:** `TAP` 24,815 · `SWIPE` 7,628 · `TASK_COMPLETE` 2,828 · `TYPE` 2,419 · `PRESS_ENTER` 651 ·
  `TASK_IMPOSSIBLE` 220 · `PRESS_BACK` 135 · `PRESS_HOME` 13. **Every episode ends in a terminal action:**
  TASK_COMPLETE 2,828 + TASK_IMPOSSIBLE 218 = 3,046. `TYPE` has `touch_coord [0,0]`, so the field is the preceding tap.
- **Elements [F]:** `element_anno.zip` has **104,674** screen files, and **all 38,708 trajectory screenshots have one**.
  The other 65,966 are grounding-only screens. Per element: `bbox` (pixels), `xml_desc` (text / content-desc strings),
  `functionality` (an LLM description, non-empty on 49/604 sampled), and sometimes `idx` / `type`.
  **There is no class, and no editable or scrollable flag beyond the split into `clickable_elements` /
  `scrollable_elements`.** This is a filtered list (~17–20 elements per screen), not a full tree.
  **A TAP lands inside ≥1 annotated box on 147/150 sampled steps (98.0%)**, with 1.7 boxes containing the point on average.
- **Devices [F]:** 1440×3120 (15,786 steps), 1440×3040, 1440×2960, 1080×2280, 1080×2400, and a few landscape.
- **AW overlap [F]:** by package, Chrome 47 episodes (74 steps) and Files-by-Google 23 episodes. Otherwise none:
  `com.google.android.apps.tasks` is Google Tasks, not `org.tasks`.
- **Conversion: moderate.** The gold element comes from tap-in-box (98%). Class and flags must be inferred or left
  empty. `need_human_check` is set on 77 steps.

## 3. GUI-Odyssey

- **Where:** `OpenGVLab/GUI-Odyssey` (old: per-step PNGs in folders; tree listing truncated at ≥38 GB) and
  `hflqf88888/GUIOdyssey` (new: **92.8 GB**, 9-part split-zip of screenshots plus 8,339 JSON files) **[F]**.
  CC-BY-4.0 **[C]**.
- **Counts [F]:** the old `all_anno.json` has 7,735 episodes and 118,791 steps (repo note
  `docs/research/A1-gui-odyssey-usability.md`, full parse). The **new `all_annot.json` has 8,334 episodes**, and its
  `step_length` sums to **127,893** — this resolves the 7,735-vs-8,334 question in A1. Devices are 6 emulator
  profiles, ~1,360–1,413 episodes each, including Pixel Tablet and Pixel Fold. 228 distinct apps.
- **Per step [F, new release]:** `screenshot, action, info, ps, description, intention, low_level_instruction,
  context, sam2_bbox`. **No a11y tree or XML.** `sam2_bbox` is a SAM2 box around the click point, non-empty on
  11/16 steps of a sampled episode. It is empty for TEXT and terminal steps.
- **Actions [F, A1]:** `CLICK`, `TEXT`, `SCROLL`, `COMPLETE`, `LONG_PRESS`, `INCOMPLETE`. There is no HOME / BACK.
  96.8% of episodes end in `COMPLETE`.
- **AW overlap [F]:** Chrome 913, Clock 519, Contacts 23, Files 16 episodes. "Simple Calendar - easy planner" appears
  in 8 episodes, and it is not confirmed to be AW's Simple Calendar Pro.
- **Conversion:** **screenshot-only for element choice.** `sam2_bbox` gives one positive box per click, with no
  candidate set and no text or class.

## 4. AITW (Android in the Wild)

- **Where:** `gs://gresearch/android-in-the-wild/`, 5 subsets of gzip TFRecords: general 13.6 GB (322 objects),
  single 13.4 GB, and google_apps / install / web_shopping ≥33–54 GB each (listing capped at 1,000 objects)
  **[F]**. License: **[U]**, not verified this pass.
- **Counts [F]:** `splits/standard.json` holds train 551,496 / val 68,967 / test 68,376 episode ids = 688,839.
- **Per step [F-s, 30 steps / 5 episodes]:** one TFRecord per step, holding `image/encoded` at **540×1080**
  (downsampled), `goal_info` (episode goal), `android_api_level` 31, `device_type` PIXEL_3, `current_activity`, and
  `results/action_type` (int) with `yx_touch` / `yx_lift` / `type_action`.
  `image/ui_annotations_{positions,text,ui_types}` are **OCR and icon-detector boxes** (TEXT, ICON_PLAY,
  ICON_NAV_BAR_RECT, …). **They are not an a11y tree: no class, no clickable flags.**
  There is no per-step instruction.
- **Actions [F-s]:** int codes 4 (dual-point tap or swipe), 3 (type), 5 (back), 6 (home), 7 (enter), 10 (complete).
  **All 5 sampled episodes end with 10 = STATUS_TASK_COMPLETE.**
- **AW overlap [F-s]:** Chrome, Google Calendar and Search packages in the sample. The F-Droid apps are not expected
  but were not counted (too large to scan).
- **Conversion:** **screenshot-only.** OCR/icon boxes give weak pseudo-candidates with no flags. Labels are known to
  be noisy (the reason AITZ exists) **[C]**.

## 5. AITZ (Android in the Zoo)

- **Where:** the original zip mirror `leosltl/Android-in-the-Zoo-mirror` (2.65 GB; zip directory **[F]**: 2,505 JSON,
  19,097 PNG, train / test × general / install / web_shopping / google_apps / single). License tag `other` **[C]**.
  `xwm/AITZ` (0.86 GB parquet) is **the AITZ test split only**, despite its file name `train-*`: it has 506 episodes /
  4,724 steps, and its image paths point to `aitz_test/test/...` **[F]**.
- **Per step [F-s, 587 episodes, 5,435 steps]:** AITW fields (`ui_positions/ui_text/ui_types` OCR and icon boxes,
  `result_action_type`, touch / lift yx) plus CoAT fields (`coat_screen_desc`, `coat_action_think`,
  `coat_action_desc`, `coat_action_result`). `coat_action_desc` is a low-level instruction ✓. **No a11y tree.**
- **Actions [F-s]:** 4 (tap/swipe) 3,830 · 10 (complete) 585 · 3 (type) 583 · 6 (home) 267 · 7 (enter) 127 ·
  5 (back) 43. **585/587 episodes end in 10.** The parquet re-encoding uses names `click / scroll / input / press /
  finished` (504/506 end `finished: success`).
- **AW overlap [F-s]:** goals mention Settings 13, Chrome 15, Clock 6, VLC 4, Contacts 3 in the sample.
- **Conversion:** **screenshot-only** (OCR boxes as above). Its value is the cleaned labels and the CoAT rationales.

## 6. GUIAct (smartphone part)

- **Where:** `yiye2023/GUIAct`. `smartphone_train_data.json` is 44 MB, images in a 4.3 GB parquet **[F]**.
  Apache-2.0 tag **[C]**.
- **Counts [F]:** train **7,246 episodes / 65,627 steps**; test 230 / 2,079.
  **All 7,246 train episode ids are AITW episode ids** (checked against `standard.json`), so this is a re-annotated
  AITW subset, not new data.
- **Per step [F]:** `question` (goal), `actions_history` (a short string, e.g. "step 0: swipe"), and `actions_label`
  with `point` / `dual_point` (absolute and relative) and `text`. **No tree and no boxes.**
- **Actions [F]:** `tap` 38,262 · `swipe` 11,670 · `answer` 7,246 · `input` 6,560 · `enter` 1,889.
  **Every episode but 2 ends in `answer`** (7,244/7,246).
- **AW overlap [F]:** goals name Settings 305, Chrome 259, Contacts 63, Clock 37, Camera 29, Files 12.
- **Conversion:** screenshot-only. The main use is the `answer` termination convention.

## 7. MobileViews

- **Where:** `mllmTeam/MobileViews`, **475 GB** **[F]**: 4 parquet (109 GB, columns `image_content` binary +
  `json_content` string; e.g. 122,301 rows in the 400k–522k file), 4 matching zips (101 GB), 101 per-app trace zips
  (265 GB), index CSVs, `AppMetadata.csv` (15,374 apps). MIT tag **[C]**.
- **Per screen [F, `OneExample.zip`: 266 states]:** screenshot (jpg); a **DroidBot state JSON** whose `views` carry
  class, text, content_description, resource_id, bounds, **clickable / editable / scrollable / checkable /
  long_clickable / focusable** and parent/children; and a **uiautomator `window_dump_*.xml`**. This is the richest
  element data in the survey.
- **Trajectories [F]:** `actions.csv` (from_state, to_state, action). Actions are **automated exploration** strings
  such as `touch <button bound_box=…>TRY THE DEMO COMPANY</button>`, `scroll down <scrollbar …>`,
  `intent am start …` and `kill_app`. **There is no natural-language goal or instruction.** The trace index lists
  19,431 apps and 628,655 states.
- **AW overlap [F, trace index]:** `org.tasks` 631 states · `com.simplemobiletools.calendar` 449 (free variant of
  Calendar Pro) · `net.osmand` 276 · `code.name.monkey.retromusic` 76 · Files-by-Google 57 ·
  `com.simplemobiletools.gallery` 39 · `com.simplemobiletools.smsmessenger` 32 · Clock 30 · `org.videolan.vlc` 24.
  That is ≈1,614 states.
- **Conversion:** **the tree is AW-compatible (best fit of any dataset).** There is no goal, so it is usable only for
  goal-free pretraining: element affordance, next-screen prediction, or synthetic instruction from the touched
  element. Size is the obstacle; per-app trace zips (2–4 GB) allow pulling only the AW apps.

## 8. Mobile3M (MobileVLM)

- **Where:** `xwk123/Mobile3M`, **338 GB** of 7z volumes for 11 app graphs: BaiduMap, QQmusic, androidesk, bili,
  ctrip, gaodeMap, kugou, ludashi, vipshop, xiaomiShop, ximalaya **[F]**. The README states the data license is
  **CC BY-NC-SA 4.0** (code Apache-2.0), while the HF tag says apache-2.0 **[F: README text]**.
- **Contents [U]:** the archives are 7z (magic bytes checked) and were not opened. The paper describes a UI
  transition graph with XML **[C]**.
- **AW overlap:** none by app name (all Chinese consumer apps) **[F: archive names]**.
- **Conversion:** [U]. Non-commercial license, no AW apps, and a 7z-only layout. Low priority.

## 9. OpenMobile — v1 (`cckevinn/OpenMobile-Data`) and v2 (`OpenMobile-2/OpenMobile-Data-v2`)

- **Release status:** **data is released.**
  - **v1:** 4 LlamaFactory JSONs (27–59 MB) plus `raw_dumps.tar.gz` (12.6 GB). No license, no arXiv link on the card
    **[F]**. I could **not verify** that this repo is the arXiv 2604.15093 release: the card names no paper.
  - **v2:** cites arXiv 2610.14157; no license tag **[F]**.
- **v1 counts [F, all 4 splits]:** **27,360 SFT examples = steps, in 5,101 episode folders.** One screenshot per
  example; coordinates on a 999×999 grid. Tool `mobile_use` with actions `click, long_press, swipe, type, answer,
  system_button, wait, terminate`. Final-turn actions across splits: click 16,933 · swipe 3,779 · type 2,224 ·
  terminate 1,935 · system_button 950 · answer 758 · long_press 619 · wait 162.
  `raw_dumps` [F-s, 15 episodes from the first 40 MB] hold `result.json` = {goal, trajectory[step, response,
  step_history, done]} plus `screenshot_stepK.png`. **No a11y tree anywhere** (an a11y-like string appears in 5 of
  27,360 examples).
- **AW overlap [F] — the important finding.** Episode folders are named `<n>_<AndroidWorld task class>`.
  **All 116 AW task names occur, on 2,499 v1 episodes.** The goals are **not** the AW templates; they are synthesised
  per app. For example, folder `ContactsAddContact` has 18 different goals, such as "Configure the Contacts app to
  automatically block calls from unidentified callers". So the data is collected **on the AndroidWorld emulator
  image, from AW task initial states**, and covers all 20 AW apps.
- **v2 counts:** the card says 11,643 trajectories / 118,582 steps / ~59 GB **[C]**. Verified **[F]**: the
  `emulator/trajectories` folder count is cross_app **3,202** + single_app **4,523** = 7,725, matching the card's
  emulator row. **2,770 of those folders are named after AW116 tasks**, and all 116 names occur. The record format is
  `{id, messages, images}` with at most the last 3 screenshots; it was verified on 386 records from the first 4 MB of
  the 796 MB `train.json`. `RL/task.jsonl` holds **2,475** tasks with `init` / `verify` assertions for *simulated*
  apps (notes 481, wechat 276, calendar 246, clock 180, …), not AW **[F]**.
- **Conversion:** screenshot-only (no element list, so gold elements would have to be re-derived). **Contamination
  warning:** same emulator, same apps, same initial states as the AW eval tasks. Using it for an AW claim must be
  disclosed. Training on it would make an AW number a different kind of claim.

## 10. AndroidLab / Android Instruct

- **Where:** the HF repo `xuyifan/android-lab-train` contains only a README (apache-2.0 tag) **[F]**. The real file is
  on Google Drive (linked from `THUDM/Android-Lab/docs/instruction_tuning.md`): `android-lab-train.zip`,
  **568,685,745 bytes**, readable over HTTP range **[F]**. The repo license is MIT (GitHub API). The data license is
  **[U]**.
- **Contents [F, both JSONs extracted from the zip]:**
  - `androidlab-xml-train.json` has **6,208 steps** in ~726 episodes (726 distinct goals; 729 `finish`).
  - `androidlab-som-train.json` has 6,053 steps with 6,053 set-of-mark JPGs.
  - Each XML example is a **compressed a11y tree as text**: `[node-id] Class ;click long-click ; ;;text : bounds:
    [x1,y1][x2,y2]`, so class, flags and bounds are all present. History is given as prior rounds' actions.
  - XML actions: `Tap` 4,183 · `finish` 729 · `Type` 496 · `Swipe` 417 · `Launch` 354 · `Back` 29.
    **`Tap` names its target as an element bbox, and 4,183/4,183 of those bboxes occur verbatim as a node's bounds in
    the input tree.** Episodes end with `finish(message=…)`.
  - SoM actions: `tap(<mark>)` 4,318 · `finish` 716 · `type` 513 · `swipe` 471 · `back` 35.
- **Apps [F]:** android-system-settings 1,510 steps · calendar 1,183 · bluecoins 1,057 · cantook 717 · maps.me 634 ·
  pimusicplayer 464 · contacts (google + plain) 384 · clock (google + plain) 259. Some screens are from an OPPO
  device with a Chinese launcher.
- **AW overlap:** Settings, Contacts and Clock only, as generic system apps **[F]**.
- **Conversion:** **easy: the gold element is exact.** The tree is text, so it must be parsed back into nodes. The
  flags present are click / long-click / focusable / scroll; editable is implied by class.

## 11. OS-Atlas grounding data (mobile domain)

- **Where:** `OS-Copilot/OS-Atlas-data`, 816 GB in total. Mobile parts **[F]**: `amex_raw.json` 328 MB (AMEX images,
  ~92 GB), `uibert_raw.json`, `widget_captioning.json`, `ricosca.json` (Rico images, 6.5 GB), and
  **`aw_mobile.json` 31.9 MB + `mobile_images.zip` 2.3 GB**. Apache-2.0 tag **[C]**.
- **`aw_mobile.json` [F, full parse]:** **15,905 screenshots from the AndroidWorld emulator, 89,860 elements**
  (5.65 per screen). Each element has `instruction` (an a11y text or description string), `bbox` (normalised 0–1),
  and `data_type` (Android class: TextView 61,645, EditText 4,251, CheckBox 4,236, Button 4,089, …).
  **No actions, goals, flags or trajectories.** Element strings mention Settings 470, Files 466, Contacts 299,
  Chrome 232, Markor 181, Camera 173, Broccoli 118, OsmAnd 89, Clock 60, OpenTracks 49, VLC 48, SMS 48, Retro 41,
  Pro Expense 36, Joplin 22, Audio Recorder 20, Tasks 13. **That is 17 of the 20 AW apps.**
- **Conversion:** element-grounding pretraining only, in AW apps. Elements are filtered (not the full candidate set)
  and carry no flags. It does not overlap AW eval tasks, but it does overlap AW apps.

## 12. UGround and Aria-UI

- **UGround** (`osunlp/UGround-V1-Data` 424 GB; `-Box` 265 GB, gated:auto, CC-BY-NC-SA-4.0) **[F: tree]**. The only
  top-level folders are `web_direct_150k_description_filtered`, `web_direct_258k_function_filtered` and
  `web_hybrid_773k_max_25qa_filtered`. **No mobile data in the released repo.**
- **Aria-UI** (`Aria-UI/Aria-UI_Data`, 174 GB, apache-2.0 tag). The mobile part is
  `mobile/aria_ui_mobile_AMEX_1077k.json` (1.6 GB) **[F: head]**: AMEX `clickable_elements` plus generated
  `instructions` and long/short captions. **It is derived from AMEX, not new screens.**
  `Aria-UI_Context-aware_Data` (93 GB) re-windows AndroidControl (113k + 43k), GUI-Odyssey (211k + 58k), AMEX (51k)
  and AITW/AITZ **[F: tree]**. Derived data, the same coverage as the sources.

## 13. LlamaTouch

- **Where:** GitHub `LlamaTouch/LlamaTouch`; data on OneDrive (`llamatouch_dataset_0521.zip`; size not probed — the
  link redirects to an interactive page). No repo license **[F: GitHub API]**.
- **Counts [F: `llamatouch_task_metadata.tsv`]:** **495 tasks, 3,281 steps**, 77 app strings; 393 self-made and 102
  from AITW. Per-step `.xml` (uiautomator), `.vh` (a11y JSON), screenshot, activity, and essential-state annotations
  **[C: dataset README]**.
- **AW overlap [F]:** Settings 24, Chrome 17, Clock 8, Contacts 3, Files 2, VLC 1 (name matches).
- **Conversion:** easy if the tree files are as documented, but it is **small (3.3k steps) and built as an
  evaluation set**.

## 14. AndroidArena, MonkeyCollection, AW-derived releases (UI-Voyager, MobileRL, V-Droid, others)

- **AndroidArena** (`AndroidArenaAgent/AndroidArena`): README says "We offer only task instructions … Annotated action
  sequences will be released later" **[F: README]**. **No trajectories.**
- **MonkeyCollection:** no dataset by that name found on HF or GitHub search **[U]**. MobileViews (§7) is the closest
  "monkey / explorer traversal" corpus.
- **MobileRL** (`THUDM/MobileRL`, MIT): its roadmap ships the eval framework and a 9B checkpoint. **No training data**
  **[F: README; HF author scan]**.
- **UI-Voyager** (`ui-voyager/UI-Voyager`, MIT): model weights and fork-point detection code. **No data**
  **[F: README]**.
- **V-Droid** (`V-Droid-Agent/V-Droid`): LoRA weights `V-Droid/V-Droid-8B-0323`. **The HF org has no datasets** **[F]**.
- **AW rollouts released by third parties [F]**, all on **AW eval tasks** and screenshot-only:
  - `gavinlaw/androidworld-qwen3vl4b-rollouts`: 17.3 GB, Apache-2.0; ~20k Qwen3-VL-4B episodes; `pilot` index 18,195
    with `is_successful` labels. `episode.json` = task_type, goal, steps{response, arguments, action, summary},
    screenshots; **no tree**.
  - `SeerRay-Lab/Android-World-Eval`: 464 rollout JSONs, MIT, Xiaomi-GUI-0 eval results.
  - `GUI-Gym/*-android-world`: per-run jsonl plus ~0.95 GB trajectory tarballs.
  - `tmpstorage/androidworld-cleaned-v2`: 9.1 GB. **The inner member is AES-encrypted (zip method 99), so it is
    unusable.**

  Using any of these for training contaminates an AW evaluation.
- **ScaleCUA-Data** (`OpenGVLab/ScaleCUA-Data`, 1.07 TB) has 20 Android annotation jsonl files (373 MB). Sample
  **[F]**: `android_navigation` has 9,934 conversation rows of "Task + Previous operations → `click(x=0.95,y=0.08)`";
  `action_grounding` has 7,180 rows. Coordinates only, with no element list in the annotations; whether the raw image
  archives hold XML is [U].

---

## Ranked table — what we can use for **element-choice** training (a tree or element boxes are required)

| Rank | Dataset | Element source (verified) | Goal | Gold element | Est. usable element-choice steps | AW-app overlap | Conversion |
|---|---|---|---|---|---|---|---|
| 1 | **AndroidControl** (official GCS) | full a11y forest per step, AW-compatible fields **[F]** | ✓ + per-step instr. | point-in-node from `x, y` | **~52k click/long_press + 6.1k input_text + 11.1k scroll**, of 83,848 actions (~74% on the train split) | system apps only (Clock 203, Settings 128, Contacts 123, Files 89 train eps) | done (`preprocess_ac.py`); needs a synthetic `status` |
| 2 | **AndroidLab XML** (Drive zip) | text-serialised a11y tree with class, flags, bounds **[F]** | ✓ | **exact** (bbox = node bounds, 4,183/4,183) | **~4.2k Tap + 0.4k Swipe** (6,208 steps) | Settings / Contacts / Clock | easy: parse the text tree back to nodes |
| 3 | **AMEX** | filtered clickable/scrollable boxes + text; no class or flags **[F]** | ✓ (no step instr.) | tap-in-box, 98.0% (147/150) | **~24.3k TAP** (+7.6k SWIPE) of 38,709 | Chrome 47 eps, Files-Google 23 | moderate: synthesise class and flags |
| 4 | **MobileViews** | DroidBot views + uiautomator XML, all flags **[F]** | ✗ (explorer actions) | exact (action string carries the element bounds) | ~600k states; ~1.6k in AW apps (org.tasks 631, Calendar 449, OsmAnd 276, …) | 9 AW app ids | goal-free pretraining only; 475 GB, pull per-app zips |
| 5 | **OS-Atlas `aw_mobile`** | filtered elements, text + class + box, no flags **[F]** | ✗ | n/a (referring-expression grounding) | 89,860 element-grounding pairs on 15,905 AW screens | 17/20 AW apps | grounding pretraining, not action choice |
| 6 | **LlamaTouch** | uiautomator XML + a11y JSON **[C]** | ✓ | point-in-node (expected) | ≤3.3k steps | Settings / Chrome / Clock | evaluation-sized; OneDrive |
| — | Aria-UI mobile | AMEX elements + generated instructions **[F]** | ✗ | n/a | duplicates AMEX | as AMEX | derived |

**Screenshot-only for our purpose** (no element list; boxes would need to be re-derived from pixels):

- **GUI-Odyssey:** 8,334 eps / 127,893 steps; SAM2 click boxes only; Chrome 913 / Clock 519 eps.
- **AITW:** 688,839 eps; OCR/icon boxes only.
- **AITZ:** ~2,504 eps / ~19k screenshots; OCR boxes + CoAT.
- **GUIAct smartphone:** 7,246 eps / 65,627 steps; re-annotated AITW.
- **OpenMobile v1 / v2:** 27,360 steps / 5,101 eps and 7,725 emulator trajectories; **AW emulator + all 116 AW task
  inits; contamination risk**.
- **ScaleCUA Android:** coordinates only.
- **Third-party AW rollouts:** AW eval tasks; contamination.
- **Mobile3M:** unknown inside the 7z archives; CC BY-NC-SA.

**Not released:** AndroidArena trajectories, MobileRL / UI-Voyager / V-Droid training data.

## What this changes

1. **AndroidControl is still the only large goal-conditioned corpus with a full AW-compatible tree. It has no
   terminal action.** Every `status` row we train on is our own synthesis. Its AW app coverage stops at the six
   system apps.
2. **AndroidLab is the only other set with an exact gold element on a real tree.** It is small (~6k steps) but cheap
   to add, and it carries an explicit `finish` convention.
3. **AMEX is the largest add-on with element boxes (~24k taps, 98% resolvable).** It lacks class and flags, so our
   element encoder must tolerate missing fields, or the fields must be imputed.
4. **The only data that covers the 14 F-Droid AW apps:**
   - MobileViews: tree, no goals.
   - OS-Atlas `aw_mobile`: elements, no actions.
   - OpenMobile v1 / v2: trajectories on AW task inits, screenshot-only, contamination risk.

   No released corpus has **goal + tree + gold element on AW apps** without also sitting on AW eval initial states.

## Not verified (open)

- AITW license and full-subset counts; AITZ full step count (the sample covered 587 of 2,504 episodes).
- The OpenMobile v1 ↔ arXiv 2604.15093 link; the OpenMobile v2 simulator split counts (card only).
- Mobile3M contents (7z); LlamaTouch zip size and contents (OneDrive).
- Whether the ScaleCUA raw archives contain XML; AMEX tap-in-box on the full set (only n=150 was sampled).
- Whether AndroidControl's `input_text` field can always be recovered from the focused node.
