# Dataset unknowns, resolved from the files (2026-11-11)

Rule followed: no fact about a dataset is taken from its card unless marked **FROM-CARD**. Every other
number below was counted from a downloaded file (**VERIFIED-FROM-FILE**) or is **UNKNOWN**. Heuristic counts
(string matching) are labelled **HEURISTIC** and are counts from files, not ground truth.

Working copies (scratch, not in the repo): data under
`/private/tmp/claude-501/-Users-[user]-Workspace-research/5b003b77-1d13-48ff-be88-93d75a03034c/scratchpad/dsverify/`
(one subdirectory per dataset), scripts under `.../scratchpad/dsverify_scripts/`, all run as
`python3 -I <script> <paths>` (the parquet reader runs in a scratch venv with pyarrow, also `-I`).
Downloaded in total: about 0.75 GB of whole files plus about 25 MB of byte-range reads (central directories,
PNG headers, parquet column chunks). No archive larger than 2 GB was fetched whole.

---

## 1. AndroidLab (THUDM/Android-Lab, "Android Instruct")

**Source.** Link in `docs/instruction_tuning.md` of the GitHub repo:
`https://drive.google.com/file/d/1s0b74VEOww9n1kMocd6RJivwaUCymEs4/view`. Downloaded with
`curl -sL "https://drive.usercontent.google.com/download?id=1s0b74VEOww9n1kMocd6RJivwaUCymEs4&export=download&confirm=t"`.

| Item | Value | Status |
|---|---|---|
| `android-lab-train.zip` | 568,685,745 bytes, sha256 `d38d7a2b…2eb026`, stored (no compression), 6,836 entries | VERIFIED-FROM-FILE |
| Contents | `androidlab-xml-train.json` (23,711,611 B, 6,208 rows), `androidlab-som-train.json` (5,175,184 B, 6,053 rows), `images/` = 6,053 JPGs in 774 episode folders, plus `__MACOSX`/`.DS_Store` junk | VERIFIED-FROM-FILE |
| Both JSONs byte-identical to the earlier probe copies in `dsprobe/androidlab/` | yes (`cmp`) | VERIFIED-FROM-FILE |
| Image file names | **all 6,053** end in `-before_labeled.jpg` (no other suffix exists) | VERIFIED-FROM-FILE |
| Images carry set-of-mark labels | yes: numbered grey tags on every element in all 4 images opened (rows 0 and 3000 of the SoM file, plus two matched pairs). Numeric tags only, no outline boxes. | VERIFIED-FROM-FILE (visual, 4/4) |
| Image resolution | **all 6,053 are 1440x3120** (JPEG SOF header) | VERIFIED-FROM-FILE |
| SoM rows that reference an image | 6,053 / 6,053, all files present, 6,053 distinct | VERIFIED-FROM-FILE |

**Q: Clean screenshots (without SoM marks) for the XML rows?** **No.** The archive has no clean image. Its
only images are the 6,053 SoM-labelled ones. The XML rows reference no image at all: no path, id or timestamp
field. Rows are llama-factory triples `{instruction, input, output}`, and the screen is given only as an XML-derived
text tree. VERIFIED-FROM-FILE.

**Q: Can an XML row be matched to a screenshot? How, and how many?** Only to the SoM-labelled screenshot, and
only by content, because there is no shared key. Method (`androidlab_match.py`): key = (goal text, sequence of
previous action *types* with the leading `Launch` removed, current action type). The SoM set has no
`Launch` action: 0 in 6,053 outputs and 0 in history, against 354 `Launch` outputs in XML.
- 5,854 XML rows are not `Launch`. **5,158 match exactly one SoM row (88.1%; 83.1% of all 6,208)**, 20 are
  ambiguous and 676 have no counterpart. The 354 `Launch` rows (home screen) have no image by construction.
  VERIFIED-FROM-FILE.
- Checks on the match: of the 434 matched `type` actions, 433 carry identical typed text. 2 of 2 pairs opened by
  eye show the same screen (maps.me "Favorites / My Places" and the lunar Calendar). Matched goals map to more
  than one SoM episode folder in only 3 of 639 cases. VERIFIED-FROM-FILE.
- Naive (goal, round) alignment is wrong: 5,133 unique matches but only 3,854 agree on the action type. The
  shift comes from the missing `Launch` step.
- **Coordinate caveat.** The XML `bounds` and `Tap element=[x1,y1,x2,y2]` are in the original device pixels.
  The matched XML screens span ≥12 geometries (max bounds 1080x2400: 1,508; 1080x2300: 691; 1440x3200: 608;
  1220x2700: 334; …), but every image is 1440x3120. So the images were rescaled, sometimes with a change of
  aspect ratio. Projecting XML boxes onto the images needs the per-row device size (max bound is the only proxy).
  VERIFIED-FROM-FILE. The exact resize rule is UNKNOWN.

**Q: Apps / packages.** No package names exist in either file. The app is given by the `{"current_app": …}`
tag in the prompt and by `Launch` targets. VERIFIED-FROM-FILE. XML rows by current app (`Launch` rows
counted under their target): android-system-settings 1,510 · calendar 1,183 · bluecoins 1,057 ·
cantook 717 · map.me 485 · pimusicplayer 464 · google-contacts 300 · maps.me 149 · clock 136 ·
google-clock 123 · contacts 84 (11 tags, 9 apps). Episodes (731, split at `Round 0`) by first launch target:
settings 229, calendar 92, cantook 86, bluecoins 81, map.me 66, pimusicplayer 59, google-contacts 43,
google-clock 22, clock 21, maps.me 20, contacts 12.

**Steps in AndroidWorld apps** (XML rows, VERIFIED-FROM-FILE counts; app identity reasoning noted):
- Settings: **1,510** rows (AndroidWorld also uses system Settings).
- Contacts: **384** rows (google-contacts 300 + contacts 84).
- Clock: **259** rows (clock 136 + google-clock 123).
- Calendar: 1,183 rows, but **this is not AndroidWorld's Simple Calendar Pro**. The string "Simple Calendar"
  appears on 0 screens, and the opened screen is a vendor lunar calendar with Chinese lunar dates. So **0** rows
  are from AndroidWorld's calendar app.
- Files, Chrome, Markor, Tasks, Expense, Recipe, OsmAnd, VLC, Retro Music, Audio Recorder, SMS: **0** rows as
  current app (they are absent from the tag vocabulary).
- Same app ≠ same build. The screens come from physical vendor phones: OPPO launcher, Chinese system UI,
  carrier/5G status bars, ≥12 screen geometries. They are not the AndroidWorld emulator image.

**Q: License.** GitHub repo `THUDM/Android-Lab`: `LICENSE` = **MIT** ("Copyright (c) 2024 xuyifan-0731"),
read from `https://raw.githubusercontent.com/THUDM/Android-Lab/main/LICENSE`, and the GitHub API reports `mit`.
VERIFIED-FROM-FILE for the code repo. The zip contains **no** LICENSE/README file, and `instruction_tuning.md`
states no separate data license. So the data's license is **UNKNOWN** beyond "the repo is MIT". The screenshots
show third-party apps.

**Q: Non-English UI labels.** Over the current-screen text of all 6,208 XML rows: **1,945 rows (31.3%)** contain
CJK text. 19,534 of 130,592 non-empty labels (15.0%) contain CJK. 407 rows are majority-CJK. 585 of 731 episodes
see at least one CJK screen. 12 goals contain CJK. VERIFIED-FROM-FILE.

**Other facts from the file.** XML output actions: tap 4,183 · finish 729 · type 496 · swipe 417 · launch 354 ·
back 29. SoM: tap 4,318 · finish 716 · type 513 · swipe 471 · back 35. XML taps are `element=[x1,y1,x2,y2]`
boxes, not points. SoM taps are mark indices (`tap(7)`), with no box in the file.

---

## 2. OS-Atlas mobile grounding, AndroidWorld part (`aw_mobile.json`)

**Source.** `https://huggingface.co/datasets/OS-Copilot/OS-Atlas-data` (file list from
`https://huggingface.co/api/datasets/OS-Copilot/OS-Atlas-data/tree/main?recursive=true`).

| Item | Value | Status |
|---|---|---|
| `mobile_domain/aw_mobile.json` | 31,860,297 B, list of **15,905** screens `{img_filename, elements:[{instruction, bbox(0-1 ltrb), data_type}]}` | VERIFIED-FROM-FILE |
| Images for aw_mobile | `mobile_domain/mobile_images.zip`, **2,325,202,926 B** (README says so, FROM-CARD; confirmed below) | VERIFIED-FROM-FILE |
| Archive listing (central directory read by HTTP range, 1.38 MB fetched; `remotezip.py`) | **15,905 entries**, all `.png`, all stored uncompressed, total 2,322,705,652 B, median 116,792 B | VERIFIED-FROM-FILE |
| Names in `aw_mobile.json` present in the zip | **15,905 / 15,905**, and 0 zip names not in the JSON (one-to-one) | VERIFIED-FROM-FILE |
| Resolution | **1080x2400 for 600/600** randomly sampled PNG IHDR headers (range reads; all PNG signatures valid) | VERIFIED-FROM-FILE (sample) |
| Elements per screen | median **3**, total 89,860 (this is a grounding subset, not the full a11y tree). Types: TextView 61,645, EditText 4,251, CheckBox 4,236, Button 4,089, … | VERIFIED-FROM-FILE |
| License | `license: apache-2.0` in README front matter. The repo has no LICENSE file. | FROM-CARD |

The whole archive (2.33 GB) is over the 2 GB rule, so it was **not** downloaded whole. 24 random images were
pulled whole by range (4.7 MB).

**Do the screens look like AndroidWorld app screens?** **Yes.** 3 of 3 opened (OpenTracks list, SMS Messenger in
Czech, Chrome new tab) are AndroidWorld emulator screens. They share the Pixel-style status bar, 1080x2400, and
the **Android "pointer location" developer overlay** (the `P: 0/1 dX dY Xv Yv Prs Size` bar at the top, plus
touch-trace lines on the Chrome screen) is on in all 3. That overlay is baked into the pixels, which matters for
training. Element text matches AndroidWorld's seeded data: OpenTracks activities "Skateboard Trek", "Mountain
Trail Hike"; Markor-style notes ("Winter is coming."); the Audio Recorder setup screen; Pro Expense feedback
text; status-bar "SMS Messenger notification: …". VERIFIED-FROM-FILE (visual, 3/3; text over the full file).

**Screens per app.** There is no package field, so the count is **HEURISTIC**: a unique keyword hit over the
non-status-bar element texts (`osatlas_apps.py`). With a median of 3 elements per screen, **10,227 screens
(64%) are unattributable** and 1,582 are ambiguous. Attributed: Broccoli 555 · Settings 500 · Markor 418 ·
Simple Calendar 334 · Files 323 · OpenTracks 300 · SMS 224 · OsmAnd 196 · Tasks 190 · Contacts 188 · Clock 174 ·
Pro Expense 154 · Chrome 145 · VLC 140 · Retro Music 77 · Gallery 65 · Camera 46 · Draw 38 · Audio Recorder 19 ·
Joplin 10. A reliable per-app count is **UNKNOWN** without image or package information.

**Language.** 713 of 15,905 screens contain Latin-extended, Greek or Cyrillic letters (e.g. Czech UI). 74 contain
CJK, Hangul, Arabic or Devanagari text. VERIFIED-FROM-FILE. So a minority of the screens run under non-English
locales.

---

## 3. AMEX (Yuxiang007/AMEX)

**Source.** `https://huggingface.co/datasets/Yuxiang007/AMEX` (tree from the HF API).

| Archive | Size (bytes) | Status |
|---|---|---|
| `AMEX/instruction_anno.zip` | 3,105,724 (3,046 episode JSONs, identical to the probe copy) | VERIFIED-FROM-FILE |
| `AMEX/element_anno.zip` | **115,510,591** compressed / 660,455,114 uncompressed, **104,674** JSONs. Downloaded whole. | VERIFIED-FROM-FILE |
| Screenshots: split zip `AMEX/screenshot.z01`…`z08` (8 x 10,737,418,240) + `AMEX/screenshot.zip` (7,581,961,664) | **93,481,307,584 total** (~93.5 GB). Not downloaded. The zip64 central directory was read from the last part by range (13.6 MB; `splitzip_cd.py`). | VERIFIED-FROM-FILE |
| Screenshot archive entries | 104,675 = 104,674 PNGs + 1 dir. **Every entry path starts with `../screenshot/`**, so a naive extractor writes outside the target directory (zip-slip). Extract with a sanitising tool. | VERIFIED-FROM-FILE |
| License | `license: cc-by-4.0` in README front matter (HF API `cardData` agrees). **No LICENSE file** in the repo. | FROM-CARD |

Element file schema: `{image_path, clickable_elements:[{bbox [x1,y1,x2,y2] px, xml_desc[], functionality, type?, idx?}],
scrollable_elements:[…], page_caption?}`. The file name is the screenshot name with `.png` replaced by `.json`.
There is no `element_anno_` prefix in the full release; only the `sample/` folder has one. VERIFIED-FROM-FILE.

Computed over **all 3,046 episodes / 38,709 trajectory steps** (`amex_analyze.py`):

| Quantity | Value | Status |
|---|---|---|
| Steps with an element-annotation file | **38,709 / 38,709 = 100%** | VERIFIED-FROM-FILE |
| Steps whose screenshot is in the screenshot archive listing | **38,709 / 38,709 = 100%** | VERIFIED-FROM-FILE |
| TAP steps whose `touch_coord` lies inside ≥1 clickable bbox | **23,960 / 24,815 = 96.55%** (855 TAPs hit no clickable box) | VERIFIED-FROM-FILE |
| Clickable elements per trajectory screen | **median 15**, mean 17.27. 171 screens have 0. Over all 104,674 element files the median is 13. | VERIFIED-FROM-FILE |
| Clickable bboxes outside `device_dim` | 159 / 668,604 (coordinates are device pixels) | VERIFIED-FROM-FILE |
| Screen resolutions (`device_dim` per step) | 1440x3120: 15,786 · 1440x3040: 13,865 · 1440x2960: 7,862 · 1080x2280: 988 · 1080x2400: 198 · 3120x1440 (landscape): 10 | VERIFIED-FROM-FILE |
| Actual PNG size = `device_dim` | 60 / 60 random trajectory screenshots (PNG IHDR decoded from ranged, deflated local entries; `amex_pngsample.py`) | VERIFIED-FROM-FILE (sample) |
| Action types | TAP 24,815 · SWIPE 7,628 · TASK_COMPLETE 2,828 · TYPE 2,419 · PRESS_ENTER 651 · TASK_IMPOSSIBLE 220 · PRESS_BACK 135 · PRESS_HOME 13 | VERIFIED-FROM-FILE |
| `need_human_check == true` | 77 steps | VERIFIED-FROM-FILE |
| Packages | 76 distinct. The largest are launchers (launcher3 4,430; nexuslauncher 2,086), Hotels.com 4,084, Agoda 1,807, Booking 1,778, … AndroidWorld packages present: **only** `com.android.chrome` (74 steps) and `com.google.android.documentsui` (7). Google Tasks (`com.google.android.apps.tasks`, 1,176) is not AndroidWorld's `org.tasks`. | VERIFIED-FROM-FILE |

---

## 4. AndroidControl

**Sources and what was read.**
- Official: `gs://gresearch/android_control/`. Listed via
  `https://storage.googleapis.com/storage/v1/b/gresearch/o?prefix=android_control/`: 20 gzip TFRecord shards,
  **49,930,448,595 B total** (2.35–2.68 GB each), plus `splits.json` (98,603 B) and `test_subsplits.json`. The
  shards are whole-file gzip and cannot be seeked, so a full action count needs about 50 GB of transfer, which is
  over the 6 GB budget. **The official shards were not streamed in full.** VERIFIED-FROM-FILE (listing).
- Full count instead from the parquet mirror `smolagents/android-control` (342 parquet files, 67.4 GB). Only the
  `episode_id`, `goal`, `actions` and `step_instructions` column chunks were read, by HTTP range through pyarrow
  (`ac_parquet_actions.py`; well under 100 MB transferred; screenshots never fetched).
- **Is the mirror faithful?** Its episode-id set equals the official `splits.json` set **exactly** (15,283 = 15,283,
  0 extra, 0 missing). The 25 complete official episodes decoded from the first 16 MB of shard 0 and 50 MB of
  shard 10 (probe copies; protobuf parsed by hand, `ac_tfrecord_head.py`) are **25/25 identical** in goal and
  action list. VERIFIED-FROM-FILE. Note: the mirror's own `train`/`test` file split is **not** the official
  split. 2,724 official-train episodes sit in its `test-*` files and 1,238 official-test episodes in its `train-*`
  files, so always re-split with `splits.json`.

**Total actions by `action_type`** (all 15,283 episodes; VERIFIED-FROM-FILE via the mirror):

| action_type | all | train (13,603 eps) | validation (137) | test (1,543) |
|---|---|---|---|---|
| click | 51,918 | 46,424 | 420 | 5,074 |
| scroll | 11,130 | 9,833 | 86 | 1,211 |
| input_text | 6,091 | 5,406 | 53 | 632 |
| wait | 5,782 | 5,159 | 56 | 567 |
| open_app | 5,697 | 5,044 | 45 | 608 |
| navigate_back | 3,034 | 2,662 | 29 | 343 |
| long_press | 167 | 157 | 1 | 9 |
| navigate_home | 29 | 29 | 0 | 0 |
| **total** | **83,848** | **74,714** | **690** | **8,444** |

**Is there a `status` or other terminate-like action?** **No.** The action-type vocabulary is exactly the 8 types
above. A regex for status/terminate/done/finish/complete/stop/success/fail/answer/impossible matches no type.
Non-null field sets are only {x,y} (click/long_press), {direction}, {text}, {app_name} and {} (wait,
navigate_back, navigate_home). VERIFIED-FROM-FILE.
- Last action per episode: click 10,534 · wait 2,312 · scroll 2,102 · input_text 189 · navigate_back 86 ·
  open_app 46 · long_press 4 · navigate_home 1. **9 episodes have zero actions** (ids 9948, 13538, 4739, 1502,
  15813, 3986, 4933, 14564, 1958).
- So the `status` rows in our curated corpus (AGENTS.md: "the final row of all 13,603 episodes, always
  `successful`") were **added by our pipeline**. They are not in the release. This agrees with commit `ba9792d5`
  ("AC status rows are ours").
- `len(actions) == len(step_instructions)` holds for 15,283/15,283 episodes. 37 step instructions contain
  completion/status words, but these are UI text ("Click on Is done to mark the task as complete"), not
  termination.

**Goals naming an app other than the one `open_app` launches.** HEURISTIC string matching (`ac_analyze.py`):
- 5,676 episodes contain ≥1 `open_app` (20 contain more than one). There are 758 distinct `app_name` values.
- **712 / 5,676 (12.5%)** goals do not contain the opened `app_name` as a normalised substring. Examples:
  Google Fit opened as "Fit"; a soft hyphen in "Audio\xadmack"; "Office Suite Pro" vs `OfficeSuite`; and goals that
  name no app at all.
- 1,085 goals mention some *other* corpus app name (≥4 chars, word boundary). This is an upper bound inflated by
  generic words ("Google", "Email", "Maps", "Currency").
- **172** goals mention ≥1 corpus app name but **none** that matches the opened app. Hand-read examples include
  real cross-app cases ("search Nike shoes" opened in Amazon or Temu; "share … using Gmail" opened in
  OfficeSuite/Audiomack) and false positives ("Google Fit" → `Fit`).
- An exact count needs an alias table and is **UNKNOWN**. Bounds: ≥ several dozen (hand-verified examples exist),
  ≤ 712.

---

## Commands and scripts (reproduce)

All scripts are in `.../scratchpad/dsverify_scripts/`:

| Script | Does |
|---|---|
| `remotezip.py` | HTTP-range ZIP central-directory listing (with retries) |
| `splitzip_cd.py` | Reads the zip64 central directory of AMEX's 9-part split archive from its last part |
| `androidlab_analyze.py`, `androidlab_match.py`, `androidlab_extra.py` | AndroidLab structure, XML↔SoM matching, apps, CJK |
| `osatlas_aw.py`, `osatlas_sample.py`, `osatlas_apps.py` | Name coverage, PNG headers and 24 whole images by range, heuristic per-app counts |
| `amex_analyze.py`, `amex_pngsample.py` | Coverage, tap-in-bbox, clickable counts, resolutions; PNG IHDR sampled by range |
| `ac_parquet_actions.py` | Column-projected parquet read of the mirror |
| `ac_tfrecord_head.py` | Hand-rolled TFRecord/tf.Example decoder for the official shard heads |
| `ac_analyze.py` | Counts, terminate check, official cross-check, app-name heuristic |

Example: `python3 -I dsverify_scripts/amex_analyze.py dsverify/amex/ins/instruction_anno dsverify/amex/el/element_anno dsverify/amex/screenshot_listing.json`.

## What this changes

- **AndroidLab cannot supply clean pixels.** Its images are SoM-marked, real-phone, rescaled to 1440x3120 and
  31% CJK. AndroidWorld-app overlap is Settings 1,510 + Contacts 384 + Clock 259 XML rows, on vendor builds.
  Usable as text (XML) supervision. Pairing with images teaches a marked-screenshot distribution.
- **OS-Atlas `aw_mobile`** is the only one of the four that holds actual AndroidWorld-emulator screens (15,905,
  1080x2400, images 1:1 with annotations). It is grounding-only (median 3 elements, no actions), and every
  viewed image carries the pointer-location overlay.
- **AMEX** is complete and clean for trajectories: 100% element coverage, and 96.6% of TAPs land in a clickable
  box. But it holds almost no AndroidWorld apps (Chrome 74 steps, Files 7). Its screenshots are a 93.5 GB split
  zip whose entry paths begin with `../`.
- **AndroidControl has no termination action.** Every `status` row in our corpus is ours. 2,312 episodes end on
  `wait`, which is the closest native end-signal.
