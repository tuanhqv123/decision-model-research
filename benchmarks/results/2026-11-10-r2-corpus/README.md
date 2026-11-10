# R2 corpus (v2): AndroidControl rows re-extracted the AndroidWorld way — build record and validation

*2026-11-10. Plan `docs/research/PLAN-2026-11-10-question-tree.md` §5 Stage 2 "R2" (feeds Q2, Q3, Q6);
audit `docs/research/2026-11-10-AUDIT-androidworld-contract.md` §2 D2/D3, §5.1, §5.3 R4.
Every number below is read from `summary.json` (written by the merge) or printed by `post_check.py`,
both in this directory. Nothing here is a model result.*

## 1. What exists

| File | Host | Rows | Bytes | sha256 |
|---|---|---|---|---|
| `compact_cur15k_v2.jsonl` | HOST_TRAIN `acv2/acdata/`, next to v1 | 14,965 | 609,617,711 | `3642d03b217dc0fe97549945d50e64d3482e4bf73c6a4a1fb5d63d40cf9a914b` |
| `compact_validation_v2.jsonl` | HOST_TRAIN `acv2/acdata/`, next to v1 | 827 | 32,243,690 | `6f15fc7e85a0c0365aa0e8a43e6869890ac098b939f209187bfba5b6cf6cfa55` |

Checksums were verified on HOST_TRAIN after the copy (identical to the build host). The build copy stays on
HOST_EMU under `acv2_rebuild/out/`. Every `img` path of both files exists on HOST_TRAIN (0 missing of 15,792).

Inputs: the v1 files (`compact_cur15k_v1.jsonl` sha256 `1927db09…4efa`, `compact_validation_v1.jsonl`
`ba62d821…3c43307`, copied read-only from HOST_TRAIN) and the 20 public AndroidControl TFRecord shards +
`splits.json` already on HOST_EMU (downloaded by the earlier attempt; not downloaded again).

Rows are **exactly** the v1 (ep, step) rows in v1 order; every v1 field is kept unchanged and in the same key
order (`post_check.py`: 0 changed rows, 0 key-order changes in either file). Added per row:

| Field | Meaning |
|---|---|
| `elements` | AndroidWorld's served list for the screen: `forest_to_ui_elements(forest, exclude_invisible_elements=True)` from `android_world@55ddaf5`, vendored verbatim in `benchmarks/train/aw_extract.py` (lines 23–163 of `representation_utils.py`, hash-pinned). Per element: `text, content_description, hint_text, class_name, bbox_pixels{x_min,y_min,x_max,y_max}, is_clickable, is_editable, is_scrollable, is_checkable, is_checked, is_selected, is_focused, is_long_clickable, is_enabled, is_visible, resource_name`, **plus `package_name`** (see §6). List order = AndroidWorld's `index`. |
| `target_el_idx` | index of the smallest element whose bbox contains the gold point (click / long_press / type_text, v1 `action.x,y`); null if none. Zero-area boxes are never chosen. |
| `target_el_exact` | that element's bbox equals the v1 gold `target` bbox (null when no element) |
| `target_el_hits_gold` | that element's centre lies inside the v1 gold `target` (tapping it reaches the target) |
| `target_el_centre_idx` | smallest element whose centre lies inside the v1 gold `target` (diagnostic for D2) |
| `effect` | t < last: the multiset of (class_name, text, content_description, bbox, all 10 flags) of `elements` differs between screen t and t+1; null on the terminal `status` row |
| `effect_app` | same, ignoring `com.android.systemui` elements (added — §6) |
| `pixel_change` | t < last: fraction of pixels differing by > 24 grey levels between screenshots t and t+1, grey, 270×600 (BICUBIC), top 30 rows (5%, status bar) dropped — the B0-diagnosis definition (`2026-11-10-b0-diagnosis/fixedpoint.py`); computed from the full-resolution PNGs in the TFRecords |
| `local_goal` | the step's AndroidControl `step_instruction` (null on the status row) |
| `local_done` | next step's instruction differs (after whitespace collapse + casefold), or t is the last action before the terminal status; null on the status row |
| `n_steps` | actions in the source episode (status row has `step == n_steps`) |
| `elements_source` | `android_world@55ddaf5 forest_to_ui_elements(exclude_invisible_elements=True)` |

Whole source episodes were read (screen t+1 and instruction t+1 are outside the v1 rows); 9,121 episodes,
0 episodes with a screenshot/tree/size count other than `n_actions + 1`.

## 2. Differential check against v1 — 0 mismatches

Each v1 field below was **recomputed from the raw TFRecords with the original v1 code paths**
(`preprocess_ac.flat_nodes / smallest / convert_action`, the type_text focus/last-click rule of
`preprocess_ac.do_shard`, `compact_ac`'s `pick_candidates` + `keep_bbox`, `ac_common.render_history_line`)
and compared with the stored v1 value.

| Check | cur15k (14,965) | validation (827) |
|---|---|---|
| row missing from the rebuild | 0 | 0 |
| `goal` | 0 | 0 |
| `action.action_type` | 0 | 0 |
| full `action` dict (incl. type_text x,y) | 0 | 0 |
| gold `target` bbox (incl. centre) | 0 | 0 |
| `W`, `H` | 0 | 0 |
| `history` (re-rendered gold actions of steps < t) | 0 | 0 |
| `cands` (sha1 of the full candidate list) | 0 | 0 |
| per-episode step count: `status` row with `step != n_steps` | 0 | 0 |
| per-episode step count: non-status row with `step >= n_steps` | 0 | 0 |

So the v2 rows describe the same screens, actions and targets as v1; only the element list is new.

## 3. Coverage of the gold target in the AndroidWorld list

| | cur15k | validation |
|---|---|---|
| spatial rows (click / long_press / type_text) | 8,387 | 474 |
| `target_el_idx` present | 8,296 (98.91%) | 468 (98.73%) |
| `target_el_exact` | 7,849 (93.59%) | 437 (92.19%) |
| `target_el_hits_gold` | 8,014 (95.55%) | 449 (94.73%) |
| `target_el_centre_idx` present | 8,368 (99.77%) | 474 (100%) |
| click: idx / exact / hits | 7,812 / 7,402 / 7,534 of 7,901 | 415 / 394 / 399 of 420 |
| long_press: idx / exact / hits | 36 / 35 / 35 of 37 | 0 / 0 / 0 of 1 |
| type_text: idx / exact / hits | 448 / 412 / 445 of 449 | 53 / 43 / 50 of 53 |

Where it is not exact (`post_check.py`): 91 / 7 spatial rows have no containing element (89 / 5 clicks); 20 / 5
type_text rows carry the gold bbox in the list but a smaller element (a child of the field) contains the
field's centre; 8 / 1 gold targets have zero area (never chosen by design); 8 / 9 gold targets cover more than
half the screen (audit T4/E3).

**Elements per screen** (all rows; the list includes status bar, navigation bar and notification icons):

| | p5 | p25 | p50 | p75 | p95 | max | mean |
|---|---|---|---|---|---|---|---|
| AW `elements`, cur15k | 21 | 40 | 63 | 97 | 155 | 1,127 | 73.8 |
| AW `elements`, validation | 17 | 38 | 59 | 94 | 148 | 264 | 70.4 |
| v1 `cands` (pick_candidates), cur15k | 7 | 19 | 34 | 56 | 84 | 328 | 39.3 |
| all a11y nodes (old `flat_nodes`), cur15k | 51 | 95 | 153 | 243 | 627 | 20,430 | 222.8 |

`com.android.systemui` elements per screen: median 14, p95 55, mean 20.2 (cur15k); median 15, mean 19.7 (val).
No screen has an empty list. 0.88% of elements (9,773 / 1,104,993) have a zero-area box.

## 4. Audit D2 — gold targets the old extraction had and AndroidWorld's drops

| | cur15k | validation |
|---|---|---|
| gold bbox in the **v1 candidate list** (exact, `gold_candidate`) | 6,882 (82.06%) | 393 (82.91%) |
| …only a candidate whose centre is in the gold (v1 rule 2) | 1,338 (15.95%) | 70 (14.77%) |
| …absent from the v1 candidate list | 167 (1.99%) | 11 (2.32%) |
| gold bbox in the **AndroidWorld list** | 7,869 (93.82%) | 442 (93.25%) |
| **dropped by AndroidWorld among those the v1 candidate list had** | **250 / 6,882 (3.63%)** | **18 / 393 (4.58%)** |
| gold bbox missing from the AW list (vs all nodes, where it exists by construction) | 518 (6.18%) | 32 (6.75%) |
| …reason: visible container (has children, no content_description, not scrollable) | 447 | 29 |
| …reason: invisible | 60 | 3 |
| …reason: both (several nodes share the bbox) | 11 | 0 |
| …an AW element's centre lies inside the gold (a tap that still reaches it) | 499 / 518 (96.33%) | 32 / 32 |
| …the smallest-containing rule still lands inside the gold | 147 / 518 (28.38%) | 8 / 32 (25.0%) |

Reading: the T6 figure (82.1% exact in the v1 candidate list) reproduces here (82.06%). Switching to
AndroidWorld's extractor **raises** exact coverage to 93.8%, because the v1 filter dropped unlabeled,
non-actionable leaves (icons inside buttons) that are often the smallest node under the click, and
AndroidWorld keeps every leaf. The D2 cost is real but small: 3.6% of the targets the v1 list had are dropped,
almost all clickable containers whose own node has no text. For 96% of all dropped targets some served
element sits inside the target, but `target_el_idx` (smallest **containing**) picks it in only 28%; a trainer
that wants a reachable label should fall back to `target_el_centre_idx` when `target_el_exact` is false.

## 5. Effect, pixel change and local-goal labels

Non-terminal rows: 11,675 (cur15k) / 690 (val); status rows are null by definition (3,290 / 137).

| | cur15k | validation |
|---|---|---|
| `pixel_change` p5 / p25 / p50 / p75 / p95 | 0.0058 / 0.093 / 0.281 / 0.562 / 0.904 | 0.0095 / 0.095 / 0.284 / 0.556 / 0.879 |
| `pixel_change > 0.002` | 96.81% | 97.54% |
| `effect` true / false | 11,557 / **118 (1.01% false)** | 678 / **12 (1.74%)** |
| `effect_app` true / false | 11,444 / **231 (1.98% false)** | 666 / **24 (3.48%)** |
| `effect` true but `effect_app` false (system-UI-only change) | 113 | 12 |
| `effect` false but `effect_app` true | 0 | 0 |

`effect` by action (true / rows): cur15k click 7,828/7,901 (99.1%), long_press 37/37, navigate_back 747/749
(99.7%), open_app 1,041/1,042, scroll 1,317/1,347 (97.8%), type_text 448/449, wait 139/150 (92.7%);
validation click 414/420 (98.6%), long_press 1/1, navigate_back 29/29, open_app 45/45, scroll 86/86,
type_text 53/53, wait 50/56 (89.3%).
`effect_app` by action: cur15k click 7,752 (98.1%), long_press 36 (97.3%), navigate_back 745 (99.5%),
open_app 1,040 (99.8%), scroll 1,292 (95.9%), type_text 444 (98.9%), wait 135 (90.0%); validation click 405
(96.4%), type_text 52 (98.1%), wait 48 (85.7%), all others 100%.

**Agreement with `pixel_change > 0.002`** (rows × [label, pixel]):

| | label&pix | label&!pix | !label&pix | !label&!pix | agreement | Cohen κ |
|---|---|---|---|---|---|---|
| `effect`, cur15k | 11,257 | 300 | 45 | 73 | 97.04% | 0.286 |
| `effect_app`, cur15k | 11,212 | 232 | 90 | 141 | 97.24% | 0.454 |
| `effect`, validation | 667 | 11 | 6 | 6 | 97.54% | 0.402 |
| `effect_app`, validation | 661 | 5 | 12 | 12 | 97.54% | 0.573 |

Per action (cur15k, `effect`): click 96.8%, long_press 97.3%, navigate_back 99.2%, open_app 99.9%, scroll
94.8%, type_text 99.3%, wait 94.0%.

Anatomy of the disagreements (19 rows of shard 0 inspected by hand, `effect` vs pixel):
- *element change, no pixel change* — mostly **real state changes too small for the pixel metric**: a
  `ToggleButton` Off→On, a `Switch` flipping `is_checked`, a counter 8000→8500, a `CheckedTextView` selection —
  exactly the toggle blindness of plan F2 — plus **status-bar-only changes** (clock minute tick, battery %,
  a new notification icon), which `effect` counts and `effect_app` does not (113 rows in cur15k);
- *pixel change, no element change* — scrolls and waits where the a11y tree did not change (canvas/WebView
  content, or a tree that did not refresh);
- *neither* — genuine no-op steps.

`local_done` (non-terminal rows): cur15k **true 9,620 / false 2,055 (82.40% true)**; validation 576 / 114
(83.48%). By action, cur15k: click 88.3%, long_press 78.4%, navigate_back 72.1%, open_app 85.0%,
**scroll 46.2%**, type_text 96.4%, wait 88.7%; validation: click 88.1%, navigate_back 75.9%, open_app 82.2%,
scroll 51.2%, type_text 100%, wait 87.5%. No non-status row has a missing `local_goal` or `local_done`.

## 6. Decisions, deviations and what they mean for the next experiments

1. **`package_name` and `effect_app` were added beyond the brief.** The smoke shard showed `effect` firing on
   rows whose only change was the status-bar clock; in AndroidWorld the clock also ticks between steps. `effect`
   is kept exactly as specified; `effect_app` drops `com.android.systemui` elements on both sides, which needs
   `package_name` in the file (it is a field AndroidWorld serves). Cost: 7.6% of the validation file's bytes.
2. **Zero-area elements are never a target** (AndroidWorld taps an element's bbox centre; audit T5). 8 + 1 rows.
3. **type_text uses the v1 point**, which is the centre of the focused editable field (v1 rule), not a
   click. Of the 37 / 10 non-exact type_text rows, 20 / 5 have the field itself in the list but a smaller child
   contains its centre; `target_el_hits_gold` is true on 445/449 / 50/53 type_text rows.
4. **AndroidControl almost never shows a no-effect step: 1.0% (`effect`) / 2.0% (`effect_app`) of non-terminal
   rows, 3.2% by pixels.** Online, after 53.9% of B0 steps the model saw the same screen again (plan F2; a
   different denominator — the model's own actions — and the pixel definition). Expert demonstrations are
   effective by construction, so this corpus cannot teach "my action did nothing"; those negatives have to come
   from AndroidWorld rollouts (Q2/Q8). The label is present and correct, it is just almost always `true`.
5. **The served list is long here.** Median 63 elements (p95 155) per AndroidControl screen vs a median of ~23
   at AndroidWorld run time (audit §5.3 R1); the median screen carries 14–15 system-UI elements. The failed VA
   arm capped the list at 14 candidates (AGENTS.md 2026-11-07 (s)); an R2 candidate model has to carry 60–150
   options or filter them, and any filter must be applied identically at serve time.
6. **`local_done` is weak for scrolls** (46% true): one instruction ("scroll down to X") usually covers several
   scrolls. 17.6% of non-terminal rows sit inside a multi-action local goal.
7. **File size.** `compact_cur15k_v2.jsonl` is 610 MB (≈32 KB of new fields per row, almost all `elements`);
   a loader that materialises every row will need several GB of RAM.

## 7. Tests and reproduction

- `tests/test_aw_extract.py`: the vendored block (lines 23–163) and each vendored definition are pinned by
  sha256 values computed from the **upstream** file; optional direct comparison with a clone via
  `AW_REPR_UTILS_REF`; synthetic forest (leaf kept, content-description container kept, scrollable container
  kept, invisible leaf dropped, plain container dropped); real-proto round trip (needs `android_env`); field set;
  smallest-containing selection (nesting, edges, zero area, ties); `target_fields`; multiset `effect` (order,
  text, bbox, toggle, multiplicity); `effect_app` vs a clock tick; `pixel_change` (status bar ignored, body
  counted); `local_done` (normalisation, last action, missing instruction).
- `tests/test_rebuild_ac_v2.py`: one synthetic episode through `build_episode`, the v1 oracle (reproduces every
  v1 field), the D2 reason (`container`), type_text target, effect/local_done/terminal nulls, a clock-tick-only
  step, and the merge (v1 fields and order preserved, 0 differential mismatches).
- Results: 24 passed / 2 skipped on the Mac (Python 3.9, no `android_env`); 25 passed / 1 skipped on HOST_EMU
  (the reference-clone comparison skips there; it passed locally with `AW_REPR_UTILS_REF` set).

```bash
# HOST_EMU, venv with protobuf + android_env (forest proto) + tfrecord (Example proto) + pillow + numpy
nice -n 19 ionice -c3 python benchmarks/train/rebuild_ac_v2.py extract --raw <raw shards> \
    --rows compact_cur15k_v1.jsonl compact_validation_v1.jsonl --work work --workers 6 --min-avail-gb 8
python benchmarks/train/rebuild_ac_v2.py merge --work work \
    --rows compact_cur15k_v1.jsonl compact_validation_v1.jsonl \
    --outs compact_cur15k_v2.jsonl compact_validation_v2.jsonl      # writes work/summary.json
python benchmarks/results/2026-11-10-r2-corpus/post_check.py \
    compact_cur15k_v1.jsonl compact_cur15k_v2.jsonl compact_validation_v1.jsonl compact_validation_v2.jsonl
```

Wall time: ~66 s per shard per worker; 20 shards on 6 workers ≈ 4.5 min; merge < 1 min. Available RAM on the
host stayed ≥ 35 GB (the extractor pauses below 8 GB). No GPU, docker, emulator or benchmark file was touched.
