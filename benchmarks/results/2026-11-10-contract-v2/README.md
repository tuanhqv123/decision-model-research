# Contract v2 (R2): the decision model chooses among AndroidWorld-extracted elements

*2026-11-10. Plan `docs/research/PLAN-2026-11-10-question-tree.md` §3 Q2/Q3, §5 Stage 2 "R2"; corpus
`benchmarks/results/2026-11-10-r2-corpus/README.md`. Code + offline measurement only: no model was trained,
loaded or run. Every number below is read from `summary.json`, written by `measure_rules.py` in this directory,
which calls the same `ac_common` functions that training, evaluation and serving call.*

## 1. What contract v2 is

| | v1 (VA track) | **v2** |
|---|---|---|
| Actions | 9 | the same 9 |
| `is_done` | no | no |
| Tracks | V, VA | **VA only** (the contract refuses V) |
| Element list | `pick_candidates` over the old all-nodes extraction, smallest first | AndroidWorld's own `forest_to_ui_elements(..., exclude_invisible_elements=True)` list, canonicalised and filtered by `element_rule` |
| Questions | action, target_box, target_el, scroll_dir, app_name, type_text | **action, target_el, scroll_dir, app_name, type_text** (no `target_box`) |
| Point | box centre, or the chosen candidate's centre | **the chosen element's bbox centre** |
| Stamped in `contract.json` | name, tracks | name, tracks, **`element_rule`** (read back by `read_contract`; an unknown or missing rule is refused) |

The state, exactly (`ac_common.build_state_v2`; no memory, trajectory or app-intent lines):

```
Goal: <goal>
Previous actions:            (or "Previous actions: none")
Step 1: <render_history_line(...)>
Elements on screen (id: label | type | state | position):
[1] Navigate up | ImageButton | click | x:0-13% y:3-9%
[2] New note | EditText | click,edit,long_click,focused | x:15-83% y:4-8%
```

- `canonical_element`: label = first non-empty of text / content_description / hint_text, whitespace collapsed,
  ≤ 40 chars; `class_short`; integer bbox; 8 flags (clickable, editable, scrollable, checkable, checked, selected,
  long_clickable, focused); package_name. It accepts the corpus shape (`class_name`, `is_*`, None strings), the
  harness shape (`awenv_bridge._element_dicts`: `class`, `clickable` and `is_clickable`, "" strings) and the old
  flat request shape (`bbox` list + `clickable`).
- `render_elements`: `[i] label | class_short | flags | x:a-b% y:c-d%`, `i` = 1..n in kept order, `(no label)`
  when the label is empty, `-` when no flag is set or the class is empty, percentages clamped to the screen.
- `target_el` criteria = `{"1": label, ..., "n": label}` (the rendered label strings). With no kept element the
  `target_el` question is omitted (a choice needs an option) — train and serve alike.

## 2. Rule table

Rules are pure functions of the canonical list and the screen size (`ELEMENT_RULES[name](elements, W, H)`),
never of the gold. Gold coverage = spatial rows (click / long_press / type_text) where SOME kept element's centre
lies inside the gold target bbox — what the scorer credits. Denominator: all spatial rows (8,387 train / 474 val).

| rule | split | len p50 | p90 | p99 | max | empty lists | **gold coverage** | `target_el_idx` kept | `target_el_centre_idx` kept |
|---|---|---|---|---|---|---|---|---|---|
| (a) `all_app` | train | 43 | 118 | 157 | 903 | 5 | 8,363 (99.71%) | 98.82% | 99.64% |
| (b) `interactive_or_labelled` | train | 29 | 67 | 99 | 582 | 43 | 7,917 (94.40%) | 82.65% | 81.50% |
| **(c) `cap40_reading`** | train | 29 | 40 | 40 | 40 | 43 | **7,249 (86.43%)** | 78.40% | 77.35% |
| (d) `cap40_interactive_first` | train | 29 | 40 | 40 | 40 | 43 | 7,133 (85.05%) | 75.06% | 73.14% |
| (a) `all_app` | val | 39 | 112 | 158 | 166 | 2 | 474 (100.00%) | 98.73% | 99.58% |
| (b) `interactive_or_labelled` | val | 27 | 62 | 92 | 119 | 8 | 450 (94.94%) | 81.86% | 81.65% |
| **(c) `cap40_reading`** | val | 27 | 40 | 40 | 40 | 8 | **412 (86.92%)** | 79.11% | 78.48% |
| (d) `cap40_interactive_first` | val | 27 | 40 | 40 | 40 | 8 | 412 (86.92%) | 78.27% | 77.00% |

## 3. The choice: `cap40_reading`

Pre-declared selection rule: among rules with train p90 list length ≤ 40, the highest train gold coverage; ties →
shorter train median. (a) and (b) fail the length gate (p90 118 and 67). (c) beats (d) on train coverage, 7,249 vs
7,133 (+116 rows, +1.38 pts). `CONTRACTS["v2"]["element_rule"] = "cap40_reading"`; a test re-applies
`measure_rules.choose_rule` to the stored train table and checks the corpus sha256s, so the choice cannot drift.
(The default was set provisionally to `cap40_reading` in the first commit, before the measurement; the measured
choice agreed.)

What the rule costs, on train: filtering system UI, zero-area, off-screen and unlabelled inert elements costs
99.71 → 94.40% coverage (−5.31 pts); the 40-item cap in reading order costs a further 94.40 → 86.43% (−7.97 pts).
By action (chosen rule, train): click 85.80% (6,779/7,901), long_press 72.97% (27/37), type_text 98.66% (443/449).
The 43 train / 8 val screens with an empty kept list have no `target_el` question at all.

## 4. Gold supervision for `target_el` (chosen rule)

Decided order: the element `target_el_idx` names if it survives the filter; else the one `target_el_centre_idx`
names if it survives; else no `target_el` supervision (the action is still supervised).

| case | train (of 8,387 spatial) | val (of 474) |
|---|---|---|
| `target_el_idx` survives | 6,575 (78.40%) | 375 (79.11%) |
| else `target_el_centre_idx` survives | 198 (2.36%) | 14 (2.95%) |
| none → no `target_el` supervision | 1,614 (19.24%) | 85 (17.93%) |

Diagnostics of that order against coverage (not used by any rule; reported because they change a decision):

| | train | val |
|---|---|---|
| supervised element whose centre is NOT inside the gold target (unscoreable label, the R4 defect class) | 198 (2.36%) | 16 (3.38%) |
| a creditable element IS kept, but the order gives no supervision | 498 (5.94%) | 26 (5.49%) |
| a creditable element IS kept, but a non-creditable one is supervised | 176 (2.10%) | 13 (2.74%) |

So 674 of the 7,249 train rows that show a creditable element (9.3%) are not supervised towards one. A third
fallback — "the smallest KEPT element whose centre lies inside the gold target" — would supervise a creditable
element on all 7,249 (86.43% of spatial rows vs 78.40% now). **Not implemented**: the gold order was decided; this
is a measured option for the operator.

## 5. Option-text ambiguity of the label criteria (chosen rule)

The question block lists the options by their label text (d1 renders `<code> <criteria value>`, codes A..Z then
AA..), so unlabelled elements all read `(no label)` and repeated labels read the same:

| | train | val |
|---|---|---|
| kept elements with an empty label | 46,401 / 405,538 (11.44%) | 2,634 / 21,439 (12.29%) |
| screens with at least one repeated option text | 12,051 / 14,965 (80.53%) | 630 / 827 (76.18%) |
| gold element unlabelled | 753 / 8,387 (8.98%) | 64 / 474 (13.50%) |
| gold option text shared with another option | 1,444 / 8,387 (17.22%) | 92 / 474 (19.41%) |

On those rows only the option ORDER (code ↔ `[i]`) separates the gold from its look-alikes. v1 avoided part of
this by writing `element at <position>` for unlabelled candidates. Not changed (criteria were specified as the
label strings); flagged as a risk to measure in the first v2 run.

## 6. State length (d1-2B tokenizer)

Tokens of the state string only (`tokenizer.json` of `LiquidAI/d1-2B@cfe5760`, no special tokens; the system
prompt, image tokens and the other question blocks are not included). The `target_el` question block is rendered
with the checkpoint's own `prompt.question_block` (exact, `d1_codes = 2`).

| rule | split | state p50 | p90 | p99 | max | `target_el` block p50 | p90 | max |
|---|---|---|---|---|---|---|---|---|
| **`cap40_reading`** | train | **777** | **1,092** | 1,193 | **1,687** | 171 | 275 | 876 |
| **`cap40_reading`** | val | **733** | **1,083** | 1,260 | **1,394** | 165 | 274 | 629 |
| `all_app` (for scale) | train | 1,118 | 2,959 | 3,935 | 27,146 | 251 | 572 | 7,735 |

## 7. Where the pieces live and what was checked

- `benchmarks/train/ac_common.py`: `canonical_element`, `ELEMENT_RULES`, `select_elements`, `render_elements`,
  `build_state_v2`, `element_inputs`, `CONTRACTS["v2"]`, v2 branch of `build_questions`, `gold_answers_v2`,
  `element_centre`, `chosen_element`; `write_contract` refuses v2 on any track set but `["VA"]`; `read_contract`
  refuses a v2 document without a known rule.
- `benchmarks/train/train_d1.py`: `--contract v2 --track VA --train-set cur15k_v2`; `make_sample` builds from
  `rec["elements"]`; `check_args` refuses `--track V|mix`, `--eval-every` (the 398 test rows carry no element
  list) and `--traj-k` before any model is loaded; no app memory is built.
- `benchmarks/train/eval_d1.py`: `--val --val-set validation_v2 --contract v2` (track defaults to VA; the rule is
  read from the checkpoint's `contract.json`); point = chosen element's centre, scored by `score_row`.
- `serving/d1_api.py`: `build_request` (no model needed); a v2 checkpoint returns `point` = chosen element's
  centre, `element` {idx, label, confidence, aw_index}, `contract`, `element_rule`, and `readouts.target_el`
  when asked; no elements or track V → HTTP 400.
- `benchmarks/aw/awenv/masking.py`: a v2 decision's element alternatives are rebuilt with
  `ac_common.select_elements` under the rule the server reports (previously the v1 list would have been used).

**Train/serve parity** (`tests/test_d1_api_v2.py`): one synthetic AndroidWorld screen (status bar, toolbar, a text
field, a checkbox, 45 rows so the cap and reading order bite, an invisible node, a clickable container AW drops),
stored the corpus way (`aw_extract.extract_elements`) and fed to `train_d1.make_sample`, and served the harness
way (vendored `forest_to_ui_elements` → `_element_dicts` → JSON → the real `/v1/decide` handler with a recording
stand-in model): state and questions are byte-identical. Dropping the bridge's `class` key from
`canonical_element` makes the test fail (checked). A dry run of `make_sample` over all 14,965 + 827 corpus rows
(images stubbed) raised nothing, and every gold answer is an option of its question.

**Not byte-identical, and why**: (1) the picture — training reads the corpus JPEG, serving the emulator's PNG,
through the same `load_image` (true of every contract); (2) the old flat request shape carries only `clickable`,
so a node with other flags set renders differently from the corpus — the current harness sends every flag.

## 8. Reproduce

```bash
# CPU only; the v2 files copied read-only from the training host (sha256 as in the R2 README)
uv run --no-project --python 3.12 --with numpy --with tokenizers python measure_rules.py \
    --train compact_cur15k_v2.jsonl --val compact_validation_v2.jsonl \
    --tokenizer <HF cache>/models--LiquidAI--d1-2B/snapshots/cfe5760ff5d5998f91d9ac56bd4b2dbd96996d41
```
About 90 s. Python ≥ 3.10 is needed only for the exact question-block count (d1's `prompt.py`); without it the
script still measures everything else.
