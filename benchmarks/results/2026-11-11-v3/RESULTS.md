# v3 seed 0 — results of the final checkpoint (ckpt_1824)

Every number below is recomputed from the files in this folder. Run card: `docs/research/RUNCARD-v3_s0.md`;
pre-registration: `docs/research/PRE-REG-2026-11-11-v3.md`. Training finished 2026-11-11 11:36 UTC (`TRAIN exit 0`).

Checkpoint used: **ckpt_1824** (the last). It is also the pre-registered choice (highest `validation_v3` step accuracy,
ties -> later step) and it ties the highest "goals with every step right" (14). Other checkpoints' files are kept in
`validation/` for the record; they are not reported.

## 1. Validation (`validation_v3`: 137 official AndroidControl validation goals, every step, 827 rows)
Teacher-forced: every step gets the demonstrator's correct history. Scorer: `run_acv2.score_row` (point inside the gold
element; terminal `status` row included).

| | ckpt_1824 |
|---|---|
| **Goals with every step right** | **14/137** |
| Steps right | 524/827 = **63.4%** |
| Steps right without the 137 terminal `status` rows | 60.6% (690 rows) |
| Terminal `status` rows right | 77.4% (106/137) |
| Action type right | 81.3% |
| First error of a failed goal | start 47 · middle 71 · end 5 |
| Premature `status` (gold is not `status`) | 31 |

Per action type (right / rows): click 238/420 · scroll 55/86 · status 106/137 · type_text 38/53 · wait 27/56 ·
navigate_back 16/29 · open_app 44/45 · long_press 0/1.

**Caveat (found 2026-11-11):** on these rows the app list is the gold app + 24 distractors (`row_app_options`), so the
gold app is always offered on `open_app` steps; open_app accuracy here is optimistic. AndroidWorld (33 real apps) and
AC-High (one fixed list) are not affected.

Arithmetic check: at 63.4% per step and these goal lengths, independent steps would give ~16 goals all right; 14 observed.
Half of 6-step goals all right needs ~89% per step.

## 2. Held-out AndroidWorld templates (`awgen_heldout_v3`: 19 templates never trained, 463 first steps)
App chosen from the emulator's 33 launchable apps: **463/463**. Not yet split by whether the goal names the app (most do);
until it is, this shows copying a named app, not inferring one.

## 3. AndroidControl-High (InfiGUI test = 1,543 official test goals, 8,444 steps; InfiGUI's scorer verbatim)
Standard protocol: every action allowed (InfiGUI's own prompt offers `terminate`). App list: one fixed list of 444
(model's 297 + every app the test phones need). The model also reads the a11y element list (most published models read
pixels only).

| | ckpt_1824 |
|---|---|
| SR (exact match) | *running* |
| Type | *running* |
| Grounding (click match) | *running* |

Interface ceiling (gold answers through the same decoding path, `eval_d1 --oracle`): **88.3%** (strict 297-app list 83.7%),
type 100%. Ceiling losses: click 82.5% (gold element not among the 40 kept, or its centre off the target), type 85.1%
(text not copyable from the input).

Files: `actest/perstep-*.jsonl` (one row per benchmark step: gold, mapped prediction, type/exact match),
`actest/eval-*-VA-val.jsonl` (raw v3 decisions), `actest/*oracle*`. ckpt_900 files (59.89% SR) are kept as an
intermediate record only.

## 4. AndroidWorld (116 tasks, official protocol)
Not run yet. Harness v5 and the config `benchmarks/aw/configs/aw116_v3_s0.yaml` are ready.
