# Stage 1 (2026-11-10): B0′ and R1 s0 on the 116 AndroidWorld tasks, official protocol

Pre-registration: `docs/research/PRE-REG-2026-11-10-contract-v1.md` (§4, A1, A2, A2-bis, A3). Board:
`docs/research/EXEC-2026-11-10.md`. Every number below is produced by the commands listed, from the files in this
directory.

## Files

| File | What | How it was made |
|---|---|---|
| `b0prime_results.jsonl` | 116 result lines, one per task, B0′ | last `"type": "result"` line of every non-abandoned `episodes/*.jsonl` in B0′'s run directory (104-task main run, harness v2, plus the 12-task completion, harness v2f; same checkpoint, contract, server, emulator host) |
| `b0prime_export.json` | step-level export, 116 episodes / 2,019 steps | frozen `benchmarks/results/2026-11-10-b0-diagnosis/export_steps.py` (md5 `9ef41501…`), unmodified, on the emulator host: `HOME=<scratch> AW_RUN_DIR=<absolute run dir> ~/vllm-env/bin/python3 export_steps.py` — absolute path because the script `expanduser`s `AW_RUN_DIR` and `HOME` is redirected (A3 item 4) |
| `b0prime_clustered_all.json` | A2-bis episode-clustered mechanism metrics | `python3 clustered.py b0prime_export.json --set all --inspected inspected_tasks.txt` (frozen script, same directory as above) |
| `r1s0_results.jsonl` | 116 result lines, R1 seed 0 (`v1_s0`, contract v1) | same extraction; 104-task main run + 12-task completion (the container died at 104), harness v4, B0′'s emulator host; every preflight health line of the run names `v1_s0` (12/12) |
| `r1s0_export.json`, `r1s0_clustered_all.json` | R1 s0 step export (116 episodes / 1,369 steps) and A2-bis metrics | as for B0′ |
| `paired_sr_s0.md`, `paired_sr_s0.json` | **the pre-registered analysis** | `python3 benchmarks/aw/paired_sr.py r1s0_results.jsonl b0prime_results.jsonl --seed-label s0 [--json]`; `paired_sr.py`, the task lists and `awenv/metrics.py` are byte-identical to the frozen commit `da5d3eb8` |
| `posthoc.py`, `posthoc.txt` | **post-hoc, not pre-registered**: outcome buckets, w/l tasks, first-encounter split | `python3 posthoc.py` |

## B0′ (E2, contract e2, overlay hidden), all 116 tasks

`python3 benchmarks/aw/paired_sr.py`'s `arm_summary` on `b0prime_results.jsonl`:

| set | official SR | full successes | Wilson 95% | SR task-completion | SR answer-only | excluded |
|---|---|---|---|---|---|---|
| 116 | 7.0% | 8/115 | [3.6, 13.1] | 8.9% (n=90) | 0.0% (n=25) | SimpleSmsReplyMostRecent (setup error) |
| 104 non-shadowed | 5.8% | 6/103 | [2.7, 12.1] | 7.7% (n=78) | 0.0% (n=25) | same |
| 12 shadowed by H11 | 16.7% | 2/12 | [4.7, 44.8] | 16.7% (n=12) | — | — |

The 12 tasks the 24-character episode-id cut hid from every earlier run hold 2 of B0′'s 8 successes
(`AudioRecorderRecordAudioWithFileName`, `RecipeDeleteMultipleRecipesWithConstraint`).

Mechanism (A2-bis, `--set all`, 115 episodes, 1,904 executed steps): no-effect pooled 63.2% [54.0, 71.0], per-episode
median 66.7%; first-encounter repeat 65.8% [59.3, 73.3] (n = 243); AUROC(confidence → effect) 0.544 [0.476, 0.611];
typing 2.0% of decisions.

## R1 seed 0 vs B0′ — pre-registered (§4, A1, A3)

| set | R1 s0 | B0′ | w (R1 only) | l (B0′ only) | both succeed | exact sign test p | A1: seeds 1–2 online |
|---|---|---|---|---|---|---|---|
| primary, 116 | **12.2%** 14/115 [7.4, 19.4] | 7.0% 8/115 [3.6, 13.1] | 9 | 3 | 5 | 0.146 | **yes** (\|w−l\| = 6, exactly the bar) |
| A3 sensitivity, 104 | 12.6% 13/103 [7.5, 20.4] | 5.8% 6/103 [2.7, 12.1] | 9 | 2 | 4 | 0.065 | yes (7) |

Task-completion split: R1 15.6% vs 8.9% (n=90); answer-only 0/25 in both. Excluded in both: SimpleSmsReplyMostRecent.
The two sets agree on the A1 decision (script) and on the sign of w − l (checked by hand: A3(3) names both, the
frozen script tests only the first). **No claim**: §4 needs three seeds (pooled p < 0.05 and all seeds moving SR the
same way). Q10 (same-host replay of E2 on 20 tasks) measured 0/20 outcome flips but with a 16.1% upper bound, so
run-to-run noise of a few tasks over 116 is not excluded.

**Mechanism (A2 item 2, frozen scripts, episode-clustered, all tasks).** Prediction written before the run: if
out-of-distribution history is a cause, R1 lowers the fixed-point rate and raises typing; within 5 / 1 points = no.

| metric | R1 s0 | B0′ | verdict vs prediction |
|---|---|---|---|
| first-encounter repeat | **41.2%** [32.8, 50.2] (n=199) | 65.8% [59.3, 73.3] (n=243) | lowered by 24.6 pts — **as predicted** |
| typing share of decisions | 2.9% (40/1,369) | 2.0% (41/2,019) | +0.9 pts — **unchanged** (within 1 pt; same absolute count) |
| no-effect, per-episode median | 25.0% | 66.7% | — |
| no-effect, pooled | 60.9% [49.7, 70.3] | 63.2% [54.0, 71.0] | — |
| AUROC(confidence → effect) | 0.675 [0.587, 0.756] | 0.544 [0.476, 0.611] | — |

**Post-hoc (not pre-registered; `posthoc.txt`).** The SR gain is mostly *termination*: 7 of R1's 9 wins are tasks
B0′ had already put in the right state but never declared done (Wi-Fi / Bluetooth / brightness and their Verify
variants, 3–6 steps); all 3 losses are R1 declaring done early. R1 declares done 81 times (14 correct, 17.3%) vs
B0′ 34 (8 correct, 23.5%); 36 tasks move from timeout to premature done (both fail). Median steps 6 vs 11. The
repeat drop is not only termination: at first encounters R1 chooses `status` 9.5% (B0′ 0.4%) and a different action
49.2% (B0′ 33.7%). R1 never chose `navigate_back` (B0′ 35 times), although contract v1 offers it.

## Run-to-run noise (Q10), host equivalence, and the overlay (Q12) — `replay_agreement.txt`

`benchmarks/aw/replay_agreement.py` (tests: `tests/test_replay_agreement.py`) compares two runs on the tasks scored
in both: outcome flips, declared-done differences, and semantic trajectories (action type, element hit, text,
direction, app — not raw pixels, not step counts). Excluded episodes are set aside, as in `paired_sr.py`.

| pair | tasks | outcome flips | done diffs | divergent trajectories |
|---|---|---|---|---|
| B0′ vs Q10 (same host, same weights) | 20 | 0 [0, 16.1%] | 0 | 6 [14.5, 51.9%] |
| B0′ vs host check (other emulator host) | 20 | 0 [0, 16.1%] | 0 | 10 [29.9, 70.1%] |
| B0-overlay vs B0′ (Q12) | 94 | 1 [0.2, 5.8%] | 8 | **80 [76.5, 90.9%]**, first divergence at step 1 on almost all |

The pointer-location overlay (H10) changes the model's input from the first touch on, and with it nearly every
trajectory; outcomes barely move (1 flip in 94). So B0-overlay's step-level diagnosis describes a different input
than the official one, while its outcome numbers are close to B0′'s. 10 tasks set aside: B0-overlay's 9 dead-a11y
setup errors and SimpleSmsReplyMostRecent (both arms).

## Why the rate is low — post-hoc, not pre-registered (`posthoc2.py`, `posthoc2.txt`, Q11 floor `q11_status_now_results.jsonl`)

**Q11, the floor:** a policy that only says `status` at step 0 (no model) scores **6.6% (7/113)**, Wilson [3.0, 12.2];
3 tasks excluded. Its 7 tasks are ones whose end state already holds (Wi-Fi/Bluetooth/brightness "Verify" variants,
`RecipeDeleteMultipleRecipesWithConstraint`). B0′ vs this floor: w = 7, l = 6 — **B0′ (7.0%) is not distinguishable
from doing nothing**; R1 s0 vs the floor: w = 7, l = 0.

| | successes | on floor tasks | **beyond the floor** |
|---|---|---|---|
| Q11 floor | 7 | 7 | 0 |
| B0′ | 8 | 1 | **7** |
| R1 s0 | 14 | 7 | **7** |

So R1 s0's whole gain over B0′ is the 6 floor tasks B0′ did not declare done; **beyond the floor both do 7 tasks.**
Success by step budget (task-completion tasks, successes/tasks): B0′ 4/25 · 3/36 · 1/18 · 0/11 for budgets 0–10 ·
11–20 · 21–40 · 41+; R1 s0 11/25 · 2/36 · 1/18 · 0/11. Nothing above 20 steps succeeds except 2 tasks.
Termination: R1 s0 declares done on 61/90 task-completion tasks (14 right, 47 wrong, wrong at a median 36% of the
step budget) vs B0′ 22/90 (8 right, 14 wrong); a wrong done comes after a median 4 acting steps (R1) and a right one
after 3 — the step count does not separate them. The 25 answer-only tasks score 0 for both (the model never emits an
answer), a ceiling of 78.4%.

## R1 seed 1 (`v1_s1`) vs B0′ — pre-registered, second of three seeds

`paired_sr_s1.md` (frozen `paired_sr.py`): primary 116 — R1 s1 **14.8% (17/115)** [9.4, 22.4] vs B0′ 7.0% (8/115);
**w = 12, l = 3**, sign test p = 0.035; 104-task sensitivity 15.5% vs 5.8%, w = 12, l = 2, p = 0.013; A1 |w−l| = 9 / 10.
Every one of the 11 preflight health lines of the run names `v1_s1` (audit OK).

| | R1 s0 | R1 s1 | B0′ |
|---|---|---|---|
| official SR (116) | 12.2% (14/115) | 14.8% (17/115) | 7.0% (8/115) |
| w / l vs B0′ | 9 / 3 | 12 / 3 | — |
| successes on the 7 floor tasks + beyond them (post-hoc) | 7 + 7 | 7 + **10** | 1 + 7 |
| first-encounter repeat (A2-bis) | 41.2% [32.8, 50.2] | 36.4% [30.3, 44.0] | 65.8% [59.3, 73.3] |
| typing share of decisions (A2) | 2.9% | **3.4%** | 2.0% |
| declared done: right / wrong (task-completion) | 14 / 47 | 17 / 50 | 8 / 14 |

Both seeds so far move SR the same way. **No claim yet:** §4 needs all three seeds and a pooled sign test; pooling the
two seeds' counts would be 21 w / 6 l, but seed 2 is still running and the rule is pinned to three. Typing: seed 0 was
within the 1-point band, seed 1 is +1.4 points above B0′ — the pre-registered "raises typing" prediction is mixed, and
the absolute rate stays low. **The done-step confidence signal did not replicate:** AUROC(confidence at done → right)
0.782 [0.663, 0.889] on seed 0 but 0.645 [0.471, 0.818] on seed 1 (CI includes 0.5) — no threshold may be fitted on it.
