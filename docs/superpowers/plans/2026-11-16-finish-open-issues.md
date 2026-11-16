# Finish the open issues (DMA-2B paper + DMAgent on mobile-use)

**Goal:** close every open issue from the 2026-11-16 review so that each published number is measured on the released model and the mobile agent is complete, tested on held-out tasks, and documented in English.

**Execution:** run in this session, in the order below. Long jobs run in the background, and only their result is reported.

## Global constraints
- Everything in English: docs, reports, commits, and the presentation.
- No private infrastructure in any file (privacy scan before every push). Push through the `tuan` worktree to `tuan` and `main`.
- Every number comes from an artifact on disk, with its denominator. Derived numbers are labelled as derived.
- Shared GPUs: cap memory, keep at least 3 GB free, and never touch other people's processes.

## Tasks

| # | Issue | Work | Done when |
|---|---|---|---|
| 1 | The paper's latency (135 ms) was measured on an older checkpoint. The agent logs ~270 ms with long element lists | Benchmark the released DMA-2B through the HTTP API on AndroidControl-High test inputs (500 steps): server-side model time and total request time, median and p95, plus input length, with GPU occupancy recorded. Also report latency against element-list length | Paper, README, HF card and presentation all quote the new numbers, with the setup stated |
| 2 | The AndroidControl-Curated scorer is a re-implementation that was never differential-tested against the authors' code | Install the authors' evaluation code in a clean venv with its real dependencies, score our predictions with it, and diff row by row against `score_curated.py` | 0 mismatches, or each mismatch explained and fixed |
| 3 | Sensitive-action gate only matches English words | Add Vietnamese words, matched with and without diacritics (xóa/xoá, gửi, thanh toán, chuyển tiền, mua, gọi, gỡ cài đặt, đặt lại, đăng xuất, chia sẻ). Write the test first | New tests pass, and the 17 old tests still pass |
| 4 | Weak scenarios: note_create never declares done; dnd_on hits 73 BLOCKED | Read both traces, find the first wrong step, fix the cause, add a regression test | Both pass with the agent declaring done, and the BLOCKED count falls |
| 5 | Results come from the 12 tasks used for tuning | Add about 8 new held-out scenarios (setup + ADB check). Run all 20 scenarios 3 times each and report the held-out set separately | Table with per-task success out of 3, plus held-out and tuned completion rates with Wilson intervals |
| 6 | Real phones: app list is hand-written; non-ASCII typing is untested | Script `make_app_list.py` builds the app list from any device (launchable packages and labels). Typing goes through ADBKeyboard when text is non-ASCII, with an ADB fallback. Test on the emulator with Vietnamese text | Vietnamese text typed into a Markor note and checked over ADB |
| 7 | Dashboard shows a still image every 1.5 s | MJPEG stream endpoint (about 3–5 fps). Click-to-tap on the stream while no task is running | Live stream visible; a manual tap lands |
| 8 | Quickstart was never run from a clean machine | Fresh clone from GitHub `main` into an empty directory and a fresh venv; follow the guide verbatim; fix every gap found | One task completes by following only the guide |
| 9 | Docs are in Vietnamese | Translate `dma_agent_vi.md` and `dma_quickstart_vi.md` into `dma_agent.md` and `dma_quickstart.md`, and the presentation into English. Update the links | No Vietnamese left in deliverables |
| 10 | Only one training seed | Train seeds 1 and 2 of the final recipe (about 12 h each on one 24 GB GPU, run in parallel when two GPUs are free). Score AC-Low, AC-High and Curated | Mean ± sd across 3 seeds in the paper |
| 11 | AGENTS.md program log is missing everything since 2026-11-13 | Add dated entries: final model, AC/Curated results, mobile agent, open limits | Gates green |
| 12 | Shared resources still running | Stop the emulator container, the DMA-2B API, the relay and the tunnels | Nothing of ours is left running |

## Needs the user (not done by the agent)
- **Next training direction.** Options: each step in both AC forms, or drop the "done" row. Only seed runs (task 10) are planned until the user chooses.
- **Rotate the LLM API key** that was pasted into the chat.
- **Test on a physical phone:** plug one in. The checklist is in the quickstart.

## Risks
- Task 2: if the authors' code disagrees with ours, the Curated numbers in the paper change. Report the result either way.
- Task 5: held-out success may come in lower than 12/12. Report it as measured.
- Task 10 uses about 24 GPU-hours. If no two GPUs are free, the seeds run one after the other.

## Status (2026-11-16, end of day)

| # | Status |
|---|---|
| 1 | Done: 123 ms median on an idle RTX 4090, 255 ms on a shared one (paper, card, presentation, README) |
| 2 | Blocked: running the authors' scorer was refused by the tool permission check. `benchmarks/train/diff_curated_official.py` is ready for the operator to run |
| 3 | Done (tests) |
| 4 | Done: dnd_on 26 → 6 steps, note_create finishes by itself; typing, keyboard and shade fixes |
| 5 | Done: 20 tasks × 3, held-out 24/24 (95% CI 86.2–100%), tuned 36/36 |
| 6 | Done in code and on the emulator; a physical phone needs the operator |
| 7 | Done: about 2 frames/s, click-to-tap, Back/Home |
| 8 | Done for every client step from a fresh GitHub download; the API step was checked by equivalence (same code and files), not by a fresh start |
| 9 | Done: agent docs, presentation and 26 research notes in English (numbers checked digit by digit) |
| 10 | Running: seeds 1 and 2 on the training host (`run_seeds_exec2.sh`), each followed by AC-Low/High evaluation (`ev_seed.sh`) |
| 11 | Done |
| 12 | Done: emulator stopped, API, relay and tunnels stopped, LLM key files deleted |

Also found and fixed on the way: private names in five previously published documents; the scanner's false positives.
