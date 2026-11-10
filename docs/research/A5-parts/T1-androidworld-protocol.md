# T1 — AndroidWorld protocol: mechanical extraction

| Requirement | What the source says (verbatim quote, <=25 words) | Where (URL + section/file) |
|---|---|---|
| 1. Total number of tasks in the suite | "116 programmatic tasks across 20 real-world Android apps" / "116 diverse tasks across 20 real-world apps" | https://arxiv.org/abs/2405.14573 — Abstract (v5). https://raw.githubusercontent.com/google-research/android_world/main/README.md — intro paragraph |
| 2. The official success metric definition | "agent_successful = task_successful if interaction_results.done else 0.0" / "Success Rates (SR) on AndroidWorld and MobileMiniWoB++" | https://raw.githubusercontent.com/google-research/android_world/main/android_world/suite_utils.py — `_run_task` (else-branch). https://arxiv.org/html/2405.14573 — §4.1.2, Table 3 caption |
| 3. Whether runs are seeded / seed convention | "We set the seed to 30 and the temperature to 0 to aid reproducibility." / "`_TASK_RANDOM_SEED` ... 'task_random_seed', 30, 'Random seed for task randomness.'" | https://arxiv.org/html/2405.14573 — §4.2 Experimental results. https://raw.githubusercontent.com/google-research/android_world/main/run.py — flag default. Also `_FIXED_SEED = 123` in suite_utils.py |
| 4. Step budget convention (max steps per episode) | "the max_steps/step_budget for each task in AndroidWorld have been updated to approximately 2x the human average completion time" | https://raw.githubusercontent.com/google-research/android_world/main/README.md — "Run the benchmark", "Task Step Limits Update" (dated 11/18/2024). Code: `_allocate_step_budget` returns `int(10 * (task_complexity))`, suite_utils.py |
| 5. Whether a checker runs on device state (and what it checks) | "AndroidWorld provides reward signals primarily by managing application state using the Android Debug Bridge (adb)" / "For tasks where system state inspection is impractical, AndroidWorld validates task completion by examining UI elements on screen." | https://arxiv.org/html/2405.14573 — §3.4 "Durable rewards from system state". Code: `task_successful = task.is_successful(env)` in suite_utils.py `_run_task` |
| 6. Number of runs per task that the paper reports | "Note that for consistency with existing literature we maintain the single-seed results in Table 3." / "Across three seeds, we observe significant performance variations: 27.6%, 26.3% and 33.2% (mean 29.0%)" | https://arxiv.org/html/2405.14573 — §5 Robustness Analysis; Table 3 in §4.1.2. Harness default: `n_task_combinations = 1` ("Number of task instances to run for each task template"), run.py |
| 7. What a published number must disclose (if the source says) | NOT STATED | Checked https://arxiv.org/html/2405.14573 (full text), https://raw.githubusercontent.com/google-research/android_world/main/README.md, https://raw.githubusercontent.com/google-research/android_world/main/docs/tasks_guide.md — no disclosure rule found |
| 8. Whether the environment exposes pixels, accessibility tree, or both | "Pixels: An RGB array representing the current screen capture of the device." / "Accessibility tree: A raw representation of the accessibility tree." / "UI elements: A list of processed UI elements extracted from the children of the accessibility tree." | https://arxiv.org/html/2405.14573 — Appendix C.1 "Observation Space" (lists Pixels, Accessibility tree, UI elements). Also §3.2 "The observation and action space" |
| 9. Number of task families / breadth categories | "Task completion tasks (e.g., send a message or edit a note) are abbreviated as 'TC' and information retrieval tasks are abbreviated as 'IR'." / `ANDROID_WORLD_FAMILY: Final[str] = 'android_world'  # Entire suite.` / `MINIWOB_FAMILY: Final[str] = 'miniwob'` / `INFORMATION_RETRIEVAL_FAMILY: Final[str] = 'information_retrieval'  # Subset.` | https://arxiv.org/html/2405.14573 — Appendix F intro; §D.2 "Task classification and generation"; Fig. 2(c) category tags. https://raw.githubusercontent.com/google-research/android_world/main/android_world/registry.py — `TaskRegistry` constants + `get_families()` (also `ANDROID_FAMILY = 'android'`, `MINIWOB_FAMILY_SUBSET = 'miniwob_subset'`) |
| 10. Whether the paper reports averaged or per-task results | "avg.name = '========= Average ========='" / `result_df.columns = [ 'num_complete_trials', 'mean_success_rate', ...` ] — per-task row plus an Average row | https://raw.githubusercontent.com/google-research/android_world/main/android_world/suite_utils.py — `process_episodes` (docstring example prints one row per `task_template` then `==========Average==========`). https://arxiv.org/html/2405.14573 — Table 3 |

## Version / date checked

- Retrieval date: **2026-11-09**.
- arXiv: **arXiv:2405.14573v5**, last revised 6 Apr 2025 (abstract page states "last revised 6 Apr 2025 (this version, v5)"); full text read at `https://arxiv.org/html/2405.14573`.
- GitHub: branch `main`, commit **55ddaf5edc6f6c25673aed343907d5b9dca66f84** (author date 2026-10-23T23:14:30Z, "Use typed AdbRequest.InstallApk in adb_utils.install_apk").
- Files actually read from the repo (via `raw.githubusercontent.com/google-research/android_world/main/`):
  - `README.md`
  - `docs/tasks_guide.md`
  - `android_world/registry.py`
  - `android_world/suite_utils.py`
  - `android_world/episode_runner.py`
  - `run.py`
- Not fetched / not present: `docs/` contains only `tasks_guide.md` (verified via GitHub contents API). No `TASK_SET` file exists; the task set is defined in `android_world/registry.py` (`TaskRegistry._TASKS`) and `info`/`miniwob` registries.
