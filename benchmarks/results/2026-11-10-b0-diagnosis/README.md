# B0 diagnosis scripts (frozen 2026-11-10 14:55 UTC)

Run on the host that holds the run directory (needs numpy + PIL): `AW_RUN_DIR=<run dir> python export_steps.py`,
then locally `python analyze_b0.py <dir holding b0_export.json and dl/aw>`; `fixedpoint.py` and `imgdiff.py` print the
fixed-point and consecutive-screen statistics directly.

`inspected_tasks.txt` lists the tasks whose B0 traces had been looked at when the diagnosis (PLAN §1, F2–F6) was
written. Every other AndroidWorld task is the **confirmation set**: the same scripts, unchanged, are run on it when B0
completes (pre-registration amendment A2). "Same screen" = <0.2% of pixels differ by >24 grey levels between the
270×600 images the model received on consecutive steps, status bar excluded.
