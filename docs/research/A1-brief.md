# A1 — Is GUI-Odyssey actually usable for OUR failure? (research brief)

Read `docs/research/ISSUES-2026-11-09.md` first (issue I2). Then execute this and write
`docs/research/A1-gui-odyssey-usability.md`. Do not edit any other file.

## Our measured situation (do not re-derive, do not dispute)

- 1.92B single-pass decision model, 8 typed readout slots: `click, long_press, scroll, type_text, open_app,
  navigate_back, wait, status`. Zero output tokens. Emits action + (for spatial types) a box on a 0-999 grid.
- AndroidWorld-116: **18.8% (15/80)** scorable. Decisive observation: on the 80 scorable episodes, **every
  episode in which the model emitted `type_text` failed** (14 episodes, 1 success, median 25 steps = budget).
  Episodes that never typed: 14/66 success, median 12 steps.
- `type_text` was **24 of 1,394 emitted actions (1.7%)**; it is **3.0% of our training labels** (450/15,000).
- `ContactsAddContact` trace: types once (conf 0.829) then never returns to the form — 16 more clicks.

## The claim under test

"Training on GUI-Odyssey trajectories will fix this." The dataset card (read 2026-11-09) already shows a
problem: for action `TYPE`, `info` is **empty** — no coordinates, no field identity. If that is true, the
corpus cannot supervise "which field do I type into", which is our exact gap.

## What to determine, with evidence

1. **The actual annotation schema, from the data not the card.** Read `all_anno.json` (33.7 MB) and a handful of
   `annotations/<episode_id>.json` from `OpenGVLab/GUI-Odyssey`, and — importantly — check the newer
   `hflqf88888/GUIOdyssey` the card recommends. Use `hf_dataset_info`, `hf_repo_files`, `hf_repo_read_file`,
   and `curl`/`bash` against `https://huggingface.co/datasets/OpenGVLab/GUI-Odyssey/resolve/main/...` if needed.
   Report the **literal JSON keys** for each action type, especially `TYPE`.
   - Does `TYPE` ever carry a target box or element index? In either version?
   - Does `ps`/`info` carry the typed string?
   - Are screenshots addressable (`screenshots/` path convention) and how many exist?
2. **Count the usable rows.** Over the whole corpus, count actions per type (CLICK/SCROLL/LONG_PRESS/TYPE/
   COMPLETE/IMPOSSIBLE/HOME/BACK). Report the total step count and the per-type percentage. This determines
   whether trajectory structure is even the majority of the corpus.
3. **Reconcile the numbers.** Card says 7,735 episodes; paper (arXiv 2406.08451) says 8,334; `size_categories`
   says 1K–10K. State the measured episode count in each version and which number is correct for what.
4. **Map to AndroidWorld's required capability.** AndroidWorld-116 tasks need: app launch, tapping, text entry
   into fields, scrolling, system buttons, and state verification. For each of our 8 readouts, state whether
   GUI-Odyssey provides supervision, and if only partly, exactly what is missing. Be explicit about `open_app`
   (the corpus has no explicit app-launch action — is it implied by a HOME→icon click?) and about scroll
   (theirs is a swipe vector; ours is 4-way direction).
5. **What the published users of GUI-Odyssey report.** GUI-Odyssey's own baselines, GUI-R1 (arXiv 2504.10458),
   and any other paper that trains on it: what did they report, on what metric, and **did any of them report
   end-to-end task success** rather than step accuracy? Give numbers + URLs.
6. **Alternatives, one table.** If GUI-Odyssey's TYPE gap is fatal, what else provides *localised* text-entry
   supervision on Android, is standard/famous, and licensed for use? Candidates to check (do not limit yourself
   to these): AndroidControl (the training split we already use — does it localise type_text?), AMEX,
   AndroidLab, AITW, GUI-Odyssey-latest, Mobile-Agent-v2 data, UI-Venus/MobileRL released data, OS-Atlas mobile
   subset, ShowUI/UGround mobile data. For each: size, licence, does it localise TYPE, HF/GitHub URL.

## Output format

`docs/research/A1-gui-odyssey-usability.md` with:
- A **verdict line first**: can GUI-Odyssey supervise our typing gap? yes / no / partly + one sentence why.
- The literal schema table (action → keys present → usable for us).
- A counts table over the real corpus.
- The mapping table (our 8 readouts × available: full / partial / none / needs derivation).
- The alternatives table.
- An explicit "what I could not verify" section.

Cite every number to a URL or to a file path + line you actually read. If you could not download something,
say so and do not guess. Note: `huggingface_hub` may be installed; `HF_TOKEN` may or may not be set.
