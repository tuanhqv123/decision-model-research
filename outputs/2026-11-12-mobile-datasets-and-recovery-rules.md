# Other mobile datasets and how to teach "notice the mistake, go back" (2026-11-12)

Source: a read-only research agent. It read the released annotation files of GUI-Odyssey and AMEX and computed the counts below.
Labels: [V] verified in the source or computed from the file; [I] inferred.

## Datasets

| Dataset | Size and episode length | Element tree | Licence | long_press / back | Overlap with AndroidWorld | Verdict |
|---|---|---|---|---|---|---|
| GUI-Odyssey (2406.08451) | 7,735 episodes, 118,791 steps, 15.4 steps per episode [V, file] | no (coordinates only) | CC-BY-4.0 | 0.41% / 0.20% [V] | Settings, Chrome, Clock; 8 Simple Calendar episodes (drop them) [V] | use: long episodes, 9x our long_press; trains the vision/box path only |
| AMEX (2407.17490) | 3,046 episodes, 38,709 steps, 12.7 per episode [V, file] | element annotations yes | CC-BY-4.0 | none / 0.35% [V] | Google Tasks (not Tasks.org), Files, Chrome [V] | use: long episodes and grounding; no recovery data |
| AITW | 715k episodes, about 8 steps per episode | OCR and icons, no tree | not stated | none / rare | GoogleApps slice close to some AW tasks [I] | maybe: a small filtered slice |
| AndroidLab Instruct (2410.24024) | 726 training episodes | XML | not stated | in the action space | Clock, Contacts, Settings [V] | no |
| OpenMobile (2604.15093), OS-Genesis (2412.19723) | collected INSIDE the AndroidWorld emulator (Simple Calendar Pro, Broccoli, Markor) [V] | | | | yes | **no: learning the test**; use their methods only |

**Back is rarer in every external set than in AndroidControl (3.2%).** Adding them would not teach going back; that behaviour has to be built.

## Rules for building recovery data without over-backing

1. **Failure share about 30%, not 50%.** VeriGUI (2604.05477, Tab. 5), success:failure mix:

   | Mix | Recovery | Loop rate | Step accuracy |
   |---|---|---|---|
   | 100:0 | 29.7% | | |
   | 70:30 | 45.5% | 26.5 (best) | |
   | 50:50 | | | 42.6 → 39.2 |
   | 90:10 | almost no gain | | |

   [V]
2. **Make the back label depend on visible evidence, and use twins.**
   - GUI-Reflection (2506.08012, §2.4) rewrites the goal so the recorded action becomes a mistake, labelled back plus a corrected action. Separately, an ineffective action is labelled with a *different* action, not back. Effect: 14.6 → 23.6 on its own environment [V].
   - BacktrackAgent (2505.20660, §4.2) keeps 20% correct actions next to 100% wrong ones. Even so, it flags 2.81% of correct actions (Tab. 6) [V].
3. **Errors go in the history, never in the targets, and recoveries per episode are capped.** OpenMobile (§4.1, App. A.4): at most 2 interventions per episode, each followed by 3 or more expert steps [V].

## Where v5 stands against these rules

- Errors sit in the history and gold is back: rule 3 holds.
- Matched twins: rule 2 holds.
- No-effect rows are labelled with another action (nc_*): rule 2 holds.
- Dose is 9.3% recovery + 9.3% twins, below the 30% optimum (rule 1). This is deliberate after pilot 1, which ran 30% without twins and reached 10% back misuse.

## AMEX: README and paper read against the files (2026-11-12)

Sources: the HF README, the arXiv 2407.17490 HTML, `AMEX/instruction_anno.zip` (parsed), and the zip directory of `AMEX/screenshot.zip` (read by HTTP range request).

- **Paper vs file: back.** Paper §A.1.5 says annotators were allowed to make mistakes, "leading to a greater prevalence of PRESS_BACK actions". The file has **135 PRESS_BACK in 38,709 steps (0.35%)**. AMEX is not a source of recovery data (M10: the file wins).
- **Where the paper's AndroidControl gain came from.** It comes from **Level 1 + Level 2** only: interactive-element boxes plus functionality text. Level 2 is GPT-4o/Gemini output, checked by humans. Level 3 trajectories were *not* used for AndroidControl, "since the action spaces are different". Gain: High 60.2 → 63.7 overall, click & long-press 55.4 → 62.8 (README Table 4). Level 3 transferred poorly even to AitW (paper Tables 5–6).
- **Actions in the file:** TAP 24,815 · SWIPE 7,628 · TASK_COMPLETE 2,828 · TYPE 2,419 · PRESS_ENTER 651 · TASK_IMPOSSIBLE 220 · PRESS_BACK 135 · PRESS_HOME 13. Every step has `touch_coord`/`lift_coord`, `package_name` and `type_text`. Every step also has an `interest_region`, which is a real box only on information-query completions; in the sample it is the placeholder [[0,0],[0,0]].
- **Elements:** each has `bbox`, `xml_desc` and `functionality`, split into clickable and scrollable. There is **no class name, no checked/editable flag and no resource id**, so contract v5's element rendering (class, on/off, `[id: …]`) cannot be reproduced. Rows would differ in format from the rows we serve.
- **App opening:** annotators go through the app drawer (paper §5), not a direct launch. The launcher prefix conflicts with our first-step `open_app` rule.
- **Screenshots:** a 9-part split zip, 8 × 10.74 GB + 7.58 GB = 93.5 GB. 104,675 PNGs (deflate), about 1 MB each at 1440×2960. **No full download is needed.** The zip64 central directory (13.6 MB, at offset 7,568,360,278 of the last part) gives each file's part number and offset, and an HTTP range request fetches just that image. Tested end to end: one sample screenshot fetched and decoded (1,088,685 bytes, PNG).
