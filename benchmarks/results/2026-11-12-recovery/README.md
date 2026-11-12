# Recovery data from AndroidControl alone (2026-11-12)

Operator idea: AndroidControl has screens and episodes, so teach the model to recover from a wrong step from the
dataset itself. No self-play.

## Why (measured on traces)

- On AndroidWorld, v3h and v4 chose `navigate_back` **0 times in 2,835 steps**. Mean P(navigate_back) is 0.008.
- AndroidControl humans use it on **880 of 27,398 steps (3.2%)**, in 613 of 4,197 episodes (14.6%).
- A wrong path inside the app is the first wrong step of 32 of the 96 scored v3h failures.

## Supply

From `compact_awlike_full_v4`:
- 2,136 screens appear in at least 2 episodes with different actions.
- That gives up to about 43k "other action from the same screen → its real next screen" pairs.

## Construction

`benchmarks/train/build_recovery_data.py`, 6 tests. For episode E at screen s, an episode F that acted differently on s supplies the real next screen s'. One row is then:

- goal: E's goal
- history: E's history up to s, plus `<F's action> → screen changed`
- screen: s'
- gold: `navigate_back`

Filters, applied in this order:
1. s is not system UI, the keyboard or the launcher.
2. s' is not on E's own path.
3. s' stays in the same app.
4. s' is a different page: Jaccard of the interactive elements < 0.5.
5. The "wrong" element's label shares no 4-letter word with the goal.

Output: 3,000 rows (cap).

## Label quality (hand audit of 20 random rows)

| | Count | Example |
|---|---|---|
| Back is the right recovery | 10 | "Bluetooth" goal, clicked "Internet" |
| Ambiguous | 6 | |
| Wrong: the "wrong" step was on a valid path | 4 | "identify plants", clicked "Take photo" |

The labels are noisy, so these rows are **not** trained on directly.

## Verify before any full training (blocking)

1. **AW recovery states.** Recorded AW states one step after the first wrong step, where going back is right (Markor note opened instead of deleting, a wrong menu, a wrong event). Measure P(navigate_back) and how often back is top-1. Today it is about 0.
2. **Misuse.** P(navigate_back) on on-path validation states where the gold is not back must not rise. AC-High, validation and false status (S1) must not get worse.
3. **Pilot.** At most 300 steps from the chosen model, once the GPU is free. Full training only if 1 rises clearly and 2 holds.
