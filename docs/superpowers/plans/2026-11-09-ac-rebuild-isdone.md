# AndroidControl Rebuild + `is_done` Readout — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the training corpus from the raw 89k AndroidControl rows so that episodes keep their step sequence, add a `noul("task complete?")` readout trained on the free structural label, and drop the 311 unscoreable rows — so the model can learn "not done yet" instead of only "done".

**Architecture:** Three stages, each independently testable. (1) A pure-stdlib rebuild script turns `records/*.jsonl` into episode-sampled `compact_*` files plus a drop-ledger. (2) `ac_common.py` gains an `is_done` question and its `noul` target, derived from the terminal-`status` invariant. (3) `train_d1.py` gains a `--task-complete-weight` loss term and a candidate cap default of 40. The existing forward pass, box loss and pointer reads are untouched.

**Tech Stack:** Python 3.11 (stdlib only for the rebuild), PyTorch + the d1-2B remote-code modules already vendored by `train_d1.py`, `pytest` for tests.

**Spec:** `docs/research/DATA-SPEC-2026-11-09.md` — every number in this plan comes from that file, and that file names the script that produced it.

## Global Constraints

- **Drop total is 311 rows (0.35%)**, itemised in the spec §6. Never drop anything not on that list.
- **Candidate filter is production `pick_candidates`** (`benchmarks/train/ac_common.py:57`). It covers the gold point **99.30%** of the time. Do not write a new filter.
- **Candidate cap must be ≥40** (median candidate count is 37). Any run under a lower cap is a lower bound and must be labelled one.
- **`previous_actions` is TEXT only.** Never put screenshots in history: 5 of 24 layers have a 256-token sliding window.
- **`is_clickable` must never appear in candidate text.** The rule "pick the clickable node nearest gold" scores 61.9% without the image, versus the model's 64.7%.
- **Hold out validation by episode id, never by row.** The 137 validation episodes are disjoint from train on both episode and image path (measured: 0 overlap).
- **Never report a per-step number as a task-success number.** They differ ~8×.
- Redact hostnames, IPs and usernames from anything written to disk.

## Review Focus

These are the failure modes the spec implies but whose tests no task would otherwise force. Each gets a test in the task that owns the code.

1. **An episode split across a row-budget boundary.** If the sampler truncates mid-episode, the rebuilt corpus silently regresses to median-1-step and nothing fails. → Task 1, `test_sampler_never_truncates_an_episode`.
2. **`navigate_home` disappearing again.** It is present in raw (29 rows) and absent from every `compact_*` derivative; a naive action-space rewrite drops it a second time. → Task 2, `test_action_vocab_includes_navigate_home`.
3. **`is_done` leaking the answer through `action`.** `status` and `is_done=YES` are the same 13,740 rows; if both are supervised the model can read one off the other and the AUROC is meaningless. → Task 3, `test_is_done_is_not_redundant_with_action`, which asserts the joint distribution and records the redundancy.
4. **A cap silently below the candidate median.** This is the exact defect that produced the false-negative VA result. → Task 4, `test_cap_below_median_candidates_is_refused`.
5. **The 130 non-copyable `type_text` rows being dropped again.** Current code does `del g["type_text"]`. → Task 2, `test_non_copyable_text_is_labelled_escalate_not_dropped`.

---

### Task 1: Episode-preserving corpus rebuild

**Files:**
- Create: `benchmarks/train/rebuild_ac_episodic.py`
- Test: `tests/test_rebuild_ac_episodic.py`

**Interfaces:**
- Consumes: nothing from earlier tasks. Reads `records/*.jsonl` whose rows have keys `ep, step, n_steps, split, goal, step_instruction, img, W, H, action, target, nodes`.
- Produces:
  - `load_episodes(root: str) -> dict[tuple[str, int], list[dict]]` — keyed `(split, ep)`, each list sorted by `step`.
  - `is_dropped(rec: dict) -> str | None` — returns the drop reason or `None`.
  - `sample_episodes(episodes, rows_budget: int, seed: int) -> list[tuple]` — whole-episode sampling.
  - `build_compact(rec: dict) -> dict` — one output row with keys `ep, step, n_steps, split, goal, history, img, W, H, action, target, cands`.
  - CLI: `--root --out --rows-budget --seed --drop-ledger`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_rebuild_ac_episodic.py
import json
from benchmarks.train.rebuild_ac_episodic import (
    is_dropped, sample_episodes, build_compact, load_episodes,
)

def _rec(**kw):
    base = dict(ep=1, step=0, n_steps=3, split="train", goal="g",
                step_instruction="s", img="imgs/1/0.jpg", W=1080, H=2400,
                action={"action_type": "wait"}, target=None, nodes=[])
    base.update(kw)
    return base

def test_drop_inverted_bbox():
    b = dict(x_min=10, x_max=5, y_min=0, y_max=10, area=0, center=[7, 5])
    assert is_dropped(_rec(target=b)) == "inverted_or_zero_area"

def test_drop_out_of_bounds():
    b = dict(x_min=0, x_max=2000, y_min=0, y_max=100, area=200000, center=[1000, 50])
    assert is_dropped(_rec(target=b)) == "out_of_bounds"

def test_drop_click_without_target():
    assert is_dropped(_rec(action={"action_type": "click"})) == "click_without_target"

def test_keep_valid_click():
    b = dict(x_min=0, x_max=10, y_min=0, y_max=10, area=100, center=[5, 5])
    assert is_dropped(_rec(action={"action_type": "click"}, target=b)) is None

def test_sampler_never_truncates_an_episode():
    # 3 episodes of 5 rows each; budget 7 must yield ONE whole episode, not 7 rows
    eps = {("train", i): [_rec(ep=i, step=s, n_steps=4) for s in range(5)]
           for i in range(3)}
    picked = sample_episodes(eps, rows_budget=7, seed=0)
    assert sum(len(eps[k]) for k in picked) <= 7
    assert all(len(eps[k]) == 5 for k in picked)      # never partial

def test_build_compact_emits_history_as_text():
    rec = _rec(step=2, n_steps=4)
    out = build_compact(rec, prev=[_rec(step=0), _rec(step=1)])
    assert isinstance(out["history"], list)
    assert out["history"] and all(isinstance(h, str) for h in out["history"])
    assert out["history"][0].startswith("Step 1:")
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `pytest tests/test_rebuild_ac_episodic.py -v`
Expected: FAIL with `ModuleNotFoundError: benchmarks.train.rebuild_ac_episodic`

- [ ] **Step 3: Implement `benchmarks/train/rebuild_ac_episodic.py`**

Signatures exactly as in **Interfaces**. Rules the implementation must follow:

- `is_dropped` returns, in this order: `"inverted_or_zero_area"` when `target` exists and
  `x_min >= x_max or y_min >= y_max or area <= 0`; `"out_of_bounds"` when `target` exists and any edge falls
  outside `[0, W] × [0, H]`; `"click_without_target"` when `action_type == "click"` and `target is None`;
  `"type_text_without_target"` when `action_type == "type_text"` and `target is None`. Otherwise `None`.
- `sample_episodes` shuffles keys with `random.Random(seed)`, accumulates **whole** episodes while
  `total + len(ep) <= rows_budget`, skips (does not truncate) an episode that would overflow, and continues
  scanning later episodes. It returns keys in the shuffled order taken.
- `build_compact` renders history from `prev` as `f"Step {i}: {action_type} {label}"`, `i` starting at 1,
  where `label` is the action's `app_name` or `text` or `""`. It copies `cands` through untouched when the
  input row already carries them; otherwise it leaves `cands` absent (the trainer recomputes from `nodes`).
- The CLI writes a drop ledger `{"<split>/<ep>/<step>": "<reason>"}` to `--drop-ledger` and prints
  `ROWS_IN=… ROWS_OUT=… DROPPED=… EPISODES=… MEDIAN_STEPS=…` as its last line.

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `pytest tests/test_rebuild_ac_episodic.py -v`
Expected: 6 passed

- [ ] **Step 5: Run the rebuild on the real corpus and check it against the spec's numbers**

```bash
python3 benchmarks/train/rebuild_ac_episodic.py \
  --root backup-2026-10-29/GPU_HOST_C/acwork/acdata \
  --out /tmp/acrebuild --rows-budget 39000 --seed 0 \
  --drop-ledger /tmp/acrebuild/drop_ledger.json
```
Expected (tolerance ±2%): `ROWS_OUT` ≈ 39,000, `DROPPED` ≈ 311 scaled to the sampled episodes,
`MEDIAN_STEPS` = **6** not 1. If `MEDIAN_STEPS` is 1, the sampler is truncating — go back to Step 3.

- [ ] **Step 6: Commit**

```bash
git add benchmarks/train/rebuild_ac_episodic.py tests/test_rebuild_ac_episodic.py
git commit -m "feat: episode-preserving AndroidControl rebuild with a drop ledger"
```

---

### Task 2: `is_done` and `escalate` labels in the shared row builder

**Files:**
- Modify: `benchmarks/train/ac_common.py:125-180` (`build_state`, `build_questions`) and `:306-340` (`gold_answers`)
- Test: `tests/test_ac_common_isdone.py`

**Interfaces:**
- Consumes: `build_state`, `build_questions`, `gold_answers`, `find_span` from Task 1's sibling module (unchanged signatures).
- Produces:
  - `is_done_gold(is_terminal: bool) -> str` — `"yes"` when `is_terminal`, else `"no"`.
  - `build_questions(track, cands=None)` now also returns key `"is_done"` with
    `{"type": "noul", "instructions": "Has the task already been completed on this screen?"}`.
  - `gold_answers(...)` now also returns `g["is_done"]` and, for non-copyable `type_text`, `g["type_text"] = "escalate"`.
  - `ACTION_VOCAB: tuple[str, ...]` — the 9 action names including `navigate_home`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_ac_common_isdone.py
import pytest
from benchmarks.train.ac_common import (
    ACTION_VOCAB, build_questions, gold_answers, is_done_gold, build_state,
)

def test_action_vocab_includes_navigate_home():
    assert "navigate_home" in ACTION_VOCAB
    assert len(ACTION_VOCAB) == 9

def test_is_done_gold_is_binary():
    assert is_done_gold(True) == "yes"
    assert is_done_gold(False) == "no"

def test_questions_include_is_done():
    q = build_questions("V")
    assert q["is_done"]["type"] == "noul"

def test_is_done_is_not_redundant_with_action():
    """status rows and is_done=yes are the same rows; record the redundancy explicitly."""
    import json
    from pathlib import Path
    p = Path("backup-2026-10-29/GPU_HOST_C/acwork/acdata/records")
    if not p.exists():
        pytest.skip("raw corpus not present")
    n_status = n_yes = n_both = 0
    for f in sorted(p.glob("*.jsonl"))[:1]:
        for line in f.open():
            r = json.loads(line)
            st = (r.get("action") or {}).get("action_type") == "status"
            yes = r["step"] == r["n_steps"]          # terminal marker row
            n_status += st
            n_yes += yes
            n_both += (st and yes)
    assert n_status == n_yes, "status rows and terminal rows must be the same set"
    assert n_both == n_status
    # Redundancy is real and must be handled by the loss, not ignored: see Task 3.

def test_non_copyable_text_is_labelled_escalate_not_dropped():
    state = "Goal: write something original"
    rec = {"action": {"action_type": "type_text", "text": "a sentence not in state"}}
    cands = [{"idx": 1, "label": "box", "pos": "x:0-1% y:0-1%",
              "bbox": {"x_min": 0, "x_max": 9, "y_min": 0, "y_max": 9,
                       "width": 9, "height": 9, "area": 81, "center": [4, 4]}}]
    g = gold_answers(rec, "V", cands, state)
    assert g.get("type_text") == "escalate", "must be labelled, not deleted"
```

- [ ] **Step 2: Run and confirm they fail**

Run: `pytest tests/test_ac_common_isdone.py -v`
Expected: FAIL — `ImportError: cannot import name 'ACTION_VOCAB'`

- [ ] **Step 3: Implement in `benchmarks/train/ac_common.py`**

- Add module-level `ACTION_VOCAB = ("click", "write", "long_press", "scroll", "wait", "navigate_back", "open_app", "terminate", "navigate_home")`. Extend `TO_OPTION`/`FROM_OPTION` with `"navigate_home"`.
- Add `def is_done_gold(is_terminal: bool) -> str: return "yes" if is_terminal else "no"`.
- In `build_questions`, add the `is_done` entry exactly as specified in **Interfaces**.
- In `gold_answers`, set `g["is_done"] = is_done_gold(rec["step"] == rec.get("n_steps"))`, and replace the
  `del g["type_text"]` branch with `g["type_text"] = "escalate"` when the action is `type_text` and the span
  is `None`. Keep the existing `del g["app_name"]` behaviour unchanged.

- [ ] **Step 4: Run and confirm they pass**

Run: `pytest tests/test_ac_common_isdone.py -v`
Expected: 5 passed (the redundancy test skips if the corpus is absent)

- [ ] **Step 5: Commit**

```bash
git add benchmarks/train/ac_common.py tests/test_ac_common_isdone.py
git commit -m "feat: add is_done noul slot and escalate label to the shared row builder"
```

---

### Task 3: `is_done` loss term with explicit de-redundancy

**Files:**
- Modify: `benchmarks/train/train_d1.py:34` (`W_LOSS`), `:215-260` (forward), `:365-390` (argparse)
- Test: `tests/test_train_task_complete.py`

**Interfaces:**
- Consumes: `ACTION_VOCAB`, `gold_answers()["is_done"]` from Task 2; the d1 slot modules `M["slots"]`, `M["prompt"]` as used by the existing `choice` path.
- Produces: CLI flag `--task-complete-weight` (float, default `0.5`); `W_LOSS["is_done"]`; refusal `SystemExit` when `D1_MAX_CANDS` < 40 and `--track` is `VA` or `mix`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_train_task_complete.py
import pytest
from benchmarks.train.train_d1 import check_cand_cap, W_LOSS

def test_cap_below_median_candidates_is_refused():
    import os
    os.environ["D1_MAX_CANDS"] = "14"
    with pytest.raises(SystemExit, match="40"):
        check_cand_cap(track="VA")

def test_cap_at_or_above_40_is_allowed():
    import os
    os.environ["D1_MAX_CANDS"] = "40"
    check_cand_cap(track="VA")      # must not raise

def test_v_track_ignores_cap():
    import os
    os.environ["D1_MAX_CANDS"] = "14"
    check_cand_cap(track="V")       # V has no candidates, must not raise

def test_loss_table_has_is_done():
    assert "is_done" in W_LOSS
    assert 0.0 < W_LOSS["is_done"] < 1.0
```

- [ ] **Step 2: Run and confirm they fail**

Run: `pytest tests/test_train_task_complete.py -v`
Expected: FAIL — `ImportError: cannot import name 'check_cand_cap'`

- [ ] **Step 3: Implement**

- Add `def check_cand_cap(track: str) -> None` in `train_d1.py`: if `track in ("VA", "mix")` and
  `int(os.environ.get("D1_MAX_CANDS", "0") or 0) < 40`, `raise SystemExit("D1_MAX_CANDS must be >= 40 ...")`
  with a message naming the 58.6% discarded-candidate defect and pointing at `DATA-SPEC-2026-11-09.md` §8.
  Call it once from `main()` before the training loop starts.
- Add `"is_done": 0.5` to `W_LOSS`, overridden by `--task-complete-weight`.
- In `forward`, where `choice`/`noul` slots are scored, add the `is_done` slot through the same
  max-pooled-form CE path used by the existing `action` slot. Do **not** add it to the box or read paths.
- Add the argparse flag with default `0.5` and a help string noting it is the `noul` weight.

- [ ] **Step 4: Run and confirm they pass**

Run: `pytest tests/test_train_task_complete.py -v`
Expected: 4 passed

- [ ] **Step 5: Smoke-run the trainer and check the new loss appears**

```bash
D1_MAX_CANDS=40 python3 benchmarks/train/train_d1.py \
  --data /tmp/acrebuild --out /tmp/run-smoke --steps 30 --limit 400 \
  --track mix --micro 1 --accum 16 --grad-ckpt --mem-frac 0.38 \
  --task-complete-weight 0.5
```
Expected: the 30-step log shows a `is_done` component in the loss breakdown, loss decreases from the
first to the last step, and the process prints `TRAIN_DONE`.

- [ ] **Step 6: Commit**

```bash
git add benchmarks/train/train_d1.py tests/test_train_task_complete.py
git commit -m "feat: is_done noul loss term and a candidate-cap guard"
```

---

### Task 4: `is_done` evaluation on held-out episodes

**Files:**
- Create: `benchmarks/train/eval_isdone.py`
- Test: `tests/test_eval_isdone.py`

**Interfaces:**
- Consumes: a trained checkpoint directory; `gold_answers` and `build_questions` from Task 2.
- Produces:
  - `auroc(scores: list[float], labels: list[int]) -> float` — rank-based, no sklearn.
  - `risk_coverage(scores, labels, coverage: float) -> float` — error rate among the top-`coverage` share by score.
  - CLI: `--ckpt --data --out` writing `{"auroc": float, "coverage": float, "risk": float, "n": int, "n_pos": int}`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_eval_isdone.py
import pytest
from benchmarks.train.eval_isdone import auroc, risk_coverage

def test_auroc_perfect_separation():
    assert auroc([0.9, 0.8, 0.1, 0.2], [1, 1, 0, 0]) == 1.0

def test_auroc_chance_is_half():
    assert abs(auroc([0.5, 0.5, 0.5, 0.5], [1, 0, 1, 0]) - 0.5) < 1e-9

def test_auroc_inverted_is_zero():
    assert auroc([0.1, 0.2, 0.9, 0.8], [1, 1, 0, 0]) == 0.0

def test_risk_coverage_at_full_coverage_is_base_rate():
    labels = [1, 0, 1, 0]
    assert risk_coverage([0.9, 0.4, 0.3, 0.1], labels, 1.0) == pytest.approx(0.5)

def test_risk_coverage_at_half_keeps_the_half_most_confident():
    labels = [1, 1, 0, 0]
    assert risk_coverage([0.9, 0.8, 0.2, 0.1], labels, 0.5) == 0.0
```

- [ ] **Step 2: Run and confirm they fail**

Run: `pytest tests/test_eval_isdone.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement `benchmarks/train/eval_isdone.py`**

`auroc` is the Mann–Whitney statistic with ties at 0.5. `risk_coverage` sorts descending by score, keeps the
first `ceil(coverage * n)`, and returns the mean of `1 - label` over those. The CLI reuses the trainer's
`make_sample` for forward passes, so the evaluated function is the trained one — assert that in a comment.

- [ ] **Step 4: Run and confirm they pass**

Run: `pytest tests/test_eval_isdone.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add benchmarks/train/eval_isdone.py tests/test_eval_isdone.py
git commit -m "feat: is_done AUROC and risk-coverage evaluation"
```

---

### Task 5: Pre-registration and the three-arm run sheet

**Files:**
- Create: `docs/research/PRE-REG-2026-11-09-episodic-isdone.md`
- Test: none (a document); verified by `scripts/check_record.py` if it gains a check.

**Interfaces:**
- Consumes: the spec's §9 arm table and §4 copyability numbers.
- Produces: a frozen pre-registration naming the arms, the metric, the seeds and the withdrawal rule.

- [ ] **Step 1: Write the pre-registration**

It must contain, in this order: (a) the claim — *episode-preserving sampling plus an `is_done` readout
raises `is_done` AUROC and `target_el` accuracy on held-out episodes, without lowering AC-v2 per-step
accuracy*; (b) the three arms from spec §9 with their row budgets; (c) the metrics and their units —
`is_done` AUROC (unit: row), `target_el` accuracy (unit: spatial row), AC-v2 per-step accuracy
(unit: of 398 steps, **explicitly not task success**); (d) seeds `0,1,2` and the per-seed tables;
(e) the **withdrawal rule**: if the seed spread of any metric exceeds the effect size, the result is
reported as inconclusive; (f) an explicit line that AndroidWorld-116 task success is **not** a metric of
this study, with the reason (MDE ≈ 14 points, `A5-parts/Q4-power-analysis.md`).

- [ ] **Step 2: Verify the numbers in the document re-derive**

Run: `python3 scripts/audit_ac_raw.py --stage 1 | tail -12` and
`python3 scripts/audit_ac_raw2.py --stage 2 | tail -12`
Expected: the figures quoted in the pre-registration match these two outputs.

- [ ] **Step 3: Commit**

```bash
git add docs/research/PRE-REG-2026-11-09-episodic-isdone.md
git commit -m "docs: pre-register the episodic rebuild and is_done readout study"
```

---

## Self-Review

**1. Spec coverage.** §1 integrity → Task 1 Step 5 asserts the output numbers. §2 label integrity + the
`is_done` derivation → Task 2. §3 candidate filter → Global Constraints + Task 3's cap guard. §4 typing
(97.6%/2.38%) → Task 2's `escalate` test. §5 action space / `navigate_home` → Task 2's vocab test and
Review Focus 2. §6 drop list → Task 1's `is_dropped` tests. §7 the training row → Task 2 `build_questions`.
§8 the cap → Task 3's `check_cand_cap`. §9 sampling → Task 1's `sample_episodes`. §10 what is not claimed
→ Task 5's pre-registration (f). No section is unimplemented.

**2. Step scan.** Every step names a file, a command, or an exact assertion. The two steps that could have
been vague — the rebuild CLI and the trainer smoke — carry the exact command and the exact expected token
of output (`MEDIAN_STEPS=6`, `TRAIN_DONE`).

**3. Type consistency.** `is_done_gold(bool) -> str` is used only by `gold_answers` and the tests.
`auroc(scores, labels)` and `risk_coverage(scores, labels, coverage)` keep the same argument order in the
tests and the CLI. `check_cand_cap(track: str) -> None` matches all four tests. `ACTION_VOCAB` is the single
source of the 9 action names; `TO_OPTION` extends it rather than duplicating it.

**4. Review Focus.** Five lines, five tests, each named in the task that owns the code (Tasks 1, 2, 3).
Review Focus 3 is the one that is genuinely awkward: `status` and `is_done=YES` are the same rows, so the
`is_done` AUROC will be inflated by anything the model learns about `action`. The task records the
redundancy in a test and does not pretend it away; the honest fix is to score `is_done` **conditioned on
`action != terminate`**, which the eval CLI must report as a second column. That is stated in Task 4's
implementation note, not left implicit.

**5. Proportion.** The spec is ~190 lines; this plan is ~300 with tests. The ratio is dominated by test
bodies, which are the part an implementer cannot infer.

---

## Execution note

Tasks 1 and 2 are independent and can run in either order. Task 3 requires Task 2. Task 4 requires a
trained checkpoint from Task 3, so its CLI work can proceed in parallel but its verification cannot.
Task 5 is last because it freezes the arms whose code Tasks 1–4 provide.
