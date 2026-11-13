"""Generative arm (Q1) evaluation with the engine mocked: rows come from eval_d1.load_rows (val_row shape) and are
scored by run_acv2.score_row with the arguments eval_d1 would pass; parse failures count as wrong, never skipped."""
import json

import pytest

from benchmarks.gen import eval_gen as E
from benchmarks.gen import gen_common as G
from benchmarks.gen import make_gen_data as D
from benchmarks.train.eval_d1 import load_rows
from benchmarks.train.run_acv2 import score_row
from tests.gen_fakes import FIXTURE_ROWS, write_fixture


class ScriptEngine:
    """Returns a fixed text per (goal, step-count) key; records the inputs it was given."""

    def __init__(self, texts):
        self.texts, self.calls = texts, []

    def run(self, goal, history, image):
        self.calls.append((goal, list(history), image.size))
        text = self.texts[len(history)]
        pieces = list(text) + ["<|im_end|>"]
        return {"text": text, "pieces": pieces, "probs": [0.9] * len(pieces), "ids": [],
                "tokens_generated": len(pieces), "hit_eos": True, "greedy_consistent": True, "model_ms": 12.0}


def _rows(tmp_path, rows=FIXTURE_ROWS[:5]):
    data = write_fixture(tmp_path, "gfix", rows)
    return load_rows(None, None, data, val=True, val_set="gfix"), data


def test_a_perfect_generator_scores_every_fixture_row(tmp_path):
    rows, _ = _rows(tmp_path)
    eng = ScriptEngine({len(r["history"]): G.target_text(r) for r in FIXTURE_ROWS[:5]})
    out, summ = E.run_eval(eng, rows)
    assert summ["step_acc"] == 1.0 and summ["type_acc"] == 1.0 and summ["parse_fail_rate"] == 0.0
    assert summ["grounding"] == 1.0  # click + type_text rows
    assert all(size[0] == 720 for _, _, size in eng.calls)  # train_d1.load_image's width
    assert [h for _, h, _ in eng.calls] == [r["history"] for r in FIXTURE_ROWS[:5]]
    assert summ["confidence"]["rows_with_confidence"] == 5 and "ece_action" in summ


def test_parse_failures_are_scored_wrong_and_reported(tmp_path):
    rows, _ = _rows(tmp_path)
    texts = {0: '{"action_type":"open_app","app_name":"Clock"}',
             1: 'I would tap the timer tab',                                    # no JSON
             2: '{"action_type":"type_text","x":185,"y":438}',                  # missing text
             3: '{"action_type":"scroll","direction":"down"}',
             4: '{"action_type":"status"'}                                      # truncated
    out, summ = E.run_eval(ScriptEngine(texts), rows)
    assert [o["correct"] for o in out] == [True, False, False, True, False]
    assert [o["pred"] for o in out] == ["open_app", "", "", "scroll", ""]
    assert summ["parse_fail_rate"] == pytest.approx(3 / 5)
    assert summ["parse_errors"] == {"no_json_object": 1, "missing_text": 1, "invalid_json": 1}
    assert out[1]["raw"] == "I would tap the timer tab" and out[1]["conf"] is None


def test_a_wrong_point_is_a_grounding_miss_not_a_type_miss(tmp_path):
    rows, _ = _rows(tmp_path, FIXTURE_ROWS[1:2])
    out, summ = E.run_eval(ScriptEngine({1: '{"action_type":"click","x":900,"y":900}'}), rows)
    assert out[0]["pred"] == "click" and out[0]["correct"] is False and out[0]["grounding"] is False
    assert summ["type_acc"] == 1.0 and summ["grounding"] == 0.0


def test_scores_equal_a_direct_eval_d1_style_call(tmp_path):
    rows, _ = _rows(tmp_path)
    texts = {0: '{"action_type":"click","x":10,"y":10}', 1: '{"action_type":"click","x":185,"y":438}',
             2: '{"action_type":"type_text","x":185,"y":438,"text":"6"}', 3: '{"action_type":"scroll","direction":"up"}',
             4: '{"action_type":"status"}'}
    out, _ = E.run_eval(ScriptEngine(texts), rows)
    for r, o in zip(rows, out):
        p, _, _ = G.parse_action(texts[len(r["previous_actions"])])
        at = p["action_type"]
        value = {"scroll": p.get("direction"), "open_app": p.get("app_name"), "type_text": p.get("text"),
                 "status": "successful"}.get(at)
        pt = (G.from_grid(p["x"], r["screenshot_width"]), G.from_grid(p["y"], r["screenshot_height"])) \
            if at in ("click", "long_press", "type_text") else None
        assert (o["correct"], o["grounding"]) == score_row(r, at, value, pt)


def test_summary_carries_the_trivial_baselines(tmp_path):
    rows, _ = _rows(tmp_path)
    _, summ = E.run_eval(ScriptEngine({i: '{"action_type":"wait"}' for i in range(5)}), rows)
    assert summ["always_click_type"] == pytest.approx(1 / 5) and summ["always_status"] == pytest.approx(1 / 5)
    assert summ["step_acc"] == 0.0 and set(summ["per_type"]) == {"open_app", "click", "type_text", "scroll", "status"}


def test_data_check_passes_on_the_fixture(tmp_path, capsys):
    data = write_fixture(tmp_path, "gfix", FIXTURE_ROWS)
    assert D.main(["--data", data, "--split", "gfix", "--check", "--out", str(tmp_path / "x.jsonl")]) == 0
    printed = capsys.readouterr().out
    assert "CHECK_PASS" in printed
    ex = [json.loads(l) for l in open(tmp_path / "x.jsonl")]
    assert len(ex) == len(FIXTURE_ROWS) and ex[0]["target"] == G.target_text(FIXTURE_ROWS[0])
    rep, hard = D.check(FIXTURE_ROWS, data)
    assert hard == 0 and rep["reparse_mismatches"] == 0 and rep["gold_scored_wrong_by_type"] == {}
    assert rep["action_types"]["click"] == 1 and sum(rep["action_types"].values()) == len(FIXTURE_ROWS)
