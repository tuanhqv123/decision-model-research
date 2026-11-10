"""Generative arm (Q1), pure-Python layer: prompt parity with R1, target conversion for every action, the coordinate
grid, the parser (incl. malformed output), the scorer arguments, the confidence readout and the contract files."""
import json
import math

import pytest
from PIL import Image

from benchmarks.gen import gen_common as G
from benchmarks.train import ac_common as A


def _rec(action, target=None, history=None, W=1080, H=2400, goal="Open the clock and add an alarm"):
    return {"ep": 7, "step": 2, "goal": goal, "history": history if history is not None else ["open_app Clock"],
            "img": "s.jpg", "W": W, "H": H, "action": action, "target": target, "cands": []}


BOX = {"x_min": 100, "x_max": 300, "y_min": 1000, "y_max": 1100}

# One row per action of ac_common.ACTION_VOCAB, with its exact expected target string.
CASES = [
    (_rec({"action_type": "click", "x": 150, "y": 1050}, BOX), '{"action_type":"click","x":185,"y":438}'),
    (_rec({"action_type": "long_press", "x": 150, "y": 1050}, BOX), '{"action_type":"long_press","x":185,"y":438}'),
    (_rec({"action_type": "type_text", "x": 150, "y": 1050, "text": "Floor lamp"}, BOX),
     '{"action_type":"type_text","x":185,"y":438,"text":"Floor lamp"}'),
    (_rec({"action_type": "scroll", "direction": "Down"}), '{"action_type":"scroll","direction":"down"}'),
    (_rec({"action_type": "wait"}), '{"action_type":"wait"}'),
    (_rec({"action_type": "navigate_back"}), '{"action_type":"navigate_back"}'),
    (_rec({"action_type": "navigate_home"}), '{"action_type":"navigate_home"}'),
    (_rec({"action_type": "open_app", "app_name": "Simple Calendar Pro"}),
     '{"action_type":"open_app","app_name":"Simple Calendar Pro"}'),
    (_rec({"action_type": "status", "goal_status": "successful"}), '{"action_type":"status"}'),
]


# ------------------------------------------------------------------------------------------- prompt parity with R1
def test_instruction_is_frozen():
    # gen_v1 checkpoints record this hash; editing INSTRUCTION is a new contract, not a fix.
    assert G.INSTRUCTION_SHA256 == "7ea632afaa30227e09d24c42c59c38019183394b6d46d41519f47980d5649a78"


def test_instruction_lists_r1s_options_with_r1s_own_descriptions_in_r1s_order():
    lines = [l for l in G.INSTRUCTION.splitlines() if l.startswith("- ")]
    names = [l[2:].split(":")[0] for l in lines]
    assert sorted(names) == sorted(A.ACTION_VOCAB)
    assert names == [A.FROM_OPTION[o] for o in A.CONTRACTS["v1"]["actions"]]
    for l, n in zip(lines, names):
        assert l == f"- {n}: {A.ACTIONS[A.TO_OPTION[n]]}"
    for d, words in A.SCROLL.items():
        assert f"{d} ({words})" in G.INSTRUCTION


def test_state_is_r1s_training_state_byte_for_byte(tmp_path, monkeypatch):
    from benchmarks.train import train_d1
    Image.new("RGB", (108, 240), (255, 255, 255)).save(tmp_path / "s.jpg")
    monkeypatch.setitem(train_d1.RUN, "contract", "v1")
    for rec, _ in CASES:
        r1_state, _, _, _ = train_d1.make_sample(dict(rec), str(tmp_path), "V", None, None)
        assert G.state_text(rec["goal"], rec["history"]) == r1_state
        assert G.user_text(rec["goal"], rec["history"]) == r1_state + "\n\n" + G.INSTRUCTION


def test_empty_history_renders_like_r1():
    assert G.state_text("g", []) == "Goal: g\nPrevious actions: none"
    assert G.state_text("g", None) == A.build_state("g", "V", None, None, [])


def test_messages_put_the_picture_first_and_one_user_turn():
    m = G.messages("g", ["click"])
    assert len(m) == 1 and m[0]["role"] == "user"
    assert m[0]["content"][0] == {"type": "image"}
    assert m[0]["content"][1]["text"] == G.user_text("g", ["click"])


# ------------------------------------------------------------------------------------------- targets
@pytest.mark.parametrize("rec,expected", CASES, ids=[c[0]["action"]["action_type"] for c in CASES])
def test_target_for_every_action_type(rec, expected):
    assert G.target_text(rec) == expected
    assert G.target_obj(rec)["action_type"] in A.ACTION_VOCAB


def test_every_vocab_action_is_covered_by_the_cases():
    assert {c[0]["action"]["action_type"] for c in CASES} == set(A.ACTION_VOCAB)


def test_point_is_the_bbox_centre_not_the_click_point():
    rec = _rec({"action_type": "click", "x": 101, "y": 1001}, BOX)  # click near the corner of its element
    o = G.target_obj(rec)
    assert (o["x"], o["y"]) == (G.to_grid(200, 1080), G.to_grid(1050, 2400))


def test_point_falls_back_to_the_click_point_without_a_bbox():
    o = G.target_obj(_rec({"action_type": "click", "x": 540, "y": 1200}))
    assert (o["x"], o["y"]) == (500, 500)


def test_spatial_without_any_point_is_refused_not_invented():
    with pytest.raises(ValueError):
        G.target_obj(_rec({"action_type": "click"}))


def test_input_text_alias_and_unknown_action():
    o = G.target_obj(_rec({"action_type": "input_text", "x": 150, "y": 1050, "text": "a"}, BOX))
    assert o["action_type"] == "type_text"
    with pytest.raises(ValueError):
        G.target_obj(_rec({"action_type": "double_tap", "x": 1, "y": 1}, BOX))
    with pytest.raises(ValueError):
        G.target_obj(_rec({"action_type": "scroll", "direction": "sideways"}))


def test_non_ascii_text_is_kept_literally():
    t = G.target_text(_rec({"action_type": "type_text", "x": 1, "y": 1, "text": "café ☕"}, BOX))
    assert "café ☕" in t and "\\u" not in t


# ------------------------------------------------------------------------------------------- coordinate grid
@pytest.mark.parametrize("size", [1080, 2400, 1440, 3120, 720])
def test_grid_round_trip_is_within_half_a_thousandth(size):
    for px in [0, 0.4, 1, 17.3, size / 3, size / 2, size - 1, size - 0.6]:
        n = G.to_grid(px, size)
        assert 0 <= n <= 999 and isinstance(n, int)
        if px <= size * 0.9995:
            assert abs(G.from_grid(n, size) - px) <= size / 2000 + 1e-9


def test_grid_clamps_out_of_screen_points():
    assert G.to_grid(-5, 1080) == 0
    assert G.to_grid(1080, 1080) == 999
    assert G.to_grid(5000, 1080) == 999
    assert G.from_grid(1500, 1000) == 999.0 and G.from_grid(-3, 1000) == 0.0


def test_every_decodable_point_is_a_centre_d1s_box_can_express():
    # d1's box edges live on the 0.000-0.999 grid; its point is (left + right) / 2 * W, i.e. k * W / 2000.
    W = 1080
    for n in range(1000):
        k = G.from_grid(n, W) * 2000 / W
        assert math.isclose(k, round(k), abs_tol=1e-6)


def test_screen_centre_is_exact():
    assert G.from_grid(G.to_grid(540, 1080), 1080) == 540.0


# ------------------------------------------------------------------------------------------- parser
@pytest.mark.parametrize("rec,expected", CASES, ids=[c[0]["action"]["action_type"] for c in CASES])
def test_parse_gives_every_target_back(rec, expected):
    parsed, err, flags = G.parse_action(expected)
    assert err is None and flags == [] and parsed == G.target_obj(rec)


@pytest.mark.parametrize("text", [
    ' {"action_type": "click", "x": 185, "y": 438} ',
    '{\n  "action_type": "click",\n  "x": 185,\n  "y": 438\n}',
    '<think></think>{"action_type":"click","x":185,"y":438}',
    '<think>tap the field</think>\n{"action_type":"click","x":185,"y":438}<|im_end|>',
    '{"action_type":"CLICK","x":185.4,"y":437.6}',
])
def test_parse_tolerates_whitespace_think_blocks_case_and_floats(text):
    parsed, err, _ = G.parse_action(text)
    assert err is None and parsed == {"action_type": "click", "x": 185, "y": 438}


@pytest.mark.parametrize("text,error", [
    ("", "no_json_object"),
    (None, "no_json_object"),
    ("click at the search bar", "no_json_object"),
    ('{"action_type":"click","x":185,', "invalid_json"),
    ("{'action_type': 'click'}", "invalid_json"),
    ('["click", 1, 2]', "no_json_object"),
    ('{"action_type":"tap","x":1,"y":2}', "unknown_action_type"),
    ('{"action":"click","x":1,"y":2}', "unknown_action_type"),
    ('{"action_type":"answer","text":"42"}', "unknown_action_type"),
    ('{"action_type":"click","x":1}', "missing_point"),
    ('{"action_type":"click","x":"185","y":"438"}', "missing_point"),
    ('{"action_type":"click","x":true,"y":2}', "missing_point"),
    ('{"action_type":"long_press","x":null,"y":2}', "missing_point"),
    ('{"action_type":"type_text","x":1,"y":2}', "missing_text"),
    ('{"action_type":"type_text","x":1,"y":2,"text":5}', "missing_text"),
    ('{"action_type":"open_app"}', "missing_app_name"),
    ('{"action_type":"scroll","direction":"sideways"}', "bad_direction"),
    ('{"action_type":"scroll"}', "bad_direction"),
])
def test_parse_rejects_malformed_output(text, error):
    parsed, err, _ = G.parse_action(text)
    assert parsed is None and err == error


def test_parse_flags_trailing_text_extra_keys_and_clamping():
    p, err, flags = G.parse_action('{"action_type":"wait","why":"loading"} and then click')
    assert err is None and p == {"action_type": "wait"} and set(flags) == {"trailing_text", "extra_keys"}
    p, err, flags = G.parse_action('{"action_type":"click","x":1200,"y":-4}')
    assert err is None and (p["x"], p["y"]) == (999, 0) and flags == ["clamped"]


def test_parse_reads_the_first_object_only():
    p, _, flags = G.parse_action('{"action_type":"navigate_back"}{"action_type":"click","x":1,"y":1}')
    assert p == {"action_type": "navigate_back"} and "trailing_text" in flags


def test_empty_text_parses_and_is_left_to_the_adapter():
    p, err, _ = G.parse_action('{"action_type":"type_text","x":1,"y":2,"text":""}')
    assert err is None and p["text"] == ""


# ------------------------------------------------------------------------------------------- decision + scorer args
def test_decision_maps_the_grid_to_pixels_of_the_sent_screenshot():
    p, _, _ = G.parse_action('{"action_type":"click","x":500,"y":250}')
    assert G.decision(p, 1080, 2400) == {"action": "click", "point": [540.0, 600.0]}
    assert G.decision(p, 1440, 3120)["point"] == [720.0, 780.0]


def test_decision_fields_per_action():
    d = G.decision(G.parse_action('{"action_type":"type_text","x":1,"y":2,"text":"hi"}')[0], 1000, 1000)
    assert d["text"] == "hi" and d["point"] == [1.0, 2.0]
    assert G.decision(G.parse_action('{"action_type":"scroll","direction":"up"}')[0], 9, 9) == \
        {"action": "scroll", "point": None, "direction": "up"}
    assert G.decision(G.parse_action('{"action_type":"open_app","app_name":"Clock"}')[0], 9, 9)["app_name"] == "Clock"
    assert G.decision(G.parse_action('{"action_type":"status"}')[0], 9, 9) == {"action": "status", "point": None}
    assert G.decision(None, 9, 9) == {"action": "", "point": None}


def test_scorer_args_mirror_eval_d1():
    assert G.scorer_args(None, 9, 9) == ("", None, None)
    assert G.scorer_args({"action_type": "status"}, 9, 9) == ("status", "successful", None)
    assert G.scorer_args({"action_type": "scroll", "direction": "down"}, 9, 9) == ("scroll", "down", None)
    assert G.scorer_args({"action_type": "open_app", "app_name": "Clock"}, 9, 9) == ("open_app", "Clock", None)
    assert G.scorer_args({"action_type": "wait"}, 9, 9) == ("wait", None, None)
    at, value, pt = G.scorer_args({"action_type": "type_text", "x": 500, "y": 500, "text": "a"}, 1080, 2400)
    assert (at, value, pt) == ("type_text", "a", (540.0, 1200.0))


# ------------------------------------------------------------------------------------------- confidence
def test_confidence_is_the_product_over_the_action_type_value_tokens():
    pieces = ['{"', 'action', '_type', '":"', 'long', '_press', '","', 'x', '":', '1', ',"y":', '2', '}', '<|im_end|>']
    probs = [0.99, 0.99, 0.99, 0.99, 0.5, 0.8, 0.3, 0.9, 0.9, 0.1, 0.9, 0.2, 0.9, 0.9]
    assert G.action_type_confidence(pieces, probs) == pytest.approx(0.5 * 0.8)


def test_confidence_counts_a_token_that_straddles_the_value_boundary():
    pieces = ['{"action_type":"', 'click"', ',"x":1,"y":2}']
    assert G.action_type_confidence(pieces, [0.9, 0.6, 0.2]) == pytest.approx(0.6)


def test_confidence_is_none_when_the_value_is_absent():
    assert G.action_type_confidence(['{"action":', '"click"}'], [0.5, 0.5]) is None
    assert G.action_type_confidence([], []) is None
    assert G.action_type_confidence(['{"action_type":""}'], [0.5]) is None


# ------------------------------------------------------------------------------------------- contract
def test_contract_files(tmp_path):
    G.write_contract(str(tmp_path))
    assert json.loads((tmp_path / "contract.json").read_text()) == {"name": "gen_v1", "tracks": ["V"]}
    prompt = json.loads((tmp_path / "gen_prompt.json").read_text())
    assert prompt["instruction"] == G.INSTRUCTION and prompt["sha256"] == G.INSTRUCTION_SHA256
    assert G.check_contract(str(tmp_path))["name"] == "gen_v1"


def test_gen_v1_is_not_a_one_pass_contract(tmp_path):
    # d1_api (ac_common.read_contract) must refuse a generative checkpoint, and gen_v1 must stay out of CONTRACTS.
    G.write_contract(str(tmp_path))
    assert "gen_v1" not in A.CONTRACTS
    with pytest.raises(KeyError):
        A.read_contract(str(tmp_path))


def test_check_contract_refuses_other_contracts_and_other_prompts(tmp_path):
    A.write_contract(str(tmp_path), "v1")
    with pytest.raises(ValueError):
        G.check_contract(str(tmp_path))
    G.write_contract(str(tmp_path))
    doc = json.loads((tmp_path / "gen_prompt.json").read_text())
    doc["sha256"] = "0" * 64
    (tmp_path / "gen_prompt.json").write_text(json.dumps(doc))
    with pytest.raises(ValueError):
        G.check_contract(str(tmp_path))
    (tmp_path / "gen_prompt.json").unlink()
    with pytest.raises(FileNotFoundError):
        G.check_contract(str(tmp_path))
