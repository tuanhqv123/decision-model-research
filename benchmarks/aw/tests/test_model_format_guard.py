"""The run refuses to start when the served checkpoint and the harness disagree on the input format (2026-11-11).

A contract-v3 checkpoint reads v3 history lines (element + effect); every other contract reads v1 lines. The harness
writes whichever `model.history_format` says. Nothing checked that the two agree, so a config typo would have fed a v3
model v1 history for all 116 tasks and produced a number. Same for the image width the checkpoint was trained at."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from run_bench import model_format_error  # noqa: E402


def test_v3_checkpoint_with_v1_history_is_refused():
    err = model_format_error({"contract": "v3", "img_width": 720}, history_format="v1", img_width=720)
    assert err and "v3" in err and "history_format" in err


def test_v3_checkpoint_with_v3_history_is_accepted():
    assert model_format_error({"contract": "v3", "img_width": 720}, history_format="v3", img_width=720) is None


def test_older_checkpoint_with_v3_history_is_refused():
    assert model_format_error({"contract": "v2a", "img_width": 720}, history_format="v3", img_width=720)
    assert model_format_error({"contract": "v2a", "img_width": 720}, history_format="v1", img_width=720) is None


def test_image_width_must_be_the_training_width():
    err = model_format_error({"contract": "v3", "img_width": 858}, history_format="v3", img_width=720)
    assert err and "858" in err and "720" in err


def test_unknown_or_missing_contract_is_refused():
    assert model_format_error({"img_width": 720}, history_format="v1", img_width=720)
    assert model_format_error({"contract": "v9", "img_width": 720}, history_format="v1", img_width=720)
