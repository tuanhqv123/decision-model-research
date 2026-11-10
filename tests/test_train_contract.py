"""The trainer asks the questions of the contract it is told, and stamps that contract on its checkpoint."""
import json

from PIL import Image

from benchmarks.train import train_d1
from benchmarks.train.train_d1 import make_sample, stamp_contract, tracks_of


def _rec(tmp_path):
    Image.new("RGB", (108, 240), (255, 255, 255)).save(tmp_path / "s.jpg")
    return {"ep": "e1", "step": 1, "goal": "Open the clock", "history": ["open_app Clock"],
            "img": "s.jpg", "W": 1080, "H": 2400, "cands": [],
            "action": {"action_type": "click", "x": 10, "y": 10},
            "target": {"x_min": 0, "x_max": 20, "y_min": 0, "y_max": 20}}


def test_samples_use_the_configured_contract(tmp_path, monkeypatch):
    monkeypatch.setitem(train_d1.RUN, "contract", "e2")
    _, qs, _, _ = make_sample(_rec(tmp_path), str(tmp_path), "V", None, None)
    assert list(qs) == ["action", "target_box", "scroll_dir", "app_name", "type_text"]
    assert "navigate_home" not in qs["action"]["criteria"]


def test_v1_samples_have_no_is_done(tmp_path, monkeypatch):
    monkeypatch.setitem(train_d1.RUN, "contract", "v1")
    _, qs, gold, _ = make_sample(_rec(tmp_path), str(tmp_path), "V", None, None)
    assert "is_done" not in qs


def test_tracks_of_training_flag():
    assert tracks_of("V") == ["V"]
    assert tracks_of("VA") == ["VA"]
    assert tracks_of("mix") == ["V", "VA"]


def test_checkpoint_is_stamped_with_its_contract(tmp_path, monkeypatch):
    monkeypatch.setitem(train_d1.RUN, "contract", "v1")
    stamp_contract(str(tmp_path), "V")
    doc = json.load(open(tmp_path / "contract.json"))
    assert doc["name"] == "v1" and doc["tracks"] == ["V"]
