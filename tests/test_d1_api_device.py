"""The API can serve on a CPU (D1_DEVICE=cpu): the model is loaded in float32 on the CPU, CUDA is never touched, and
/health still answers. The default stays the GPU in bfloat16, as the released model was trained and measured."""
import importlib
import sys
from types import ModuleType, SimpleNamespace as NS

import pytest

pytest.importorskip("fastapi")
import torch  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from tests.test_d1_api_v2 import AC, SERVING, Recorder  # noqa: E402


class DeviceRecorder(Recorder):
    def __init__(self):
        super().__init__()
        self.placed = []

    def cuda(self):
        self.placed.append("cuda")
        return self

    def to(self, device):
        self.placed.append(str(device))
        return self


def _load(monkeypatch, tmp_path, env):
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    AC.write_contract(str(ckpt), "e2", tracks=["V"])
    model, loads = DeviceRecorder(), []
    stub = ModuleType("transformers")
    stub.AutoModel = NS(from_pretrained=lambda *a, **k: loads.append(k) or model)
    monkeypatch.setitem(sys.modules, "transformers", stub)
    monkeypatch.setenv("D1_MODEL", str(ckpt))
    for k in ("D1_MEM_FRAC", "D1_API_TOKEN", "D1_DEVICE", "D1_DTYPE"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.syspath_prepend(SERVING)
    sys.modules.pop("d1_api", None)
    return importlib.import_module("d1_api"), model, loads


@pytest.fixture(autouse=True)
def _unload():
    yield
    sys.modules.pop("d1_api", None)


def test_cpu_device_loads_float32_on_cpu_and_never_touches_cuda(monkeypatch, tmp_path):
    def no_cuda(*a, **k):
        raise AssertionError("CUDA was called on a CPU server")
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", no_cuda)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", no_cuda)
    api, model, loads = _load(monkeypatch, tmp_path, {"D1_DEVICE": "cpu", "D1_MEM_FRAC": "0.4"})
    assert model.placed == ["cpu"]
    assert loads[0]["dtype"] is torch.float32
    h = TestClient(api.app).get("/health").json()
    assert h["ok"] is True and h["device"] == "cpu" and h["gpu_mem_gb"] is None


def test_default_stays_gpu_bfloat16(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *a, **k: 0)
    api, model, loads = _load(monkeypatch, tmp_path, {})
    assert model.placed == ["cuda"]
    assert loads[0]["dtype"] is torch.bfloat16
    assert TestClient(api.app).get("/health").json()["device"] == "cuda"


def test_dtype_can_be_forced(monkeypatch, tmp_path):
    api, model, loads = _load(monkeypatch, tmp_path, {"D1_DEVICE": "cpu", "D1_DTYPE": "bfloat16"})
    assert loads[0]["dtype"] is torch.bfloat16
