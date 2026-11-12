"""Unlikelihood for avoid-only rows (contract v4, 2026-11-12): -log(1 - p(avoided option)). Needs torch (training host)."""
import math
import os
import sys

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_d1 import unlikelihood  # noqa: E402


def test_unlikelihood_value_and_gradient_direction():
    v = torch.tensor([2.0, 0.0, -1.0], requires_grad=True)
    logp = torch.log_softmax(v, -1)
    loss = unlikelihood(logp, 0)
    p0 = math.exp(2.0) / (math.exp(2.0) + 1 + math.exp(-1.0))
    assert abs(float(loss) - (-math.log(1 - p0))) < 1e-5
    loss.backward()
    assert v.grad[0] > 0  # gradient descent lowers the avoided option's score


def test_unlikelihood_is_finite_when_the_avoided_option_is_certain():
    logp = torch.log_softmax(torch.tensor([50.0, 0.0]), -1)
    assert torch.isfinite(unlikelihood(logp, 0))
