"""Pause and resume a training run without changing it: a run paused at step k and resumed ends with the same weights
as the same run never paused (same data order, same learning-rate schedule, same optimizer state, same random
streams). A pause is asked for with a signal and taken at the end of an optimizer step."""
import os
import random
import signal
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks", "train"))
from resume_state import PAUSED_EXIT, PauseFlag, has_state, load_state, save_state  # noqa: E402

N_RECS, MICRO, TOTAL = 7, 2, 12


def build(seed=0):
    torch.manual_seed(seed)
    model = torch.nn.Linear(4, 3)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 3) * (1 - s / TOTAL))
    return model, opt, sched


def train(model, opt, sched, rng, order, cursor, epoch, start, stop, data):
    """The trainer's loop in miniature: the order is reshuffled at each epoch end, samples draw from `rng` (as
    make_sample does) and from torch (as dropout and stochastic rounding do)."""
    for step in range(start, stop):
        if cursor + MICRO > len(order):
            rng.shuffle(order)
            cursor, epoch = 0, epoch + 1
        x = torch.stack([data[k] * (1 + rng.random()) for k in order[cursor:cursor + MICRO]])
        cursor += MICRO
        loss = (model(x + 0.01 * torch.randn_like(x)) ** 2).mean()
        loss.backward()
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
    return cursor, epoch


def data():
    g = torch.Generator().manual_seed(1)
    return [torch.randn(4, generator=g) for _ in range(N_RECS)]


def test_a_paused_and_resumed_run_ends_with_the_same_weights_as_an_unpaused_one(tmp_path):
    d = data()
    m1, o1, s1 = build()
    train(m1, o1, s1, random.Random(0), list(range(N_RECS)), 0, 0, 0, TOTAL, d)

    m2, o2, s2 = build()
    rng, order = random.Random(0), list(range(N_RECS))
    cursor, epoch = train(m2, o2, s2, rng, order, 0, 0, 0, 5, d)
    save_state(str(tmp_path / "resume"), step=5, model=m2, opt=o2, sched=s2, rng=rng, order=order, cursor=cursor,
               epoch=epoch)
    torch.manual_seed(999)                                  # a new process: its random streams differ
    m3, o3, s3 = build(seed=7)
    rng3 = random.Random(123)
    st = load_state(str(tmp_path / "resume"), model=m3, opt=o3, sched=s3, rng=rng3)
    assert st["step"] == 5
    train(m3, o3, s3, rng3, st["order"], st["cursor"], st["epoch"], st["step"], TOTAL, d)
    for a, b in zip(m1.parameters(), m3.parameters()):
        assert torch.equal(a, b)
    assert s1.get_last_lr() == s3.get_last_lr()


def test_a_half_written_state_is_not_resumed_from(tmp_path):
    m, o, s = build()
    d = str(tmp_path / "resume")
    assert not has_state(d)
    save_state(d, step=1, model=m, opt=o, sched=s, rng=random.Random(0), order=[0], cursor=0, epoch=0)
    assert has_state(d)
    os.makedirs(d + ".tmp", exist_ok=True)                  # a save interrupted mid-way leaves only .tmp
    open(d + ".tmp/state.pt", "w").write("partial")
    assert load_state(d, model=m, opt=o, sched=s, rng=random.Random(0))["step"] == 1


def test_a_pause_signal_is_remembered_until_the_step_ends():
    flag = PauseFlag()
    flag.install()
    assert not flag.is_set()
    os.kill(os.getpid(), signal.SIGUSR1)
    assert flag.is_set() and PAUSED_EXIT == 75
    flag.uninstall()
