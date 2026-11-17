"""Pause and resume a training run without changing what it learns.

A run asked to pause (SIGTERM or SIGUSR1) finishes its current optimizer step, writes its whole state and exits with
PAUSED_EXIT. Started again with the same arguments, it continues from that state: model weights, optimizer and
scheduler state, the data order and cursor, and every random stream (Python's sampler RNG, torch CPU and CUDA). A run
paused at step k therefore follows the same data and learning-rate schedule as one never paused. This lets a run
yield a shared GPU to other people and continue later (`gpu_share.py`).

The state is written to `<dir>.tmp` and renamed, so an interrupted save never replaces a good one.
"""
import os
import shutil
import signal
import threading

import torch

PAUSED_EXIT = 75          # EX_TEMPFAIL: "paused, start me again"
STATE = "state.pt"


def has_state(path: str) -> bool:
    return os.path.isfile(os.path.join(path, STATE))


def save_state(path: str, *, step, model, opt, sched, rng, order, cursor, epoch):
    tmp = path + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    torch.save({"step": step, "model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                "rng": rng.getstate(), "order": list(order), "cursor": cursor, "epoch": epoch,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
               os.path.join(tmp, STATE))
    old = path + ".old"
    shutil.rmtree(old, ignore_errors=True)
    if os.path.isdir(path):
        os.rename(path, old)
    os.rename(tmp, path)
    shutil.rmtree(old, ignore_errors=True)


def load_state(path: str, *, model, opt, sched, rng) -> dict:
    """Restore everything into the given objects; returns step (the next one to run), order, cursor, epoch."""
    st = torch.load(os.path.join(path, STATE), map_location="cpu", weights_only=False)
    model.load_state_dict(st["model"])
    opt.load_state_dict(st["opt"])
    sched.load_state_dict(st["sched"])
    rng.setstate(st["rng"])
    torch.set_rng_state(st["torch_rng"])
    if st.get("cuda_rng") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda_rng"])
    return {"step": st["step"], "order": st["order"], "cursor": st["cursor"], "epoch": st["epoch"]}


class PauseFlag:
    """Set by SIGTERM or SIGUSR1; the training loop checks it after each optimizer step."""

    def __init__(self, signals=(signal.SIGTERM, signal.SIGUSR1)):
        self._event, self._signals, self._old = threading.Event(), signals, {}

    def install(self):
        for s in self._signals:
            self._old[s] = signal.signal(s, lambda *_: self._event.set())
        return self

    def uninstall(self):
        for s, h in self._old.items():
            signal.signal(s, h)
        self._old = {}

    def is_set(self) -> bool:
        return self._event.is_set()
