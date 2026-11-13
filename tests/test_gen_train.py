"""Generative arm (Q1), model side, with the model and processor mocked: the supervised example (prompt ids + answer
ids, loss mask), the answer-only loss, the trainer's parity with R1 (row order, schedule, frozen embedding,
contract stamped last) and greedy decoding with token probabilities."""
import json
import math
import os
import random
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from benchmarks.gen import gen_common as G
from benchmarks.gen import gen_model as M
from benchmarks.gen import train_gen as T
from tests.gen_fakes import FIXTURE_ROWS, FakeModel, FakeProcessor, write_fixture


def _img():
    from PIL import Image
    return Image.new("RGB", (72, 160))


# ------------------------------------------------------------------------------------------- example + mask
def test_example_is_prompt_then_answer_and_only_the_answer_is_supervised():
    proc = FakeProcessor()
    ans = '{"action_type":"click","x":185,"y":438}'
    ex, labels, n_prompt = M.build_example(proc, "Open Clock", ["open_app Clock"], _img(), ans)
    ids = ex["input_ids"][0].tolist()
    prompt = M.encode_prompt(proc, "Open Clock", ["open_app Clock"], _img())["input_ids"][0].tolist()
    assert ids[:n_prompt] == prompt and n_prompt == len(prompt)
    assert ids[n_prompt:] == proc.tokenizer.encode(ans + "<|im_end|>")
    assert labels[0, :n_prompt].eq(-100).all() and labels[0, n_prompt:].tolist() == ids[n_prompt:]
    assert ex["attention_mask"].shape == ex["input_ids"].shape and ex["attention_mask"].all()
    assert "pixel_values" in ex and "image_grid_thw" in ex
    assert ids.count(1) == 1 and ids[0] == 1  # exactly one BOS: the template's own


def test_prompt_ends_with_the_empty_think_block_and_carries_the_r1_state():
    s = M.prompt_string(FakeProcessor(), "g", ["click"])
    assert s.endswith("<|im_start|>assistant\n<think></think>")
    assert G.user_text("g", ["click"]) in s


def test_answer_loss_equals_the_shifted_full_sequence_loss():
    proc, model = FakeProcessor(), FakeModel()
    ex, labels, n_prompt = M.build_example(proc, "g", [], _img(), '{"action_type":"wait"}')
    loss, st = M.answer_loss(model, ex, labels, n_prompt)
    full = model(**ex).logits[0]
    ref = F.cross_entropy(full[:-1], labels[0, 1:], ignore_index=-100)
    assert float(loss) == pytest.approx(float(ref), rel=1e-6)
    assert st["n_answer"] == labels.shape[1] - n_prompt and st["n_tokens"] == labels.shape[1]


def test_to_device_casts_only_floating_inputs():
    ex = {"input_ids": torch.tensor([[1, 2]]), "pixel_values": torch.ones(2, 3)}
    out = M.to_device(ex, "cpu", torch.bfloat16)
    assert out["input_ids"].dtype == torch.long and out["pixel_values"].dtype == torch.bfloat16


# ------------------------------------------------------------------------------------------- parity with train_d1
def _train_d1_draws(n, steps, micro, accum, seed):
    """train_d1.main's loop, transcribed line for line (cursor over range(n); rng.shuffle on exhaustion)."""
    rng = random.Random(seed)
    order = list(range(n))
    cursor, out = 0, []
    for _ in range(steps):
        for _ in range(accum):
            if cursor + micro > len(order):
                rng.shuffle(order)
                cursor = 0
            out.extend(order[cursor:cursor + micro])
            cursor += micro
    return out


@pytest.mark.parametrize("n,steps,micro,accum", [(14965, 938, 1, 16), (50, 7, 1, 16), (37, 5, 2, 8)])
def test_draw_order_is_train_d1s(n, steps, micro, accum):
    assert T.draw_order(n, steps * micro * accum, micro, 0) == _train_d1_draws(n, steps, micro, accum, 0)
    assert T.draw_order(n, steps * micro * accum, micro, 3) == _train_d1_draws(n, steps, micro, accum, 3)


def test_lr_schedule_is_train_d1s():
    f = T.lr_lambda(30, 938)
    for s in (0, 1, 29, 30, 400, 937):
        ref = min(1.0, (s + 1) / 30) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / 938)))
        assert f(s) == pytest.approx(ref)


def test_records_come_from_train_d1s_loader(tmp_path):
    from benchmarks.train.train_d1 import load_records
    data = write_fixture(tmp_path, "gfix", FIXTURE_ROWS)
    # Same function from the same file (imported flat by train_gen, as a package path here).
    same = os.path.realpath(load_records.__code__.co_filename)
    assert os.path.realpath(T.load_records.__code__.co_filename) == same
    assert os.path.realpath(T.load_image.__code__.co_filename) == same
    assert T.load_records(data, "gfix", 0, 0) == load_records(data, "gfix", 0, 0)


# ------------------------------------------------------------------------------------------- trainer end to end
def _base_dir(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    for f, text in {"LICENSE": "LFM Open License v1.0", "config.json": '{"base": true}',
                    "generation_config.json": '{"g": 1}', "modeling_fake.py": "# remote code",
                    "README.md": "card", "contract.json": '{"name": "v1"}'}.items():
        (base / f).write_text(text)
    (base / ".cache").mkdir()
    return str(base)


def test_trainer_runs_freezes_the_embedding_and_stamps_the_contract_last(tmp_path, monkeypatch):
    data = write_fixture(tmp_path, "gfix", FIXTURE_ROWS)
    base = _base_dir(tmp_path)
    model, proc = FakeModel(), FakeProcessor()
    monkeypatch.setattr(T, "load", lambda path, device, dtype: (model, proc))
    seen = []
    real = T.build_example
    monkeypatch.setattr(T, "build_example", lambda p, goal, hist, img, ans: seen.append(ans) or real(p, goal, hist,
                                                                                                         img, ans))
    emb0 = model.emb.weight.detach().clone()
    mix0 = model.mix.weight.detach().clone()
    vis0 = model.vis.weight.detach().clone()
    out = tmp_path / "run"
    rc = T.main(["--data", data, "--out", str(out), "--init", base, "--train-set", "gfix", "--steps", "3",
                 "--micro", "1", "--accum", "4", "--lr", "1e-2", "--warmup", "1", "--grad-ckpt", "--freeze-embed",
                 "--optim", "adamw", "--device", "cpu", "--dtype", "float32"])
    assert rc == 0 and model.ckpt
    assert torch.equal(model.emb.weight, emb0)                       # tied embedding / head frozen
    assert not torch.equal(model.mix.weight, mix0) and not torch.equal(model.vis.weight, vis0)
    assert len(seen) == 12                                           # 3 steps x micro 1 x accum 4
    lines = open(T.os.path.join(data, "compact_gfix.jsonl")).read().splitlines()
    random.Random(0).shuffle(lines)
    expected = [G.target_text(json.loads(lines[i])) for i in T.draw_order(len(lines), 12, 1, 0)]
    assert seen == expected                                          # R1's rows, in R1's order
    ck = out / "ckpt"
    assert json.loads((ck / "contract.json").read_text()) == {"name": "gen_v1", "tracks": ["V"]}
    assert G.check_contract(str(ck))["name"] == "gen_v1"
    assert (ck / "config.json").read_text() == '{"base": true}'      # the release's config, as train_d1 keeps it
    assert (ck / "LICENSE").exists() and (ck / "modeling_fake.py").exists() and (ck / "NOTICE_MODIFIED.txt").exists()
    assert not (ck / "README.md").exists() and not (ck / ".cache").exists()
    assert json.loads((ck / "args.json").read_text())["saved_at_step"] == 3
    log = [json.loads(l) for l in (out / "train.log").read_text().splitlines()]
    assert log and {"loss", "tok_acc", "exact", "lr", "gnorm", "step"} <= set(log[0])


# ------------------------------------------------------------------------------------------- greedy decoding
class _GenModel(torch.nn.Module):
    """generate() emits `script` token by token; logits put `peak` probability on a chosen token per step."""

    def __init__(self, script, chosen=None, peak=4.0):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))
        self.script, self.chosen, self.peak = script, chosen or script, peak
        self.calls = []

    def generate(self, input_ids, generation_config=None, return_dict_in_generate=True, output_logits=True, **kw):
        self.calls.append((generation_config, set(kw)))
        logits = []
        for c in self.chosen:
            v = torch.zeros(1, 300)
            v[0, c] = self.peak
            logits.append(v)
        seq = torch.cat([input_ids, torch.tensor([self.script])], 1)
        return SimpleNamespace(sequences=seq, logits=tuple(logits))


def test_engine_greedy_decode_returns_text_tokens_probs_and_pieces():
    proc = FakeProcessor()
    out_text = '{"action_type":"click","x":1,"y":2}'
    script = proc.tokenizer.encode(out_text + "<|im_end|>")
    eng = M.GenEngine(_GenModel(script), proc, max_new_tokens=64)
    g = eng.run("g", ["click"], _img())
    assert g["text"] == out_text and g["tokens_generated"] == len(script) and g["hit_eos"]
    assert g["greedy_consistent"] and "".join(g["pieces"]) == out_text + "<|im_end|>"
    p = math.exp(4.0) / (math.exp(4.0) + 299)
    assert all(x == pytest.approx(p) for x in g["probs"])
    assert G.action_type_confidence(g["pieces"], g["probs"]) == pytest.approx(p ** len("click"))
    cfg, kw = eng.model.calls[0]
    assert cfg.do_sample is False and cfg.num_beams == 1 and cfg.repetition_penalty == 1.0
    assert cfg.max_new_tokens == 64 and cfg.eos_token_id == 2
    assert cfg.return_dict_in_generate and cfg.output_logits
    assert {"pixel_values", "image_grid_thw", "attention_mask"} <= kw


def test_engine_flags_a_decode_that_is_not_the_argmax_and_a_missing_eos():
    proc = FakeProcessor()
    script = proc.tokenizer.encode('{"action_type":"wa')
    chosen = list(script)
    chosen[3] = 99
    g = M.GenEngine(_GenModel(script, chosen), proc).run("g", [], _img())
    assert g["greedy_consistent"] is False and g["hit_eos"] is False
