"""Model side of the generative arm: load LFM3-VL-2B, render the prompt, build one supervised example, decode greedily.

ONE prompt path for training, evaluation and serving: `encode_prompt` renders the chat with
`add_generation_prompt=True, enable_thinking=False`, so every prompt ends with `<|im_start|>assistant\n<think></think>`
and the answer follows directly. Training appends the answer tokens (target JSON + `<|im_end|>`) to exactly those
prompt ids, and the loss covers only the appended tokens. The template's own rendering of the same assistant turn is
`<think></think>{answer}<|im_end|>\n`; benchmarks/gen/cpu_check.py verifies token for token that the two agree.
"""
import os
import sys
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gen_common import END_OF_TURN, messages  # noqa: E402

BASE = "LiquidAI/LFM3-VL-2B"
BASE_REV = "23387beda43c60310348e83e25f78f33c3140d3a"  # pinned: train, eval and serve run the same remote code
DEFAULT_MAX_NEW_TOKENS = 64  # longest gold answer incl. <|im_end|>: 40 tokens (cur15k_v1), 45 (validation_v1)


def load(path, device="cuda", dtype=torch.bfloat16):
    """(model, processor). A Hub id is loaded at the pinned revision; a local directory as it is."""
    from transformers import AutoModelForImageTextToText, AutoProcessor
    rev = None if os.path.isdir(path) else BASE_REV
    model = AutoModelForImageTextToText.from_pretrained(path, trust_remote_code=True, dtype=dtype, revision=rev)
    processor = AutoProcessor.from_pretrained(path, trust_remote_code=True, revision=rev)
    return model.to(device), processor


def prompt_string(processor, goal, history):
    return processor.apply_chat_template(messages(goal, history), add_generation_prompt=True,
                                         enable_thinking=False, tokenize=False)


def encode_prompt(processor, goal, history, image):
    """Processor outputs for the prompt alone (input_ids, attention_mask, pixel_values, image_grid_thw).
    `add_special_tokens=False`: the template already starts with the BOS token."""
    return processor(text=[prompt_string(processor, goal, history)], images=[image], return_tensors="pt",
                     add_special_tokens=False)


def answer_ids(processor, answer):
    """Token ids the loss covers: the answer text and the end-of-turn token (so the model learns to stop)."""
    return processor.tokenizer(answer + END_OF_TURN, add_special_tokens=False)["input_ids"]


def build_example(processor, goal, history, image, answer):
    """(model inputs, labels, n_prompt). labels = -100 on every prompt position, the answer ids after it."""
    enc = encode_prompt(processor, goal, history, image)
    prompt = enc["input_ids"]
    ans = torch.tensor([answer_ids(processor, answer)], dtype=prompt.dtype)
    ids = torch.cat([prompt, ans], dim=1)
    ex = {k: v for k, v in enc.items() if k not in ("input_ids", "attention_mask")}
    ex["input_ids"] = ids
    ex["attention_mask"] = torch.ones_like(ids)
    labels = torch.full_like(ids, -100)
    labels[0, prompt.shape[1]:] = ans[0]
    return ex, labels, prompt.shape[1]


def to_device(ex, device, dtype):
    """Floating inputs (pixel_values) take the model's dtype; integer inputs keep theirs."""
    return {k: (v.to(device, dtype) if torch.is_tensor(v) and v.is_floating_point() else
                v.to(device) if torch.is_tensor(v) else v) for k, v in ex.items()}


def answer_loss(model, ex, labels, n_prompt):
    """Mean token cross-entropy over the answer tokens only. Logits are computed for those positions alone
    (`logits_to_keep` = the positions that PREDICT them), so the 128k-vocabulary projection never runs over the
    ~1,300 prompt positions. Returns (loss, stats)."""
    n_ans = labels.shape[1] - n_prompt
    dev = ex["input_ids"].device
    keep = torch.arange(n_prompt - 1, n_prompt - 1 + n_ans, device=dev)
    out = model(**ex, logits_to_keep=keep, use_cache=False)
    logits = out.logits[0].float()
    tgt = labels[0, n_prompt:].to(logits.device)
    loss = F.cross_entropy(logits, tgt)
    with torch.no_grad():
        hit = logits.argmax(-1) == tgt
    return loss, {"tok_acc": float(hit.float().mean()), "exact": float(hit.all()), "n_answer": int(n_ans),
                  "n_tokens": int(labels.shape[1])}


class GenEngine:
    """Greedy decoding with the probabilities of the emitted tokens. Used by eval_gen and serving/gen_api."""

    def __init__(self, model, processor, max_new_tokens=DEFAULT_MAX_NEW_TOKENS):
        from transformers import GenerationConfig
        self.model, self.processor = model, processor
        self.tok = processor.tokenizer
        self.eos = self.tok.convert_tokens_to_ids(END_OF_TURN)
        self.max_new_tokens = int(max_new_tokens)
        # Greedy and nothing else. transformers 5.x fills every field left None from the checkpoint's own
        # generation_config (the card's sampling defaults: do_sample, temperature 0.1, top_k 50, repetition_penalty
        # 1.1), so the fields that act under greedy decoding are pinned explicitly: do_sample=False, num_beams=1 and
        # repetition_penalty=1.0 — a repetition penalty is a logits processor that runs even under greedy and would
        # rewrite repeated JSON characters and digits. The sampling-only fields it fills in are inert. `run`
        # verifies at every step that the emitted token is the argmax of the RAW logits (`greedy_consistent`).
        self.gen_cfg = GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=self.max_new_tokens,
                                        repetition_penalty=1.0, eos_token_id=self.eos,
                                        pad_token_id=self.tok.convert_tokens_to_ids(self.tok.pad_token)
                                        if self.tok.pad_token else self.eos,
                                        return_dict_in_generate=True, output_logits=True)

    @property
    def device(self):
        return next(self.model.parameters()).device

    @property
    def dtype(self):
        return next(self.model.parameters()).dtype

    @torch.inference_mode()
    def run(self, goal, history, image):
        """-> {text, ids, probs, pieces, tokens_generated, hit_eos, greedy_consistent, model_ms}."""
        enc = to_device(dict(encode_prompt(self.processor, goal, history, image)), self.device, self.dtype)
        n_prompt = enc["input_ids"].shape[1]
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        out = self.model.generate(**enc, generation_config=self.gen_cfg)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        model_ms = (time.time() - t0) * 1000
        new = out.sequences[0, n_prompt:].tolist()
        probs, consistent = [], True
        for step_logits, t in zip(out.logits, new):
            lp = step_logits[0].float()
            probs.append(float(torch.softmax(lp, -1)[t]))
            consistent &= int(lp.argmax()) == t
        pieces, prev = [], ""
        for k in range(len(new)):
            cur = self.tok.decode(new[:k + 1], skip_special_tokens=False)
            pieces.append(cur[len(prev):])
            prev = cur
        return {"text": self.tok.decode(new, skip_special_tokens=True), "ids": new, "probs": probs,
                "pieces": pieces, "tokens_generated": len(new), "hit_eos": bool(new) and new[-1] == self.eos,
                "greedy_consistent": bool(consistent), "model_ms": model_ms, "n_prompt": n_prompt}
