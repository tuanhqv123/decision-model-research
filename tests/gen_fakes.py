"""Tiny stand-ins for the LFM3-VL processor and model, so the generative arm's plumbing is testable on CPU.

The fake tokenizer has one token per character plus a few special tokens; the fake processor renders the same
chat structure as LFM3-VL's template (BOS, user turn with one image placeholder, `<think></think>` generation
prompt) and expands the image placeholder to 4 tokens, as a 2x2-merged 4x4 patch grid would.
"""
import json
import os
from types import SimpleNamespace

import torch
from PIL import Image

SPECIAL = {"<pad>": 0, "<|startoftext|>": 1, "<|im_end|>": 2, "<|image|>": 3}
N_IMAGE_TOKENS = 4
VOCAB = 300


class FakeTokenizer:
    bos_token, bos_token_id = "<|startoftext|>", 1
    eos_token, eos_token_id = "<|im_end|>", 2
    pad_token, pad_token_id = "<pad>", 0

    def encode(self, text):
        ids, i = [], 0
        while i < len(text):
            for s, sid in SPECIAL.items():
                if text.startswith(s, i):
                    ids.append(sid)
                    i += len(s)
                    break
            else:
                ids.append(10 + min(ord(text[i]), VOCAB - 11))
                i += 1
        return ids

    def __call__(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return {"input_ids": self.encode(text)}

    def convert_tokens_to_ids(self, tok):
        return SPECIAL[tok]

    def decode(self, ids, skip_special_tokens=False):
        inv = {v: k for k, v in SPECIAL.items()}
        return "".join(("" if skip_special_tokens else inv[i]) if i in inv else chr(i - 10) for i in ids)


class FakeProcessor:
    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.image_processor = SimpleNamespace(merge_size=2)
        self.saved_to = None

    def apply_chat_template(self, msgs, add_generation_prompt=False, enable_thinking=True, tokenize=False):
        assert tokenize is False and enable_thinking is False
        (user,) = [m for m in msgs if m["role"] == "user"]
        text = "".join("<|image|>" if c["type"] == "image" else c["text"] for c in user["content"])
        s = "<|startoftext|><|im_start|>user\n" + text + "<|im_end|>\n"
        if add_generation_prompt:
            s += "<|im_start|>assistant\n<think></think>"
        return s

    def __call__(self, text, images, return_tensors="pt", add_special_tokens=True):
        assert add_special_tokens is False and len(text) == 1 and len(images) == 1
        ids = []
        for t in self.tokenizer.encode(text[0]):
            ids.extend([3] * N_IMAGE_TOKENS if t == 3 else [t])
        ids = torch.tensor([ids])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids),
                "pixel_values": torch.ones(16, 3), "image_grid_thw": torch.tensor([[1, 4, 4]])}

    def save_pretrained(self, out):
        self.saved_to = out
        with open(os.path.join(out, "processor_config.json"), "w") as fh:
            fh.write("{}")


class FakeModel(torch.nn.Module):
    """Embedding (tied to the head) -> mixer -> head. Uses pixel_values so the image path carries gradient."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.emb = torch.nn.Embedding(VOCAB, 8)
        self.mix = torch.nn.Linear(8, 8)
        self.vis = torch.nn.Linear(3, 8)
        self.config = SimpleNamespace(use_cache=True)
        self.ckpt = False

    def get_input_embeddings(self):
        return self.emb

    def get_output_embeddings(self):
        return self.emb  # tied, as in LFM3-VL

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.ckpt = True

    def forward(self, input_ids, attention_mask=None, pixel_values=None, image_grid_thw=None, logits_to_keep=0,
                use_cache=None, labels=None):
        h = torch.tanh(self.mix(self.emb(input_ids)))
        if pixel_values is not None:
            h = h + self.vis(pixel_values).mean(0)
        h = torch.cumsum(h, dim=1)  # causal mixing: position t sees 0..t
        idx = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        return SimpleNamespace(logits=h[:, idx] @ self.emb.weight.T)

    def save_pretrained(self, out, safe_serialization=True):
        torch.save(self.state_dict(), os.path.join(out, "model.pt"))
        with open(os.path.join(out, "config.json"), "w") as fh:
            fh.write('{"saved": true}')


def compact_row(ep, step, action, target=None, history=None, W=1080, H=2400, goal="Open Clock and set a timer"):
    return {"ep": ep, "step": step, "goal": goal, "history": history or [], "img": f"imgs/{ep}/{step}.jpg",
            "W": W, "H": H, "action": action, "target": target, "cands": [],
            "history_source": "rendered_gold_actions_v1"}


def write_fixture(tmp_path, name, rows):
    """compact_<name>.jsonl + a small screenshot per row, in the acdata layout."""
    for r in rows:
        p = tmp_path / r["img"]
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (r["W"] // 10, r["H"] // 10), (200, 200, 200)).save(p)
    with open(tmp_path / f"compact_{name}.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return str(tmp_path)


BOX = {"x_min": 100, "x_max": 300, "y_min": 1000, "y_max": 1100}
FIXTURE_ROWS = [
    compact_row(1, 0, {"action_type": "open_app", "app_name": "Clock"}),
    compact_row(1, 1, {"action_type": "click", "x": 150, "y": 1050}, BOX, ["open_app Clock"]),
    compact_row(1, 2, {"action_type": "type_text", "x": 150, "y": 1050, "text": "5"}, BOX,
                ["open_app Clock", "click"]),
    compact_row(1, 3, {"action_type": "scroll", "direction": "down"}, None, ["open_app Clock", "click", "type_text 5"]),
    compact_row(1, 4, {"action_type": "status", "goal_status": "successful"}, None,
                ["open_app Clock", "click", "type_text 5", "scroll down"]),
    compact_row(2, 0, {"action_type": "navigate_back"}, None, [], W=1440, H=3120),
    compact_row(2, 1, {"action_type": "wait"}, None, ["navigate_back"], W=1440, H=3120),
    compact_row(2, 2, {"action_type": "long_press", "x": 700, "y": 700},
                {"x_min": 600, "x_max": 800, "y_min": 600, "y_max": 800}, ["navigate_back", "wait"], W=1440, H=3120),
]
