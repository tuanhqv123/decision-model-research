"""Probe: can the on-device Qwen2.5-VL (mlx_lm) process a raw ScreenSpot image?"""
import os, json, glob, sys
from datasets import load_dataset

MODEL = "mlx-community/Qwen2.5-VL-3B-Instruct-4bit"

ds = load_dataset("ZhuOnR/ScreenSpot", split="test")
row = ds[5]
img = row["image"]
inst = row["instruction"]

print("instruction:", inst, "data_type:", row["data_type"], "src:", row.get("data_source"))
print("image size:", img.size, "mode:", img.mode)
img.save("/tmp/_ss_probe.png")

try:
    from mlx_lm import load, generate
    import mlx_lm
    print("\nLoading", MODEL, "(this may take ~30s)...")
    model, tok = load(MODEL)
    print("load OK; model type:", type(model).__name__)
    # try to pass image; mlx vision via <|image|> placeholder and image kwarg
    prompt = (
        '<|im_start|>system\nYou are a GUI grounding agent. '
        'Return the element matching the instruction as a JSON bbox '
        '{"bbox":[x1,y1,x2,y2], "confidence":0.0-1.0}.\n<|im_end|>\n'
        f'<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\nInstruction: {inst}\n'
        '<|im_end|>\n<|im_start|>assistant\n'
    )
    try:
        out = generate(model, tok, prompt=prompt, max_tokens=128, image="/tmp/_ss_probe.png")
        print("\nIMAGE GENERATE OK:\n", out)
    except Exception as e:
        print("\nimage generate FAILED:", type(e).__name__, e)
        # fallback: text-only
        try:
            out2 = generate(model, tok, prompt=prompt, max_tokens=64)
            print("\ntext-only fallback OK:\n", out2)
        except Exception as e2:
            print("\ntext-only also FAILED:", type(e2).__name__, e2)
except Exception as e:
    print("\nLOAD FAILED:", type(e).__name__, e)
