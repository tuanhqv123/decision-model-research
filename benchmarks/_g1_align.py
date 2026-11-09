"""G1 de-risk: does EmbeddingGemma 2 multimodal separate correct from wrong candidate actions?

Loads google/embeddinggemma-2 (text+image 440M, bf16) via sentence-transformers 6.1.0.
For AC-v2 test rows: encode state = (goal + screenshot) and each candidate action text,
compute cosine similarity, report top-1 alignment (gold ranks #1) for multimodal vs text-only.
"""
import json, time, os, random, sys
import numpy as np
from PIL import Image
import torch
from sentence_transformers import SentenceTransformer

random.seed(0)
dev = "mps" if torch.backends.mps.is_available() else "cpu"
t0 = time.time()
m = SentenceTransformer("google/embeddinggemma-2",
                        model_kwargs={"torch_dtype": torch.bfloat16},
                        config_kwargs={"audio_config": None})   # text+image = 440M
if dev != "cpu":
    m.eval().to(dev)
print(f"[load] {time.time()-t0:.0f}s | params {sum(p.numel() for p in m.parameters())/1e6:.0f}M | {dev}", flush=True)

PREFIX = "task: classification | query: "

def _norm(x):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2 and x.shape[0] == 1:
        x = x[0]
    x = x / (np.linalg.norm(x) + 1e-9)
    return x

def state_emb(goal, imgpath):
    img = Image.open(imgpath).convert("RGB")
    with torch.no_grad():
        e = m.encode({"text": PREFIX + goal, "image": [img]}, convert_to_numpy=True)
    return _norm(e)

def text_emb(texts):
    with torch.no_grad():
        e = m.encode([PREFIX + t for t in texts], convert_to_numpy=True,
                     normalize_embeddings=True)
    return np.asarray(e, dtype=np.float32)

rows = [json.loads(l) for l in open("benchmarks/data/acv2_test.jsonl")]
good = [r for r in rows if os.path.exists(r["image"])]
print(f"rows {len(rows)} | with image {len(good)}", flush=True)
if len(good) == 0:
    sys.exit("no usable rows; check image paths")

N = int(os.environ.get("N", "46"))
good = good[:N]
tm = tt = n = 0
for i, r in enumerate(good):
    cands = r["candidates"]; gold = r["gold"]
    st_m = state_emb(r["goal"], r["image"])
    ce = text_emb(cands)                       # [K, d]
    sims_m = (st_m @ ce.T)
    rk_m = int(np.argsort(-sims_m)[0])
    st_t = text_emb([r["goal"]])[0]
    sims_t = (st_t @ ce.T)
    rk_t = int(np.argsort(-sims_t)[0])
    tm += (rk_m == gold); tt += (rk_t == gold); n += 1
    if i < 8:
        print(f"  ep={r['episode_id']} step={r['step']} gold={gold} | (multi)rank={rk_m} | (text)rank={rk_t}", flush=True)

print(f"\n=== G1 n={n} === top-1 (image+goal)={tm}/{n}  (text-only goal)={tt}/{n}", flush=True)
