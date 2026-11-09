"""Auxiliary GUI-grounding samples on the SAME screens as a training set (no extra images), to strengthen the
visual grounding the decision model lacks (UGround / OS-Atlas / Aguvis-style grounding pre-training).

Two kinds, each its own sample (never packed with a decision sample: under `set` attention the state reads
every question, so a step instruction packed next to the high-level decision would leak its target):
  aux_instr : state = the step's low-level instruction ("Click on the search icon"), box = its gold element
  aux_label : state = an element's visible label ("Search"), box = that element (unique label on the screen)

Usage: python make_aux.py <acdata> <train_set> <out_name> <n_instr> <n_label> [seed]
  e.g. python make_aux.py acdata cur15k aux7k 4000 3500
"""
import json
import random
import sys
from collections import Counter

data, train_set, out_name, n_instr, n_label = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
seed = int(sys.argv[6]) if len(sys.argv) > 6 else 0
rng = random.Random(seed)

# every step's own instruction = the later decisions' history entry
instr = {}
for line in open(f"{data}/compact_train.jsonl"):
    r = json.loads(line)
    for j, h in enumerate(r["history"]):
        instr[(r["ep"], j)] = h

rows = [json.loads(l) for l in open(f"{data}/compact_{train_set}.jsonl")]
out_i, out_l = [], []
for r in rows:
    base = {"ep": r["ep"], "step": r["step"], "img": r["img"], "W": r["W"], "H": r["H"]}
    W, H = r["W"], r["H"]
    t = r.get("target")
    ins = instr.get((r["ep"], r["step"]), "").strip()
    if r["action"]["action_type"] in ("click", "long_press", "type_text") and t and ins:
        out_i.append({**base, "kind": "aux_instr", "text": ins, "target": t})
    labels = Counter(c["label"].strip().lower() for c in r["cands"] if c["label"].strip())
    ok = [c for c in r["cands"]
          if 2 <= len(c["label"].strip()) <= 40 and labels[c["label"].strip().lower()] == 1
          and (c["bbox"]["x_max"] - c["bbox"]["x_min"]) * (c["bbox"]["y_max"] - c["bbox"]["y_min"]) <= 0.2 * W * H]
    if ok:
        c = rng.choice(ok)
        out_l.append({**base, "kind": "aux_label", "text": c["label"].strip(), "target": c["bbox"]})
rng.shuffle(out_i)
rng.shuffle(out_l)
out = out_i[:n_instr] + out_l[:n_label]
rng.shuffle(out)
with open(f"{data}/compact_{out_name}.jsonl", "w") as f:
    for r in out:
        f.write(json.dumps(r) + "\n")
print(f"pools: instr {len(out_i)}, label {len(out_l)} -> wrote compact_{out_name}.jsonl: {len(out)} "
      f"({min(n_instr, len(out_i))} instr + {min(n_label, len(out_l))} label); screens all from {train_set}")
print("examples:", [(r["kind"], r["text"][:60]) for r in out[:4]])
