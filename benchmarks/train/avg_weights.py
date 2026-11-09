"""Mean of replica safetensors (fp32 accumulate, bf16 out). Usage: avg_weights.py out.safetensors in1 in2 ..."""
import sys
import torch
from safetensors.torch import load_file, save_file
out, ins = sys.argv[1], sys.argv[2:]
acc = None
for p in ins:
    sd = load_file(p)
    if acc is None:
        acc = {k: v.float().clone() for k, v in sd.items()}
    else:
        assert acc.keys() == sd.keys(), "replica key mismatch"
        for k, v in sd.items():
            acc[k] += v.float()
save_file({k: (v / len(ins)).to(torch.bfloat16).contiguous() for k, v in acc.items()}, out)
print(f"averaged {len(ins)} replicas -> {out} ({len(acc)} tensors)")
