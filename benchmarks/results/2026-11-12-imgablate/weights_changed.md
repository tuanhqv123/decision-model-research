# Which parts of the model fine-tuning changed (measured 2026-11-12)

v4 `ckpt_2482/model.safetensors` against the d1-2B base snapshot (`LiquidAI/d1-2B`, both revisions), tensor by tensor
(shared names, same shape; "changed" = max |difference| > 0; relative change = ||a - b|| / ||b||, max over the group):

| Group (name match) | Changed | Max relative change |
|---|---|---|
| vision (name contains vision / siglip / visual) | 132/132 | 0.1171 |
| projector (projector / connector / mm_) | 6/6 | 0.0425 |
| language and heads (everything else) | 285/286 | 0.0466 |

The unchanged tensor is the tied embedding / LM head, frozen in training (`args.json: freeze_embed: true`; train log:
"[freeze] embeddings/LM head frozen (tied=True)"). 424 shared tensors in total. v4 weights md5 `fa8ba04b…` (identical on
the three hosts that hold a copy).
