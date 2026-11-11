# Training throughput on the 24 GB 4090 (v2a state, 720 px, measured 2026-11-11)

| micro x accum | grad-ckpt | length bucketing | samples/s | peak GB | note |
|---|---|---|---|---|---|
| 1 x 16 | on | — | **2.36–2.38** | 11.1 | used |
| 4 x 4 | on | no | 2.02 | 14.3 | padded width ~2,840 tokens vs 2,260 per sample |
| 4 x 4 | on | yes (`--bucket`) | 1.96 | 14.3 | padded width 2,515 — padding reduced, still slower |
| 8 x 2 | on | yes | OOM | — | |
| 1 x 16 / 2 x 8 / 4 x 4 | off | — | OOM | — | activations do not fit without checkpointing |

d1's own runner also right-pads (`runner._rows`) and groups similar lengths (`_batches`); only attention runs
unpadded (`d1::varlen`). Conclusion: on this card the step is compute-bound at micro 1; `--bucket` stays as an option
(off by default). A ~2x speed-up needs fewer tokens per sample (image size, element cap) or another GPU.
