# v6 (run `final`) vs v4 — pre-registered gates, 2026-11-13

Run `final`: init v4 ckpt_2482, contract v6, data compact_cur_final (episode stop pairs kept, `curate_v5.is_stop_pair`)
+ awgen_v4, 2,313 steps x 16, lr 5e-6, seed 0. Evaluated with the same scripts and data as v4.
Numbers below are produced by `compare_v4_v6.py` (output in `compare_v4_v6.out`) and the `*.summary.json` files.

| Gate | v4 | v6 | Direction |
|---|---|---|---|
| AC-Low Type / Grounding / SR (InfiGUI scorer) | 96.6 / 90.1 / 89.3 | 96.4 / 89.4 / 88.8 | worse; paired steps 104 wins / 148 losses, sign test p = 0.007 |
| AC-High Type / Grounding / SR | 74.3 / 78.4 / 61.7 | 73.3 / 78.4 / 61.3 | tie; 442 wins / 483 losses, p = 0.19 |
| AC-High episodes with every step right | 344 / 1,543 | 323 / 1,543 | worse |
| AC-High `status` predicted (test has no correct `status` step) | 1,171 / 8,444 | 1,100 / 8,444 | slightly fewer |
| AC-High `navigate_back` predicted (gold 343) | 249 | 444 | over-predicted |
| Validation steps right | 520 / 827 | 521 / 827 | tie |
| Validation episodes all right | 14 / 137 | 19 / 137 | better (small n) |
| Validation false `status` (non-status rows) | 64 / 690 | 72 / 690 | worse |
| Validation `status` recall | 123 / 137 | 125 / 137 | tie |
| Validation back misuse | 16 / 798 | 23 / 798 | worse |

**Verdict: v6 does not replace v4.** Keeping the episode stop pairs did not reduce premature `status`
(validation false `status` rose 64 → 72; on AC-High it fell only 1,171 → 1,100), and v6 over-predicts
`navigate_back`. AC-Low is significantly worse on paired steps. The paper keeps v4 (DMA-2B) as the reported model.
v6 was therefore not run on AndroidWorld.

Not pushed: the per-step and eval `.jsonl` files (they contain dataset text with e-mail addresses); they stay local.
