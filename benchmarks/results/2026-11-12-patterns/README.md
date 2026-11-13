# Zero-shot pattern probes (v4, no training), 2026-11-12

`benchmarks/train/probe_patterns.py` on two inputs:

- **validation_v4:** 827 rows. 430 rows have a gold target element. M3 positives 634; negatives 10 (cross-episode branches that `build_recovery_data.recover_back_rows` finds in validation).
- **AndroidControl test sample:** 2,012 rows (`compact_actest_v4.jsonl` of the 380-episode AC-High sample used for gate 2). 1,224 rows have a gold target element. M3: 1,550 positives, 26 negatives.

Files:
- `probe_v4_validation.json`
- `probe_v4_actest_sample.json`
