---
license: other
license_name: mixed
pretty_name: DMA-2B results
tags:
  - gui-agent
  - mobile-agent
  - android
  - androidcontrol
  - androidworld
  - evaluation
size_categories:
  - 1K<n<10K
---

# DMA-2B results: evaluation outputs and training logs

Everything measured for [DMA-2B](https://huggingface.co/tuanhqv123/DMA-2B) and its training seeds
([seed 1](https://huggingface.co/tuanhqv123/DMA-2B-seed1), [seed 2](https://huggingface.co/tuanhqv123/DMA-2B-seed2)):
the model's prediction for every evaluated step, the per-step verdicts of the scorers, the summaries every reported
number is computed from, and the training logs. Paths mirror the code repository, so every script there reads these
files where it expects them.

## Where the reported numbers come from

| Result | Files |
|---|---|
| AndroidControl-Low / High, released model (seed 0): 90.50% / 74.21% | `benchmarks/results/2026-11-16-ac-noterm/` — `eval-ours_actest{_low,}_noterm-VA-val.jsonl` (model outputs with probabilities), `perstep-ours_actest{_low,}_noterm.jsonl` (InfiGUI-R1 verdict per step), `paper_numbers.json` |
| Training seeds 1 and 2, and the seed table (Low 90.69 ± 0.23, High 73.97 ± 0.44) | `benchmarks/results/2026-11-16-ac-noterm/seeds/` and `seeds_summary.json` |
| Base model d1-2B: 59.79% / 40.16% | `benchmarks/results/2026-11-12-base/` |
| AndroidControl-Curated (Easy 89.72, Hard 77.02, Hard box 73.60) and the differential test against the authors' scorer | `benchmarks/results/2026-11-16-ac-noterm/curated_*.json` |
| Confidence, calibration and the act/ask gate | `benchmarks/results/2026-11-16-ac-noterm/conf-*-achigh-nt.jsonl`, `benchmarks/results/2026-11-12-calibration/` |
| Latency (RTX 4090, 500 AndroidControl-High steps) | `benchmarks/results/2026-11-16-latency/` |
| Earlier recipe (ablation, 89.34 / 69.73) | `benchmarks/results/2026-11-12-v4/` |
| Training logs and arguments of seeds 0, 1, 2 | `training/exec2_s{0,1,2}/` (`args.json`, `train.log`, `train.out`); run card `docs/RUNCARD-exec2_s0.md` |

Earlier experiments of the same project (AndroidWorld runs with trajectories and screenshots, probes and ablations of
earlier checkpoints) are kept under their dated folders in `benchmarks/results/`; their notes explain each run.

## File formats

- `eval-*.jsonl`: one line per step: `episode_id`, `step`, the gold and predicted action, the chosen element
  (`el_idx`) and point, the action probabilities (`probs`, `conf`) and element confidence (`el_conf`), and the latency.
- `perstep-*.jsonl`: one line per step: `episode_id`, `step`, `gt`, `pred`, `type_match`, `exact_match` (InfiGUI-R1
  scorer).
- `score_*.out`: the scorer's summary line.

## License

The model outputs and logs are released by the authors. Ground-truth actions and task text come from AndroidControl
and AndroidControl-Curated, and the AndroidWorld trajectories and screenshots from AndroidWorld's emulator tasks; use
that content under the licenses of those datasets.
