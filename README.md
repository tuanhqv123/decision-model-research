# Decision Model Research (a.k.a. the "Jev/d1" project)

> We're building an open **decision model** — a network that answers with a *calibrated, typed decision* (choice / score / yes-no) in one forward pass, no text generation — and deploying it to mobile.

This is a research program, not a general repo. Start here, then go to `AGENTS.md` for the full map.

## What we're studying
- **Jev** (TypeSafe AI) — proprietary frontier of "System One" decision models. The target.
- **d1 / d1-3B** (Liquid AI) — open-weight decision models. The blueprint.

## The plan (phases)
1. **Understand** — Jev + d1 + the category → `notes/`
2. **Build** — post-train/fine-tune an open model into our own decision model → `models/`
3. **Benchmark** — Decision Index + a custom mobile/edge bench → `benchmarks/`
4. **Deploy** — edge/mobile (llama.cpp / CoreML / ONNX / MLX) → `mobile/`
5. **Paper** — optimize, organize, write → `paper/`

## Layout
```
AGENTS.md       master research map (read this)
notes/          per-topic notes
references/     citation ledger + bib + verbatim captures
papers/         2024→now literature
benchmarks/     evaluators / Decision Index
models/         our model design + training recipe
mobile/         edge/mobile deployment research
paper/          our paper drafts
docs/visual/    answer-me-with-html pages
```

## How we research
- `agent-browser` (Chrome) → pull primary sources verbatim into `references/raw/`.
- `subagents` → bounded parallel research per topic seam.
- `answer-me-with-html` → one-page visual explainers for relations/architecture.
- Citation rule: every claim carries a `bibkey` from `references/ledger.md`.
