# How the field benchmarks decision models

Synthesis from the Decision Index reproduction kit [`apolinario/decision-index`][decisionIndexKit]
(edition 0.3, shipped with the benchmark the live board uses), cross-referenced with
[`benchmark.md`](./benchmark.md) (Decision Index semantics) and the Jev/d1 notes.

## The contract (what a decision-model benchmark asks)

A *typed decision engine* receives a `state` + a set of typed `questions` and returns **one
answer per question with a probability for every option** — no free text. The three primitives:

| primitive | meaning | example answer shape |
|---|---|---|
| `choice` | pick one of N (2–255) options | `{"type":"choice","choice":k,"probabilities":{k:p}}` |
| `noul` | yes/no, return P(yes) | `{"type":"noul","noul":p_yes}` |
| `score` | numeric score | numeric + confidence |

Every supplied option must get a probability. This is the zero-output-token contract the
category is defined by; it is also exactly what our own bracket-numeric scoring produces.

## Decision Index methodology (edition 0.3)

The live board's index = a **frozen, rebuildable multi-benchmark suite** of typed decisions,
plus private tests nobody can train on.

### 1. Frozen suite, rebuilt from pinned sources
- Suite is **not redistributed** (some sources don't allow republishing, e.g. RAGTruth contexts).
  The kit ships the **recipe**; you rebuild ~7 GB from pinned revisions and need to accept the
  **gated HLE** terms + HF login.
- Public suite = **37 scored benchmarks** across **5 areas**:
  - Knowledge & Reasoning 25.8% · Language Understanding 25.8% · Retrieval & Classification 20.0% ·
    Tools & Automation 18.3% · Arts & Human Taste 10.0%.
  - Gold ★ benchmarks weigh **1.2** inside their area.
- Benchmarks include BFCL, ToolRet, API-Bank, BANKING77, CLINC150, RouterBench, Home appliances,
  SGD, ContractNLI, ANLI, BPoMP, Humicroedit, POP909, cfcolor, GPQA Diamond, WinoGrande, HellaSwag,
  GSM8K, ChessBench, MuSR, SATA-Bench, BRIGHT, Amazon ESCI, ACOS, FinEntity, iSarcasmEval, VAST,
  NLI4CT, CRUXEval, CLadder, HLE, Habermas Machine, PhishNChips, MMLU-Pro, BBH, RAGTruth, HoVer,
  When2Call, New Yorker (several shown but not counted).

### 2. Anti-cheating rules (encoded in the runner, not left to the reader)
- **No truncation** — an engine that can't fit a request raises `Unsupported`; counts as wrong.
- **No option filtering** — every option gets a probability or the response is rejected.
- **No prompt tuning** — one fixed rendering per benchmark for every entrant.
- **Unanswered = wrong** — coverage-adjusted up front.
- **Linked cases stay whole** — multi-request cases count only if every linked request succeeded.
- **Option-leak fixes** — GSM8K rebuilt so each distractor is another problem's answer, not
  `gold ± offsets` (so rules that never read the problem score exactly 1/N); ToolRet/BRIGHT drop
  unanswerable queries; duplicate + dev-leak rows dropped.

### 3. Scoring (the index)
Per benchmark: `raw = native_score × answered/requests` (coverage first), then
**chance-corrected**: `skill = clip((raw − chance)/(1 − chance), 0, 1)` — **0 = random guessing,
100 = perfect**. Chance depends on the metric (accuracy, macro-F1, nDCG@10, Brier, case-exact
accuracy, etc.) and is computed honestly per benchmark.

Then the **Decision Index = 100 × weighted mean of the five skill areas** (gold ★ ×1.2).
`breadth_skill` uses the same weights in a geometric mean. Raw (uncorrected) index is reported
as a secondary score.

### 4. Public vs private (why the board is the hard part)
- **Full score = 20% public + 50% private-same + 30% private-new**, where each private part is
  put on the public scale using untracked stock models as the yardstick. Most of the score comes
  from tests **nobody can train on**, so a public-only advantage counts for little.
- Models whose Full scores differ by <0.9 points tie (within sampling noise).
- **Latency gate**: a model is only "Jev-like" if its median/mean/p80 latency is ≤1000 ms/request
  on one RTX PRO 6000; slower models are not added to the board.

## How Jev / d1 are measured
- Jev is the reference entrant in the kit's parity tests (57.89 on 0.2.1 public; tied with the
  board's entrants on 0.3 public per `tests/test_index03.py`).
- d1 reported **48.57 on Decision Index v0.2.1** (self-scored; no independent replication found).
- The score is a single chance-corrected number that rewards **skill and breadth**, not latency or
  calibration per se (though probability outputs are required and some benchmarks use Brier).

## Where our benchmark fits (the gap we're targeting)
The Decision Index optimizes **skill** (chance-corrected accuracy) **with coverage as a cost** but
does **not** measure **calibration-reliability** the way we do. Our harness (`benchmarks/`
`calib_metrics.py` + `cascade.py`) measures the axis DI under-measures:
- **ECE / Brier** (calibration),
- **coverage–risk curve** (selective prediction — act/ask/refuse/escalate),
- **cascade / gate** at a target risk.

So we are **not "one benchmark"**: the field's reference benchmark is the Decision Index (37
public + private + latency gate), and our harness is a **complementary calibration-reliability
probe** that can be layered on top of the same typed-decision contract.

## Plugging our model into the Decision Index
The kit's `transformers` engine scores **every option in one forward pass over a shared prompt** —
the same trick our LFM2.5 bracket-numeric scoring already does. So our on-device LFM2.5 head is
DI-ready: `python -m decision_index run --engine transformers --model mlx...` (needs the suite
rebuilt in a Python ≥3.10 env with torch/transformers, and the HLE terms accepted).

Related notes: [`benchmark.md`](./benchmark.md) · [`calibration.md`](./calibration.md) ·
[`decision-contract.md`](./decision-contract.md) · [`d1.md`](./d1.md) ·
[`action-selection-feature.md`](./action-selection-feature.md).
