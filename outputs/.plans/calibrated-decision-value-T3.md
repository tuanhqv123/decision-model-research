# T3 brief — production failure modes

Read this, then write `outputs/.drafts/calibrated-decision-value-research-failure-modes.md`.
Do not modify anything else in the repo.

## Question

Under what conditions does a calibrated act/ask/refuse gate silently become unsafe — and has anyone
published a method that survives it?

## Find

1. **Distribution shift and threshold non-transfer**: conformal prediction under covariate shift, label
   shift, checkpoint change, and population change. Which guarantees are conditional and which are
   assumption-free? Key results with citations (e.g. Angelopoulos & Bates; SCRC; Mondrian/group-conditional
   conformal; adaptive conformal inference).
2. **Correlated / clustered errors**: why exchangeability fails when rows share episodes, users, or tasks;
   how clustering changes effective sample size and coverage.
3. **Base-rate change and selective risk**: what happens to a fixed threshold when prevalence moves.
4. **Measurement artefacts**: ambiguous ground truth, multiple valid actions, scorer gaming, box/point
   tolerance — and how much of a reported "error rate" is the metric rather than the model.
5. **Monitoring / re-certification**: published methods for detecting gate degradation in deployment
   (online coverage audits, drift detection, periodic re-fit) and the evidence they work.

## Rules

- Prefer peer-reviewed; label preprints.
- For every guarantee, state the assumption it needs. A guarantee without an assumption is a red flag.
- Record exact numbers (coverage/risk deltas) with the setting they were measured in.
- If a remedy is proposed but not measured, say so.
- Record dead links and exact search terms.

## Output format

- Failure-mode table: Mode | What breaks | Published evidence | Remedy | Remedy evidence.
- A short "what a deployer must monitor" list derived only from cited work.
- A blunt paragraph: is a per-checkpoint certificate defensible as a product property? (Answer may be no.)
- Exact search terms used.
