# T2 brief — audience requirements

Read this, then write `outputs/.drafts/calibrated-decision-value-research-audience.md`.
Do not modify anything else in the repo.

## Question

For each of three audiences, what would make a calibrated single-pass act/ask/refuse gate matter to them —
and what evidence do they actually accept?

- **A. CCF-A reviewer** (ICML / NeurIPS / TMLR / CHI / UIST): what makes a reliability contribution
  publishable when the accuracy is a tie? Find reviewer guidance, accepted-paper patterns, and the
  difference between a mechanism paper, a benchmark paper, and a systems paper.
- **B. Builder** (mobile/agent product): what threshold, latency, and failure rate make a gate shippable?
  Find primary product/engineering docs on on-device inference, human-in-the-loop confirmation, and
  irreversible actions.
- **C. End user**: which concrete harms does an ask-before-act gate prevent (irreversible tap, wrong
  purchase, message to the wrong recipient, destructive delete, medical/automotive action), and what is the
  cost of asking (time, friction, automation bias, asking too often)?

## Find

1. Venue/reviewer standards: ICML and TMLR review criteria, NeurIPS checklist, CHI/UIST contribution types.
   Quote the actual criteria text where possible.
2. Product evidence: Apple/Google on-device ML constraints, Android/iOS permission & confirmation patterns,
   published latency budgets for on-device models, any shipped "ask before acting" UX.
3. Human-factors evidence on confirmation prompts and automation bias: when does an ask help, when does it
   cause reflex approval?
4. A concrete "value unit": what one number would a builder put in a ship/no-ship decision?

## Rules

- Primary documents only (venue pages, standards, official docs). No SEO blog summaries as sole support.
- Distinguish "a product does X" from "X was measured to help".
- If a claim is an inference, label it inference.
- Record dead links and exact search terms.

## Output format

- One section per audience: what they need, what they accept as proof, what would make them dismiss it.
- A table: Audience | Decision | Required number | Acceptable source | Current gap.
- A short list of concrete end-user harms with the cost of asking for each.
- Exact search terms used.
