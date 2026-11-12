# Format compliance note (venue-templates workflow)

- Target: ICML 2026 main-track style, used as the format for an internal preprint. ICML 2027 instructions were not published at the time of writing.
- Official source: https://icml.cc/Conferences/2026/AuthorInstructions (checked 2026-11-12). The style package is https://media.icml.cc/Conferences/ICML2026/Styles/icml2026.zip.
- Rules from the instructions:
  - main body up to 8 pages; references and appendices unlimited, in the same PDF;
  - LaTeX only;
  - submissions must be anonymized.
- This file uses `\usepackage[preprint]{icml2026}` (the package's own option) with anonymous author placeholders.
- The class and style files are copied unchanged from the official zip. No margins, fonts or spacing are overridden.
- Build: `tectonic main.tex` (XeTeX engine, runs BibTeX itself).
