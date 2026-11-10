# T3 — AndroidLab protocol (mechanical extraction)

Primary source verified: **arXiv:2410.24024** resolves to *"AndroidLab: Training and Systematic Benchmarking of Android Autonomous Agents"* (Xu et al., v2, 4 Nov 2024, cs.AI). Repository verified live: **https://github.com/THUDM/Android-Lab** (MIT, 328 stars, 6 commits).

Sources fetched:
- Abstract: https://arxiv.org/abs/2410.24024
- Full text (HTML v2): https://arxiv.org/html/2410.24024v2
- Repo README: https://github.com/THUDM/Android-Lab (raw: https://raw.githubusercontent.com/THUDM/Android-Lab/main/README.md)
- Linux/Docker setup doc: https://raw.githubusercontent.com/THUDM/Android-Lab/main/docs/prepare_for_linux.md

| Requirement | What the source says (verbatim, <=25 words) | Where (URL) |
|---|---|---|
| 1. Total task count | "AndroidLab benchmark includes predefined Android virtual devices and 138 tasks across nine apps built on these devices." | https://arxiv.org/abs/2410.24024 |
| 2. Task categories / app coverage | "we design 138 tasks, including 93 Operation Tasks and 45 Query Tasks." Apps: "Bluecoins", "Calendar", "Cantook", "Clock", "Contacts", "Maps.me", "PiMusic", "Setting", "Zoom"; per-app counts 15/14/12/27/15/15/12/23/5 | https://arxiv.org/html/2410.24024v2 (Table 4) |
| 3. Official success metric | "The task is considered successfully executed when all sub-goals are completed." Query tasks: "advanced LLMs verify if the model's predicted results match the standard answers" | https://arxiv.org/html/2410.24024v2 (§3.2.3) |
| 4. Step budget | "we set a maximum execution step limit of 25 for each task, with a 3-second interval" | https://arxiv.org/html/2410.24024v2 (§5.1) |
| 5. Whether runs are seeded | NOT STATED. Nearest determinism statement: "We use ADB commands at the start of each evaluation to set the machine's time and virtual geolocation to predetermined values." | https://arxiv.org/html/2410.24024v2 (§3.2.2) |
| 6. Number of runs per task reported | NOT STATED as a count. "We generate by greedy search for each task of all models." No repeat/trial count given | https://arxiv.org/html/2410.24024v2 (§5.1) |
| 7. Emulators and parallelism | Uses AVDs: "the pre-packaged AVD (Android virtual device) image". Paper gives no parallel count; repo: "We support parallel testing... Each concurrent session takes up approximately 6G of memory and 9G of storage space." | https://arxiv.org/html/2410.24024v2 ; https://github.com/THUDM/Android-Lab |
| 8. Text-only and multimodal variants reported separately | "We design the SoM mode for the multimodal models (LMMs) and the XML mode for the text-only models (LLMs)" — separate result rows per mode | https://arxiv.org/html/2410.24024v2 (Table 1) |
| 9. Published results for any model <= 4B params | NOT STATED. No model of <=4B params appears in Table 1; the smallest reported is Qwen2-7B (XML 4.35 SR, SoM 3.62 SR as Qwen2-VL-7B-Instruct) | https://arxiv.org/html/2410.24024v2 (Table 1) |

## Relation to AndroidWorld

Only statement found in the fetched sources (AndroidLab paper, §2 Related Work):

> "AndroidWorld [Rawles et al. (2024)] offers reward signals for 116 tasks across 20 real-world apps but does not support instruction-tuning data construction."

URL: https://arxiv.org/html/2410.24024v2

No other fetched source (repo README, repo Linux setup doc, abstract page) states a relation to AndroidWorld.
