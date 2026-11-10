# T5 — Training-recipe ingredient checklist (mechanical extraction)

Scope: what each paper *literally lists*. No inference, no verdicts. `no` = the ingredient is not mentioned in the abstract or the fetched text; where text exists on an adjacent-but-different concept it is quoted and labelled `(closest text)`.

| Paper (arXiv id) | Ingredient | Present? | Verbatim evidence (<=20 words) | URL |
|---|---|---|---|---|
| MAI-UI (2512.22047) | multi-step trajectory data | yes | "we build a self-evolving data pipeline to collect and synthesize diverse multi-step trajectories" | https://arxiv.org/html/2512.22047v1 |
| MAI-UI (2512.22047) | interaction history / previous-actions state rendering | yes | "denotes the history context of previous actions and observations" | https://arxiv.org/html/2512.22047v1 |
| MAI-UI (2512.22047) | action-space size stated | yes | "click, long_press, type, swipe, drag, system_button, wait, terminate, answer, ask_user, mcp_call" (Table 1, 11 rows) | https://arxiv.org/html/2512.22047v1 |
| MAI-UI (2512.22047) | resolution / aspect-ratio handling | yes | "we resize images to half their original resolution" | https://arxiv.org/html/2512.22047v1 |
| MAI-UI (2512.22047) | coordinate representation | unclear | "Leveraging the relative coordinate mechanism of Qwen3-VL" (range not stated) | https://arxiv.org/html/2512.22047v1 |
| MAI-UI (2512.22047) | dedicated grounding loss | yes | "a combination of a format reward and a point-in-box reward" | https://arxiv.org/html/2512.22047v1 |
| MAI-UI (2512.22047) | negative or contrastive samples | no | not mentioned in abstract or fetched text | https://arxiv.org/html/2512.22047v1 |
| MAI-UI (2512.22047) | rejection sampling / data filtering | yes | "Newly generated trajectories are filtered via rejection sampling, retaining only high-quality examples" | https://arxiv.org/html/2512.22047v1 |
| MAI-UI (2512.22047) | MLLM judge / automated verification | yes | "examined by an MLLM-as-a-judge module" | https://arxiv.org/html/2512.22047v1 |
| MAI-UI (2512.22047) | progress or termination supervision | yes | "terminate: Marks the task as complete with status (success or fail)" | https://arxiv.org/html/2512.22047v1 |
| MAI-UI (2512.22047) | per-task-type balancing | no | (closest text) "Tasks are dynamically stratified into four difficulty levels" — difficulty, not task type | https://arxiv.org/html/2512.22047v1 |
| MAI-UI (2512.22047) | number of seeds / repeated runs reported | no | not mentioned in abstract or fetched text | https://arxiv.org/html/2512.22047v1 |
| MobileRL (2509.18119) | multi-step trajectory data | yes | "An episode yields a trajectory τ = ((s0,a0),(s1,a1),…,(sT,aT))" | https://arxiv.org/html/2509.18119v2 |
| MobileRL (2509.18119) | interaction history / previous-actions state rendering | no | (closest text) "the agent observes only the current state s_t" — history appears only in AndroidControl-High eval | https://arxiv.org/html/2509.18119v2 |
| MobileRL (2509.18119) | action-space size stated | yes | "Tap, Type, Swipe, Long Press, Launch, Back, Home, Wait, Finish" (Table 5, 9 rows) | https://arxiv.org/html/2509.18119v2 |
| MobileRL (2509.18119) | resolution / aspect-ratio handling | yes | "All images were used at their full resolution without compression." | https://arxiv.org/html/2509.18119v2 |
| MobileRL (2509.18119) | coordinate representation | yes | "represents screen positions with bounding boxes aligned to XML data" | https://arxiv.org/html/2509.18119v2 |
| MobileRL (2509.18119) | dedicated grounding loss | no | not mentioned in abstract or fetched text | https://arxiv.org/html/2509.18119v2 |
| MobileRL (2509.18119) | negative or contrastive samples | yes | "we maintain a maximum positive-to-negative trajectory ratio of 1:2" | https://arxiv.org/html/2509.18119v2 |
| MobileRL (2509.18119) | rejection sampling / data filtering | yes | "any task producing all-zero rewards for two consecutive epochs enters a three-epoch cooldown" | https://arxiv.org/html/2509.18119v2 |
| MobileRL (2509.18119) | MLLM judge / automated verification | yes | "paired with verifiable rewards and VLM-based reward models" | https://arxiv.org/html/2509.18119v2 |
| MobileRL (2509.18119) | progress or termination supervision | yes | "reshapes the reward function based on completion length, assigning higher returns to shorter solutions" | https://arxiv.org/html/2509.18119v2 |
| MobileRL (2509.18119) | per-task-type balancing | no | (closest text) "down-weights persistently unsolvable tasks" — difficulty, not task type | https://arxiv.org/html/2509.18119v2 |
| MobileRL (2509.18119) | number of seeds / repeated runs reported | yes | "All test set results are averaged over three runs to mitigate the impact of randomness." | https://arxiv.org/html/2509.18119v2 |
| GUI-Libra (2602.22190) | multi-step trajectory data | yes | "we therefore aggregate trajectory data from multiple public sources that cover both web and mobile domains" | https://arxiv.org/html/2602.22190v2 |
| GUI-Libra (2602.22190) | interaction history / previous-actions state rendering | yes | "the interaction history (previous actions), and the current screenshot" | https://arxiv.org/html/2602.22190v2 |
| GUI-Libra (2602.22190) | action-space size stated | yes | "We consider 13 common action types for web and mobile control" | https://arxiv.org/html/2602.22190v2 |
| GUI-Libra (2602.22190) | resolution / aspect-ratio handling | no | not mentioned in abstract or fetched text | https://arxiv.org/html/2602.22190v2 |
| GUI-Libra (2602.22190) | coordinate representation | yes | "normalized coordinates in [0,1000] for Qwen3-VL-based models" (absolute pixels for Qwen2.5-VL) | https://arxiv.org/html/2602.22190v2 |
| GUI-Libra (2602.22190) | dedicated grounding loss | unclear | "applies action-aware token reweighting to emphasize action and grounding tokens" (no separate loss term) | https://arxiv.org/html/2602.22190v2 |
| GUI-Libra (2602.22190) | negative or contrastive samples | no | (closest text) "success-adaptive scaling to downweight unreliable negative gradients" — gradients, not samples | https://arxiv.org/html/2602.22190v2 |
| GUI-Libra (2602.22190) | rejection sampling / data filtering | yes | "We discard steps with re-prediction accuracy below 0.3" | https://arxiv.org/html/2602.22190v2 |
| GUI-Libra (2602.22190) | MLLM judge / automated verification | yes | "prompt Qwen3-VL-32B-Instruct to predict a bounding box … keep a step only if" | https://arxiv.org/html/2602.22190v2 |
| GUI-Libra (2602.22190) | progress or termination supervision | yes | "13 common action types … Click, Write, Terminate, Swipe, Scroll" — Terminate is a supervised action | https://arxiv.org/html/2602.22190v2 |
| GUI-Libra (2602.22190) | per-task-type balancing | no | (closest text) "we downsample early steps … and further downsample mobile-domain trajectories" | https://arxiv.org/html/2602.22190v2 |
| GUI-Libra (2602.22190) | number of seeds / repeated runs reported | yes | "we report the average across four runs for all experiments" | https://arxiv.org/html/2602.22190v2 |
| AndroidControl-Curated (2510.18488) | multi-step trajectory data | unclear | "post-trained on just 2.4k curated samples" — step vs trajectory unit not stated | https://arxiv.org/html/2510.18488v1 |
| AndroidControl-Curated (2510.18488) | interaction history / previous-actions state rendering | no | not mentioned in abstract or fetched text | https://arxiv.org/html/2510.18488v1 |
| AndroidControl-Curated (2510.18488) | action-space size stated | no | not mentioned in abstract or fetched text | https://arxiv.org/html/2510.18488v1 |
| AndroidControl-Curated (2510.18488) | resolution / aspect-ratio handling | no | not mentioned in abstract or fetched text | https://arxiv.org/html/2510.18488v1 |
| AndroidControl-Curated (2510.18488) | coordinate representation | unclear | "for each interactive UI element e_k on screen I, we can obtain its minimal bounding box B_k" | https://arxiv.org/html/2510.18488v1 |
| AndroidControl-Curated (2510.18488) | dedicated grounding loss | yes | "we design a smooth reward function based on a 2D Gaussian kernel" | https://arxiv.org/html/2510.18488v1 |
| AndroidControl-Curated (2510.18488) | negative or contrastive samples | no | not mentioned in abstract or fetched text | https://arxiv.org/html/2510.18488v1 |
| AndroidControl-Curated (2510.18488) | rejection sampling / data filtering | yes | "trained on only 2,400 high-quality samples selected from AndroidControl-Curated" | https://arxiv.org/html/2510.18488v1 |
| AndroidControl-Curated (2510.18488) | MLLM judge / automated verification | yes | "We leverage a large language model (LLM) as an Automated Reviewer" | https://arxiv.org/html/2510.18488v1 |
| AndroidControl-Curated (2510.18488) | progress or termination supervision | no | not mentioned in abstract or fetched text | https://arxiv.org/html/2510.18488v1 |
| AndroidControl-Curated (2510.18488) | per-task-type balancing | yes | "we propose an Action Type Proportional Optimization strategy" | https://arxiv.org/html/2510.18488v1 |
| AndroidControl-Curated (2510.18488) | number of seeds / repeated runs reported | no | not mentioned in abstract or fetched text | https://arxiv.org/html/2510.18488v1 |
| AndroidWorld (2405.14573) — env side only | multi-step trajectory data | no | not mentioned in abstract or fetched text | https://arxiv.org/html/2405.14573v5 |
| AndroidWorld (2405.14573) — env side only | interaction history / previous-actions state rendering | no | not mentioned in abstract or fetched text | https://arxiv.org/html/2405.14573v5 |
| AndroidWorld (2405.14573) — env side only | action-space size stated | no | (closest text) "The action space includes: … gestures … Text input … Navigation" — no count given | https://arxiv.org/html/2405.14573v5 |
| AndroidWorld (2405.14573) — env side only | resolution / aspect-ratio handling | yes | "The screenshot resolution is 2400×1080×3" | https://arxiv.org/html/2405.14573v5 |
| AndroidWorld (2405.14573) — env side only | coordinate representation | yes | "Click-based actions (click, long press): Simulates touch events at specified coordinates" | https://arxiv.org/html/2405.14573v5 |
| AndroidWorld (2405.14573) — env side only | dedicated grounding loss | no | not mentioned in abstract or fetched text | https://arxiv.org/html/2405.14573v5 |
| AndroidWorld (2405.14573) — env side only | negative or contrastive samples | no | not mentioned in abstract or fetched text | https://arxiv.org/html/2405.14573v5 |
| AndroidWorld (2405.14573) — env side only | rejection sampling / data filtering | no | not mentioned in abstract or fetched text | https://arxiv.org/html/2405.14573v5 |
| AndroidWorld (2405.14573) — env side only | MLLM judge / automated verification | no | (closest text) "reward signals primarily by managing application state using the Android Debug Bridge (adb)" | https://arxiv.org/html/2405.14573v5 |
| AndroidWorld (2405.14573) — env side only | progress or termination supervision | yes | "each task includes dedicated initialization, success-checking, and tear-down logic" | https://arxiv.org/html/2405.14573v5 |
| AndroidWorld (2405.14573) — env side only | per-task-type balancing | no | not mentioned in abstract or fetched text | https://arxiv.org/html/2405.14573v5 |
| AndroidWorld (2405.14573) — env side only | number of seeds / repeated runs reported | yes | "We set the seed to 30 and the temperature to 0" | https://arxiv.org/html/2405.14573v5 |

## Source quality

| Paper (arXiv id) | What was read | URL read |
|---|---|---|
| MAI-UI (2512.22047) | Abstract; then full text (all 5 sections + 3.5/3.6 analysis; appendix not present in HTML) | https://arxiv.org/abs/2512.22047 + https://arxiv.org/html/2512.22047v1 |
| MobileRL (2509.18119) | Abstract; then full text incl. Appendices B.1–B.3 (hyperparameters, action space, image resolution) and D.5/E | https://arxiv.org/abs/2509.18119 + https://arxiv.org/html/2509.18119v2 |
| GUI-Libra (2602.22190) | Abstract; then full text incl. §4.1.4–4.1.5 (filtering), §5.3, Appendix B (action space, coordinate systems), Appendix C.3 | https://arxiv.org/abs/2602.22190 + https://arxiv.org/html/2602.22190v2 |
| AndroidControl-Curated (2510.18488) | Abstract; then full text (all of §1–§4; paper is 5 sections, ~560 HTML lines) | https://arxiv.org/abs/2510.18488 + https://arxiv.org/html/2510.18488v1 |
| AndroidWorld (2405.14573) | Abstract; then full text incl. §3.2, Appendix C.1–C.2 (observation/action space), §4.2 | https://arxiv.org/abs/2405.14573 + https://arxiv.org/html/2405.14573v5 |

**Abstract-only evidence:** none. Every table cell above is sourced from the full-text HTML URL listed in its own row; the abstract pages were read for all five papers but are not the sole source for any cell.

**Notes on cell counts (all from the cited HTML):** MAI-UI action table = 11 rows; MobileRL action table = 9 rows; GUI-Libra = "13 common action types" in text.
