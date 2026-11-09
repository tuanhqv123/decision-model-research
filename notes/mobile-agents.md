# Mobile Agents — SOTA, Benchmarks & Our Improvable Feature

> Goal: map current mobile/on-device agent SOTA, the benchmarks that define it, then pick **one feature** our decision models can improve. Companion to `notes/agents.md` (decision models as an agent's action-selection layer).

## 1. The landscape (who builds mobile agents)

Mobile agents drive a phone by observing the screen (screenshot / UI XML) and issuing taps/swipes/types. Two big classes:

- **Cloud/multimodal agents** — powerful VLM drives the phone (highest accuracy, but slow & uploads UI state each step).
- **On-device / local agents** — privacy + low latency, but limited model capacity.

| System | Year | Idea | Source |
|---|---|---|---|
| AppAgent | 2023–24 | Multimodal agent mimics taps/swipes; no backend access; human-like action space | https://dl.acm.org/doi/full/10.1145/3706598.3713600 |
| AutoDroid | 2023 | Arbitrary tasks on any Android app; functionality-aware UI representation; exploration-based memory injection; multi-granularity query optimization | http://arxiv.org/pdf/2308.15272v3 |
| MobileUse | 2025 | Hierarchical *reflection-driven* GUI agent; long-horizon + error recovery + cold-start | https://proceedings.neurips.cc/paper_files/paper/2025/file/3994410d63ec68ce9a66011a34c9a2c4-Paper-Conference.pdf |
| CORE | 2025 | **Cloud + local LLM collaboration** to reduce UI exposure (local filters, cloud plans) | https://proceedings.neurips.cc/paper_files/paper/2025/file/636f1f78f7096caf126afaa27deb7b53-Paper-Conference.pdf |
| Hi-Agent | 2025 | Trainable hierarchical VL agent: high-level reasoning model + low-level action model, jointly optimized | https://arxiv.org/html/2510.14388v2 |
| PalmClaw | 2026 | Native on-device agent framework for phones | https://arxiv.org/html/2607.13027v1 |
| K²-Agent | 2026 | Hierarchical: declarative ("knowing what") + procedural ("knowing how") co-evolved | https://proceedings.iclr.cc/paper_files/paper/2026/file/b078eaadc927d7f7ae67371c1363195b-Paper-Conference.pdf |
| Minitap | 2026 | Multi-agent, 100% AndroidWorld (116 tasks), surpass human (80%): cognitive separation, catch silent input failures, escape action loops | https://arxiv.org/html/2602.07787 |
| UI-R1 | 2026 | **RL to enhance action prediction** of GUI agents (UI-R1-3B); rule-based RL over MLLMs | https://github.laiyagushi.com/lll6gg/UI-R1 |
| MIRAGE | 2026 | Mobile agents with **implicit reasoning + generative world models** (no long textual thoughts) | https://arxiv.org/html/2606.04627v2 |

## 2. The benchmarks (how SOTA is measured)

- **AndroidWorld** (Google DeepMind) — the *dominant* dynamic benchmark: 116 programmatic tasks across 20 real apps, deterministic reward. Now **saturating** (>90%).
- **MobileWorld** — 201 tasks / 20 apps, agent-user interactive + MCP-augmented; harder, closer to real use.
- **Mobile-Bench-v2** — VLM mobile agents; stable reward, multi-path.
- **VenusBench-Mobile** — user-intent-driven, realistic online.
- **MobiFlow** — real-world via trajectory fusion (no system-level reward).
- **A3 (Android Agent Arena)** — dynamic, live apps, gap in existing static/offline eval.
- **PhoneWorld** — reusable pipeline converting real GUI trajectories into controllable phone-use envs.
- **SimuWoB** — simulating real-world mobile apps for fast/faithful benchmarking.
- Others (earlier): AITW (Android in the Wild), ScreenSpot (grounding), MobileAgentBench, AppBench.

## 3. Where SOTA actually fails (the recurring pain points)

1. **Long-horizon reliability** — context pollution from reasoning traces, silent text-input failures, action loops (Minitap's diagnosis).
2. **Partially observable / high-dynamic state** — one screenshot per action isn't enough; need world models or implicit reasoning (MIRAGE, high-dynamic GUI).
3. **Cost + UI exposure + latency** — full UI state uploaded to cloud every step (CORE).
4. **On-device capacity gap** — local models are accurate enough but low success rate (PalmClaw, CORE).
5. **Grounding / next-action prediction** — textual reasoning makes it slow & expensive; grounding from screenshots is brittle.
6. **No calibrated confidence** — agents don't know *when they're wrong*; they keep going until failure. This is our edge.

## 4. The single feature we can improve (decision)

**Calibrated, on-device next-action selection with a confidence gate.**

Instead of a VLM emitting a long textual thought then a discrete action (slow, uncalibrated, expensive), use a **decision model** that:
- Reads the screen (screenshot / UI state) + the task + the *candidate actions* (the actionable affordances extracted/grounded), and
- Returns a **calibrated probability over each candidate action** in **one forward pass, zero output tokens** (`choice` over K affordances), plus a confidence score.

Then a **confidence gate** decides: *act* (top action, enough confidence), *ask the user* (ambiguous), *refuse/warn* (can't safely proceed), or *escalate* (hand off to a stronger cloud model). This is exactly a `choice`/`noul` decision — our core primitive.

**Why this is improvable / non-obvious:** SOTA action prediction is (a) slow because it generates text, (b) never states its own uncertainty, and (c) un-calibrated, so it cannot be safely *trusted* to hand off. Calibration lets us do **selective prediction** (act only when confident) and build a **cascade** (on-device decision model first, escalate only when uncertain) — which is the same pattern that makes decision models win on speed/cost in the Jev literature.

**Competing/harder features we are NOT picking** (for now): hierarchical long-horizon planning (K²/Minital), generative world models (MIRAGE), cross-app service-composed reliability (UIAnchor), UI-exposure reduction (CORE).

## 5. Open questions to research (next step)
- How does SOTA currently produce/predict the next action (UI-R1 RL, Hi-Agent low-level head, SoM grounding, MIRAGE implicit reasoning)? What's the input representation & loss?
- How is confidence/uncertainty (if at all) modeled for action selection today? Any conformal/ece work on GUI agents?
- Where does the K-candidate set come from (grounding extractor vs. model proposes)?
- Is there an existing "decision-model-for-GUI" prior art (e.g., a `choice` head over affordances)?

Sources: gathered 2026-10-09 via web search. See `references/INDEX.md` for the full URL record.
