# Trend 2026: Memory, RL, CoT cho GUI Agents — Pros/Cons (nối vào quyết định của dự án)

> Ngày: 2026-10-29. Tổng hợp từ primary sources (arXiv/ACL), mỗi claim có bibkey. Mục đích: trả lời câu hỏi
> của user — "CoT / RL / memory có thực sự phù hợp không, pros/cons?" — để quyết định hướng xây dựng cho
> d1-2B (single-pass decision model của dự án). Không đoán, chỉ nêu điều verify được.

---

## 1. CoT / Reasoning — KẾT LUẬN: KHÔNG đáng đầu tư cho trục của mình

### Nguồn quan trọng nhất: "Does Chain-of-Thought Reasoning Help Mobile GUI Agents?" (ACL 2026 Findings, Zhang/Gao/Xu, BUPT+Tsinghua)
Đây là **empirical study đầu tiên** (6 cặp VLM: Gemini, Claude, AgentCPM-GUI, GLM-4.5V, Qwen3-VL-4B/8B) trên
ScreenSpot + AndroidControl (static) + AndroidWorld (interactive).

**Kết quả chính — rất quan trọng cho dự án:**
1. **Static benchmarks: reasoning gần như không cải thiện, thậm chí giảm.**
   - AndroidControl: Gemini Thinking chỉ hơn 0.8% (54.4%).
   - ScreenSpot: giảm tới **29.7%** ở nhiều setup.
2. **AndroidWorld (interactive): model-specific.** Claude Thinking +6.3%, Gemini/Qwen3-VL **giảm**.
3. **Reasoning và non-reasoning THẤT BẠI trên các task KHÁC NHAU** — lợi và hại **triệt tiêu lẫn nhau**.
4. **Chi phí token tăng 3.11× → 14.78×** (ScreenSpot: 37.6 → 238.5 token) — tăng latency + tiền, không rõ lợi.
5. **Reasoning budget nhỏ (128 token) là tốt nhất**; nhiều hơn thì **giảm dần** (76.83 → 77.81 (128) → 76.90 (unconstrained)).
6. **Phần lớn lỗi do benchmark** (47.8% weak-evaluation; 14.9% static-input) **không phải do model** — ủng hộ
   luận điểm của dự án: benchmark static có lỗi, đừng đổ lỗi cho model.

### Magma-R1 (arXiv 2510.18488) — bổ trợ: CoT có thể HẠ grounding
Paper nói rõ: long CoT SFT *hurt* grounding. Giải pháp của họ là **action-aware SFT** (reweight token về
action/grounding) + RL. → Nghĩa là **CoT không phải "kim chỉ nam"**, nó còn cần được sửa.

### GUI-Libra (arXiv 2602.22190) — xác nhận cùng điều
- "standard SFT with long CoT often **hurts grounding**."
- ASFT + conservative GRPO (KL trust region) mới giữ được grounding khi có CoT.

### 🎯 Pros/Cons CoT cho DỰ ÁN
| Pros | Cons |
|---|---|
| Có thể giúp task khó / multi-step (Claude Thinking +6.3% AndroidWorld) | **Tăng token 3–15×** → phá vỡ trục "0-token, latency thấp" của mình |
| Duy trì được với budget nhỏ (128 token) | CoT dài **hạ grounding** (Magma-R1, GUI-Libra) |
| — | Kết quả **model-specific**, không nhất quán |
| — | Reasoning/non-reasoning fail trên task khác nhau → không phải "giải pháp chung" |

**KẾT LUẬN CHO DỰ ÁN:** CoT **KHÔNG phù hợp** với trục "calibrated single-pass 0-token" của mình. Nó vừa
phá latency, vừa hạ grounding, vừa không nhất quán. Trục của mình (quyết định trực tiếp, đọc logits, xác suất
calibrated) **mạnh hơn** về latency/calibration — và bằng chứng ACL 2026 cho thấy **reasoning không phải là
điều kiện cần để giỏi GUI**.

---

## 2. RL — KẾT LUẬN: ĐÂNG ĐẦU TƯ, đây là con đường có bằng chứng nhất

### Trend 2026: RL là "động cơ" chính của GUI agents
| Paper | Cách RL | Kết quả |
|---|---|---|
| **MobileRL** (ICLR 2026) | AdaGRPO online, sparse terminal reward, difficulty-adaptive replay/filter/spa | 27.6 → 50.2 (SFT) → 56.8 (reason SFT) → **72.0** (RL) |
| **GUI-Libra** (NeurIPS 2026) | conservative GRPO + **KL trust region** (đối phó partial verifiability) | base 30.4 → **42.6** AndroidWorld |
| **Magma-R1** (AC-Curated) | GRPO one-stage, **dense 2D-Gaussian grounding reward** | 2.4k curated → 3-4B = 235B |
| **UI-AGILE** (arXiv) | RL + inference-time grounding; nêu vấn đề "ineffective/sparse reward" | — |
| **HyperClick** (arXiv) | Self-critiqued RL: correctness + **confidence-alignment reward** | — |

### Điểm chung của các paper RL thành công (quan trọng):
1. **Reward phải DENSE, không binary.** Magma-R1: `exp(−‖c−p‖²/2σ²)` thay cho hit/miss. UI-AGILE/MobileRL cũng
   than "sparse reward = no effective signal".
2. **Partial verifiability là vấn đề lớn.** GUI-Libra: nhiều hành động đúng nhưng chỉ verify 1 → cần **KL trust
   region** (β=0.001) để ổn định.
3. **Class imbalance.** Magma-R1: stratified sampling theo action type.
4. **RL hiệu quả NHẤT sau SFT tốt.** MobileRL: SFT → +22.6, RL → +15.2. GUI-Libra: base 3.5 → ASFT 13.0 → +RL 25.2.

### 🎯 Pros/Cons RL cho DỰ ÁN
| Pros | Cons |
|---|---|
| Bằng chứng mạnh nhất để nâng **grounding** (Magma-R1: 2.4k → 235B) | **Online RL cần emulator + compute lớn** (MobileRL: 2k task, nhiều AVD) |
| GUI-Libra làm "offline RL" (không cần online data) → vẫn +12.2 | Cần **reward design** khó (dense, xử lý partial verifiability) |
| Nâng được **cả AC-v2 lẫn AndroidWorld** | Rủi ro overfit / mất calibration (RLVR làm confidence phồng lên ~0.99) |
| Phù hợp với việc **đóng gap grounding 56.5 → 70+** của mình | — |

**KẾT LUẬN CHO DỰ ÁN:** RL **đáng đầu tư nhất** — đặc biệt **dense grounding reward** (Magma-R1) + **KL trust region**
(GUI-Libra) để vs partial verifiability. Đây là con đường có bằng chứng để nâng AC-v2 và AndroidWorld.

---

## 3. Memory — KẾT LUẬN: ĐÁNG làm, NHƯNG phải chọn đúng cơ chế

### Trend 2026: từ text-summary → latent/continuous memory
| Paper | Cơ chế | Điểm |
|---|---|---|
| **Mem-W** (2605.09317, LV-NUS) | **latent memory token** (trajectory-to-latent compressor) — memory native vào embedding | +30.0 |
| **MementoGUI** (2605.18652) | working + episodic memory, multimodal (ROI image) | UI-Venus-1.5-8B: AMS 54.58→67.69 |
| **CoMEM / Auto-scaling** (2510.09038) | continuous memory 8 embeddings/traj (giữ visual cues) | giảm context, giữ ảnh |
| **FocusMem** (2608.04530) | factorize content/readout/trust trong latent memory | — |
| **AndroTMem** (2603.18429) | anchored memory (intermediate-state anchors) | — |
| **MemGUI-Agent** (2606.19926) | proactive context management | TSR 5.56→72.22 |

### Phát hiện quan trọng từ ablation:
- **MementoGUI**: bỏ ROI image khỏi working memory → AMS 67.69→64.22. → **Visual grounding trong memory là cần
  thiết, không chỉ là text summary.** (Điều này TRÁI với cái mình thử TF-IDF text memory — giải thích vì sao
  mình bị **−6.0**.)
- **Benchmark under-represent memory**: MemGUI-Bench chỉ 5.2–11.8% task là memory-related → **memory chưa được
  đo lường đúng** trong các benchmark hiện có.
- **Từ passive records → active task-driving states** (2606.31612): "retrieving a value does not reveal its current
  role in the workflow" — memory cần biết trạng thái, không chỉ lưu.

### 🎯 Pros/Cons memory cho DỰ ÁN
| Pros | Cons |
|---|---|
| **Cần cho multi-step** (AndroidWorld, long-horizon) | Model single-pass của mình cần **đổi cơ chế** — không đơn giản thêm prompt |
| **Latent/continuous memory** giữ visual cue, giảm context (tốt cho on-device latency) | Complexity: cần **trajectory compressor / encoder** |
| Bằng chứng mạnh: Mem-W +30, MementoGUI +13 AMS | **Memory cần được train** (self-distillation, outcome-aware supervision) |
| — | Text memory thuần (như TF-IDF của mình) **không đủ** — mình đã test, −6.0 |

**KẾT LUẬN CHO DỰ ÁN:** Memory **đáng làm** nhưng **phải là latent/multimodal**, không phải text-summary (mình
đã thử text → −6.0). Cách đơn giản nhất cho d1-2B: **compact latent memory token** (theo Mem-W) hoặc **ROI
crop** (theo MementoGUI) — để giữ visual cue. Đây là hướng để không thua trên multi-step.

---

## 4. Tổng hợp cho dự án (thứ tự ưu tiên)

| Hướng | Đáng làm? | Lý do | Cách làm |
|---|---|---|---|
| **RL + dense grounding reward** | ✅ **ƯU TIÊN #1** | Bằng chứng mạnh nhất để **đóng gap grounding 56.5 → 70+**, nâng cả AC-v2 & AndroidWorld | Magma-R1 (2D-Gaussian reward) + GUI-Libra (KL trust region) |
| **Latent/multimodal memory** | ✅ #2 | Cần cho multi-step, không thua họ; text memory đã thất bại | Mem-W latent token hoặc MementoGUI ROI crop |
| **CoT/reasoning** | ❌ **KHÔNG** | ACL 2026: không cải thiện nhất quán, hạ grounding, tăng token 3–15× phá latency | Bỏ — trục của mình là quyết định trực tiếp + calibration |

### Điểm mấu chốt cho paper
- **CoT không phải là yêu cầu để giỏi GUI** — bằng chứng ACL 2026 + Magma-R1 + GUI-Libra đều chỉ ra CoT có thể
  **hạ** grounding. Trục "0-token calibrated" của mình **mạnh hơn** về latency + calibration.
- **Cái mình cần để KHÔNG thua trên multi-step = memory (latent/multimodal) + RL (dense reward)**, KHÔNG phải
  bắt chước reasoning.
- **Benchmark static có lỗi** (ACL 2026: 47.8% weak-eval; Magma-R1: ~20% annotation lỗi) → ủng hộ nội dung paper
  về việc audit benchmark.

---

## Bibkey / Source

- [cotACL2026] Zhang, Gao, Xu. *Does Chain-of-Thought Reasoning Help Mobile GUI Agents? An Empirical Study.* Findings of ACL 2026. arXiv 2503.16788 / aclanthology.org/2026.findings-acl.392
- [magmaR1] *AndroidControl-Curated + Magma-R1.* arXiv 2510.18488.
- [GUIlibra] Rui Yang et al. *GUI-Libra.* arXiv 2602.22190 (NeurIPS 2026), gui-libra.github.io
- [MobileRL] *MobileRL: Online Agentic RL for Mobile GUI Agents.* arXiv 2509.18119 (ICLR 2026)
- [MemW] Guibin Zhang, ... Shuicheng Yan. *Mem-W: Latent Memory-Native GUI Agents.* arXiv 2605.09317
- [MementoGUI] *MementoGUI: Learning Agentic Multimodal Memory Control.* arXiv 2605.18652
- [CoMEM] Wu et al. *Auto-scaling Continuous Memory for GUI Agent.* arXiv 2510.09038
- [FocusMem] Zhang et al. *FocusMem: Factorizing Content, Readout, Trust.* arXiv 2608.04530
- [AndroTMem] *AndroTMem.* arXiv 2603.18429
- [MemGUI-Bench] *MemGUI-Bench.* arXiv 2602.06075
- [UIAGILE] *UI-AGILE: Advancing GUI Agents with Effective RL.* arXiv 2507.22025
- [HyperClick] *Enhancing Trustworthy GUI Grounding via Self-Critiqued RL.* arXiv 2510.27266
- [MemGUI-Agent] *MemGUI-Agent: End-to-End Long-Horizon Mobile GUI Agent.* arXiv 2606.19926
- [TSR] *A Task-State Representation for Long-Horizon Mobile GUI Agents.* arXiv 2607.00502
- [WhatMem] *What Memory Do GUI Agents Really Need? From Passive Records to Active Task-Driving States.* arXiv 2606.31612
