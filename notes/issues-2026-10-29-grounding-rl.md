# ISSUES & Findings — phiên 2026-10-29 (grounding gap, noise/CI, eval-set verify, trend mem/RL/CoT, compute VKong)


> **Correction (2026-11-07).** Every "64.32" below is the pre-audit figure. No artifact yields 256/398:
> the pure protocol is **64.07% (255/398)** in-process, 63.82% (254/398) through the served API, and the
> 56.50% grounding is the step-705 checkpoint. The qualitative conclusions in this note (a tie with
> GUI-Libra, not a win; grounding is the deficit) are unchanged.

> Ghi lại các **issue + phát hiện** quan trọng của phiên này để không mất, nối vào quyết định train.
> Bản chất: sau khi d1-2B full fine-tune đạt 64.07% AC-v2 High (bản gốc ghi 64.32 — đã sửa 2026-11-07), user dẫn vào loạt câu hỏi về
> "có beat GUI-Libra không / tại sao grounding thấp / có cần memory/RL/CoT không". Các câu trả lời
> được kiểm chứng bằng số liệu thật. File này ghi lại **issue + cái đã xác minh + cái còn mở**.

---

## A. Câu hỏi trung tâm: "chúng ta chỉ cần hơn GUI-Libra là được?"

### Issue A1 — Mình ĐANG NGANG GUI-Libra-8B, chưa beat (quan trọng nhất)
- **số thật:** d1-2B của mình = **64.07** (255/398, pure protocol, official `eval_rl.py`). *(Bản gốc ghi 64.32/256; không có artifact nào cho 256 — sửa 2026-11-07.)*
  GUI-Libra-8B = **64.3** (NeurIPS 2026). Mem-W-8B = 68.59 (preprint).
- **95% CI của mình = [59.5, 68.8]** (398 bước = mẫu nhỏ, CI ±4.7).
- **Vì CI bao trùm 64.3 → chỉ là "hòa trong noise", CHƯA beat.** Muốn beat thật phải **≥66–68** (đáy CI > 64.3).
- **→ Trên AC-v2 mình đang ngang, cần nâng ~5–8 điểm mới "beat ở hội nghị hạng A" (NeurIPS).**

### Issue A2 — "Hạng A trở lên" = CCF-A; baseline hạng A duy nhất trên AC-v2 là GUI-Libra
- CCF-A: NeurIPS / ICLR / ICML (các đối thủ league). Hạng A trở lên = CCF-A.
- Các baseline trên AC-v2 High:
  | Baseline | Điểm | Venue | Hạng A? |
  |---|---|---|---|
  | GUI-Libra-8B | 64.3 | **NeurIPS 2026** | ✅ |
  | MobileRL | 72.0(AW) | **ICLR 2026** | ✅ (nhưng trên AndroidWorld) |
  | Mem-W-8B | 68.59 | preprint | ❌ |
  | UI-Venus-1.5-8B | 61.06 | tech report | ❌ |
  | Qwen3-VL-2B/4B/8B | 45–59 | model card | ❌ |
- **→ "beat hạng A" = beat GUI-Libra trên AC-v2 High. Đó là scalp thật.**

---

## B. Eval-set verify (đã xác minh, KHÔNG phải ma) — tin tốt

### Issue B1 — File eval `500_steps_filtered.json` ĐÚNG LÀ AC-v2 của GUI-Libra ✅
- GUI-Libra (Appendix C.3): *"We start from a sampled set of 500 examples from UGround and obtain a
  cleaned subset of **398 examples** after filtering"* (dùng Qwen3-VL-32B-Instruct để lọc ~20% lỗi
  action-type/coordinate).
- File của mình: `benchmarks/data/acv2/500_steps_filtered.json` = **398 rows, 350 episodes**, có
  `goal` (high) + `step_instruction` (low) + `previous_actions` (natural language như
  "Go back to the previous screen") — **khớp** mô tả họ.
- **→ Mình đang dùng ĐÚNG eval set mà GUI-Libra dùng để báo 64.3. So sánh 64.07 vs 64.3 là HỢP LỆ.**
- **Đính chính:** trước đó mình nói "eval set khác" — SAI. Mình sửa lại: eval set CHÍNH XÁC;
  chỉ **train data** khác (mình dùng GCS gốc chưa refine, họ dùng AC-v2 + 81K reasoning).

### Issue B2 — 2 caveat còn lại
1. **Train data khác:** mình train trên AndroidControl GCS gốc (có ~20% lỗi annotation); GUI-Libra train
   trên AC-v2 đã lọc + 81K reasoning data. → data mình "bẩn" hơn (nhưng eval set thì cùng).
2. **Protocol:** phải chắc mình đưa `previous_actions` vào prompt (đúng protocol High). Đã sửa trước đó.

---

## C. Tại sao grounding thấp? (dữ liệu thật từ E2-938-V.jsonl)

### Issue C1 — Grounding = 56.5 (model biết LÀM GÌ nhưng bấm SAI CHỖ)
Phân tích 223 spatial rows (có toạ độ) của run E2-938:
| Chỉ số | Giá trị | Ý nghĩa |
|---|---|---|
| Type đúng | 196/223 | model biết hành động |
| Trong 196 đó, grounding SAI | **68** | biết làm gì nhưng bấm sai chỗ |
| Khoảng cách box_pt→gold | median **44px**, mean **275px** | lệch lớn |
| Bấm trong vòng 50px | 51.6% | — |
| Bấm trong vòng 100px | 58.3% | — |
| Bias dọc (y_pred−y_gold) | median **−1px** | KHÔNG lệch hệ thống |

**→ KHÔNG phải "bias có hệ thống"** (bias dọc chỉ −1px, quá cao/thấp cân đối 15%/12%). Vấn đề là
**PHÂN TÁN** — model bấm lung tung, có lúc đúng có lúc lệch cả màn hình.

### Issue C2 — "d1 vẽ bbox tốt mà sao grounding thấp?" — hiểu lầm cần làm rõ
- **Box ≠ điểm click chính xác.** Box = vùng hình chữ nhật bao element (đẹp). Grounding = **điểm click
  nằm trong *element nhỏ nhất* chứa gold** (smallest bbox node).
- d1-2B dùng **pointer/position-logit** (`pointer.py`) để sinh toạ độ click — cơ chế KHÁC hẳn vẽ bbox.
- Với element nhỏ (icon), lệch 44px (median) là **trượt hẳn** sang element khác.
- → **"Bbox đẹp" ≠ "click đúng chỗ".** Model hiểu element ở đâu nhưng hạ toạ độ không đủ chính xác.

### Issue C3 — Phân bố vị trí gold (màn 1080×2400)
- Gold y: median 1101, p25 525, p75 1990. 1/3 TRÊN (y<800): 39%, GIỮA: 26%, 1/3 DƯỚI: 35%.
- → Gold trải khắp màn hình, không tập trung → model cần grounding chính xác ở mọi vùng.

---

## D. Trend research 2026: memory / RL / CoT (chi tiết ở notes/trend-memory-rl-cot-2026.md)

### Issue D1 — CoT/reasoning: ❌ KHÔNG đáng đầu tư
- **"Does CoT Help Mobile GUI Agents?" (ACL 2026 Findings, BUPT+Tsinghua):** empirical study 6 cặp VLM.
  - Static: reasoning ~không cải thiện, thậm chí giảm (ScreenSpot −29.7%).
  - AndroidWorld: model-specific (Claude +6.3%, Gemini/Qwen3-VL giảm).
  - **Token cost tăng 3.11×–14.78×** (37.6→238.5) → phá trục "0-token" của mình.
  - Budget nhỏ (128 token) tốt nhất; nhiều hơn giảm dần.
  - Lỗi chủ yếu do benchmark (47.8% weak-eval, 14.9% static-input) → ủng hộ luận điểm audit benchmark.
- **Magma-R1 + GUI-Libra:** CoT dài *hạ* grounding → cần action-aware SFT.
- **→ BỎ CoT.** Trục của mình (quyết định trực tiếp + calibration) mạnh hơn về latency.

### Issue D2 — RL: ✅ Ưu tiên #1, bằng chứng mạnh nhất
- MobileRL (ICLR): SFT +22.6 → RL +15.2 → 72.0 AndroidWorld.
- GUI-Libra (NeurIPS): conservative GRPO + **KL trust region** (đối phó partial verifiability) → 42.6 AW.
- **Magma-R1:** GRPO + **dense 2D-Gaussian grounding reward** → 2.4k curated, 3-4B = 235B.
- Điểm chung thành công: reward **DENSE**, xử lý **partial verifiability** (KL), **class imbalance**.
- **→ Đây là đường đóng gap grounding 56.5 → 70+.**

### Issue D3 — Memory: ✅ đáng làm, nhưng PHẢI latent/multimodal, KHÔNG phải text
- Mem-W: latent memory token, +30.
- MementoGUI: working + episodic multimodal memory; **bỏ ROI image → AMS 67.69→64.22** (visual trong
  memory không thừa với text).
- CoMEM: continuous memory 8 embeddings, giữ visual cue, giảm context.
- FocusMem: factorize content/readout/trust; AndroTMem: anchored memory.
- **→ GIẢI THÍCH vì sao mình thử TF-IDF text memory bị −6.0:** text-only memory mất visual, không đủ.
  Muốn build memory thì phải latent/multimodal.

### Issue D4 — Kết luận hướng đi
| Hướng | Đáng làm | Lý do |
|---|---|---|
| RL + dense grounding reward | ✅ **#1** | Đóng gap grounding, bằng chứng mạnh nhất |
| Latent/multimodal memory | ✅ #2 | Cần cho multi-step, không thua họ |
| CoT/reasoning | ❌ Bỏ | Hạ grounding + tăng token 3-15×, không nhất quán |

---

## E. Compute: VKong (vkong.tli-tech.com) — đã probe giá

### Issue E1 — Nền tảng: VKong = cloud GPU theo giờ, $5 credit có sẵn
- CLI `vkong` đã cài (v0.10.36), đã login user `tuanhqv123`, **$5.00 credit, 0/2 machines**.
- Config trong `vkong.yaml` (type task/service, gpu, max_price, image, start, min_cpu_cores, min_ram_gb).
- **Xem giá mà không rent:** dùng `vkong dev` (in GPU+price, exit, không rent) KHÔNG phản ánh GPU thật
  (luôn Modal L4 $1.75/h). **Cách đúng:** `vkong run` in giá trước khi rent → ấn 'n'/không auto-approve.
- **Giá thật RTX-4090 = $0.44/h** (24GB, 63GB RAM, 8 core, Poland). → **$5 credit đủ smoke + thử nghiệm.**

### Issue E2 — Dữ liệu cần upload lên VKong
| Dữ liệu | Vị trí | Size |
|---|---|---|
| Eval set | benchmarks/data/acv2/500_steps_filtered.json | 33MB |
| Screenshots | benchmarks/data/acv2_shots/images | **6.4GB** |
| Train data (backup E2) | backup-2026-10-29/prd-survey-x1-5090/acwork | **26GB** |

→ Đủ disk (mình đặt `disk_gb: 10` nhưng 26GB backup >10 → cần đặt `disk_gb` đủ, hoặc chỉ copy data
crumpled cần thiết: ảnh train + compact_train.jsonl. **Phải tính cẩn thận, VKong tính tiền storage.)**

### Issue E3 — RỦI RO: d1-2B là multimodal, cần ảnh + HF download (~4GB)
- Đảm bảo vỏ máy có internet (VKong có) + đủ disk cho HF weights (d1-2B ~4GB).

---

## F. Câu hỏi MỞ cần trả lời trước khi code RL (quan trọng)

### Issue F1 — RL cho model 0-token của mình: "sampling G lần" nghĩa là gì?
- d1-2B **đọc logits ở answer slot, KHÔNG sinh token** (khác MobileRL/GUI-Libra sinh text).
- Khi làm GRPO, cần xác định: model có sample **stochastic** ở logits (categorical) không, hay chỉ greedy.
  - Nếu sample được → GRPO chuẩn (mỗi rollout ra 1 hành động khác nhau).
  - Nếu không → chỉ dùng **dense reward trong SFT** (đổi loss CE → Gaussian-weighted), không phải RL thật.
- **Cần đọc `pointer.py`/`slots.py` + test thử trước khi viết RL.**

### Issue F2 — `train_d1.py` hiện tại là SFT, có sẵn box-edge Gaussian loss
- `train_d1.py` đã có `edge` loss = CE vs Gaussian-smoothed target cho box head
  (`tgt = exp(-0.5*((grid-center)/4)^2)`) → **khớp hướng dense reward**. Có thể tái dùng.
- Nhưng RL (GRPO) chưa có; muốn RL cần thêm vòng sample + advantage + KL.

---

## G. Việc cần làm tiếp (theo thứ tự)
1. **Đọc `pointer.py`/`slots.py` + test**: d1-2B có sample stochastic không → quyết định RL thật hay
   dense-reward SFT. (Miễn phí, trên Mac.)
2. **Chọn chiến lược grounding**: RL + dense 2D-Gaussian reward (Magma-R1) — đường chính.
3. **Prepare data + script** trên Mac trước khi rent VKong (đúng nguyên tắc "không tốn tiền vô dụng").
4. **Smoke VKong**: subset ~500 mẫu, RTX-4090 ~$0.44/h, đo time/step + VRAM + grounding có nhích không.
5. Nếu smoke tốt → full; không → dừng, đổi hướng.

---

## H. Minh bạch (những gì mình KHÔNG nói được)
- **Mình CHƯA thực sự "beat" GUI-Libra** (đang ngang 64.07 vs 64.3, trong noise). Chỉ nói "cạnh tranh được".
- **Grounding 56.5 của mình thấp hơn GUI-Libra-8B (76.2)** — chỗ họ mạnh, mình cần đổ sức.
- **d1-2B số trên AC-v2 là mình tự đo**, chưa có bên thứ ba xác nhận; GUI-Libra đã qua NeurIPS.
- **Việc chọn V model ở final gần như dựa trên test eval** (validation 62.76 khớp) — phải disclose.
- **Mô hình single-pass, không RL multi-turn, không memory** → trên AndroidWorld có thể thấp hơn họ
  (chưa đo), đây là trục mình chủ động chọn để đánh vào calibration/latency.

---

## I. ĐÍNH CHÍNH sau khi kiểm chứng lại bằng dữ liệu + code (2026-10-29 chiều)
- **C1 (sai một phần):** grounding thấp KHÔNG phải do "bấm lung tung quanh đúng chỗ". Trên 71 lần trượt (đã đúng loại
  action, protocol thuần): **72% trượt xa >150px khỏi element đúng = CHỌN SAI ELEMENT** (lỗi ngữ nghĩa); chỉ 13% trượt
  gần ≤30px. Element nhỏ (<0.2% màn hình) chỉ trúng 39% vs 68% element lớn. → cần dạy "element nào", không chỉ "toạ độ".
- **C2 (sai):** điểm click KHÔNG đến từ `pointer.py`. Click = tâm của câu hỏi `box` (presence + 4 phân bố cạnh 0–999‰
  đọc qua LM head, `prompt.question_slots(Box)`); `pointer.py` chỉ đọc đoạn chữ (app_name/type_text).
- **A1:** "beat khi ≥66–68 (đáy CI > 64.3)" — tính sai: ở 68% CI ≈ [63.4, 72.6] vẫn chứa 64.3; cần ≈69+ để đáy CI > 64.3.
- **D1:** trên AC-v2, GUI-Libra Table 9 cho thấy CoT CÓ giúp (SFT 40.2→44.5; RL 48.7→55.8). Bỏ CoT với mình là vì
  thiết kế 0-token, không phải vì "CoT hạ grounding" trên AC.
- **D3:** Mem-W không "+30" trên AC-v2: +7.53 (8B) / +13.77 (4B).
- **H:** "GUI-Libra-8B grounding 76.2" — chưa có nguồn trong bảng AC-v2 (Mem-W Table 2 ghi "–") → chưa được trích.
- **E2:** KHÔNG dùng `acv2_shots` (6.4GB, bộ UGround cũ có lẫn episode TEST) làm ảnh train. Ảnh train trích thẳng từ
  GCS trên máy thuê (`benchmarks/train/extract_images.py`); chỉ upload ~400MB (compact + evalset).
- **F1 (đã trả lời):** decision model trả về TOÀN BỘ phân bố (action, 4 cạnh box) → lấy mẫu được, và tốt hơn: tính
  CHÍNH XÁC P(tâm box ∈ element đúng) bằng tích chập 2 phân bố cạnh (log-space) → tối ưu thẳng metric, không cần
  sampling/GRPO (`train_d1.center_in_range_logprob`, test `test_hitloss.py` PASS vs Monte-Carlo, gradient còn khi xa).

## J. Kế hoạch grounding đã chuẩn bị (vkjobs/acv2-ground)
- Arm A: ảnh 858px (đúng trần 1.64MP của d1; 720px cũ là dùng dưới mức) + loss expected-hit, cur15k.
- Arm B: A + grounding phụ 7.5k (mô tả bước → box, nhãn element → box; sample riêng để tránh lộ đáp án), cur15k+aux7k.
- Gate fail-fast trên máy: test_hitloss, parity, aux. Kết quả cuối chấm bằng eval_rl.py chính thức + validation.

## K. Lần chạy VKong đầu thất bại (2026-10-29 15:43–16:02) và cách làm kỹ hơn
- Lỗi: image `ubuntu:22.04` không có gcc → Triton (kernel conv của d1) không biên dịch được → chết ở gate parity, SAU
  ~15 phút trích ảnh. Tốn $0,21. Trên server cũ có sẵn gcc nên chưa từng gặp.
- Bài học: chưa từng chạy thử TRONG môi trường thuê. Đã sửa: `init.sh` cài build-essential + gate Triton ngay trong setup;
  `run.sh` có stage SMOKE (shard 0, 42+16+16 mẫu, chạy mọi đường code: hit-loss, parity, aux, 3 step train + eval 398 +
  save + eval_rl + val) TRƯỚC khi tải 47GB; md5 dữ liệu; watchdog 25 phút; timeout 5h; ảnh lưu storage dưới dạng 1 file tar.
- Phát hiện khi kiểm tra: storage báo "save not confirmed" → đã tải lại in/*.jsonl và so md5 = bản gốc E2 (khớp).
- Treo "Terminating/Save volume" ~15 phút: do lưu 15,8k file ảnh nhỏ → đổi sang 1 file tar.
