# Thay self-play bằng cách nào tốt hơn (2026-11-11)

Mọi số đo của mình nằm trong `benchmarks/results/2026-11-11-v3/postmortem/`. Số liệu bài báo do một agent đọc từ trang
arXiv ngày 2026-11-11. Đó là số **của họ, trên benchmark của họ**, và phải đối chiếu với PDF trước khi đưa vào paper.

## 1. Vì sao lượt self-play vừa rồi là cách kém (đo được)
- Mỗi bài chạy **một lần**, luôn chọn phương án xác suất cao nhất. Kết quả 9/54 bài thành công, trong đó **8 bài v3 vốn
  đã giải được**, chỉ **1 bài mới** (OsmAndMarker) (`selfplay_value.out`). Học lại từ đó chủ yếu là học lại điều đã biết.
- Model tất định: cùng màn hình thì ra cùng quyết định. Chạy lại cùng cách sẽ ra cùng kết quả, nên không khám phá được gì.

## 2. Hai số đo cho thấy hướng đúng
| Đo | Số | File |
|---|---|---|
| AC-High, đáp án đúng nằm trong top-k của model | loại hành động: top1 81,0 → top2 **93,5** → top3 97,8% · phần tử click: top1 60,4 → top3 **73,9** → top5 77,1% (trần 83,3%) | `topk_actest.out` |
| Lấy một mẫu theo xác suất (T = 1) | 51,2%, **kém hơn** chọn hạng 1 (60,4%) | `topk_actest.out` |
| Độ tự tin dự báo click có làm màn hình đổi không (AndroidWorld) | theo loại hành động AUROC 0,698 · theo phần tử 0,553 (gần đoán mò) | `conf_noeffect.out` |

Ý nghĩa:
- Lỗi của model phần lớn nằm ở **lựa chọn thứ 2–3**. Thử các lựa chọn hàng đầu thì có khả năng tìm ra bài thành công mới.
- Lấy mẫu tự do thì hại, nên phải lấy mẫu **giới hạn trong top-k**.
- Độ tự tin về phần tử gần như không dự báo được gì, nên **không** nên chỉ rẽ nhánh ở các bước model kém tự tin.

## 3. Chứng cứ từ bài báo (không dùng model ngoài, chỉ dùng bộ chấm AndroidWorld)
| Bài | Số họ báo cáo | Gần với mình ở điểm nào |
|---|---|---|
| AppVLM 2502.06395 (3B) | AW: chỉ học AndroidControl 11,4 → học lại nhiều vòng từ các lần chạy được bộ chấm xác nhận (RFT) 17,9 / 23,2 / 30,5 / 35,0 → **37,8** (Tab. 2–4) | Gần nhất: xuất phát chỉ từ AC, ~11% (mình 14%), không dùng model ngoài, mỗi bài được chạy **nhiều lần** |
| UI-Voyager 2603.24533 (4B) | RFT 3 vòng: 37 → 73. GRSD (tìm bước rẽ giữa lần thất bại và lần thành công của cùng bài, rồi học bước đúng tại đó) **81,0** so với GRPO/PPO 76 | Bước rẽ tìm bằng so khớp ảnh màn hình (SSIM), không cần LLM; học chỉ bằng likelihood, hợp với các đầu chọn của mình |
| Generalization in Online RL 2603.07432 (7B) | +26,1% trên instance chưa thấy, nhưng chỉ **+15,7%** trên template chưa thấy và **+8,3%** trên app chưa thấy | **Rủi ro chính:** cái tăng chủ yếu nằm trong các template đã học |
| MobileRL 2509.18119 | 7B: SFT 56,8 → RL 72,0. Loại bỏ positive replay mất 7,5 điểm | Cần tới 256 lần chạy song song; mình có 2 máy giả lập |

Đính chính bản ghi: trong `AGENTS.md`, "StepReflect 2608.05587, 882 transition, 41,4%" là sai. Bài này tên là
**Q-Reflector**, dùng 880 transition, 41,82% âm, nhãn do GPT-5.2 tạo, và không thỏa ràng buộc "không dùng model ngoài".

## 4. Phương án đề xuất: chạy lặp, khám phá top-k, học tại bước rẽ
1. **Khám phá có giới hạn:** mỗi bài luyện tập chạy k lần. Ở mỗi bước, lấy mẫu chỉ trong top-3 của từng đầu chọn (lấy
   từ xác suất máy chủ đã trả về, không đổi model). Lần chạy thứ nhất giữ nguyên chọn hạng 1 làm mốc.
2. **Nhãn chỉ lấy từ bộ chấm AndroidWorld và từ việc màn hình có đổi hay không:**
   - lần chạy thành công: học lại các bước đã đi (imitate);
   - **bước rẽ** giữa lần thất bại và lần thành công của cùng bài: học bước đúng ở đúng màn hình đó (kiểu GRSD);
   - `status` đúng (bộ chấm nói đã xong) và `status` sai (báo xong khi chưa xong);
   - bước không đổi màn hình: học "đừng chọn".
3. **Học lại** (v4 = dữ liệu v3 + dữ liệu mới), rồi lặp lại vòng.
4. **Chống nhiễm template:**
   - để riêng một phần ba template của AndroidWorld, không bao giờ đem học;
   - không bao giờ học trên seed 30;
   - luôn báo hai con số: template đã học và template chưa học.

## 5. Cổng quyết định trước khi tốn công học lại (không cần học lại)
**Đo pass@k:** khoảng 30 bài mà v3 đang thất bại, seed luyện tập (khác 30), k = 4 lần khám phá top-3 mỗi bài. Chạy trên 2
máy giả lập, ước khoảng 3 giờ.
- **Đi tiếp** nếu số bài có ít nhất 1 lần thành công tăng **≥ 5 bài** so với chọn hạng 1. Khi đó vòng RFT có dữ liệu mới
  để học.
- **Dừng hướng này** nếu tăng < 5 bài. Khi đó thử đổi đầu vào trước: bỏ cắt 40, kiểm tra dòng Progress, và học "gõ chữ
  sau khi click ô nhập".
- **Dự đoán trước (để có thể sai):** bài có ngân sách > 20 bước vẫn 0, vì độ chính xác mỗi bước lũy thừa theo số bước.

## 6. Việc rẻ chạy song song (không cần máy giả lập)
- Kiểm tra dòng `Progress`: dùng đúng đầu vào self-play đã ghi, chỉ đổi số bước, đo xác suất `status` đổi bao nhiêu.
- Bỏ cắt 40: chạy lại 656 dòng AC-High có hơn 40 ứng viên.
