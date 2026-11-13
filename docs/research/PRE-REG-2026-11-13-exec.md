# Đăng ký trước: model thực thi cho harness LLM + decision model (exec1, 2026-11-13)

Viết **trước khi train**. Lý do và bằng chứng từ các nhóm khác: `docs/research/2026-11-13-LLM-PLUS-DECISION-HARNESS-vi.md`.

## 1. Model này dùng để làm gì

Trong harness, hai bên chia việc như sau:
- **LLM:** lập kế hoạch, soạn chữ cần gõ, trả lời câu hỏi, và quyết định khi nào xong **cả task**.
- **Decision model:** nhận một **mục tiêu nhỏ** (một hoặc vài bước) rồi tự làm. Khi xong mục tiêu nhỏ, model báo `status`.

DMA-2B hiện tại chưa làm được vai trò này. Đo trên validation, khi nhận mục tiêu nhỏ:
- Model làm đúng 76,2% số bước (526/690).
- Nhưng gần như không báo xong: chỉ 12/334 lần (3,6%).
- Lý do: lúc train, `status` của model chỉ có nghĩa là "xong cả task".

## 2. Dữ liệu

Script: `benchmarks/train/build_exec_data.py`, có test ở `test_build_exec_data.py`. Mọi số dưới đây do script in ra.

**Nguồn:** toàn bộ tập train của AndroidControl. Script `build_ac_full_v2.py` dựng lại từ bản ghi gốc, kết quả:
- 13.594 episode, 88.308 dòng. 9 episode lỗi bị bỏ, 0 dòng của tập test.
- Kiểm tra chéo với 4.197 episode đã dựng hôm 11/11: 27.454/27.454 dòng khớp, 0 sai khác.

**Không có:**
- dữ liệu AndroidWorld;
- dòng `answer`;
- dòng "xong cả task";
- tăng cường no-change (chain, avoid).

**Cách chia episode.** Dùng hạt giống `seed:ep`, nên mỗi episode luôn rơi vào cùng một nhóm, dù chạy ở máy nào hay lần nào:

| Nhóm | Episode | Dòng | Nội dung |
|---|---|---|---|
| `goal` (AC-High) | 4.624 | 25.340 | Chỉ có mục tiêu chung. Mọi bước hành động đều giữ. **Không có dòng `status`.** |
| `span` | 8.970 | 49.374 | Mỗi dòng có thêm `Step instruction: <mục tiêu nhỏ>`. |
| `span_done` | (cùng 8.970) | 23.933 | Màn hình ngay sau bước cuối của mục tiêu nhỏ, nhãn `status`. |
| **Tổng** | 13.594 | **98.647** | `status` chiếm 24,3% số dòng. |

**Mục tiêu nhỏ được tạo thế nào:**
- Gộp các câu lệnh từng bước liên tiếp của AC. Một câu lệnh có thể kéo dài nhiều bước, đến khi `local_done`.
- Mỗi mục tiêu nhỏ gồm 1–4 câu lệnh, phân bố 1: 12.564, 2: 7.049, 3: 3.303, 4: 1.017.
- Các câu lệnh được nối bằng ", then".
- Những dòng nằm giữa một mục tiêu nhỏ dạy model rằng **chưa xong**.

**Hợp đồng đầu vào/đầu ra:** giữ `v4`: cùng câu hỏi, cùng cách hiển thị trạng thái, cùng ảnh 720 px. Mỗi dòng tự quy định có hiện câu lệnh hay không, qua trường `show_instruction` mà `ac_common.instruction_included` đọc.

**Không có nhãn BLOCKED:** AndroidControl không có mục tiêu nhỏ nào thất bại. Harness nhận biết bị chặn qua ba tín hiệu:
- xác suất thấp;
- màn hình không đổi;
- vượt số bước cho phép của một mục tiêu nhỏ.

## 3. Công thức train (cố định)

Các thiết lập giống v3/v4, chỉ khác dữ liệu:
- khởi tạo từ `LiquidAI/d1-2B` (revision `cfe5760`);
- 1 epoch: 98.647 dòng = **6.166 bước × 16**;
- lr 1e-5, warmup 30, cosine;
- embedding gắn (tied) đóng băng; vision tower vẫn train;
- grad-ckpt, micro 1, seed 0.

Thời gian ước tính: khoảng **13,4 giờ** trên RTX 4090 (2,05 mẫu/giây, đo ở lần train trước trên cùng máy). Mỗi 1.000 bước ghi đè một checkpoint, để chạy tiếp được nếu bị ngắt; máy train chỉ còn khoảng 9 GB đĩa trống. **Chỉ dùng checkpoint cuối.** Không chọn checkpoint và không dừng sớm theo checkpoint giữa chừng (bài học M17).

## 4. Đánh giá (cố định)

Mọi đánh giá dùng checkpoint cuối, trên **tập test AndroidControl** (1.543 episode):

1. **Vai trò thực thi**, dùng `compact_actest_exec.jsonl` với `--instruction-per-row` và chấm bằng `score_exec.py`. File này có 1.530 episode `span`; 13 episode thiếu câu lệnh bị loại. Gồm 8.364 dòng hành động và 2.513 dòng "xong"; mục tiêu nhỏ cuối mỗi episode không có màn hình sau nó nên không có dòng "xong". Các chỉ số:
   - tỉ lệ bước đúng;
   - báo xong sai (khi đang giữa một mục tiêu nhỏ, và khi ở bước cuối của nó);
   - recall của "xong";
   - AUROC của p(status).
2. **AC-Low:** chấm bằng scorer InfiGUI, 8.444 bước, có `--instruction`.
3. **AC-High:** cùng scorer, 8.444 bước, không có câu lệnh.

Ngoài ra, chạy validation (`compact_validation_exec`, 137 episode) như một kiểm tra phụ. Kết quả này không dùng để quyết định gì.

## 5. Mốc so sánh: DMA-2B (v4 ckpt_2482) trên đúng các file trên

| Chỉ số | DMA-2B |
|---|---|
| Validation, vai trò thực thi: bước đúng | 76,23% (690 dòng) |
| Validation, vai trò thực thi: recall "xong" | 3,59% (12/334) |
| Validation, vai trò thực thi: AUROC p(status) | 0,736 |
| **Test, vai trò thực thi: bước đúng** | **74,06%** (8.364 dòng) |
| Test, vai trò thực thi: báo xong sai giữa mục tiêu nhỏ / ở bước cuối | 12/5.851 / 1/2.513 |
| **Test, vai trò thực thi: recall "xong"** | **0,64%** (16/2.513); precision 55,2% (16/29) |
| Test, vai trò thực thi: AUROC p(status) | 0,679 |
| AC-Low SR / AC-High SR | 89,3 / 61,7 |

## 6. Điều kiện dùng model (một seed)

Model mới thay DMA-2B trong harness **chỉ khi đạt cả ba**:
1. **Vai trò thực thi trên test:** recall "xong" ≥ 80%, báo xong sai giữa một mục tiêu nhỏ ≤ 5%, và AUROC p(status) ≥ 0,90.
2. **AC-Low SR ≥ 88,3**, tức DMA-2B − 1,0.
3. **Tỉ lệ bước đúng trong vai trò thực thi trên test ≥ 73,06%**, tức mức của DMA-2B (74,06%) − 1,0.

Cách đọc kết quả:
- AC-High chỉ được báo cáo, không dùng làm điều kiện. Model mới không học "xong cả task", nên số AC-High chặt (không có bước kết thúc) dự kiến sẽ tăng. Nhưng một seed không đủ để khẳng định.
- Mọi chênh lệch dưới 2 điểm được ghi là "không phân biệt được với nhiễu giữa các seed" (seed spread của chương trình khoảng sd 2).

## 7. Dự đoán (để kiểm lại)

1. Recall "xong" tăng mạnh, từ 3,6%, vì đây là kỹ năng duy nhất mới được dạy.
2. Báo xong sai trên AC-High chặt giảm về gần 0, vì không còn dòng "xong cả task". Hôm 12/11, DMA-2B báo xong sai 1.171 lần trên 8.444 bước.
3. Rủi ro: model báo xong ngay ở bước cuối của mục tiêu nhỏ mà chưa thực hiện bước đó. Chỉ số "báo xong sai ở bước cuối" sẽ phát hiện lỗi này.

## 8. Thứ tự sau khi train (chưa chạy)

Chỉ khi đạt mục 6 mới làm tiếp:
1. Gắn model vào harness AndroidWorld: LLM ra mục tiêu nhỏ, decision model thực thi.
2. Chạy so sánh đã đăng ký trước ở mục 5 của tài liệu harness: cùng LLM, ba cấu hình.
