# Harness LLM + decision model: tính khả thi, hiệu quả, và số liệu từ các nghiên cứu khác (2026-11-13)

Câu hỏi: nếu ghép một LLM (lập kế hoạch) với decision model của mình (thực thi), thì có làm được không, có lợi
không, và các nhóm khác đã đo được gì. Mọi số dưới đây lấy từ paper gốc (link ở cuối) hoặc từ file kết quả của mình;
chỗ nào là suy luận của mình thì ghi rõ.

## 1. Số liệu từ các hệ đã làm

| Hệ | Kiến trúc | Kết quả | Thời gian / chi phí | Nhận định của tác giả |
|---|---|---|---|---|
| **Jev-Mobile** (arXiv 2609.30186) | VLM (Qwen3.8-Max) ra **mục tiêu cục bộ**, quyết định xong / bị chặn, và cung cấp chữ cần gõ. Jev (decision model) đọc accessibility tree dạng text, chọn ID ứng viên do chương trình sinh, trả về DONE hoặc BLOCKED cho mục tiêu cục bộ. Không dùng ảnh. | AndroidWorld: **79%**, so với 84% khi VLM quyết từng bước (−5 điểm) và 78% của SeeAct-V | Trên các lần thành công: thời gian **−32,7%** (197 s → 133 s), chi phí API **−73,4%** ($0,274 → $0,073), thời gian bộ thực thi −94,5% | Tách "lập kế hoạch thưa" và "thực thi dày" thì giảm số lần gọi VLM mà vẫn thích nghi được. Giới hạn: mục tiêu không có trong tree thì không bấm được; chỉ gõ được ASCII; lịch sử dài cần nén. Tác giả chỉ khẳng định cho thiết kế bộ thực thi này. |
| **"When Faster Decisions Do Not Make Better Agents"** (arXiv 2610.13092), 4.500 lần chạy | Đặt Jev **vào chỗ của bước chọn hành động** trong agent có sẵn; đầu vào chỉ là text | AndroidWorld: T3A **57,5% → 44,3%** (−13,2 điểm; thắng 9, thua 55); FastAgent **50,0% → 31,9%** (−18,1); AutoDroid 18,7% → 16,4% (không có ý nghĩa thống kê) | Thời gian ra quyết định giảm 74–82% (T3A 7,24 s → 1,34 s), nhưng **thời gian cả task chỉ −10,6% (T3A) và +0,7% (FastAgent)**. Chi phí mỗi lần thành công của T3A **tăng 7,7%**. FastAgent ra quyết định nhiều hơn 44,1%. | Phần tiết kiệm ở bước quyết định bị bù lại bởi suy luận phụ và số bước nhiều hơn. **Phục hồi lỗi thất bại và báo xong sớm** làm hỏng task (ví dụ bài Wi‑Fi: báo xong khi chưa đạt). Nên đánh giá theo "task thành công", khả năng phục hồi, và tài nguyên trên mỗi task thành công. |
| **V-Droid** (arXiv 2503.15937) | Chương trình sinh ứng viên từ accessibility tree; **Llama-3.1-8B đã fine-tune làm bộ chấm** (một token Có/Không cho mỗi ứng viên). LLM điền app, chữ, câu trả lời (12,4% số hành động). GPT-4 giữ bộ nhớ làm việc. | AndroidWorld **59,5%**, AndroidLab 38,3% | **0,7 s mỗi quyết định**, 4,3 s mỗi bước. Bộ nhớ do LLM giữ chiếm 3,0 s. | "Kiểm tra dễ hơn tạo ra". Chọn trong tập hành động liệt kê được thì tránh lỗi định dạng. Bỏ bộ nhớ LLM thì SR còn 46,1% (bộ nhớ theo quy tắc) hoặc 40,0% (chỉ lịch sử). Verifier chưa fine-tune thì 0%. |
| **UGround / SeeAct-V** (arXiv 2410.05243) | GPT-4o lập kế hoạch, UGround (model grounding) chỉ ra vị trí trên ảnh | AndroidWorld: GPT-4o + UGround-V1-7B **44,0%**, so với M3A (GPT-4 chọn trên tree) 30,6%. AndroidControl 500 bước: High 48–50, Low 62–66 | – | **"Lỗi lập kế hoạch là nguyên nhân chính của thất bại trên mọi benchmark."** Bộ grounding phụ thuộc planner bên ngoài. |
| **AndroidControl** (arXiv 2406.03679) | PaLM-2S fine-tune, chỉ đọc accessibility tree | Có câu lệnh từng bước (LL): 80,7; chỉ có mục tiêu (HL): 65,5 (in-domain). Ngoài miền: khoảng 78 và 58 | – | Khoảng cách LL–HL là phần "lập kế hoạch". Ngoài miền HL giảm mạnh hơn LL nhiều. |
| **DMA-2B (của mình)** | decision model 2B, đọc ảnh + danh sách phần tử | AC-Low 89,3; AC-High 61,7. AndroidWorld 21,6% khi chạy một mình (91 task không thành công: 51 báo xong sớm, 40 hết số bước) | 135 ms mỗi quyết định; một bước emulator mất 1.624 ms. UI-TARS-2B đo trên RTX 5090: 1,7 s mỗi bước, khoảng 90 token. | – |

## 2. Các bằng chứng này nói gì

1. **Tách vai đúng thì giữ được độ thành công.** Jev-Mobile chỉ mất 5 điểm so với VLM quyết từng bước, nhưng giảm
   1/3 thời gian và 3/4 chi phí. Điều kiện để làm được: VLM giữ mọi quyết định cần suy nghĩ (chia mục tiêu, khi nào
   xong, chữ cần gõ, xử lý khi bị chặn). Decision model chỉ lo mục tiêu cục bộ.
2. **Thay thẳng LLM bằng decision model ở cùng chỗ thì thua.** Mất 13–18 điểm. Lỗi nằm đúng ở báo xong sớm và phục
   hồi kém. Đây cũng là hai lỗi chính của DMA-2B trên AndroidWorld (51 trên 91 task).
3. **Tiết kiệm ở bước quyết định không tự biến thành tiết kiệm ở cả task.** Thời gian cả task do emulator và các lần
   gọi LLM còn lại quyết định (T3A: −10,6%; của mình: model 0,135 s so với emulator 1,6 s; V-Droid: bộ nhớ LLM chiếm
   3,0 trên 4,3 s). Lợi ích thật đến từ việc **gọi LLM ít lần hơn**, không phải từ việc model chạy nhanh hơn.
4. **Planner quyết định trần thành công** (UGround: lỗi lập kế hoạch là chính). Chất lượng LLM quan trọng hơn chất
   lượng bộ thực thi, khi bộ thực thi đã đủ tốt.
5. **Chọn trong danh sách thì ổn định hơn sinh text** (V-Droid, Jev-Mobile, và AC-Low 89,3 của mình). Bộ thực thi
   phải được train cho đúng việc đó. Verifier không train thì 0% (V-Droid).

## 3. Tính khả thi với hệ của mình

| Thành phần | Đã có | Còn thiếu |
|---|---|---|
| Bộ thực thi chọn hành động + phần tử từ danh sách, có xác suất | DMA-2B: AC-Low 89,3; AUROC (xác suất hành động × phần tử) 0,812 trên AC-Low | Ngưỡng phải chọn trên tập validation (con số "80% số bước sai ≤ 5%" hiện chọn ngay trên test nên lạc quan) |
| Harness AndroidWorld (đọc màn hình, thực thi, chấm theo giao thức chính thức) | Có (bản D) | Chỗ gắn LLM: gọi LLM, truyền mục tiêu cục bộ, nhận DONE / BLOCKED |
| Mục tiêu cục bộ nhiều bước, và tín hiệu DONE / BLOCKED từ bộ thực thi | **Chưa có.** Dữ liệu AC-Low chỉ có câu lệnh cho từng bước | Phải tạo dữ liệu: gộp các câu lệnh liên tiếp trong một episode AC thành một mục tiêu cục bộ, gắn nhãn "xong mục tiêu" ở bước cuối. Đây là augmentation từ AndroidControl, không dùng emulator. |
| Chữ cần gõ, câu trả lời, khi nào xong | – | Giao cho LLM, như Jev-Mobile và V-Droid |

**Điểm mấu chốt:** nếu LLM phải viết câu lệnh cho **từng bước** (đúng kiểu AC-Low) thì LLM bị gọi ở mọi bước, và
lợi ích chỉ còn ở chỗ LLM không cần nhìn ảnh hay trả toạ độ. Muốn được lợi như Jev-Mobile (−32,7% thời gian, −73,4%
chi phí) thì bộ thực thi phải làm được **mục tiêu nhiều bước** và tự báo DONE / BLOCKED.

## 4. Kỳ vọng hiệu quả (suy luận của mình, chưa đo)

- **Độ thành công:** bị giới hạn bởi LLM planner. Theo Jev-Mobile, nên đặt mục tiêu "không thấp hơn quá 5 điểm" so
  với cùng LLM quyết từng bước. Không nên kỳ vọng vượt LLM chạy một mình.
- **Thời gian và chi phí:** giảm gần như tỉ lệ với số lần gọi LLM được bỏ đi. Tốc độ của model (135 ms) không phải yếu
  tố quyết định, vì emulator mất khoảng 1,6 s mỗi bước.
- **Rủi ro:** (1) phần tử không có trong accessibility tree; (2) bộ thực thi báo DONE sai cho mục tiêu cục bộ (cùng
  loại lỗi báo xong sớm); (3) đổi nhiều bước lấy ít lần gọi LLM (FastAgent: thêm 44% số quyết định); (4) gửi nội dung
  màn hình cho LLM bên ngoài là vấn đề riêng tư.

## 5. Đề xuất đo cho công bằng (đăng ký trước khi chạy)

So trên AndroidWorld 116 task, seed 30, **cùng một LLM** trong ba cấu hình:
(a) LLM quyết từng bước, đọc danh sách phần tử;
(b) LLM ra mục tiêu cục bộ, DMA-2B thực thi;
(c) DMA-2B chạy một mình (21,6%).

Báo cáo: tỉ lệ thành công, số lần gọi LLM mỗi task, thời gian mỗi task, chi phí mỗi task thành công, và số task báo
xong sớm. Điều kiện "đạt": (b) không thấp hơn (a) quá 5 điểm, và thời gian mỗi task giảm ít nhất 25%.

## Nguồn

- Jev-Mobile: https://arxiv.org/abs/2609.30186 (HTML: kiến trúc, Bảng thời gian/chi phí, phần giới hạn)
- When Faster Decisions Do Not Make Better Agents: https://arxiv.org/abs/2610.13092 (Bảng 2, 5, 6, 8, 10)
- V-Droid: https://arxiv.org/abs/2503.15937
- UGround: https://arxiv.org/abs/2410.05243 (bảng AndroidWorld, AndroidControl; phần kết luận)
- AndroidControl: https://arxiv.org/abs/2406.03679 (Bảng 4, 5, 7, 8; phụ lục D.1, D.3)
- Số của mình: `benchmarks/results/2026-11-12-v4/actest/` (AC-High/Low), `benchmarks/results/2026-11-12-v14targeted/`
  (AndroidWorld), `benchmarks/results/2026-11-12-calibration/`; thời gian UI-TARS-2B đo trên GPU_HOST_C, 40 bước AC-High, batch 1, bf16
