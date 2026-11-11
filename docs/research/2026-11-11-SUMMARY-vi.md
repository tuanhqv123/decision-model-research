# Tổng hợp ngày 2026-11-11 — vì sao điểm thấp, đã quyết gì, đang huấn luyện gì, sai ở đâu và đã sửa thế nào

Viết cho người đọc chưa theo dõi cuộc trao đổi. Số liệu nào cũng có file nguồn đi kèm.

## 1. Đang chạy

**Huấn luyện v2a bản 1** trên GPU 4090 của máy huấn luyện, bắt đầu 12h57 (giờ VN), dự kiến xong 14h55; sau đó tự đánh giá trên
`validation_v2` (827 bước AndroidControl) và `awgen_heldout` (463 dòng của 19 dạng bài không cho học). Script: `~/acv2/run_v2a_s0.sh`.
Xong sẽ: chép checkpoint sang máy phục vụ, thử một yêu cầu thật ("Turn wifi on." trên màn hình chính → phải ra `open_app settings`),
rồi chạy 116 bài thi (`benchmarks/aw/hostscripts/stage2_chain_v2a.sh`). Bản đăng ký trước: `docs/research/PRE-REG-2026-11-11-v2a.md`.

## 2. Vì sao R1 chỉ đạt 12–15% (đo được) — `docs/research/2026-11-11-DIAGNOSIS-why-low.md`

| # | Nguyên nhân | Số liệu |
|---|---|---|
| 1 | 25 bài hỏi thông tin cần hành động `answer`, mô hình không có | luôn 0/25 |
| 2 | Tên app chép từ đề; đề không nêu tên app → mở `wifi`, `timer`, `''`… | 29–36% lần mở app không mở được; 24–25 bài không bao giờ mở được app |
| 3 | Mô hình đoán toạ độ từ ảnh, không chọn phần tử trong cây a11y | 30–37% cú bấm rơi vào phần tử không bấm được |
| 4 | Báo "xong" sớm (thường sớm đúng 1 bước) | 21–26 bài; 103/171 lần báo xong sai trên `val` là ngay trước bước cuối |
| 5 | Học bước lẻ, lịch sử luôn đúng, không có ví dụ sửa sai | kể cả cho lịch sử đúng, chỉ 15–17/137 bài `val` đúng hết mọi bước; 0/21 bài ≥ 9 bước |

## 3. Đã thống nhất

* **Một mô hình, học bắt chước, không sinh chữ, không mô hình ngoài** (kể cả lúc tạo dữ liệu).
* **Đầu vào mỗi bước:** đề + **1 ảnh màn hình hiện tại** (720 px) + lịch sử dạng chữ + danh sách phần tử hiện tại (≤ 40, giữ 127/130
  nút đúng trên màn hình AndroidWorld thật) + danh sách app của máy. Không đưa tên bài, độ phức tạp, số bước tối đa.
* **Đầu ra (chọn, không sinh chữ):** loại hành động (9), phần tử (số thứ tự), app (trong danh sách), chiều cuộn; chữ cần gõ chép từ
  đề (98.4% chữ gõ của AndroidControl có nguyên văn trong đề).
* **Định dạng khớp AndroidWorld:** đối chiếu mã gốc (`json_action.py`, `actuation.py`, `suite_utils.py`); chiều cuộn AndroidControl ↔
  AndroidWorld khớp (đo trên 158 dòng); ngữ cảnh mô hình 128k token, đầu vào thật ~2,300 token. Hợp đồng: `docs/research/CONTRACT-v3-io.md`.
* **Được dùng "bài luyện tập cùng dạng"** (cùng 116 mẫu đề, tham số khác) — README/bài báo AndroidWorld cho phép, không có luật cấm;
  không bao giờ dùng 116 bài thi; ghi rõ; giữ 1/3 dạng bài không học để đo khả năng làm dạng mới.
* **Danh sách app** lấy từ chính máy giả lập (`pm list packages` ∩ tên AndroidWorld mở được) — thông tin của máy, không phải của bài.
* **Bước đầu chỉ được "mở app" hoặc "báo xong"** — luật cố định của agent, không dùng thông tin bài; ghi rõ.
* **Thêm 2 ảnh cũ: để sau**, khi đã có điểm của mô hình đầu tiên.
* **Dữ liệu chỉ gồm AndroidControl + AndroidWorld.** AMEX (bạn quyết bỏ), AndroidLab (không có ảnh sạch, máy thật giao diện tiếng Trung),
  OS-Atlas (ảnh có lớp đo cảm ứng in sẵn, giao diện nhiều ngôn ngữ, ~3 phần tử/màn hình) đều đã mở file kiểm và bỏ.

## 4. Đang huấn luyện: v2a

| | |
|---|---|
| Hợp đồng | `v2a` = chọn phần tử từ cây a11y + chọn app từ danh sách (+ luật bước đầu ở máy chủ) |
| Dữ liệu | `cur15k_v2` (14,965 bước AndroidControl, có cây phần tử kiểu AndroidWorld) + `compact_awgen_v2a` (1,736 dòng "đề → app" sinh bằng bộ sinh đề AndroidWorld trên ảnh màn hình chính, không chạy bài) |
| Giữ lại | 19/80 dạng bài một app không cho học (`benchmarks/aw/awenv/heldout_templates.json`) |
| Công thức | như R1: 1 vòng (1,044 bước × 16), lr 1e-5, warmup 30, embedding đóng băng, phần đọc ảnh vẫn học |
| Tốc độ | 2.36 mẫu/giây, 11.1 GB — cấu hình nhanh nhất đo được trên GPU này (xem §6) |
| Code | `benchmarks/train/ac_common.py` (`v2a`, `app_options`, `row_app_options`, `restrict_first_step`, `decode_app_choice`), `train_d1.py`, `eval_d1.py`, `serving/d1_api.py`, `benchmarks/aw/hostscripts/make_aw_app_rows.py`; test `benchmarks/train/test_app_choice.py` (11) + harness 249 pass |

Chưa giải quyết trong v2a: 25 bài hỏi thông tin (không có dữ liệu `answer` khi không chạy giả lập), báo xong sớm, bài nhiều bước.

## 5. Sai sót phát hiện hôm nay và cách sửa

| Sai | Phát hiện thế nào | Sửa |
|---|---|---|
| Giả lập chết giữa đợt; 10 bài sau bị ghi là "đã chấm" | đọc kết quả từng bài của R1 bản 3 | `sweep_infra.py` (+4 test): dời bài chết sang thư mục riêng, chạy lại; bài "hỏng lây" luôn chạy lại; lưu nhật ký giả lập |
| Đợt chạy treo cả giờ khi giả lập chết | log không có bài mới 60+ phút | `watchdog_runbench.sh` |
| `preflight.py` đếm cả file đã loại; `rebuild_index.py` ghi tên bài rỗng | lỗi khi chạy lại | xử lý bằng tay, ghi vào sổ chạy |
| "AndroidControl bài nào cũng kết thúc bằng xong" | mở file gốc: 83,848 hành động, 0 `status` | dòng `status` là do `preprocess_ac.py:126-128` của chúng ta thêm; đã sửa hồ sơ |
| "AndroidControl ~22% bài cùng loại AndroidWorld", "không có bài Wi-Fi" | đếm trên toàn bộ 13,603 bài train | thật ra 4,416 bài (32%), 78 bài Wi-Fi/Bluetooth/độ sáng; chỉ app riêng của AndroidWorld là vắng |
| "117 dòng mở app không nêu tên app" | đọc từng dòng | ~75 chỉ viết khác, ~5 nhãn sai, ~37 thật sự không nêu tên |
| Huấn luyện ở 858 px | đo ảnh đã lưu: 720×1600 | ảnh AndroidControl lưu ở 720 px → 858 là phóng to ảnh mờ; chuyển về 720 |
| Nhãn app cho bài Browser là Chrome | đọc đề: "mở task.html trong file manager" | đổi thành `files` |
| Sinh đề thử ra khác bài thi | thiếu `% 2**32` trong công thức của mình | harness đúng; công thức gốc tái tạo đúng từng bài thi |
| Tăng tốc bằng 2–4 mẫu/lần | đo: 4 mẫu + grad-ckpt chậm hơn (2.02 vs 2.36); tắt grad-ckpt thì hết bộ nhớ | huấn luyện đệm các mẫu cho bằng nhau; giữ 1 mẫu/lần + grad-ckpt |

## 6. Tối ưu tốc độ (tài liệu d1 + đo)

* Tài liệu d1: `system_one_batch` **gói nhiều yêu cầu không đệm**; `model.compile(mode="reduce-overhead")` chạy câu hỏi đơn bằng
  CUDA graphs (16 → 8 ms trên 4090), mỗi hình dạng mới phải "làm nóng".
* Hàm huấn luyện của chúng ta (`train_d1.forward`) dùng `eng._rows` + `attention_mask` → **có đệm**; vì vậy gộp nhiều mẫu không nhanh hơn.
  Huấn luyện kiểu gói không đệm là cách tăng tốc thật (ước tính, chưa đo) nhưng cần viết lại phần ghép mẫu và test so khớp.
* Lúc chạy 116 bài, thời gian chủ yếu nằm ở giả lập (cài bài, chờ màn hình), không ở mô hình (~0.1 s/bước) → compile ít tác dụng.

## 7. Các quyết định đang chờ / việc tiếp

1. Điểm v2a trên `val` và trên 19 dạng bài chưa học; kiểm tra yêu cầu thật trên máy phục vụ; chạy 116 bài; so cặp với R1.
2. R1 dừng ở 2/3 bản theo yêu cầu (hai bản đều hơn mô hình cũ, nhưng luật đặt trước cần đủ 3 bản → không kết luận).
3. Sau v2a: thêm 2 ảnh cũ; huấn luyện gói không đệm; hành động `answer` (cần dữ liệu từ giả lập).
