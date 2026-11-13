# Phân tích lại sau lượt v3 (2026-11-11): sai ở đâu, vì sao, sửa thế nào. Mọi khẳng định đều có số và file

**Lập luận và phản biện cho từng kết luận:** `2026-11-11-REASONING-v3-vi.md`. Ở đó có những chỗ phản biện đã bác một phần kết luận dưới đây.

**Quy tắc của tài liệu này:** mỗi dòng "đo được" có một file trong `benchmarks/results/2026-11-11-v3/postmortem/`
(script + output), chạy lại sẽ ra đúng số đó. Dòng nào là kiến thức từ bài báo thì ghi nguồn và nói rõ là số **của
họ, trên benchmark của họ**. Dòng nào là dự đoán thì ghi **giả thuyết**, kèm cách đo và điều kiện để bỏ.
Số nào không tính lại được thì đã bị gỡ (xem §6).

## 1. Tóm tắt (chỉ số đo)
d1-mobile v3 chọn **từng bước**: AC-High 70,6% (bỏ `status`) / 67,1% (chuẩn), 8.444 bước. **Tự làm trọn bài**:
AndroidWorld **14,0%** (16/114, `aw_facts.out`). Năm nguyên nhân dưới đây đều có số đo. Nguyên nhân 1 và 3 đã được
kiểm định thống kê hoặc đúng tuyệt đối trên dữ liệu (4.197/4.197). Các nguyên nhân còn lại mới là quan sát, chưa phải
thí nghiệm có đối chứng; đó là việc phải làm tiếp.

## 2. Sự thật đo được

| # | Đo | Số | File |
|---|---|---|---|
| F1 | SR AndroidWorld chính thức (hàm `official_sr` của repo; loại MarkorMergeNotes vì bộ chấm lỗi, SimpleSmsReplyMostRecent vì lỗi khởi tạo) | **16/114 = 14,0%**; 16/116 Wilson 95% 8,7–21,2 | `aw_facts.out` |
| F2 | Thành công theo độ dài bài (step budget) | ≤10: 11/48 · 11–20: 5/39 · 21–30: 0/8 · >30: 0/21 | `aw_facts.out` |
| F3 | Bài thất bại: báo xong khi chưa xong / không báo, hết bước | 49 / 51 (trên 100) | `aw_facts.out` |
| F4 | Bước mà dòng lịch sử (model nhìn thấy) ghi "không đổi" | 663/1.476 = 44,9%; **gồm cả `wait`**. Ở bài được chấm: 595 = wait 151 + scroll 155 + click 285 + open_app 4 | `aw_facts.out`, `challenge_aw.out` |
| F5 | Ngay sau một bước vô ích, model làm lại y hệt | 534/609 = 87,7%. Lặp **chắc chắn lãng phí** (lần sau cũng không đổi): click 188 + scroll 142 = **≥330**; `wait` lặp 131, có 37 lần hợp lý. Sau click ô nhập không đổi: click tiếp 57, gõ 11 | `aw_facts.out`, `challenge_aw.out` |
| F6 | Bài thất bại có một hành động lặp liên tiếp ≥5 lần | 46/100 | `aw_facts.out` |
| F7 | Bài chỉ chấm qua `answer` | 0/25 | `aw_facts.out` |
| F8 | App AndroidWorld có **0 dòng học** | **19/33**; dòng học ở app ngoài AW: 18.788/27.454 | `train_facts.out` |
| F9 | Thành công theo độ phủ app (app chính của bài có dòng học hay không) | có: **8/25 = 32,0%** · không: **8/89 = 9,0%** · Fisher một phía **p = 0,0072** | `app_coverage.out` |
| F9b | Cùng phép chia, trong từng nhóm độ dài | ≤10 bước: 7/16 vs 4/32 (p = 0,021) · 11–20: 1/7 vs 4/31 · >20: 0/2 vs 0/26 | `app_coverage.out` |
| F10 | Dòng học ngay sau một bước vô ích | 539/27.454 = 2,0% (đáp án ở các dòng đó: click 261, status 122, wait 62, scroll 54, …) | `train_facts.out` |
| F11 | Dòng `status` nằm ở **bước cuối** của bản ghi | **4.197/4.197**; dòng không phải bước cuối mà có nhãn `status`: **0** | `train_facts.out` |
| F12 | Sau khi bấm vào ô nhập chữ, đáp án kế tiếp là `type_text` | 577/699 = 82,5% | `train_facts.out` |
| F13 | Bước AW có >40 ứng viên (bị cắt) | **454/1.541 = 29,5%**; `New Event` trên màn hình 155 bước, bị cắt **155/155**, ở cả 17 bài Simple Calendar (v3: 0/17) | `aw_cut.out` (danh sách dựng lại khớp lựa chọn thật 899/903) |
| F14 | Dòng học có >40 ứng viên | 740/6.863 mẫu = 10,8% | `cap_tokens.out` |
| F15 | Độ dài chữ khi **không cắt** (trạng thái + phương án) / giới hạn model | trung vị 775, p99 2.659, lớn nhất 6.456 token / **128.000** | `cap_tokens.out` |
| F16 | Giữ được đáp án trên màn hình >40 ứng viên: cắt theo thứ tự đọc (R0) / R1 / R2 | train 1.285 / 1.427 / **1.465** trên 1.697 · val 59 / 67 / 68 trên 80 · AC-High 487 / **547** / 543 trên 656 | `rule_retention.out` |
| F17 | Cấm lặp lúc chạy (mask), 112 bài ghép cặp | v3 16/112 · mask 19/112 (thắng 4, thua 1, p = 0,375; 3/4 bài thắng là bài "cho không"); lặp y hệt 86,5% → 62,2%; báo xong sai 49 → 56 | `mask_paired_full.out` |
| F18 | Self-play: bộ chấm trả lời sau **mỗi** bước | 395/395 bước, 0 lỗi; 8 bài đạt trạng thái xong (báo đúng bước sau: 5, báo muộn: 2, không báo: 1); **18** bài báo xong khi bộ chấm nói chưa xong | `selfplay_facts.out` (37 bài, đang chạy) |
| F19 | Tốc độ model mỗi quyết định | p50 139,5 ms, p95 174,6 ms | `aw_facts.out` |

## 3. Nguyên nhân: mỗi cái kèm số, và mức độ chắc chắn

### N1. Học trên app khác app thi (chắc chắn nhất: có kiểm định)
19/33 app AW không có dòng học nào (F8). Bài thuộc app có dữ liệu thành công 32% so với 9% (p = 0,007, F9).
**Phản biện tự đặt ra:** app quen (Settings, Clock, Contacts) có bài ngắn hơn, nên khác biệt có thể do độ dài bài. Trong nhóm
bài ≤10 bước, khác biệt vẫn còn: 7/16 (44%) so với 4/32 (12,5%), p = 0,021 (F9b). Ở nhóm 11–20 bước thì **không thấy khác
biệt** (14% so với 13%). Kết luận được bảo vệ: thiếu dữ liệu app làm hỏng cả bài ngắn. Kết luận **không** được bảo vệ:
chỉ thêm dữ liệu app là đủ cho bài dài.

### N2. Chỉ học bắt chước với lịch sử luôn đúng, nên không biết thoát khi tự sai (quan sát, chưa có đối chứng)
45% số bước không làm màn hình đổi (F4). Sau mỗi bước như vậy, 88% lần model làm lại y hệt (F5). Trong khi đó, chỉ 2% dòng
học có bước trước vô ích (F10). Bài dài hơn 20 bước: 0/29 (F2).
Lý thuyết (Ross & Bagnell, DAgger, 2011): học bắt chước chỉ thấy trạng thái của người làm mẫu, nên lỗi cộng dồn theo độ dài
bài. Số đo **khớp** với lý thuyết này, nhưng mình **chưa** làm thí nghiệm tách riêng nó.
Một lối tắt đã bị loại bằng số: cấm lặp lúc chạy không tăng SR có ý nghĩa: 16 → 19/112, p = 0,375, 3/4 bài thắng là bài "cho không" (F17).

### N3. Nhãn "báo xong" gắn theo vị trí (chắc chắn: đúng tuyệt đối trên dữ liệu)
4.197/4.197 dòng `status` là bước cuối của bản ghi, 0 dòng ở chỗ khác (F11). AndroidControl gốc không có hành động kết thúc,
mình tự thêm vào (`benchmarks/train/preprocess_ac.py:5` và `:134`: một bước `status` cho mỗi bản ghi, đặt trên màn hình sau hành động cuối). Như vậy nhãn "xong" chỉ mang một tín hiệu: vị trí trong bản ghi mẫu. **Phản biện (xem REASONING L3):** model **không** chỉ dựa vào vị trí. Bước nó báo xong tăng theo độ dài bài (Spearman 0,46), nhưng ở bài dài vẫn báo quá sớm (trung vị bước 8 với bài có ngân sách >20 bước).
Hệ quả đo được: 49 bài báo xong khi chưa xong (F3); trên AC-High model dự đoán `status` 539 lần, trong khi benchmark không
có bước kết thúc (`RESULTS.md` §3). Thí nghiệm ngày 2026-11-10 xác nhận cùng cơ chế: nhãn `is_done` trùng 137/137 với
`action == status` (`benchmarks/results/2026-11-10-episodic-s0-fails-prereg.md`).
Nhãn thay thế đã có trên máy (F18): bộ chấm trả lời ở 395/395 bước.

### N4. Cắt danh sách ở 40: không phải giới hạn của model, và chưa từng được kiểm tra trên v3
Model nhận 128.000 token; không cắt thì p99 chỉ 2.659 token (F15). Căn cứ duy nhất cho số 40 là một **so sánh quan sát**
ngày 2026-11-07 trên một bản model khác (`benchmarks/results/2026-11-07-list-length-is-causal.md`). Đó không phải thí
nghiệm: màn hình nhiều phần tử vốn khó hơn.
Cái giá đo được: 29,5% số bước AW bị cắt, `New Event` mất 155/155 lần, 0/17 bài Simple Calendar (F13); 10,8% dòng học
bị che một phần (F14).
**Giới hạn của kết luận:** 0/17 Simple Calendar cũng trùng với N1 (app không có dòng học) và với N5 (8/17 bài cần
`answer`). Riêng phần cắt chưa được tách ra; đó là thí nghiệm T1 ở §4.

### N5. Thiếu hành động `answer`
25/116 bài chỉ chấm qua `answer`: 0/25 (F7). Trần điểm hiện tại là 91/116 = 78,4%.

## 4. Giải pháp. Phần "hiệu quả" đều là **giả thuyết**, kèm trần đo được và cách bỏ

| # | Việc | Sửa | Trần đo được (tối đa bao nhiêu bài) | Chứng cứ ủng hộ | Thí nghiệm / điều kiện bỏ |
|---|---|---|---|---|---|
| T1 | **Bỏ cắt 40** (hoặc nới), thử offline trước | N4 | 9 bài Simple Calendar không cần `answer` (+7,9 điểm nếu làm được hết) | F13, F15 | (a) v3, cùng checkpoint, 656 dòng AC-High >40 ứng viên: không cắt vs cắt 40. (b) Nếu (a) không tụt: chạy 17 bài Simple Calendar. Bỏ nếu (a) tụt so với cắt 40. Phương án dự phòng: R2 (F16: 1.465 vs 1.285 trên train) |
| T2 | **Self-play trên đúng app thi** (seed ≠ 30), giữ bài thành công rồi học lại, lặp nhiều vòng | N1, N2 | 89 bài thuộc app không có dòng học | F9; **DigiRL (arXiv 2406.11896), trên AitW, không phải AW:** học bắt chước (SFT) 17,7% → học lại từ quỹ đạo tự thu đã lọc 57,8% → RL 67,2% | Bỏ nếu bộ dev AW-luyện-tập không tăng ≥5 điểm sau vòng 1 |
| T3 | **Nhãn "đã xong" từ bộ chấm** ở mỗi bước | N3 | 49 bài báo sai | F11, F18 (18 lần báo sớm và 2 lần báo muộn đã gắn được nhãn) | Số lần báo xong khi chưa xong trên dev phải giảm |
| T4 | **"Đừng chọn"** cho bước vô ích (unlikelihood loss) | N2 | 46 bài lặp ≥5 lần | F5, F10 | Tỉ lệ lặp sau bước vô ích (87,7%) phải giảm trên dev |
| T5 | Thêm hành động `answer` | N5 | 25 bài | F7 | 25 bài hỏi thông tin |
| T6 | RL với thưởng từ bộ chấm, sau T2–T5 | N2 | — | MAI-UI Bảng 9 (`references/raw/maiui2026-androidworld-table9.txt`): 2B SFT 45,1 → RL 49,1 (+4,0), harness của họ | Chỉ làm khi SR trên dev ≥ ~30% |

**Đã đo là không giúp có ý nghĩa:** cấm lặp lúc chạy (F17, đủ 112 bài). **Không làm nữa vì không chạm nguyên nhân nào ở trên:** thêm
dữ liệu hoặc epoch chỉ từ AndroidControl, tinh chỉnh cho AC-High, v3i (AC-Low).

## 5. Thứ tự
1. T1(a): offline, không học lại, rẻ nhất.
2. Tạo bộ dev AW-luyện-tập (seed riêng, không bao giờ đem học) làm thước đo cho mọi thay đổi sau.
3. Vòng 1 gồm T2 + T3 + T4, học v4, đo trên dev, rồi đo 116 bài với ≥3 seed (một seed trên 116 bài có sai số chuẩn ~3,2
   điểm, 95% khoảng ±6,3).
4. T5, sau đó T6.

## 6. Những chỗ bản trước (commit `8c0ceeb7`) sai hoặc không có chứng cứ, đã sửa
| Bản trước | Sự thật | Vì sao sai |
|---|---|---|
| Bước bị cắt: 201/1.541 = 13% | **454/1.541 = 29,5%** | không tìm lại được cách đếm cũ; số mới đã kiểm bằng lựa chọn thật của model (899/903) |
| Sau khi bấm ô nhập, gõ 536/619 = 87% | **577/699 = 82,5%** | không tái tạo được định nghĩa cũ; định nghĩa mới nằm trong script |
| "Bước đầu mở đúng app 98/115" | Chỉ chứng minh được: 115/116 bài, hành động đầu là `open_app` | không có định nghĩa "đúng" có thể chạy lại, nên đã gỡ |
| "Thử 9 bài bị cắt" | Cả **17** bài Simple Calendar bị cắt (9 không cần `answer`) | đếm sót các bài hỏi thông tin |
| "14,0% so với 45–56% của MAI-UI-2B / UI-Venus-1.5-2B" | Số của họ dùng harness khác, không nêu số lần chạy hay giới hạn bước (`outputs/2026-11-11-published-ac-aw-metrics.md` §2) | M4: không so số khác harness. Chỉ dùng làm bối cảnh, không xếp hạng |
| "Bài thành công tập trung ở app quen" (không số) | 32% vs 9%, p = 0,007; còn đúng ở bài ≤10 bước, không thấy ở bài 11–20 bước | trước đó chưa đo |
| "Cấm lặp không cứu được" (một nửa) | Đúng trên nửa A (12/56 = 12/56); nửa B chưa xong | đã ghi rõ phạm vi |
| "DigiRL nâng mạnh" (không số) | 17,7 → 57,8 (lọc rồi học lại) → 67,2 (RL), trên AitW | trước đó chưa trích số |
