# Mọi số liệu hiện có (2026-11-12)

Mỗi số đều kèm mẫu số. Các bảng AndroidWorld được dựng bằng `benchmarks/aw/aw_report.py` từ trace đã lưu. Trace (không có ảnh chụp màn hình) nằm trong `benchmarks/results/2026-11-12-armsAB/androidworld_armsAB.tgz` và `benchmarks/results/2026-11-12-maskenv/androidworld_maskenv.tgz`. Chuẩn chấm: AndroidWorld chính thức, 116 bài, seed 30, 1 lần chạy (pass@1); SR = tổng điểm / số bài được chấm.

## 1. AndroidWorld: các lần chạy

| Model | Harness / môi trường | Điểm | SR | So cặp với lần chạy trước nó |
|---|---|---|---|---|
| v3h | V8 | 17/113 | 15,0% | – |
| v4 | V8 | 15,5/115 | 13,5% | – |
| v4 | V11 (sửa câu trả lời, ghi policy, fresh_wait) | 20,5/115 | 17,8% | so với v4/V8: thắng 5, thua 0 |
| trộn 0,3·v3 + 0,7·v4 | V11 | 18/114 | 15,8% | so với v4/V11: thắng 6, thua 9 |
| v4 | V11 + mask | 20,5/113 | 18,1% | thắng 2, thua 2 |
| **v4** | **V12 + sửa môi trường (SIM, Chrome)** | **25/115** | **21,7%** | so với v4/V11: thắng 5, thua 1 (p = 0,22) |
| v4 | V14 (thêm định dạng câu trả lời, ẩn bàn phím) + sửa Markor/OsmAnd | đang chạy 39 bài bị ảnh hưởng; 77 bài còn lại giữ kết quả của V12 | – | – |

**Bài bị loại vì lỗi môi trường (đều đã sửa):**
- OsmAndMarker: file `map_markers_db` rỗng, thuộc root;
- SimpleSmsReplyMostRecent: lỗi SIM;
- SystemCopyToClipboard: lỗi Clipper, chỉ xảy ra một lần.

## 2. AndroidWorld theo độ khó chính thức (`task_metadata.json`: easy 61 / medium 36 / hard 19)

### v4 + V11 (arm A)
| Difficulty | scored n | score | full | SR % | avg steps | avg steps (success) | avg steps (failure) | avg optimal steps | excluded |
|---|---|---|---|---|---|---|---|---|---|
| easy | 61/61 | 19.0 | 19 | 31.1 | 8.7 | 5.4 | 10.1 | 5.0 | 0 |
| medium | 35/36 | 1.0 | 1 | 2.9 | 12.3 | 7.0 | 12.4 | 9.1 | 1 |
| hard | 19/19 | 0.5 | 0 | 2.6 | 18.3 | - | 18.3 | 18.1 | 0 |
| all | 115/116 | 20.5 | 20 | 17.8 | 11.3 | 5.5 | 12.6 | 8.4 | 1 |
### merge 0.3 + V11 (arm B)
| Difficulty | scored n | score | full | SR % | avg steps | avg steps (success) | avg steps (failure) | avg optimal steps | excluded |
|---|---|---|---|---|---|---|---|---|---|
| easy | 61/61 | 17.0 | 17 | 27.9 | 8.8 | 5.8 | 10.0 | 5.0 | 0 |
| medium | 35/36 | 1.0 | 1 | 2.9 | 13.4 | 8.0 | 13.6 | 9.1 | 1 |
| hard | 18/19 | 0.0 | 0 | 0.0 | 22.1 | - | 22.1 | 18.5 | 1 |
| all | 114/116 | 18.0 | 18 | 15.8 | 12.3 | 5.9 | 13.5 | 8.4 | 2 |
### v4 + V11 + mask
| Difficulty | scored n | score | full | SR % | avg steps | avg steps (success) | avg steps (failure) | avg optimal steps | excluded |
|---|---|---|---|---|---|---|---|---|---|
| easy | 60/61 | 19.0 | 19 | 31.7 | 8.4 | 5.5 | 9.8 | 5.1 | 1 |
| medium | 35/36 | 0.0 | 0 | 0.0 | 12.1 | - | 12.1 | 9.1 | 1 |
| hard | 18/19 | 1.5 | 1 | 8.3 | 12.4 | 7.0 | 12.8 | 18.5 | 1 |
| all | 113/116 | 20.5 | 20 | 18.1 | 10.2 | 5.5 | 11.2 | 8.5 | 3 |
### v4 + V12 (env fixes)
| Difficulty | scored n | score | full | SR % | avg steps | avg steps (success) | avg steps (failure) | avg optimal steps | excluded |
|---|---|---|---|---|---|---|---|---|---|
| easy | 61/61 | 21.0 | 21 | 34.4 | 8.5 | 5.5 | 10.1 | 5.0 | 0 |
| medium | 36/36 | 3.0 | 3 | 8.3 | 13.0 | 6.7 | 13.6 | 9.0 | 0 |
| hard | 18/19 | 1.0 | 0 | 5.6 | 19.3 | - | 19.3 | 18.5 | 1 |
| all | 115/116 | 25.0 | 24 | 21.7 | 11.6 | 5.7 | 13.2 | 8.4 | 1 |

**Theo mức độ phức tạp kiểu MobileRL**, ceil(complexity), với v4 + V12:

| Mức | Thành công | SR | Số bước TB khi thành công |
|---|---|---|---|
| C1 | 19/45 | 42,2% | 5,3 |
| C2 | 5,5/40 | 13,8% | 7,2 |
| C3 | 0/9 | 0% | – |
| C4 | 0,5/21 | 2,4% | – |

## 3. AndroidControl (v4 ckpt_2482)

**AC-High** (scorer InfiGUI, tập test 1.543 bài / 8.444 bước):
- đúng từng bước: 5.213/8.444 = 61,7%;
- bài đúng mọi bước: 344/1.543 = 22,3%;
- số lần model đòi dừng sai (tập test không có bước dừng nào): 1.171/8.444.

**AC-Low:**
- đúng từng bước: 7.544/8.444 = 89,3%;
- bài đúng mọi bước: 932/1.543 = 60,4%.

**Validation** (827 dòng / 137 bài):
- đúng từng bước: 520/827 = 62,9%;
- bài đúng mọi bước: 14/137;
- báo xong sai: 64/690;
- quay lại sai: 16/798.

**Thử thay ảnh bằng ảnh xám** (validation, cùng v4):

| | Ảnh thật | Ảnh xám |
|---|---|---|
| Đúng từng bước | 519/827 | 337/827 |
| Chọn đúng loại hành động | 80,0% | 50,8% |

## 4. Thử pattern với model hiện tại (v4, chưa train)

| | Validation | Mẫu AC 2.012 dòng |
|---|---|---|
| **M1: chọn bằng choice** (argmax) | 286/430 | 821/1.224 |
| M1: chọn lại bằng một noul cho mỗi ứng viên | 261/430 | 772/1.224 |
| M1: kết hợp hai cách | 285/430 | 823/1.224 |
| M1: đáp án đúng nằm trong top-5 | 388/430 | 1.084/1.224 |
| **M3: AUROC** của noul "bước vừa rồi có tiến gần mục tiêu không" | 0,68 (10 mẫu âm) | **0,76** (1.550 dương / 26 âm) |

## 5. Lần train lại (contract v6): chưa xong

| Ở ckpt_250, trên validation | Lần 1 | Lần 2 | Lần 3 | v4 |
|---|---|---|---|---|
| Đúng từng bước | 489 | 485 | 492 | 520 |
| Báo xong sai (trên 690) | 67 | 108 | 106 | 64 |
| Quay lại sai (trên 798) | 44 | 23 | 35 | 16 |

- **Lần 1 và 2** bị dừng theo quy tắc cảnh báo sớm. Sau đó chính dữ liệu của run v3 bác bỏ quy tắc này: chỉ số giữa chừng không dự báo được chỉ số cuối (AGENTS.md M17).
- **Lần 3** dừng ở bước 630 để nhường GPU dùng chung cho khách. Run sẽ chạy lại trên GPU_HOST_A.
