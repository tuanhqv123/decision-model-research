# Thay "tool cố định" bằng cơ chế tổng quát: rút ra từ cookbook của TypeSafe (đọc trực tiếp 2026-11-12)

Nguồn: docs.typesafe.ai, gồm 11 cookbook, 2 pattern và trang giới hạn của Jev 1.13. Bản text nằm trong scratchpad; bản README lưu ở `references/raw/`.

## Các quy tắc TypeSafe tự viết cho decision model (trích nguyên văn)

1. **Re-ranking: một câu hỏi cho mỗi cặp (truy vấn, ứng viên).** "uses one System One question per query-candidate pair to raise top-1 accuracy from 5% to 30% and top-10 accuracy from 38% to 65%" (40 truy vấn, mỗi truy vấn 30 ứng viên).
2. **Progressive disclosure, kiểm hai lượt** (skill_suggestion): đọc rẻ cả 194 lựa chọn, sau đó "re-reads only the top three, now with each skill's full description … and is free to reject". Kết quả: "cut wrong skill loads by half and needless ones by nearly two thirds".
3. **Đếm và tính toán để code làm, model chỉ phán từng mục:** "Jev does not count reliably … When you want to count items matching some criteria, iterate in code over the candidates and ask one question for each, then add up the answers yourself."
4. **Ngày tháng:** "Jev reads dates as text, not as ordered quantities … Every part of a date is a small closed set … Code assembles the parts into a real date and owns everything after that, including ordering, duration, offset, and weekday."
5. **Hiểu theo nghĩa đen:** "Where interpretation is unavoidable, split it into two literal questions and combine them in code."
6. **Fan-out:** "putting all of the questions your system needs in a single request, and then using code to decide what is relevant … adding more questions usually has little effect on response time." Hỏi gộp hay hỏi tách cho cùng kết quả (parallel_questions: độ lệch chuẩn 0,0 trên phần lớn câu hỏi).
7. **Độ tin cậy là trục thứ hai:** "The score tells you what; confidence tells you whether to act on it." classification_using_confidence: nhóm tự tin thì đúng 93%, phần còn lại 50%; phần còn lại này được báo ở mức phân loại cao hơn.
8. **Kiểm từng luật bằng noul** (sde_cascade): "ask yes/no questions … one per rule it must follow", rồi code quyết định.

## Ba cơ chế tổng quát cho agent của mình (thay cho tool cố định)

### M1. Chọn phần tử hai tầng: shortlist rồi chấm từng ứng viên (quy tắc 1 + 2)
- **Hiện tại:** một câu choice chọn trong danh sách 20–70 phần tử. Click sai đích là **435/1.199** lỗi đầu tiên trên AC-High; trên AndroidWorld là "sai cách thao tác app" và "chọn sai phần tử".
- **Làm:**
  - **Tầng 1** lấy top-k (ví dụ 5) từ chính câu choice hiện có.
  - **Tầng 2** hỏi **một noul cho mỗi ứng viên**: "Hành động tiếp theo có tác động lên phần tử này không?". Đầu vào tầng 2 là state gọn, chỉ phần tử đó, hàng của nó và ngữ cảnh.
  - Code chọn ứng viên có xác suất cao nhất. Nếu không ứng viên nào đủ tin cậy thì quay về đáp án của tầng 1.
- **Nhãn:** lấy thẳng từ AndroidControl (gold element = có, các ứng viên khác = không). Không cần dữ liệu tổng hợp.
- **Bằng chứng sẵn có:** đã đo ngày 2026-11-07: shortlist theo điểm dự đoán của chính model chứa đáp án đúng ở **78,5%** dòng, trong khi grounding hiện tại là **56%**. Thiết kế hai tầng khi đó đã được đăng ký trước nhưng chưa chạy.
- **Đo offline được ngay** (grounding trên validation và AC-High) trước khi chạy AndroidWorld.

### M2. "Map–reduce" cho câu trả lời và mọi phép tính (quy tắc 3 + 4 + 5)
Đây là cách thay cho bảng ngày, máy tính, note và đánh dấu trùng. Một cơ chế chung, không viết riêng cho từng loại bài.
- **Model phán từng mục** (fan-out, cùng một lượt):
  - một noul cho mỗi hàng trên màn hình: "Hàng này có thỏa điều kiện trong mục tiêu không?";
  - một choice "mục tiêu hỏi kiểu đáp án nào": {một mục, đếm, tổng, danh sách, có/không, ngày, không hỏi}.
- **Code gom kết quả:**
  - tích lũy các hàng thỏa điều kiện qua các lần cuộn (khóa theo chữ của hàng, nên tự loại trùng);
  - tách số, thời lượng và ngày bằng regex (cookbook pre-parsed value);
  - tính đếm, tổng, đổi đơn vị;
  - ghép danh sách bằng dấu phẩy (đúng định dạng checker AndroidWorld yêu cầu);
  - so trùng giữa các ghi nhận;
  - định dạng rồi gửi `answer`.
- **Ngày trong form:** model chọn các phần của ngày (tương đối/tuyệt đối, thứ, tháng, ngày). Code tính ngày thật từ đồng hồ cố định của AndroidWorld (15/10/2023).
- **Nhóm lỗi nhắm tới:**
  - hỏi thông tin: 21 bài (thấy mà không trả lời, đếm, cộng, danh sách nhiều mục, ngày);
  - tìm trùng;
  - các bài form ngày/giờ.
- **Nhãn:** câu "hàng có thỏa không" cần dữ liệu tổng hợp từ các danh sách trong AndroidControl, dùng điều kiện theo nghĩa đen dựng từ chữ của hàng. Đây là phần rủi ro nhất.

### M3. Kiểm chứng tiến độ bằng màn hình kế tiếp (quy tắc 7 + 8)
- **Làm:** sau mỗi hành động, ở lượt kế tiếp hỏi thêm một noul: "Hành động vừa rồi có đưa mục tiêu tới gần hơn không?". Nếu model **tự tin** là không, code bấm Back rồi thử ứng viên thứ hai.
- **Khác với câu "đã xong chưa" đã thất bại:** câu đó trùng nhãn với hành động `status` và chỉ nhìn màn hình *trước* khi làm. Câu này nhìn màn hình *sau* khi làm, tức là dùng thông tin mới mà một lượt đọc đơn lẻ không có.
- **Nhãn:**
  - có tiến độ = cặp (màn hình t, hành động gold, màn hình t+1) trong AndroidControl;
  - không tiến độ = các nhánh sai lấy từ episode khác, đã dựng sẵn trong `build_recovery_data`.
- **Nhóm lỗi nhắm tới:** lặp đến hết ngân sách (40 bài), chọn sai rồi đi tiếp.
- **Rủi ro:** quay lại sai. Vì vậy Back chỉ do code quyết khi model tự tin cao (theo quy tắc 7), không phải là một nhãn hành động.

## Không dùng
- **Tool cố định từng loại:** bảng ngày, note, đánh dấu trùng riêng lẻ. M2 thay hết.
- **Tool clipboard:** kênh đặc quyền.
- **OCR:** là model ngoài.
- **Ngưỡng trên p(status):** đã đo là có hại ở mọi ngưỡng (M12).

## Thứ tự đề xuất
1. **Final lần 3 (đang train)** sửa phần dừng sớm, là nhóm lỗi lớn nhất. Đo nó trước.
2. **Làm M1 trước.** Nhãn lấy sẵn từ AndroidControl, đo offline được ngay, có tác dụng cho mọi bài.
3. **Làm M3 tiếp theo.** Nhãn từ AndroidControl cộng dữ liệu nhánh sai đã có, đo offline bằng AUROC trên tập để riêng.
4. **Làm M2 sau cùng,** cho nhóm hỏi thông tin và tính toán. Dữ liệu tổng hợp, rủi ro cao nhất, nhưng gỡ được nhiều nhất trong nhóm bị chặn.
5. **Train một lần cho M1 + M3** (cùng là câu hỏi mới trên dữ liệu AndroidControl), có kiểm tra sớm, rồi so cặp với final.

## Kế hoạch hiện tại (2026-11-12, buổi chiều)

1. **Mốc nền mới v4 + harness V14, đang chạy.** Đủ 116 bài trên 6 emulator đã sửa (Markor, OsmAnd, SIM). V14 = định dạng câu trả lời + ẩn bàn phím. Model v4 phục vụ trên GPU_HOST_A. Lượt này trả lời hai câu: lỗi môi trường cũ còn không, và harness tăng được bao nhiêu khi model giữ nguyên.
2. **Final lần 3 (contract v6), đang train, không dừng nữa.**
   - Quy tắc dừng sớm ở ckpt_250 đã bị chính dữ liệu của mình bác bỏ. Ở run v3, chỉ số giữa chừng lên xuống mạnh và không dự báo được kết quả cuối (ledger M17).
   - Chỉ cổng 1 trên checkpoint cuối mới quyết định.
   - Xu hướng các checkpoint giữa sẽ đo lại sau khi train xong, vì checkpoint được giữ.
3. **Thử pattern với model hiện tại, không train** (`benchmarks/train/probe_patterns.py`, v4, 827 dòng validation):
   - **M1:** re-rank top-5 phần tử bằng một noul cho mỗi ứng viên, so với chọn bằng choice;
   - **M3:** AUROC của noul "bước vừa rồi có tiến gần mục tiêu không?". Mẫu dương là màn hình sau bước của người làm mẫu, mẫu âm là màn hình sau một nhánh sai.

   Nếu có tín hiệu thì mới đưa vào train (một lần, M1 + M3). Nếu không có thì ghi thành kết quả âm.
4. **Sau đó:** chạy 13 bài nhắm trúng, rồi 116 bài cho final với V14, so cặp với mốc nền. Paper dùng số tốt nhất đo được.

## Rà soát 2026-11-12 (theo yêu cầu "kiểm tra lại cho kỹ")

**Những gì đã kiểm và đúng:**
- **Trọng số v4:** md5 `fa8ba04b…` giống nhau trên cả ba máy (GPU_HOST_A `init_v4`, máy train `v4_s0/ckpt_2482`, máy serving `v4_s0_ckpt2482`). Server v4 dùng cho lượt có mục tiêu đúng là model đã chạy ở v4 + V12.
- **Chính sách harness ghi trong episode:** `answer_format: true`, `hide_keyboard_after_type: true`, `fresh_wait_s: 20`, `mask: false`.
- **Ẩn bàn phím chạy thật:** 19 bước gõ, 17 lần bàn phím được ẩn sau đó. Hai lần còn lại bàn phím không hiện.
- **Môi trường:** mỗi container đều báo `APP_OK` cho markor và osmand_markers khi khởi động.

**Những lỗi tìm ra và đã sửa trong lần rà soát này:**

| # | Lỗi | Hệ quả nếu không sửa | Đã làm |
|---|---|---|---|
| 1 | Preflight viết cứng cổng 8701 | Không chạy được server nào khác 8701 | Lấy URL từ `model.url` trong config, có commit |
| 2 | 6 chain ghi chung `/tmp/preflight.out` | Kết quả preflight của các emulator ghi đè lên nhau | Mỗi emulator ghi một file riêng |
| 3 | Chạy lại đủ 116 bài cho v4 + V14 | Tốn khoảng 2,5 giờ vô ích | Chỉ chạy 39 bài thật sự bị ảnh hưởng; 73 bài giữ kết quả v4 + V12 (sẽ khai rõ khi báo số) |
| 4 | **Train final chiếm GPU dùng chung với khách: 31,9/32,6 GB, chỉ còn 0,7 GB** (dịch vụ của khách tăng bộ nhớ, thêm một dịch vụ mới) | Dịch vụ của khách có thể OOM | Dừng train ở bước 630; chuyển sang GPU_HOST_A (24,5 GB, gần như trống) sau khi lượt 39 bài xong |
| 5 | GPU_HOST_A còn 21 GB đĩa, mà giữ mọi checkpoint thì cần khoảng 30 GB | Đĩa đầy giữa lúc train | Xóa pilot và checkpoint cũ của mình (bản trộn trọng số vẫn còn trên máy serving). Final lưu một checkpoint ghi đè mỗi 500 bước |
