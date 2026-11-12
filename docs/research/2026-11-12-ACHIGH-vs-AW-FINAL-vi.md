# AC-High điểm cao mà AndroidWorld thấp: vì sao, và kế hoạch chốt (2026-11-12)

Mọi số dưới đây đều tính lại từ file. Một agent độc lập, đọc file mà không biết trước kết luận, đã kiểm lại toàn bộ (chỉ đọc, không sửa).

**Nguồn dữ liệu**
- AC-High: `benchmarks/results/2026-11-12-v4/actest/perstep-v4_s0_ckpt_2482_actest_v4.jsonl` (model v4 ckpt_2482, chấm bằng scorer của InfiGUI).
- AndroidWorld: `benchmarks/results/2026-11-12-maskenv/aw116_v4g_s0{,_b,_c}` (v4 + harness V12, seed 30) và `compare_A_AG.out`.

## 1. Hai benchmark không đo cùng một thứ

| | AndroidControl-High (offline) | AndroidWorld (online) |
|---|---|---|
| Model thấy gì ở bước t | Lịch sử và màn hình **của người làm mẫu** (teacher-forced): `eval_d1.py:42,48-49,220-225`; lịch sử dựng từ hành động gold: `add_history_v3.py:32-39` | Màn hình do **chính các bước trước của model** tạo ra |
| Chấm gì | **Từng bước**, so với đúng hành động người đã làm (`vendor/infigui_eval.py:83-122`). **Không bước nào kiểm tra mục tiêu đã đạt chưa** | **Trạng thái cuối** của máy, do checker của tác vụ xác định |
| Dừng / trả lời | Không có bước dừng: hàng `status` bị bỏ ở `make_actest_v3.py:5`. Không có câu trả lời | Model phải tự tuyên bố xong. 25/116 bài chỉ chấm câu trả lời. Không tuyên bố xong thì 0 điểm, kể cả khi máy đã đúng trạng thái (luật upstream) |
| Sai một bước | Bước sau vẫn bắt đầu từ màn hình đúng (lỗi không cộng dồn) | Lỗi cộng dồn; phải tự nhận ra và quay lại |
| Đường đi | Phải trùng đường của người. Riêng click và chữ được chấm dễ: click lọt vào hộp chứa phóng to 1,2× cũng tính, chữ chỉ cần là chuỗi con | Đường nào cũng được |

"High" chỉ có nghĩa là chỉ dẫn đưa cho model là mục tiêu, không phải từng bước. Nó không có nghĩa là chấm theo mục tiêu.

## 2. Cùng thước đo "cả bài đúng" thì hai bên ngang nhau

- **AC-High:** đúng từng bước **61,7%** (5.213/8.444), nhưng đúng **mọi bước** của một bài chỉ **344/1.543 = 22,3%**, dù đã được dắt tay ở mỗi bước.
- **AndroidWorld:** **25/115 = 21,74%** (24 bài trọn điểm + 2 bài nửa điểm; OsmAndMarker lỗi setup nên bị loại).

Tỉ lệ cả bài đúng trên AC-High tụt dần theo độ dài bài:

| Số bước | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | ≥12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Đúng cả bài | 69/116 | 37/122 | 70/200 | 60/221 | 49/256 | 19/173 | 18/138 | 11/102 | 4/65 | 1/42 | 2/30 | 4/78 |
| % | 59 | 30 | 35 | 27 | 19 | 11,0 | 13,0 | 10,8 | 6,2 | 2,4 | 6,7 | 5,1 |

Gộp các bài từ 6 bước trở lên: 59/628 = 9,4%.

AndroidWorld tụt theo độ phức tạp đúng như vậy: 22/67 bài khi độ phức tạp < 1,5; 2,5/25 khi 1,5–2,5; 0/12 khi 2,5–4,5; 0,5/11 khi lớn hơn. AC-Low (chỉ dẫn từng bước) đạt 89,3% mỗi bước nhưng cũng chỉ 932/1.543 = 60,4% cả bài.

**Kết luận:** hai benchmark không mâu thuẫn. Con số "AC-High tốt" là số theo từng bước. Đo theo cả bài, năng lực của model khoảng 22% ở cả hai nơi.

Hai con số này không cùng một đại lượng, và các khác biệt kéo theo hai hướng ngược nhau:
- AW **khắt khe hơn** ở chỗ lỗi cộng dồn, phải tự dừng, phải trả lời, và gặp app lạ;
- AW **dễ hơn** ở chỗ đường nào cũng được.

Hai hiệu ứng tình cờ bù nhau. Dùng 22,3% làm chỉ báo, không dùng làm dự đoán.

## 3. Cùng một lỗi ở cả hai nơi: dừng sai lúc

- **AC test không có bước dừng nào là đúng.** Vậy mà model vẫn chọn `terminate` ở **1.171/8.444 bước (13,9%)**, tức là muốn dừng khi bài chưa xong.
- **AndroidWorld:**
  - **49 bài tuyên bố xong khi chưa xong**, cộng 2 bài nửa điểm.
  - Ngược lại, **2 bài máy đã đúng trạng thái nhưng model không dừng**, nên bị 0 điểm: `SystemBrightnessMinVerify` và `RecipeDeleteMultipleRecipesWithConstraint` (`task_successful=True`, `score=0`).
  - 40 bài hết ngân sách bước.
- **Lỗi đầu tiên trên AC-High** (1.199 bài sai):
  - chọn sai loại hành động: 684;
  - đúng loại nhưng sai đích hoặc giá trị: 515, trong đó click sai đích 435;
  - dừng sớm: 154.

Như vậy biết dừng đúng lúc là kỹ năng thiếu nhất. AC-High theo từng bước không thấy được nó, nhưng số lần `terminate` trên AC test đo được nó mà không cần emulator.

## 4. AMEX (arXiv 2407.17490): đọc kỹ paper, đối chiếu với file

1. **AMEX chỉ báo cáo kết quả theo từng bước** (ScreenSpot, AitW, AC). Paper tự nhận không chạy được AndroidWorld vì "SPHINX is not deployable" và viết rằng đánh giá online mới chính xác hơn. Vì vậy không có bằng chứng AMEX giúp tăng tỉ lệ hoàn thành tác vụ.
2. **Phần có ích là dữ liệu hiểu màn hình (Level 1/2:** hộp phần tử và chức năng), không phải chuỗi thao tác. Level 3 chuyển giao kém giữa hai cách thu thập: huấn luyện trên AMEX rồi kiểm tra trên AitW chỉ đạt 56,2, so với 76,3 khi huấn luyện trên chính AitW. Mức tăng trên AC không nhất quán giữa hai phiên bản:
   - paper: 49,8 → 61,7, nhưng phải dùng dữ liệu gấp 4;
   - so cùng số mẫu: 49,8 → 52,8;
   - hình trong README: 60,2 → 63,7.
3. **Paper nói khác file.** Paper (§A.1.5) viết rằng người gán nhãn được phép làm sai nên PRESS_BACK nhiều. File: 135/38.709 bước = 0,35%, ít hơn AC test (4,06%) khoảng 12 lần. 2.527/3.046 episode mở đầu bằng thao tác vuốt ở launcher; không có hành động mở app. AMEX không dạy được việc quay lại hay dừng đúng lúc, mà đó lại là hai lỗi chính của mình.

Có thể tải từng ảnh bằng HTTP range (đã thử thành công), không cần tải cả 93,5 GB. Nhưng với các lý do trên, **không dùng AMEX cho đợt train này**.

## 5. Kế hoạch chốt (đăng ký trước khi có kết quả v5)

**Hiện tốt nhất đã đo:** v4 + harness V12 = **25/115 = 21,74%**. So với A (V11): thắng 5, thua 1, p = 0,22.

1. **v5:** một lần train gộp, dữ liệu như RUNCARD: chuỗi 4 episode để học dừng đúng lúc, dòng quay lại kèm dòng đối chứng, hiển thị on/off và `[id: …]`.
   - Lần chạy đầu hết bộ nhớ (OOM) ở bước ~40. Nguyên nhân là chính giới hạn 0,42 của mình chạm trần khi gặp một dòng chuỗi dài, không phải do tiến trình khác.
   - Đã chạy lại với giới hạn 0,48, lưu mỗi 250 bước.
2. **Cổng 1 (offline, so với v4, cùng file và cùng scorer).** Không qua thì bản cuối dùng v4.
   - (a) AC-High cả bài đúng ≥ 344/1.543, và McNemar theo episode không thua.
   - (b) Số lần `terminate` trên AC test giảm rõ so với 1.171/8.444.
   - (c) Validation: số bài đúng mọi bước không giảm.
   - (d) P(back | gold ≠ back) ≤ của v4 + 0,02.
   - AC test chỉ dùng làm chẩn đoán; benchmark chính vẫn là AndroidWorld.
3. **Cổng 2 (13 bài AndroidWorld nhắm trúng, chạy song song):**
   - dừng sớm: NotesIsTodo, MarkorDeleteNewestNote, ExpenseDeleteMultiple, MarkorDeleteNote, RecipeDeleteMultipleRecipes, SimpleCalendarDeleteOneEvent;
   - không dừng hoặc hết bước: SystemBrightnessMinVerify, RecipeDeleteMultipleRecipesWithConstraint, MarkorCreateNote;
   - giữ các bài đang đúng: SystemWifiTurnOn, RecipeDeleteSingleRecipe, ExpenseDeleteSingle, SimpleCalendarEventsOnDate.
4. **Số cuối:** chạy đủ 116 bài với harness **V13** (V12 + `answer_format`, đã đóng gói và test, md5 e4fb3991), không mask, trên 6 emulator, ghép cặp với v4 + V12. Báo cáo SR chính thức theo quy ước điểm/N, kèm thắng/thua và nhóm lỗi.

Không hứa trước con số. Số được báo là số đo được.
