# Kế hoạch v4: sửa tất cả lỗi trong một lần học, nhắm khoảng 40% AndroidWorld (2026-11-12)

Bản này thay bản cùng tên viết trước đó trong ngày (bản chỉ sửa tín hiệu "no change").

**Mục tiêu do người vận hành đặt:** tiến gần các model 2B–4B hiện có, khoảng 40% trên AndroidWorld (khoảng 46/114 bài có chấm
điểm). v3 hiện đạt 16/114 = 14%.
- So sánh với số trong bài báo chỉ để định hướng. Họ dùng harness khác, nên không xếp hạng trực tiếp được (quy tắc M4).
- Mốc gần nhất là AppVLM 3B: chỉ học AndroidControl được 11,4%; thêm các lần chạy AW được bộ chấm xác nhận thì lên 37,8%
  (số của họ, Tab. 4).

Chứng cứ: `docs/research/2026-11-12-DATA-AUDIT-vi.md` và `benchmarks/results/2026-11-12-*`.


## 0. Cập nhật 2026-11-12: v4 chỉ dùng dữ liệu có sẵn và augment, không chạy máy giả lập
Người vận hành chốt: lần huấn luyện này phải sửa mọi lỗi, dùng dữ liệu đang có cộng augment, không đợi các lần chạy AW mới.

| Nhánh | Làm bằng augment như thế nào | Có thể sửa trọn vẹn không |
|---|---|---|
| W0 harness | Dòng lịch sử ghi theo màn hình của bước kế tiếp; có test | Có |
| W1 vòng lặp | Bỏ 52 dòng dạy lặp; thêm 10% dòng có một thất bại giả định và 3% dòng có 2–3 thất bại liên tiếp. Nhãn tác dụng vẫn là "danh sách phần tử có đổi không", giống hệt lúc chạy | Có về cơ chế: đã đo trên dữ liệu, tỉ lệ lặp giảm từ 19,2% xuống 1,6% |
| W2 báo xong sớm | Ghép hai episode AndroidControl cùng app thành goal hai phần ("A. Then B."), nối bằng `navigate_home`. Ở màn hình "xong phần A" nhãn là đi tiếp; `status` chỉ ở cuối. **Không dùng self-play** (người vận hành đã bỏ) | Có về cơ chế (P3, P4); chỉ cho loại lỗi "dừng sau phần đầu" (xem 0b) |
| W3 `answer` | Thêm hành động `answer` và ô `answer_el` (chọn phần tử có chữ là đáp án). Dữ liệu là câu hỏi sinh theo mẫu từ màn hình AndroidControl, đáp án là chữ của đúng một phần tử (ví dụ "What is the title at the top?", "What is the time of the first alarm?") | Dạy được cách trả lời bằng sao chép. Câu hỏi dạng đếm hoặc cộng tổng (khoảng 8/26 loại đáp án) thì **không** |
| W4 bỏ cắt 40 | Quy tắc mới `nocap_reading_noime`, áp cho cả lúc học và lúc chạy | Có |
| Focus | Đặt `is_focused = False` trong dữ liệu học, vì AW không bao giờ báo cờ này | Có |
| W5 quy trình trong 19 app chưa có dữ liệu | **Không làm trong v4.** Không có dữ liệu AW để học, và self-play đã bị bỏ | Không. Đây là phần giới hạn mức trần |

**Kiểm tra trước khi học, không tốn GPU:**
- tỉ lệ lặp sau "no change" thấp hơn sau "changed";
- tỉ lệ `status` theo bước không còn đỉnh ở bước 5–8;
- "New Event" có trong danh sách;
- dòng `answer` có đáp án nằm đúng trong danh sách phần tử;
- bộ test của harness, của code học và của server đều xanh.

**Kiểm tra sau khi học:** thăm dò P1–P5, AC-High, rồi AW 116 bài.

## 0b. Rà soát lại kế hoạch từ góc nhìn khác (2026-11-12)
Mỗi điểm là một câu hỏi "nếu kế hoạch sai thì sai ở đâu", kèm số đo.

1. **"Báo xong sớm" thực chất gồm ba lỗi khác nhau** (`2026-11-12-expect/premature_goals.out`).
   - Có 49 lần báo xong sai. **14 lần là bài phải trả lời bằng chữ**: model đã tới đúng màn hình, nhưng không có hành động
     `answer` nên báo xong. Ví dụ SimpleCalendarLocationOfEvent click đúng "100 Main St Chicago", tức chính là đáp án.
     Nhóm này do W3 sửa, không phải W2.
   - Trong 35 lần còn lại, 18 lần có goal nhiều phần **và** hành động cuối là kiểu Save/Delete/Create. Đây là dừng sau khi
     xong phần đầu (thêm 1 trên 3 công thức, xóa 1 trên nhiều sự kiện). Ghép episode trong W2 đúng là nhắm vào loại này.
   - Phần còn lại là lỗi **khác**, không phải lỗi kết thúc: bấm nhầm nút gửi ("SMS"), gõ sai nội dung ("clipboard content"
     thay vì nội dung clipboard). Augment kết thúc không sửa được các lỗi này. Ghi rõ để không kỳ vọng sai.
   - Phép đếm "nhiều phần" và "kiểu Save" dùng regex thô, nên là ước lượng.
2. **Hồ sơ dự án có ghi "sụp khi có hơn 40 ứng viên"** (−17,3 / −23,8, ngày 2026-11-07).
   - Đó là thiết kế cũ, đầu ra là tọa độ: danh sách dài làm hỏng đầu ra tọa độ. Hợp đồng v3 chọn phần tử theo số thứ tự,
     là cơ chế khác.
   - Nhưng rủi ro "danh sách dài làm chọn kém" vẫn có thật. **Phải kiểm tra trước khi học:** chạy v3 không giới hạn trên
     656 dòng AC-High có hơn 40 ứng viên (vài phút GPU), so với khi cắt ở 40. Nếu sụp thì giữ một giới hạn cao hơn kèm thứ
     tự ưu tiên, thay vì bỏ hẳn.
3. **`answer` có thể bị dùng quá sớm.** Nếu mọi dòng `answer` đều ở bước 0 (câu hỏi về chính màn hình đang thấy), model sẽ
   học "câu hỏi → trả lời ngay". Bài AW lại cần đi tới đúng màn hình trước.
   - Cách sửa: dòng `answer` đặt ở **cuối một episode AndroidControl thật** ("<goal A>, then tell me <câu hỏi về màn hình
     cuối>"). Các bước trước giữ nhãn thật, chỉ màn hình cuối có nhãn `answer`.
4. **Phần tử giả định trong W1 đang chọn ngẫu nhiên**, nên là ví dụ âm dễ. Trên AW, model lặp lại một phần tử **trông hợp
   lý**.
   - Cách sửa: ưu tiên phần tử cùng loại với đáp án đúng hoặc nằm gần nó, để có ví dụ âm khó hơn. Vẫn đo lại hai tiêu chí
     trên dữ liệu.
5. **Tỉ lệ trộn.** Dòng augment không được lấn át dữ liệu gốc. Đặt trần: W1 khoảng 13%, W2 khoảng 15%, W3 khoảng 5% số
   dòng.
   - Lý do: bài học 2026-11-07 cho thấy đổi trọng số `status` đơn thuần đã thất bại (−27 lần báo xong sai đổi lấy −26 lần
     báo xong đúng). Phải thêm **thông tin**, không chỉnh trọng số.
6. **Rò rỉ sang bài đánh giá:** không còn, vì v4 không dùng dòng nào từ AndroidWorld ngoài 1.736 dòng chọn app có sẵn
   trong v3 (19 template đã để dành).

## 1. Bản đồ lỗi của v3 (seed 30, 114 bài, `expected_gain.out`)
| Nhóm lỗi | Số bài | Nguyên nhân đã đo | Hướng sửa |
|---|---|---|---|
| Báo xong quá sớm | **35** | P(status) theo thứ tự bước: dữ liệu tăng từ 0 lên 28% ở bước 5–8; v3 ở trạng thái chưa xong đạt đỉnh ở bước 5. Model đọc phần goal còn lại rất yếu (P4: −8 điểm). Dữ liệu không có ví dụ "xong một phần, goal còn phần khác" | W2 |
| Kẹt vòng lặp | **26** (17 bài ≤ 20 bước) | Model bỏ qua "no change" (P1: −0,3 điểm), chép lại lần thất bại (P2: +1,3). Dữ liệu dạy lặp: 19,2% so với 5,4% | W1 |
| Cần trả lời bằng chữ (`answer`) | **25** | Hợp đồng không có hành động `answer`, nên trần của nhóm này là 0 | W3 |
| Đi lan man rồi hết bước | 12 | 19/33 app AW không có dòng huấn luyện nào; model không biết quy trình trong app | W5 |
| Cắt danh sách ở 40 phần tử | trùng các nhóm trên | 454/1.541 bước AW có hơn 40 ứng viên; nút "New Event" bị cắt 155/155 lần; Simple Calendar 0/17 | W4 |
| Harness ghi nhãn tác dụng sai | trùng các nhóm trên | 9% số click "no change" là đọc quá sớm | W0 |

## 2. Các nhánh sửa, tất cả vào cùng v4
Mỗi nhánh có **phép đo riêng** (thăm dò hoặc chỉ số chuyên biệt), nên vẫn biết nhánh nào có tác dụng mà không phải học
lại nhiều lần.

### W0: sửa harness (không cần học lại)
- Dòng lịch sử của bước k được chốt ở đầu bước k+1, so với danh sách phần tử của bước k+1. Đây đúng là định nghĩa lúc tạo
  dữ liệu (`add_history_v3.py`). Có test.
- **Đo:** tỉ lệ "no change" mà ảnh lại đổi từ 1% trở lên, trên AW.

### W1: tín hiệu "no change" (vòng lặp)
- Bỏ 52 dòng mục tiêu dạy lặp vô ích.
- Thêm khoảng 10% dòng mà lịch sử có "click phần tử khác → no change", nhãn giữ nguyên.
- Thêm khoảng 3% dòng có 2–3 lần thất bại liên tiếp.
- **Đã thử trên dữ liệu:** lặp lại sau "no change" giảm từ 19,2% xuống 1,6%; thông tin của chữ "no change" tăng ×20.
- **Cổng thăm dò:** P1 ≤ −5 điểm; P2 phải giảm (v3 đang tăng).

### W2: báo xong sớm
Đây là nhóm lớn nhất, 35 bài.

**(a) Goal nhiều phần, ghép từ hai episode AndroidControl cùng app.**
- Goal mới: "A. Then B."
- Ở màn hình cuối của A, nhãn là **đi tiếp** (bước đầu của B) chứ không phải `status`. `status` chỉ ở cuối B.
- Nguồn có đủ: 615 episode bắt đầu từ màn hình chính và 400 app (`v4_sizing.out`). Episode ghép dài khoảng 13 dòng, nên tỉ
  lệ `status` sẽ không còn tập trung ở bước 5–8.
- Sẽ chọn B bắt đầu từ màn hình chính và nối bằng `navigate_home`, để chuỗi màn hình vẫn thật.

**(b) Nhãn `status` lấy từ chính AndroidWorld, dùng bộ chấm hỏi sau mỗi bước.**
- Chạy trên instance luyện tập (seed khác 30), không dùng model ngoài.
- Trạng thái bộ chấm nói **đã xong** → dòng `status` dương.
- Trạng thái **chưa xong** mà model chọn `status` → **dòng âm cho `status`**. Cần thêm vào `train_d1.py` một thành phần loss
  unlikelihood, −log(1 − p_status), chỉ áp lên các dòng này. Có test.

**(c) Cổng thăm dò:**
- P3 (thêm 10 dòng `wait` vào đầu lịch sử) gần như không đổi.
- P4 (thêm vế việc chưa làm vào goal) phải giảm P(status) ít nhất 30 điểm (v3: −8).
- Trên các trạng thái AW chưa xong ở bước 3–5, P(status) trung bình phải giảm một nửa.

### W3: hành động `answer` (25 bài)
- **Hợp đồng v4:** thêm `answer` vào lựa chọn hành động, và một ô câu trả lời trỏ vào **chữ của một phần tử** trên màn hình
  (sao chép, giống cách `type_text` trỏ vào goal). Harness gửi `answer` theo đúng giao thức AW (`interaction_cache`).
- **Dữ liệu:** instance luyện tập của 25 template trả lời, đáp án lấy từ chính tham số của task (không cần model ngoài).
  Ở các màn hình mà đáp án xuất hiện nguyên văn trong chữ của một phần tử, tạo dòng "answer → phần tử đó".
- **Phải đo trước:** bao nhiêu trong 25 template có đáp án xuất hiện nguyên văn trên màn hình. Câu hỏi dạng đếm hoặc cộng
  tổng thì không sao chép được. Đã biết: 16 câu trả lời chuỗi, 8 số, 1 ngày, 1 giờ (mâu thuẫn tổng 26 so với 25 vẫn chưa
  giải quyết).
- **Cổng:** độ chính xác `answer` trên instance luyện tập của các template được để dành.

### W4: bỏ cắt 40 phần tử
- Học và phục vụ với danh sách đầy đủ theo quy tắc của v3.
- 10,8% số dòng có hơn 40 ứng viên. Số token p99 là 2.659, tối đa 6.456, vẫn rất xa giới hạn 128k (`cap_tokens.out`).
- **Cổng:** "New Event" luôn có mặt trong danh sách. Độ chính xác trên 656 dòng AC-High có hơn 40 ứng viên không giảm.

### W5: dữ liệu quy trình trong app AW (RFT), sau khi v4 qua các cổng
- Chạy v4 trên instance luyện tập của các template **không để dành**. Một phần ba số template được để dành và không bao giờ
  đem học.
- Lấy các lần bộ chấm xác nhận thành công làm dòng bắt chước, lấy các bước rẽ giữa lần thất bại và lần thành công của cùng
  bài, rồi học lại thành v4.1, v4.2.
- Đây là đòn bẩy mạnh nhất theo mốc AppVLM (11,4 → 37,8), và là nguồn duy nhất không dùng model ngoài cho 19 app chưa có
  dữ liệu.
- Mức ngẫu nhiên khi khám phá chọn bằng số đo trên AW (tỉ lệ thành công mới trên mỗi lần chạy), không đặt tay.

## 3. Kỳ vọng
Đây là ước lượng, chưa phải kết quả. Mỗi số là khoảng, kèm lý do.

| Nhánh | Nhóm bài | Kỳ vọng thêm | Lý do / rủi ro |
|---|---|---|---|
| W1 + W0 | 17 bài kẹt lặp (≤ 20 bước) | +2 đến +5 | Masking phá vòng lặp cơ học chỉ cứu thật khoảng 1 bài; v4 học được hành động đúng nên kỳ vọng hơn. Rủi ro: chuyển sang lỗi báo xong sớm |
| W2 | 35 bài báo xong sớm | +4 đến +10 | Không biết bao nhiêu bài đang đi đúng hướng lúc báo xong; nhãn của bộ chấm sẽ cho biết |
| W3 | 25 bài trả lời | +3 đến +8 | Phụ thuộc tỉ lệ đáp án sao chép được; đo trước |
| W4 | Simple Calendar và các màn hình dài | +1 đến +4 | Bài thêm sự kiện còn cần gõ chữ đúng |
| **v4 cộng lại** | | **+10 đến +27 bài, tức khoảng 23–37%** | Các nhánh tương tác với nhau, nên không cộng thẳng được |
| W5, các vòng RFT | 12 bài lan man và các bài trong app chưa có dữ liệu | phần còn lại để tới khoảng 40% | Theo mốc AppVLM; harness khác nên không bảo đảm |

## 4. Học và kiểm tra
1. **Dữ liệu v4** = v3 + W1 + W2a + W3 + W4, kèm run card. Kiểm tra trên dữ liệu trước khi học:
   - tỉ lệ lặp sau "no change" thấp hơn sau "changed", và thông tin ≥ 10 lần v3;
   - phân bố `status` theo bước phẳng hơn;
   - "New Event" có trong danh sách.
2. **Học v4:** cùng công thức v3 (lr 1e-5, embedding đóng băng, 1 epoch), thêm loss unlikelihood cho các dòng âm của W2b.
   - Thứ tự: v4 trước, rồi một vòng AW luyện tập để lấy dòng W2b và W5, rồi v4.1.
3. **Cổng theo thứ tự:**
   - các phép thăm dò P1–P4;
   - AC-High không thấp hơn v3 quá 1 điểm;
   - `answer` trên template để dành;
   - AW 116 bài, seed 30, harness W0, với v3 và v4.
   - Báo theo từng nhóm lỗi ở mục 1. SR cuối cùng chạy 3 seed.
