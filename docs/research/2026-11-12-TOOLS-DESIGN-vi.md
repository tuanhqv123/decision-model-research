# Cho decision model dùng tool: thiết kế cho các lỗi "bị ràng buộc chặn" (2026-11-12)

## Ý tưởng

Model đã giỏi một việc: đọc mục tiêu, **chỉ ra** đúng phần tử, và **chép** đúng đoạn chữ có trong đầu vào. Cái nó thiếu là phép tính, trí nhớ qua nhiều màn hình, và những thông tin không có trong cây phần tử.

Vì vậy ta để một **tool tất định** (code thường, không phải model) làm phần tính và phần nhớ, rồi **in kết quả thành các dòng chữ trong state**. Model chỉ cần trỏ hoặc chép đúng dòng. Như vậy vẫn là một lượt đọc và không sinh token. Ràng buộc "không model ngoài" vẫn giữ, vì tool không phải model. Ngoại lệ duy nhất là OCR, xem bảng dưới.

| Lỗi bị chặn | Tool (tất định, chạy ở harness) | Dòng thêm vào state | Model cần làm gì | Dạy bằng dữ liệu nào (lấy từ AndroidControl) | Bài có thể gỡ |
|---|---|---|---|---|---|
| Tính ngày ("Thứ Năm tới", "ngày mai", "2 tuần nữa") | Đọc ngày giờ trên máy, tính sẵn bảng ngày | `Dates: today Sun Oct 15 2023 · tomorrow Mon Oct 16 · this Thu Oct 19 · next week Oct 22–28 · in two weeks Sun Oct 29` | Chép đúng ngày bằng câu hỏi `type_text`, hoặc chọn ô lịch có nhãn khớp | Viết lại các bài AndroidControl có gõ ngày theo kiểu tương đối, rồi thêm dòng Dates chứa đáp án | 4–7 (Calendar thêm/xóa sự kiện, các câu hỏi về ngày) |
| Giờ + khoảng thời gian ("+1 tiếng", "lúc 8:00 pm" → "20:00") | Đổi mọi giờ có trong mục tiêu sang các dạng tương đương và các mốc cộng thêm | `Times: 8:00pm = 20:00 · 13:00 +30m = 13:30 · +1h = 14:00` | Chép | Tương tự, viết lại từ dữ liệu AndroidControl | 1–3 |
| Cộng tổng, đổi đơn vị, đếm qua nhiều màn hình | **Bộ nhớ:** hành động mới `note` lưu chữ của phần tử được chọn. Tool cộng và đổi đơn vị | `Notes: 2:15:00 · 0:45:00 → total 3:00:00 = 180 min` | Bấm `note` trên từng mục cần ghi, rồi `answer` bằng cách chép dòng tổng | Tổng hợp từ các bài AndroidControl có danh sách (ghi n mục rồi trả lời tổng) | 2–4 |
| Tìm bản trùng qua nhiều màn hình | (a) Gom chữ từng hàng của danh sách và đánh dấu hàng giống hệt; (b) với bản khó: bộ nhớ `note` lưu chữ của màn hình chi tiết, rồi tool so các note | (a) `… · same as row 3` / `… · seen before`; (b) `Notes: 2 = 5 (identical)` | (a) Chọn hàng bị đánh dấu rồi xóa; (b) mở từng biến thể, ghi note, xóa bản tool báo trùng | Tổng hợp từ danh sách AndroidControl | 1–2. Bản dễ +1 (ExpenseDeleteDuplicates). Bản khó 0–1, vì các nhiễu cùng tên chỉ khác nhau ở trường ẩn và cần khoảng 18 bước trên ngân sách 24 (code upstream `recipe.py`) |
| Màu hoặc biểu tượng mang ý nghĩa (độ ưu tiên) | Lấy màu chủ đạo của phần tử nhỏ từ ảnh chụp | Thêm `color: red` vào nhãn phần tử | Nối "high priority" với màu đỏ | Không có nhãn trong AndroidControl, phải dựa vào kiến thức sẵn có của d1. **Không chắc** | 0–2 |
| Gợi ý clipboard | Đọc clipboard của máy (bằng chính hàm `get_clipboard` của AndroidWorld) | `Clipboard: I think, therefore I am.` | Chép bằng `type_text` | Thêm dòng Clipboard vào các dòng gõ chữ của AndroidControl khi mục tiêu nhắc tới clipboard (dạng tổng hợp) | 2 |
| Chữ chỉ có trong ảnh (hóa đơn, chi tiêu) | **OCR** | `Text in image: 2023-03-23, Monitor Stand, $33.22` | Chép | **OCR là một model.** Dùng thì vi phạm ràng buộc "không model ngoài", **người vận hành phải quyết định** | 4 nếu được phép |

**Tổng:** lạc quan khoảng 10–15 bài. Thực tế khoảng 5–8 bài nếu không có OCR.

## Điều kiện để làm đúng

- **Đúng giao thức.** Agent nào cũng được dùng thông tin này:
  - ngày giờ có sẵn trên thanh trạng thái;
  - clipboard: agent dán được;
  - màu: là pixel agent vẫn nhìn thấy;
  - phép tính: làm phía agent.

  Paper sẽ khai báo rõ đây là một "decision model có tool".
- **Khớp giữa train và serve.** Các dòng tool phải được dựng bằng cùng một hàm ở cả hai phía (`ac_common`), như mọi phần khác của state. Mỗi tool cần một contract mới (v7) và phải train lại.
- **Kiểm tra sớm như thường lệ.** Dòng mới có thể gây hại bất ngờ, như dòng "trạng thái đã đúng" đã làm (báo xong sai 108/690).

## Thứ tự đề xuất

1. Để bản final hiện tại chạy xong và đo. Nó là mốc nền.
2. Trong lúc đó code tool layer: harness, contract v7, dữ liệu tổng hợp, kèm test. Thứ tự làm:
   1. Dates/Times và Clipboard: rẻ nhất, gỡ nhiều bài nhất;
   2. bộ nhớ `note` kèm cộng tổng;
   3. đánh dấu trùng;
   4. màu.
3. Train v7 một lần, tiếp từ checkpoint final, rồi so cặp với final.

## Đối chiếu với hướng dẫn của chính TypeSafe (Jev), đọc 2026-11-12

Nguồn: danh sách cộng đồng awesome-jev-use-cases, trích các trang của TypeSafe. Bản lưu nằm ở `references/raw/awesome-jev-use-cases-README-2026-11-12.md`; ledger `typesafe-jev-limits-2026`.

**Giới hạn TypeSafe tự ghi cho Jev 1.13:**
- "It is not a calculator. Keep math in code, which includes counting and date comparison, and use Jev to extract the parts." Điều này đúng với thiết kế ở trên: code tính, model chỉ trỏ hoặc chép.
- Hai cookbook đi theo đúng kiểu này:
  - **Date extraction:** model trích các phần của ngày, code giải ra ngày thật và kiểm tra.
  - **Pre-parsed value extraction:** regex tìm sẵn các ứng viên (email, số, tiền), model chọn, code chuẩn hóa.
- "Accuracy falls when the state carries content unrelated to the question." Vì vậy dòng tool chỉ được thêm khi mục tiêu cần đến nó (ví dụ mục tiêu có chữ chỉ ngày tương đối, hoặc nhắc tới clipboard). Không thêm vào mọi bước.

**Hệ quả cho thiết kế: một biến thể có thể gọn hơn cho ngày và giờ.**
- Thay vì in cả bảng ngày, hỏi model thêm một câu dạng *choice* trong cùng lượt đọc (kiểu "speculative fan-out"). Ví dụ "ngày đích so với hôm nay": {hôm nay, ngày mai, thứ Hai … Chủ Nhật tuần này / tuần sau, sau N ngày/tuần}.
- Code tính ra ngày thật rồi gõ vào ô (kiểu function calling: model chọn tham số có kiểu, code thực thi).
- Cách này cần thêm một câu hỏi vào contract và cần dữ liệu tổng hợp để dạy.
- Sẽ chọn giữa "in bảng" và "câu hỏi choice" khi bắt đầu làm v7.

**Về OCR:** `awlevin/typesafe-computer-use` dùng OCR để đọc màn hình cho Jev. Việc này cho thấy các hệ dùng decision model thường ghép thêm OCR làm tool, nhưng không thay đổi ràng buộc "không model ngoài" của mình. Việc quyết định vẫn thuộc về người vận hành.

## AndroidWorld nói gì về tool, và các bên khác làm thế nào (đọc nguồn gốc 2026-11-12)

Nguồn: code upstream `android_world@55ddaf5`, paper v5, bảng xếp hạng (bản CSV lưu ngày 2026-11-12 tại `references/raw/androidworld-leaderboard-2026-11-12.csv`) và paper của từng agent.

**Luật của AndroidWorld**
- **Không có luật nào cấm tool.** README chỉ mô tả agent đọc ảnh chụp và các phần tử UI qua AndroidEnv, chọn một hành động rồi thực thi. Không có câu nào cấm.
- **Agent tham chiếu M3A cũng dùng thêm thứ ngoài model:** mỗi bước nó viết một bản tóm tắt bằng LLM làm bộ nhớ ("can be used as memory … shared between different apps"), và prompt có mẹo do người viết sẵn.
- **Paper tự nêu điểm yếu:** agent thất bại ở các việc cần trí nhớ, chép nội dung giữa các app, và nhân số. Bài hỏi thông tin được chấm bằng các phép count / sum / identity.
- **Đồng hồ thiết bị bị đặt cố định ở 15/10/2023, 15:34 UTC.** Vì vậy bảng ngày tính từ đồng hồ máy là tất định.

**Tập hành động (`json_action.py`)**
- Gồm: click, double_tap, scroll, swipe, input_text, navigate_home, navigate_back, keyboard_enter, open_app, status, wait, long_press, answer.
- Không có hành động clipboard hay ngày tháng.
- Clipboard chỉ đọc được qua hàm helper `adb_utils.get_clipboard_contents`, và hàm này đưa app Clipper lên màn hình trước khi đọc.

**Bảng xếp hạng: tự khai báo, không kiểm chứng**
- **Model Type:**
  - **Model** = "one LLM/VLM call";
  - **AI agent** = "several LLM calls … or an LLM wrapped into an advanced agent with memory, subgoal planning, etc.".
- **Screen Representation:**
  - Screenshot: 32;
  - Screenshot + A11y tree: 12;
  - A11y tree: 4;
  - để trống: 1.
- Không có cột riêng cho tool; tool được khai trong cột Note. 44/49 dòng ghi Number of trials = 1.

**Các agent khác**
- **Bộ nhớ / ghi chú rất phổ biến:** M3A, Mobile-Agent-v3 (Notetaker), Agent S2 (`save_to_knowledge`), V-Droid (tóm tắt các bước).
- **OCR hiếm.** Chỉ Agent S2 mô tả dùng Tesseract, và không rõ nó có bật khi chạy AndroidWorld hay không. Mobile-Agent-v2/E dùng OCR nhưng không có số AndroidWorld.
- **Không nguồn nào nhắc tới tool máy tính hay tool ngày giờ.**
- **MAI-UI báo số "without … external tools".** V-Droid và MobileRL thì train trên chính các task của AndroidWorld.

**Kết luận cho mình**

1. Lớp tool tất định (bảng ngày, note kèm cộng tổng, đánh dấu hàng trùng, màu pixel) **phù hợp với chuẩn mực** của AndroidWorld. Khi công bố phải khai từng tool. Xếp nhóm **"AI agent"** là trung thực nhất (vì có bộ nhớ). Nếu xếp "Model", cột Note phải liệt kê đủ các tool.
2. **Bỏ tool đọc clipboard.** Lý do:
   - đây là kênh adb đặc quyền, không phải thông tin hiện trên màn hình;
   - hàm đọc làm thay đổi màn hình (đưa Clipper lên);
   - nó có thể biến bài "dán" thành "gõ".

   Bài clipboard nên được giải bằng hành động trên màn hình (nhấn giữ ô → Paste, hoặc chip gợi ý trên bàn phím). Nếu vẫn dùng thì phải báo kết quả có và không có tool trên 3 bài clipboard.
3. **OCR:** hiếm trên bảng xếp hạng và trái với ràng buộc "không model ngoài". Giữ nguyên là không dùng, trừ khi người vận hành đổi ràng buộc.
4. **Khi báo cáo số AndroidWorld của mình**, khai đủ:
   - số lần chạy (1);
   - phiên bản ngân sách bước;
   - không dùng dữ liệu AndroidWorld khi train;
   - harness và các tool đi kèm.
