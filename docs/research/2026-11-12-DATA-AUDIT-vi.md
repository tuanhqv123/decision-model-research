# Kiểm định dữ liệu v3 đã học: model học được gì, vì sao, và hậu quả trên AndroidWorld (2026-11-12)

Mọi số đều tính lại được từ các script trong `benchmarks/results/2026-11-12-data-audit/`, mỗi script ghi kết quả vào file
`.out` cùng tên. Tài liệu này đi sâu hơn `2026-11-12-PATTERNS-data-vi.md`. Bản đó chỉ ra **chỗ** model hỏng; bản này trả lời
**vì sao** dữ liệu khiến model hỏng ở đó.

- **Dữ liệu huấn luyện:** `compact_awlike_full_v3.jsonl`, gồm 27.454 dòng và 4.197 episode AndroidControl. Danh sách phần
  tử được trích bằng chính `forest_to_ui_elements` của AndroidWorld.
- **Model:** v3, checkpoint ckpt_1824.

## Tóm tắt trong năm câu
1. Model **không đọc** chữ "→ no change". Đổi nó thành "→ screen changed" trên đúng những trạng thái v3 đã gặp thì 97% quyết
   định giữ nguyên.
2. Model bỏ qua chữ đó là **hợp lý với dữ liệu**. Trong dữ liệu, chữ này chỉ mang 0,09% thông tin về nhãn đúng. Ở chỗ có
   tín hiệu thì tín hiệu còn dạy ngược: sau "no change", nhãn đúng **lặp lại** nhiều gấp 6 lần (14,7% so với 2,3%).
3. Nguyên nhân là chính dữ liệu:
   - nhãn "no change" vừa hiếm (1,9%) vừa hay sai (với scroll, 46% trường hợp ảnh thực ra đã đổi ≥ 1%);
   - phía sau nhãn ấy thường là người gán nhãn làm lại đúng thao tác vừa hỏng, và 52/79 lần làm lại cũng không ăn.
4. Báo xong sớm đi theo **thứ tự bước**. Tỉ lệ `status` trong dữ liệu tăng từ 0% ở bước 0 lên khoảng 28% ở bước 5–8. Xác
   suất báo xong của v3 ở trạng thái **chưa xong** trên AW có cùng hình dạng, đỉnh ở bước 5.
5. Thêm dòng vào dữ liệu thôi thì chưa đủ. Phải làm cho chữ "no change" **đúng** và **có ích** trong dữ liệu, thì model mới có
   lý do để học đọc nó.

## L1. Model dùng thông tin gì? (thí nghiệm phản thực tế, `nochange_sensitivity.out`)
- **Trạng thái dùng để đo:** 124 trạng thái self-play seed 1001, mỗi trạng thái nằm ngay sau một hành động (không phải `wait`)
  bị ghi là "→ no change".
- **Cách đo:** gửi lại đúng đầu vào (ảnh, goal, lịch sử, danh sách phần tử) cho cùng server.
- **Kiểm tra phép đo:** gửi lại y nguyên thì 124/124 cho đúng hành động đã ghi. Model tất định, phép đo đáng tin.

| Biến thể dòng lịch sử cuối | Lặp lại hành động vừa hỏng | Báo xong |
|---|---|---|
| A. Như thật: "… → no change" | 40/124 (32%) | 9 |
| B. Đổi thành "… → screen changed" | 38/124 (31%) | 10 |
| C. Xóa hẳn dòng | 63/124 (51%) | 1 |

- A và B cho cùng hành động 98%, cùng hành động và cùng điểm 97%. Model **bỏ qua** việc hành động có tác dụng hay không.
- C khác A rõ: tỉ lệ lặp tăng từ 32% lên 51%. Model **có đọc** "mình vừa làm gì". Thói quen nó học được là "thường không làm
  lại đúng việc vừa làm", và thói quen đó không phụ thuộc vào kết quả của việc vừa làm.

**Lỗ hổng còn mở:**
- Hầu hết các trạng thái là click (119/124). Kết luận cho scroll và gõ chữ chưa đủ mẫu.
- Self-play có bật masking. Vì thế đây là các lần đầu gặp; vòng lặp kéo dài (86% trên seed 30) không được đo lại ở đây.

## L2. Vì sao model học bỏ qua: chữ "no change" trong dữ liệu (`effect_information.out`, `nochange_truth.out`)
**Hiếm:** 412/22.026 dòng (1,9%). Không tính các dòng sau `wait`.

**Không mang thông tin, còn dạy ngược:**

| Hành động trước, tác dụng | n | Nhãn đúng lặp lại nó | Nhãn đúng là `status` | Nhãn đúng là `wait` |
|---|---|---|---|---|
| click, đổi | 15.724 | 2,3% | 21,2% | 5,4% |
| click, **không đổi** | 293 | **14,7%** | 14,7% | 12,3% |
| scroll, đổi | 1.733 | 46,6% | 13,2% | 1,2% |
| scroll, **không đổi** | 90 | 37,8% | **33,3%** | 7,8% |

- I(chữ tác dụng ; loại hành động đúng) = **0,0015 bit**, tức 0,09% độ bất định của nhãn.
- Một model tối ưu trên dữ liệu này có lý do bỏ qua chữ đó. Nếu có học được gì từ nó thì sẽ là "không đổi thì làm lại",
  đúng là thứ hỏng trên AW.

**Thường sai, xét theo ảnh:**

| Hành động mà lịch sử ghi "không đổi" | Ảnh đổi dưới 1% | 1–5% | 5–20% | 20% trở lên |
|---|---|---|---|---|
| click (293) | 75% | 8% | 12% | 5% |
| scroll (90) | 54% | 13% | 23% | 9% |

Đọc tay 41 dòng scroll và 25 dòng click (`effect_vs_pixels.out`, `click_nochange.out`) thấy ba nguồn sai:
- **Danh sách phần tử không thấy thay đổi:** vuốt carousel ảnh (ảnh đổi 5–50%), chấm sao `RatingBar`, chọn AM/PM, bàn phím số
  nhập tiền. Thao tác **có tác dụng** nhưng bị ghi "không đổi".
- **Thao tác đã hoàn tất việc nhưng màn hình không đổi:** bấm "SAVE" rồi `status`. 43/293 click "không đổi" được theo ngay
  bằng `status`.
- **Màn hình đang tải:** "không đổi" rồi `wait` với cùng câu mô tả bước, 36/293.

**Người gán nhãn tự lặp** (`retry_outcome.out`):
- Có 79 lần nhãn đúng làm lại đúng thao tác vừa "không đổi". Sau đó 27 lần màn hình đổi, **52 lần vẫn không đổi**.
- Trong 79 lần đó, 51 lần ảnh thực ra có đổi (≥ 1%). Phần lớn là carousel, tức nhãn sai chứ không phải người lặp vô ích.

## L3. Hậu quả trên AndroidWorld (`../2026-11-12-patterns/situation_gap.out`)
- Bước ngay sau một hành động không tác dụng chiếm **28% số bước AW**, nhưng chỉ 1,4% dữ liệu huấn luyện. Các bước này
  chiếm **73% số bước hỏng**.
- Trên AW, "không đổi" gần như luôn đúng nghĩa đen, tức thao tác thật sự không ăn. Các trường hợp hay gặp:
  - bấm vào ô đã được chọn;
  - bấm "Save" không có tác dụng (44 lần);
  - cuộn tới cuối danh sách.

  Ngược lại, trong dữ liệu AndroidControl, "không đổi" thường có nghĩa là "đã ăn nhưng danh sách phần tử không thấy". Cùng
  một chữ mang hai nghĩa ở hai môi trường.

## L4. Báo xong sớm là pattern theo vị trí bước (`status_by_step_train.out`, `status_by_step_aw.out`)
| Bước | Dữ liệu: P(status) | AW, trạng thái chưa xong: P(status) trung bình của v3 | AW, chưa xong: v3 chọn status |
|---|---|---|---|
| 1 | 6,2% | 0,5% | 0% |
| 3 | 16,2% | 6,0% | 2,0% |
| 4 | 22,0% | 11,0% | 8,2% |
| 5 | 25,6% | **17,8%** | **20,5%** |
| 6–8 | 28–29% | 8–13% | 3–13% |
| 15+ | 22,1% | 6,5% | 0,9% |

- Ở 15 trạng thái **đã xong**, P(status) trung bình từ bước 3 trở đi là 52–99%. Model **có** phân biệt được phần nào giữa xong
  và chưa xong; không phải chỉ đếm bước.
- Nhưng ở trạng thái chưa xong, xác suất báo xong tăng mạnh trong khoảng bước 3–5, đúng vùng mà dữ liệu dạy rằng episode hay
  kết thúc (episode trung vị dài 6 dòng; chỉ 3% dài hơn 15 dòng).
- Dữ liệu không có ví dụ nào kiểu "màn hình trông như xong, nhưng goal còn phần khác". `status` luôn là dòng cuối
  (4.197/4.197).

**Lỗ hổng:** chỉ có 15 trạng thái đã xong, nên không tính AUROC giữa xong và chưa xong một cách đáng tin.

## L5. Lệch giữa lúc học và lúc chạy ở ô nhập (`../2026-11-12-patterns/focus_skew.out`, `ime_before.out`)
- **Ô đang được chọn:** AW không bao giờ báo `is_focused` (0/18.470 phần tử). Dữ liệu báo ô đang được chọn ở 584/629 màn hình
  sau khi click vào ô.
- **Click vào ô trong dữ liệu:** 674/699 lần làm màn hình đổi, chủ yếu vì bàn phím bật lên.
- **Click vào ô trên AW:** ô thường đã được chọn sẵn và bàn phím đã có, nên click thêm không đổi gì.

## Hệ quả cho dữ liệu (thay mục 5 của bản PATTERNS)
Mục tiêu: làm cho dòng tác dụng **đúng** và **có ích**, rồi mới thêm khối lượng.

1. **Sửa nhãn tác dụng.** Dùng thêm ảnh làm nhân chứng thứ hai: danh sách phần tử không đổi nhưng ảnh đổi ≥ 5% thì ghi là
   "đổi" (carousel, sao, picker).
   - Ở lúc chạy, harness phải dùng **cùng quy tắc** đó.
   - Trên AW, ảnh gần như luôn khác vì con trỏ nhấp nháy. Ngưỡng phải đo trên ảnh AW trước; đây là việc phải đo, chưa có số.
2. **Bỏ hoặc gắn cờ các lần người gán nhãn lặp vô ích:** 52 dòng nhãn đúng là lặp lại thao tác vừa không đổi, và lần lặp đó
   cũng không đổi. Nếu không bỏ, dữ liệu dạy vòng lặp.
3. **Thêm dòng "không đổi, rồi làm việc khác"** với hành động chèn vào đúng sự thật vật lý (D1–D4 của bản PATTERNS): gõ xong
   rồi bấm lại ô, bấm vào ô đã được chọn, cuộn lên khi đang ở đầu danh sách, bấm vào mục đang được chọn.
   - **Tiêu chí đo được ngay trên dữ liệu, trước khi học:** sau khi sửa, P(lặp | không đổi) phải **thấp hơn** P(lặp | đổi),
     và I(chữ tác dụng ; nhãn) phải tăng rõ so với 0,0015 bit.
4. **Báo xong:** thêm các tình huống "xong một phần, chưa xong goal" (D5, nối hai episode cùng app), để tỉ lệ `status` theo
   bước không còn là một đường cong theo vị trí.
5. **Focus:** chọn một trong hai: bỏ cờ `focused` khỏi dữ liệu học, hoặc sửa harness để đọc được focus.

**Kiểm tra sau khi học:**
- Chạy lại đúng thí nghiệm L1 trên v4.
- Đạt nếu: đổi "no change" thành "screen changed" làm quyết định đổi ở **phần lớn** trạng thái (không còn 97% giống hệt), và tỉ
  lệ lặp trong biến thể A giảm rõ dưới 32%.
- Đồng thời AC-High không được giảm quá 1 điểm.
