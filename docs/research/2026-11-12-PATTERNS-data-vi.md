# Model v3 học sai pattern nào, và cần thêm dữ liệu gì (2026-11-12)

Mọi số trong tài liệu này đều tính lại được từ các script và file kết quả trong `benchmarks/results/2026-11-12-patterns/`.
- **Dữ liệu huấn luyện** là `compact_awlike_full_v3.jsonl`: 27.454 dòng, 4.197 episode.
- **Dữ liệu AndroidWorld** là các lần chạy v3 ckpt_1824 trên seed 30, không bật masking: 1.463 bước có chấm điểm.

Tài liệu này thay cho `2026-11-12-PLAN-passk-gate-vi.md`. Bản đó tối ưu cách khám phá trước khi biết model sai ở đâu, nên
tạm hoãn.

## 1. Cách tìm pattern (để không phải chọn tay)
Mỗi bước được gắn một **tình huống**, gồm ba đặc trưng mà cả dữ liệu huấn luyện lẫn log AndroidWorld đều có:
1. hành động ngay trước đó;
2. hành động đó có làm màn hình đổi không, đúng như dòng lịch sử model đọc được ("→ screen changed" hay "→ no change");
3. hành động đó có phải là click vào ô nhập chữ không.

Với từng tình huống, mình đếm bốn thứ:
- số lần nó xuất hiện trong dữ liệu huấn luyện và trên AndroidWorld;
- tỉ lệ bước hỏng trên AndroidWorld, tức hành động không tác dụng hoặc báo xong sai;
- dữ liệu dạy làm gì trong tình huống đó;
- v3 thực tế làm gì.

Tình huống được ưu tiên theo **số bước hỏng**: đó là chỗ sửa xong thì được nhiều nhất. Không có ngưỡng nào do mình tự đặt.

## 2. Kết quả (`situation_gap.out`)
Trên AndroidWorld có **493 bước hỏng**: 444 bước hành động không tác dụng và 49 lần báo xong quá sớm.

| Tình huống (hành động trước, tác dụng, có phải ô nhập) | Dòng huấn luyện | Bước AW | AW/huấn luyện | Bước hỏng trên AW | % tổng bước hỏng | Dữ liệu dạy | v3 làm trên AW |
|---|---|---|---|---|---|---|---|
| click, **không đổi**, không phải ô nhập | 268 | 201 | 14× | 161 (80%) | **33%** | click 62%, status 16%, wait 13% | click 87% |
| scroll, **không đổi** | 90 | 143 | 30× | 143 (100%) | **29%** | scroll 46%, status 33% | scroll 99% |
| click, đổi, không phải ô nhập | 15.050 | 450 | 0,6× | 71 (16%) | 14% | click 61%, status 22% | click 78%, status 10% |
| click vào ô nhập, **không đổi** | 25 | 68 | 51× | 57 (84%) | **12%** | gõ chữ 60%, click 36% | click 84%, gõ chữ 16% |
| gõ chữ, đổi | 1.664 | 59 | 0,7× | 21 (36%) | 4% | click 90% | click 92% |

Cột "AW/huấn luyện" so tỉ lệ một tình huống chiếm trong số bước AW với tỉ lệ nó chiếm trong số dòng huấn luyện.

## 3. Pattern 1: "hành động vừa rồi không có tác dụng"
**Đây là chỗ hỏng lớn nhất: 73% số bước hỏng.**

| | Dữ liệu huấn luyện | AndroidWorld |
|---|---|---|
| Bước ngay sau một hành động không tác dụng | **387 dòng = 1,4%** | **416 bước = 28%** |
| Sau click không tác dụng, click lại **đúng phần tử đó** | 45/539 = 8% (`train_patterns.out`) | 540/625 = 86% (`aw_patterns.out`) |
| Sau scroll không tác dụng, scroll tiếp **cùng hướng** | 34/41 (`detail_checks.out`) | 142/142 |
| Sau click ô nhập không tác dụng, click lại đúng ô đó | gõ chữ 15 / click 8 (chỉ 23 dòng) | 120/132 = 91% |

**Model học sai ở đâu:**
- **Click:** dữ liệu không dạy lặp lại (chỉ 8%), nhưng tình huống này gần như không có trong dữ liệu. Model chưa từng học được
  rằng dòng "→ no change" phải làm quyết định thay đổi.
  - Khi lặp, màn hình giống hệt và lịch sử chỉ thêm một dòng, nên đầu vào gần như trùng với bước trước. Model cho ra đúng
    quyết định cũ. Đây là suy luận từ cơ chế, chưa được đo trực tiếp.
- **Scroll:** chính dữ liệu dạy sai. 34 trên 41 dòng scroll tiếp cùng hướng sau khi màn hình không đổi. Cần đọc tay 41 dòng
  này: có thể đó là danh sách tải chậm, hoặc nhãn "không đổi" bị sai.
- **Ô nhập:** trong dữ liệu, click vào ô gần như luôn làm màn hình đổi (674/699). Lý do là bàn phím bật lên: 637/699 màn hình
  sau click có phần tử bàn phím (`focus_skew.out`).
  - Trên AW, ô nhập thường đã được chọn sẵn, bàn phím đã có từ trước, nên click thêm không đổi gì: 12/37 trường hợp trong
    self-play, cả 12 đều có bàn phím từ trước (`aw_patterns.out`).
  - Ngoài ra AW **không bao giờ** báo `is_focused` (0/18.470 phần tử), trong khi dữ liệu có 584/629 màn hình báo ô đang được
    chọn. Đây là một lệch giữa lúc học và lúc chạy.

**Model click vào đâu mà không có tác dụng (`detail_checks.out`):**
- 330/353 cú click như vậy rơi vào phần tử **có thể bấm**. Đứng đầu là:
  - ô nhập đã chứa chữ vừa gõ: 65 + 25 + 15 lần. Model gõ xong rồi cứ bấm lại vào ô thay vì đi tiếp;
  - "Save" bấm mà không có tác dụng: 44 lần;
  - ô lịch không nhãn: 86 lần.
- Vì vậy dữ liệu bổ sung phải mô phỏng **các tình huống đó**, không phải bấm vào phần tử không bấm được.

## 4. Pattern 2: báo xong quá sớm (10% số bước hỏng)
- 49 lần báo xong sai. Bước trung vị là 5. 47/49 lần ngay sau một click, và 37/49 lần click đó có làm màn hình đổi
  (`aw_patterns.out`).
- Dữ liệu: `status` luôn là dòng cuối, có 4.197 dòng như vậy, và 80% đứng ngay sau một click làm màn hình đổi.
  - Sau một click làm màn hình đổi, 22% số dòng huấn luyện là `status`.
  - Episode ngắn: trung vị 6 dòng, p90 là 11 dòng, chỉ 3% dài hơn 15 dòng (`train_patterns.out`).
- **Pattern model đã học:** "vài bước, một click thành công, vậy là xong". Trên AW, 33/49 lần báo xong sai rơi vào bài có
  ngân sách trên 10 bước.

## 5. Dữ liệu cần thêm, theo thứ tự số bước hỏng

Mọi cách tạo dữ liệu dưới đây chỉ dùng dữ liệu AndroidControl đang có và các quy tắc kiểm chứng được, không dùng model
ngoài. Mỗi cách đều phải giữ đúng sự thật vật lý: hành động được chèn vào phải thực sự không có tác dụng trên màn hình đó.

| # | Sửa cho | Cách tạo dòng mới | Vì sao đúng về vật lý | Nguồn có sẵn (cần đếm chính xác) |
|---|---|---|---|---|
| D1 | Gõ xong rồi bấm lại vào ô (65+25+15 lần) | Lấy dòng ngay sau một lần gõ chữ; chèn vào lịch sử "click [ô vừa gõ] → no change"; nhãn giữ nguyên (dữ liệu: click 90%, thường là Save hoặc nút tiếp theo) | Ô đã chọn sẵn và bàn phím đã có, nên click không đổi gì | khoảng 1.664 dòng tình huống "gõ chữ, đổi" |
| D2 | Bấm vào ô đã chọn sẵn (12% số bước hỏng) | Lấy dòng có nhãn gõ chữ mà ô đã được chọn và bàn phím đã có; chèn "click [ô đó] → no change"; nhãn vẫn là gõ chữ | Như trên | khoảng 194 dòng (179 + 15 trong `ime_before.out`) |
| D3 | Scroll mãi cùng hướng (29%) | Lấy màn hình danh sách ngay sau khi mở app; chèn "scroll up → no change" (đang ở đầu danh sách); nhãn giữ nguyên. Song song, đọc tay 41 dòng dạy scroll cùng hướng, nếu sai nhãn thì sửa | Ở đầu danh sách thì cuộn lên không đổi gì | Cần đếm |
| D4 | Click lại phần tử vừa không tác dụng (33%) | Dùng 268 dòng có thật. Thêm biến thể: chèn "click [phần tử đang được chọn, `selected`] → no change" trước dòng gốc | Bấm vào tab hay mục đang được chọn thường không đổi gì. **Cần kiểm chứng**, vì không phải app nào cũng vậy | Cần đếm |
| D5 | Báo xong sớm (10%) | Nối hai episode cùng app ("A, rồi B"); ở màn hình cuối của A thì nhãn là đi tiếp (`navigate_home` hoặc `open_app`), không phải `status` | Màn hình "A xong" là có thật, chỉ là goal chưa xong | Cần đếm số cặp cùng app |
| D6 | Lệch focus giữa lúc học và lúc chạy | Bỏ cờ `focused` khỏi đầu vào và khỏi phép so "có đổi không" khi học, hoặc sửa harness để đọc được focus. Phải chọn một trong hai | Làm cho lúc học và lúc chạy nhìn cùng một kiểu | Toàn bộ dữ liệu |

**Kiểm tra trước khi học lại (không cần GPU dài, không cần máy giả lập):**
- Dựng một tập đánh giá **AW-NC**. Mỗi mẫu là đầu vào model đã thực sự thấy trong self-play seed 1001, tại bước ngay sau
  một hành động không tác dụng. `state_in` đã được ghi, nên gửi lại đúng y nguyên được.
- Đo tỉ lệ model chọn lại đúng hành động vừa thất bại:
  - v3 hiện tại: dự kiến khoảng 86%;
  - v4 (thêm D1–D4): phải giảm rõ, và AC-High không được giảm quá 1 điểm.
- Báo xong sớm: đo P(status) trên các trạng thái self-play mà bộ chấm nói **chưa xong**. Tỉ lệ này phải giảm, và P(status)
  ở trạng thái **đã xong** không được giảm.

**Điều chưa biết, nói thẳng:**
- Bước hỏng không phải là bài hỏng.
- Masking đã chặn lặp một cách cơ học: tỉ lệ lặp giảm từ 86,5% xuống 62,2%, nhưng SR chỉ từ 16 lên 19/112 (p = 0,375), còn
  báo xong sớm lại tăng từ 49 lên 56.
- Chặn lặp thôi thì chưa đủ. Model phải **biết làm gì thay thế**. D1–D5 dạy đúng điều đó vì nhãn là hành động đúng, không phải
  lựa chọn hạng hai của model. SR tăng bao nhiêu thì chưa biết; tập kiểm tra ở trên trả lời trước khi tốn công học.

## 6. Pattern có thật nhưng không giải bằng thêm dữ liệu kiểu trên
- **Cắt 40 phần tử:** 454/1.541 bước có hơn 40 ứng viên. Đây là việc sửa đầu vào, đã có kế hoạch riêng.
- **Hành động `answer`:** 25 bài. Cần thêm hành động mới vào hợp đồng.
- **19/33 app không có dữ liệu:** kiểm định "click trong app chưa thấy hỏng nhiều hơn" đã **bị bác** (36,3% so với 43,3%).
  Không xếp ưu tiên khi chưa có số đo mới.
