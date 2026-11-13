# Lập luận và phản biện cho post-mortem v3 (2026-11-11)

Tài liệu này đi kèm `2026-11-11-POSTMORTEM-v3-vi.md`. Nó ghi lại **cách** từng kết luận được rút ra, những câu hỏi
phản biện mình tự đặt ra từ góc nhìn khác, số đo trả lời từng câu, và lỗ hổng nào còn mở. Mọi file nhắc tới nằm trong
`benchmarks/results/2026-11-11-v3/postmortem/`.

## 0. Phương pháp
1. **Một khẳng định = một con số + một file chạy lại được.** Không có file thì chỉ được gọi là giả thuyết.
2. **Nói rõ định nghĩa và mẫu số.** Ví dụ "bước vô ích" ở đây là định nghĩa mà **model nhìn thấy**: dòng lịch sử
   `→ no change`, do `ac_common.effect_changed` tính, giống hệt lúc học. Không phải `ui_changed` của harness.
3. **Mỗi luận điểm phải qua ít nhất một câu hỏi phản biện** có thể bác bỏ nó: biến gây nhiễu, cách giải thích khác,
   lặp lại trên lượt chạy khác, cơ chế ở cấp nhỏ hơn.
4. **Xếp loại sau phản biện:** ĐỨNG VỮNG · ĐỨNG VỮNG CÓ ĐIỀU KIỆN · BỊ BÁC MỘT PHẦN · BỊ BÁC · CHƯA KIỂM ĐỊNH.
5. **Quan sát khác thí nghiệm.** Mọi phép so trên một lượt chạy có sẵn là quan sát. Chỉ can thiệp có đối chứng (đổi một
   thứ, giữ nguyên phần còn lại) mới chứng minh được quan hệ nhân quả. Lỗ hổng nào còn mở đều kèm thí nghiệm can thiệp
   để đóng nó.

---

## L0. Bản thân con số 14,0% có tin được không?
**Khẳng định:** v3 đạt 16/114 = 14,0% trên AndroidWorld theo giao thức chính thức (`aw_facts.out`, hàm `official_sr` của repo).

| Câu hỏi phản biện | Trả lời bằng số |
|---|---|
| Harness có lỗi làm tụt điểm không? | Năm sai lệch so với `suite_utils` gốc (H1–H5) đã sửa, mỗi cái có test trích dòng code gốc. Gate A đạt 41/41, kiểm tra hệ quả của từng hành động trên máy giả lập (`AGENTS.md`, nhật ký 2026-11-10). |
| Loại bài khỏi mẫu số có đúng luật không? | Loại 2 bài: MarkorMergeNotes (bộ chấm lỗi adb) và SimpleSmsReplyMostRecent (lỗi khởi tạo). Theo quy tắc `NaN` của `suite_utils`, cài trong `awenv.metrics.official_sr`. |
| Một seed có đủ không? | Không đủ để so chênh lệch nhỏ. Sai số chuẩn ở mức 14% trên 116 bài là 3,2 điểm (95% ±6,3). Chạy lại cùng máy: 0/20 kết quả đổi (Q10, 2026-11-10), nên nhiễu chủ yếu đến từ khác seed hay khác instance, không phải chạy lại. |

**Xếp loại: ĐỨNG VỮNG** như một con số một seed. **Lỗ hổng:** chưa có ≥3 seed.

---

## L1. "Học trên app khác app thi" (N1)
**Khẳng định gốc:** 19/33 app AW không có dòng học; bài thuộc app có dòng học thành công nhiều hơn.

| # | Câu hỏi phản biện | Số đo | File |
|---|---|---|---|
| 1 | App quen chỉ có bài ngắn hơn? | Trong nhóm bài ≤10 bước vẫn còn: 7/16 vs 4/32, p = 0,021. Nhóm 11–20 bước: **không khác** (1/7 vs 4/31) | `app_coverage.out` |
| 2 | Các bài "cho không" (trạng thái đích có sẵn từ đầu) có thổi phồng nhóm app quen? | Có rủi ro thật: agent không làm gì (Q11) giải được 7 bài, **6/7 thuộc Settings**. Bỏ 7 bài đó: v3 **7/19 (36,8%) vs 8/88 (9,1%), p = 0,005**. Hiệu ứng vẫn còn | `floor_tasks.out`, `coverage_replication_v3.out` |
| 3 | Có lặp lại trên các lượt chạy độc lập khác không? | Trên mọi bài: R1 seed 0 36% vs 5,6% (p = 0,0003); R1 seed 1 40% vs 7,8% (p = 0,0003); B0′ 8% vs 6,7% (p = 0,55; B0′ chỉ ngang agent không làm gì). Bỏ bài cho không: R1 s0 15,8% vs 4,5% (p = 0,10), R1 s1 21,1% vs 6,7% (p = 0,07). **Cùng chiều, gấp 3–3,5 lần, nhưng không có ý nghĩa thống kê** | `coverage_replication.out` |
| 4 | Ngưỡng "có dòng học" có phải chọn sau khi nhìn kết quả? | App có dòng học thấp nhất là files (155 dòng); mọi app còn lại có 0. Ngưỡng >0 và ≥100 cho cùng một phép chia | `train_facts.out` |
| 5 | Cơ chế: ở app lạ, click vô ích nhiều hơn? | **Không.** Click vô ích ở app chưa học 240/662 = 36,3%, ở app quen 45/104 = 43,3% (z = −1,38) | `challenge_aw.out` Q-D |
| 6 | Hay đơn giản app quen là app có giao diện dễ (bật/tắt trong Settings)? | **Không tách được** bằng dữ liệu quan sát: "app" và "loại bài" trùng nhau | — |

**Xếp loại: ĐỨNG VỮNG CÓ ĐIỀU KIỆN.** Có thật ở cấp bài cho v3 (p = 0,005 sau khi bỏ bài cho không), cùng chiều trên hai
lượt R1. **Bị bác:** cơ chế "click không ăn ở app lạ" (câu 5).
**Lỗ hổng mở:** nhiễu giữa app và độ khó của loại bài (câu 6); cơ chế thật chưa biết.
**Thí nghiệm đóng lỗ hổng (can thiệp):** thêm dữ liệu self-play **chỉ** cho app có 0 dòng, giữ nguyên mọi thứ khác, đo trên seed
chưa học. Nếu nhóm app 0 dòng không tăng thì N1 không phải nguyên nhân.

---

## L2. "Không biết thoát khi tự sai" (N2)
**Khẳng định gốc:** 45% bước vô ích; 88% lặp y hệt sau bước vô ích; chỉ 2% dòng học có bước trước vô ích.

| # | Câu hỏi phản biện | Số đo | File |
|---|---|---|---|
| 1 | Hai cách đếm "vô ích" (663 và 573) có mâu thuẫn? | Không. 663 là định nghĩa model thấy (dòng lịch sử) = 595 bước ở bài được chấm + 68 bước của MarkorMergeNotes. 573 là `ui_changed`, một phép so khác của harness | `challenge_aw.out` |
| 2 | Mọi bước "không đổi" đều vô ích? | **Không.** 595 bước gồm: `wait` 151 (đứng yên là bình thường), scroll 155, click 285 (phần tử tương tác 191, ô nhập chữ 71, phần tử không tương tác 23), open_app 4. **"45% vô ích" phóng đại**, vì tính cả `wait` | `challenge_aw.out` Q-A |
| 3 | Lặp lại có lúc hợp lý? | 469 lần lặp y hệt (bài được chấm): `wait` 131, trong đó 37 lần lần sau có tác dụng (chờ tải là hợp lý). click 196, chỉ 8 lần có tác dụng. scroll 142, **0** lần có tác dụng. Lặp chắc chắn lãng phí: **≥ 330** (click 188 + scroll 142) | `challenge_aw.out` Q-E |
| 4 | Lỗi cụ thể nhất? | Sau một click vào ô nhập chữ mà màn hình không đổi, model **click tiếp 57 lần, gõ chữ chỉ 11 lần**. Trong dữ liệu học, sau click vào ô nhập, đáp án là gõ chữ 82,5% (577/699) | `challenge_aw.out`, `train_facts.out` |
| 5 | Thiếu dữ liệu sửa sai có phải nguyên nhân? | Mới là **tương quan**: 2,0% dòng học có bước trước vô ích. Giải thích khác: model quyết định tất định, màn hình không đổi và lịch sử chỉ thêm một dòng, nên quyết định lặp lại | `train_facts.out` |
| 6 | Chỉ cần chặn lặp là đủ? | Đủ 112 bài ghép cặp: không chặn 16/112, chặn 19/112 (thắng 4, thua 1, sign test p = 0,375). **3/4 bài thắng là bài "cho không"** (agent không làm gì cũng giải được). Chặn lặp có giảm lặp (86,5% → 62,2%), nhưng bài báo xong sai tăng từ 49 lên 56: vòng lặp chuyển thành báo xong sớm. Lặp lại là triệu chứng | `mask_paired_full.out` |

**Xếp loại:** triệu chứng **ĐỨNG VỮNG**, đã chỉnh số: ≥330 lần lặp lãng phí, lỗi "click lại thay vì gõ" 57/68. Nguyên nhân
"thiếu dữ liệu sửa sai" **CHƯA KIỂM ĐỊNH**.
**Thí nghiệm đóng lỗ hổng:** thêm dòng "đừng chọn" (từ self-play) và dòng lịch sử có lỗi, giữ nguyên phần còn lại. Đo tỉ
lệ lặp sau bước vô ích và tỉ lệ "gõ sau khi click ô nhập" trên bộ dev. Bỏ nếu cả hai không đổi.

---

## L3. "Nhãn báo xong gắn theo vị trí" (N3)
**Khẳng định gốc:** nhãn `status` chỉ nằm ở bước cuối bản ghi, nên model học báo xong theo vị trí.

| # | Câu hỏi phản biện | Số đo | File |
|---|---|---|---|
| 1 | Sự thật về nhãn? | 4.197/4.197 nhãn `status` ở bước cuối, 0 ở chỗ khác. Do `preprocess_ac.py:5,134` thêm vào | `train_facts.out` |
| 2 | Model báo xong ở bước giống độ dài bản ghi mẫu? | Báo xong sai: trung vị bước 6 (IQR 5–9). Nhãn `status` khi học: trung vị 6 (IQR 4–8). **Trùng nhau** | `challenge_aw.out` Q-B, `train_status_position.out` |
| 3 | Nếu model chỉ dựa vào vị trí, bước báo xong phải **không** đổi theo độ dài bài. Có đổi không? | **Có đổi:** trung vị 5 (bài ≤10 bước) → 7 (11–20) → 8 (>20); Spearman 0,46 (n = 49). Vậy model **có** đọc trạng thái một phần | `challenge_aw.out` Q-C |
| 4 | Báo xong đúng thì sao? | Bài thành công cũng báo ở trung vị bước 6, nên bước báo xong không phân biệt được đúng hay sai | `challenge_aw.out` Q-B |
| 5 | Có nhãn tốt hơn không? | Có, đã thu: bộ chấm trả lời ở 395/395 bước self-play. 18 lần báo xong sớm, 2 lần báo muộn đã được gắn nhãn | `selfplay_facts.out` |

**Xếp loại:** sự thật về nhãn **ĐỨNG VỮNG** (đúng tuyệt đối). "Model chỉ báo xong theo vị trí": **BỊ BÁC MỘT PHẦN**, vì
bước báo xong có tăng theo độ dài bài. Điều đứng vững: ở bài dài, model báo xong quá sớm (trung vị bước 8 với bài có
ngân sách >20 bước), và nhãn hiện tại không có ví dụ "chưa xong" nào ở các màn hình trông giống màn kết thúc.
**Lỗ hổng mở:** dòng `Progress: step n` trong đầu vào có phải là tín hiệu vị trí model dựa vào không.
**Thí nghiệm đóng lỗ hổng:** dùng đúng đầu vào đã ghi của self-play (`state_in`), chỉ đổi số bước trong dòng `Progress`, đo
xác suất `status` thay đổi bao nhiêu. Không cần học lại.

---

## L4. "Cắt 40 phần tử làm hại" (N4)
| # | Câu hỏi phản biện | Số đo | File |
|---|---|---|---|
| 1 | Model có nhận được danh sách đầy đủ không? | Giới hạn 128.000 token; không cắt thì p99 2.659, lớn nhất 6.456 token | `cap_tokens.out` |
| 2 | Số 40 dựa trên gì? | Một so sánh quan sát ngày 2026-11-07 trên model khác, không phải thí nghiệm | `2026-11-07-list-length-is-causal.md` |
| 3 | Cắt có thật sự xảy ra trên AW? | 454/1.541 bước = 29,5%; `New Event` bị cắt 155/155 lần. Danh sách dựng lại khớp lựa chọn thật của model 899/903 | `aw_cut.out` |
| 4 | 0/17 Simple Calendar có phải do cắt? | **Không tách được:** app này cũng có 0 dòng học (N1) và 8/17 bài cần `answer` (N5) | — |
| 5 | Bỏ cắt có làm chậm không? | Chưa đo độ trễ; số token tăng ở p99 từ 1.706 lên 2.659 | — |
| 6 | v3 có dùng được phương án thứ 41 trở đi không? | Chưa biết: lúc học chưa bao giờ có | — |

**Xếp loại:** "40 không phải giới hạn của model" **ĐỨNG VỮNG**; "cắt gây ra 0/17" **CHƯA KIỂM ĐỊNH**.
**Thí nghiệm:** T1(a): cùng checkpoint, 656 dòng AC-High có >40 ứng viên, so không cắt với cắt 40, đo độ chính xác và độ
trễ. Sau đó 9 bài Simple Calendar không cần `answer`.

---

## L5. "Thiếu `answer`" (N5)
Số học thuần: 25/116 bài chỉ chấm qua `answer`, v3 được 0/25, trần điểm 78,4%. **Phản biện:** có thêm `answer` thì
cũng chưa chắc làm được 25 bài, vì diễn đạt được không có nghĩa là giải được. 25 là **trần**, không phải mức tăng dự kiến.
**Xếp loại: ĐỨNG VỮNG** (như một trần).

---

## L6. Giải pháp
Mọi mức tăng là **giả thuyết**. Số từ bài báo là số **của họ, trên benchmark của họ**. Ví dụ DigiRL, arXiv 2406.11896,
trên AitW: SFT 17,7% → học lại quỹ đạo tự thu đã lọc 57,8% → RL 67,2%. Số đó chỉ cho thấy hướng đi đã thành công ở nơi
khác, không dự báo con số của mình. Mỗi giải pháp có điều kiện bỏ trong bản post-mortem §4.

---

## 7. Những gì phản biện đã thay đổi so với bản post-mortem trước
| Trước | Sau phản biện |
|---|---|
| "45% bước vô ích" | Phóng đại, vì tính cả `wait`. Đúng hơn: ≥330 lần lặp chắc chắn lãng phí, cộng lỗi "click lại thay vì gõ" 57/68 |
| "Model học báo xong theo vị trí" | Bị bác một phần: bước báo xong tăng theo độ dài bài (ρ = 0,46). Đứng vững: báo quá sớm ở bài dài |
| "Ở app lạ click không ăn" (ngầm hiểu) | Bị bác ở cấp bước (36,3% vs 43,3%) |
| "Học trên app khác là nguyên nhân" | Có điều kiện: đứng ở cấp bài cho v3 (p = 0,005 sau khi bỏ bài cho không); trên R1 cùng chiều nhưng không có ý nghĩa thống kê; chưa tách khỏi độ khó loại bài |

## 8. Lỗ hổng còn mở, xếp theo chi phí đóng
1. **Dòng `Progress` có phải tín hiệu vị trí?** Can thiệp trên đầu vào đã ghi. Không học lại, vài phút GPU.
2. **Bỏ cắt 40 trên v3** (T1a). Không học lại, khoảng 1 giờ GPU.
3. **Một seed:** chạy v3 thêm 2 seed (2 máy giả lập, khoảng 3 giờ mỗi seed).
4. **App hay độ khó loại bài?** Can thiệp bằng dữ liệu self-play cho riêng app 0 dòng. Cần học lại.
5. **Thiếu dữ liệu sửa sai có phải nguyên nhân lặp?** Can thiệp bằng dòng "đừng chọn" và lịch sử có lỗi. Cần học lại.
