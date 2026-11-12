# Bản tổng hợp cuối (2026-11-12): sửa gì, học được gì, áp dụng gì

Mỗi số dưới đây đều tính từ file và có ghi mẫu số. Các số ở phần 1 đã được một agent độc lập tính lại (chỉ đọc, không biết trước kết luận). Phân tích chi tiết nằm ở `2026-11-12-ACHIGH-vs-AW-FINAL-vi.md`.

## 1. Hiện trạng đã đo

- **AndroidWorld** (v4 + harness V12, seed 30): **25/115 = 21,74%**. Gồm 24 bài trọn điểm và 2 bài nửa điểm; OsmAndMarker lỗi setup nên bị loại khỏi mẫu số.
- **Các bài thất bại:**
  - 49 bài báo "xong" khi chưa xong, thêm 2 bài nửa điểm;
  - 40 bài hết ngân sách bước, trong đó 2 bài máy **đã đúng trạng thái** nhưng model không dừng.
- **AC-High** (cùng model v4): 61,7% từng bước. Nhưng chỉ **344/1.543 = 22,3%** bài đúng mọi bước, dù model được đưa lịch sử của người ở mỗi bước. Với bài từ 6 bước trở lên, con số này còn 59/628 = 9,4%.

## 2. Những gì học được

| # | Bài học | Bằng chứng |
|---|---|---|
| L1 | AC-High là điểm **từng bước, có lịch sử của người**, không phải điểm hoàn thành bài. Tính theo cả bài thì nó ngang AndroidWorld (22,3% so với 21,7%). | `eval_d1.py:42,48-49,220-225`; scorer không kiểm tra mục tiêu (`vendor/infigui_eval.py:83-122`) |
| L2 | Kỹ năng thiếu nhất là **dừng đúng lúc**. Không có bước dừng nào là đáp án đúng trên AC test, vậy mà model vẫn đòi dừng **1.171/8.444 bước**, rải trên 546 bài. Trong đó 460 lần rơi đúng bước thao tác thật cuối cùng; theo đáp án gốc, 566 lần lẽ ra phải là vuốt và 420 lần là click. | `perstep-v4_s0_ckpt_2482_actest_v4.jsonl` |
| L3 | **Bước khử trùng lặp của v5 phá đúng dữ liệu dạy dừng.** Cặp "thao tác cuối + `status`" còn đủ ở 4.191 bài trước khi curate, nhưng sau curate chỉ còn 1.918. | đếm trên `compact_awlike_full_v5` / `compact_cur_v5` |
| L4 | Lỗi đầu tiên trên AC-High: **sai loại hành động 684**, sai đích hoặc giá trị 515 (riêng click sai đích 435), dừng sớm 154. | cùng file |
| L5 | **AMEX** chỉ báo điểm từng bước, không chạy online. File chỉ có BACK 0,35%, ngược với lời paper. Chuỗi thao tác của AMEX chuyển giao kém sang dữ liệu khác (56,2 so với 76,3). | file + paper |
| L6 | **AndroidControl-Curated / Magma-R1** (2510.18488): | HF `batwBMW/AndroidControl_Curated`; GitHub `src/eval/` |
| | • Chỉ phát hành **tập test đã làm sạch**: 8.435 dòng, 1.540 bài; không có tập train. | |
| | • Paper cho biết 2.400 mẫu train được *chọn từ chính AndroidControl-Curated*. | |
| | • Lịch sử trong file là **câu mô tả từng bước do người viết**. | |
| | • Trong file `task-improved`, lịch sử **chứa luôn câu hướng dẫn của bước hiện tại** ở 8.377/8.377 dòng, và script eval đưa nguyên lịch sử vào prompt (`evaluate_actions_androidControl_vllm.py:110,125`, `prompt.py:5`). | |
| | • Con số "SR 75–80%" của họ là điểm từng bước kiểu AC, không phải hoàn thành bài trên AndroidWorld. | |
| L7 | Những điều đáng học từ hai paper: **chất lượng hơn số lượng**; **cân theo loại hành động**; **dữ liệu hiểu màn hình** giúp grounding. | |

## 3. Áp dụng trong bản train cuối ("final")

| Vấn đề | Cách sửa trong final | Trạng thái |
|---|---|---|
| Dừng sớm (L2) | **Mới:** giữ nguyên mọi cặp "thao tác cuối + `status`" (4.191/4.191), có test `test_stop_pairs_survive_dedupe_and_click_cap` | **mới, sửa L3** |
| Dừng sớm khi bài có nhiều phần | Nối 2 bài thành chuỗi (491 chuỗi): ở màn hình xong phần A, đáp án là `navigate_home` chứ không phải `status` | có từ v5 |
| Lặp khi bấm mà không có gì đổi | Các dòng `nc_*` ("vừa bấm mà không đổi gì → làm khác đi"): 3.627 dòng | có từ v4 |
| Không biết quay lại | 3.468 dòng `recover_back`, mỗi dòng kèm một dòng đối chứng (`recover_twin`) | có từ v5, **chưa đo riêng** |
| Phần tử không có nhãn, công tắc bật/tắt | Contract v5 hiển thị `[id: …]` và on/off | có từ v5 |
| Dữ liệu trùng lặp, mất cân bằng (L7) | Mỗi (màn hình, hành động, đích) giữ tối đa 2 dòng; click ≤ 45% | có từ v5 |
| Câu trả lời dạng số | Harness V13: `answer_format` (+1 bài khi chấm lại offline) | harness |

**Dữ liệu final:** `compact_cur_v6` có 38.195 dòng, cộng 1.736 dòng `awgen_v4`.
- **Tỉ lệ hành động:** click 45,3% · status 16,3% · back 11,7% · type 7,1% · open_app 6,0% · scroll 5,2% · wait 4,5%.
- **Train:** khởi tạo từ v4 ckpt_2482, **2.496 bước × 16 = đúng 1 epoch** trên 39.931 dòng, lr 5e-6, lưu checkpoint mỗi 250 bước.
  - Bản đầu chạy 1.000 bước, tức chỉ học 40% số dòng; đã khởi động lại ở bước 60.
  - v4 cũng train đúng 1 epoch. Unsloth khuyên 2 epoch, hoặc 3–4 nếu ít dữ liệu.

## 4. Những gì không sửa trong final, và vì sao

- **App chưa từng thấy** (62/95 bài thất bại nằm trong 7 app không có trong AndroidControl): muốn sửa thì phải có dữ liệu của chính các app AndroidWorld, tức là học trước đề. Điều này bị cấm (không self-play, không khám phá app). AMEX cũng không có các app này.
- **Nhãn sai trong AndroidControl train:**
  - bộ đã làm sạch chỉ có phần test;
  - lọc bằng chính các model của mình thì sẽ xóa luôn những câu khó;
  - dùng LLM bên ngoài để sửa nhãn thì vi phạm ràng buộc.
- **Dữ liệu hiểu màn hình kiểu AMEX:** AMEX không có class/flag/id nên lệch định dạng v5, và không có app của AndroidWorld.

## 5. Cổng kiểm tra (đăng ký trước; không qua thì bản cuối là v4 + V13)

1. **Cảnh báo sớm ở ckpt_250** (chạy eval validation trên một GPU khác).
   - Dừng run nếu P(back | gold ≠ back) vượt mức của v4 quá 0,02.
   - Dừng run nếu số lần model báo `status` khi đáp án không phải status vượt mức của v4.
2. **Cổng 1** (checkpoint cuối, so với v4, cùng file và cùng scorer). Cần đạt cả bốn:
   - AC-High: số bài đúng cả bài ≥ 344/1.543, và McNemar theo bài không thua;
   - số lần `terminate` giảm rõ so với 1.171/8.444;
   - validation: số bài đúng mọi bước không giảm;
   - P(back | gold ≠ back) ≤ của v4 + 0,02.
3. **Cổng 2:** chạy 13 bài AndroidWorld nhắm trúng lỗi.
   - dừng sớm: 6 bài;
   - không dừng hoặc hết bước: 3 bài;
   - giữ bài đang đúng: 4 bài.
4. **Số cuối:** chạy đủ 116 bài với harness V13, trên 6 emulator, ghép cặp với v4 + V12 (21,74%).

## 6. Đối chiếu với hướng dẫn Unsloth "train your own decision model" (đọc 2026-11-12)

Hướng dẫn này biến một LLM thường (Qwen3.5 / Gemma 4 / Llama 3.2) thành decision model:
- gắn thêm một head kiểu Clef để chấm điểm từng lựa chọn sau một lần đọc, không sinh chữ;
- train bằng LoRA r16, lr 2e-4, head lr 1e-4, 2 epoch;
- dữ liệu dạng `state` / `questions` / `gold`;
- **hiệu chỉnh xác suất (calibrate) trên phần dữ liệu để riêng** sau khi train.

Hướng dẫn không nói gì về ảnh hay vision.

| Điểm | Unsloth | Mình | Kết luận |
|---|---|---|---|
| Định dạng dữ liệu | state + questions + gold, một lần đọc | giống | đúng |
| Mô hình xuất phát | LLM thường, head mới | d1-2B đã là decision model, head có sẵn | khác điểm xuất phát, không phải lỗi |
| LoRA hay full | LoRA (để head mới học từ đầu) | full fine-tune, lr 5e-6 | đều hợp lệ; lr full FT nhỏ hơn lr LoRA là bình thường |
| Số epoch | 1–2, ít dữ liệu thì 3–4 | 0,4 epoch | **đã sửa thành 1 epoch** |
| Hiệu chỉnh sau train | `calibrate()` trên holdout | chưa làm (ECE hành động của v4 trên validation là 0,19) | chỉ ảnh hưởng độ tin cậy, không đổi hành động được chọn (argmax) vì cổng done đang tắt. Sẽ hiệu chỉnh trên validation sau cổng 1 |

## 7. Model có dùng ảnh không? Thử thay ảnh bằng ảnh xám (2026-11-12)

Cùng v4, cùng 827 dòng validation. Lần thứ hai thay mọi ảnh chụp màn hình bằng ảnh xám cùng kích thước. File: `benchmarks/results/2026-11-12-imgablate/`.

| | Ảnh thật | Ảnh xám |
|---|---|---|
| Đúng từng bước | **519/827** | **337/827** (−22 điểm) |
| Bài đúng mọi bước | 14/137 | 6/137 |
| click | 234/420 | 121/420 |
| type_text | 37/53 | 3/53 |
| scroll | 40/86 | 7/86 |
| navigate_back | 18/29 | 7/29 |
| wait | 22/56 | 39/56 (khi không thấy gì, model chọn chờ) |
| open_app | 45/45 | 45/45 |
| Độ chính xác chọn loại hành động | 80,0% | 50,8% |
| Grounding | 65,5% | 59,9% |

Ở 216 dòng, model đúng khi có ảnh và sai khi ảnh xám. Ngược lại chỉ có 34 dòng.

**Kết luận:** model **dùng ảnh nhiều**, chủ yếu để quyết định *làm gì*: thấy bàn phím thì gõ, thấy danh sách dài thì vuốt, thấy màn hình đã xong. Việc *chọn phần tử nào* phụ thuộc vào ảnh ít hơn (65,5% → 59,9%), vì danh sách phần tử đã có sẵn dạng chữ.

**Việc có thể làm sau** (không thuộc lần train này): ảnh đang ở độ rộng 720 px, trong khi d1 hỗ trợ tới 858 px. Muốn dùng 858 px thì phải dựng lại kho ảnh AndroidControl từ bản gốc, vì ảnh đang lưu chỉ có 720 px.

## 8. AndroidGen (arXiv 2504.19298; HF `zai-org/androidgen-llama-3-70b`), đọc 2026-11-12

**AndroidGen là gì:** agent LLM gọi model nhiều lần mỗi bước và chỉ đọc chữ (cây a11y, không có ảnh), viết ra lệnh dạng code, kèm chain-of-thought. Gồm bốn mô-đun:
- **ExpSearch:** lấy 1 quỹ đạo thành công giống nhất làm ví dụ mẫu;
- **ReflectPlan:** một lần gọi LLM riêng để lập và cập nhật danh sách mục tiêu con;
- **AutoCheck:** kiểm tra theo luật, không dùng LLM;
- **StepCritic:** GPT-4o lọc quỹ đạo.

**Kết quả trên AndroidWorld:**

| Cấu hình | SR |
|---|---|
| GPT-4o | 46,8 |
| Llama-3-70B fine-tune | 35,3 (dễ 55,8 / vừa 12,5 / khó 5,9) |
| GLM-4-9B fine-tune | 29,2 |

Paper không ghi số bài, seed hay ngân sách bước.

**Ablation:**
- ReflectPlan +11,7; ExpSearch +12,6; AutoCheck +1,8.
- Dữ liệu lọc so với không lọc: 25,2 so với 16,2.

**Vì sao không so được với mình:**
- **Dữ liệu train thu trong chính AndroidWorld:** khoảng 300 nhiệm vụ do GPT-4o viết dựa trên các nhiệm vụ AndroidWorld, chạy trên emulator AndroidWorld.
- **ExpSearch thêm chính các lượt chạy eval vào kho truy xuất** (`androidgen.py`; `allow_same=True`).

Theo luật của mình, cả hai điều trên đều là học trước đề.

**Ý tưởng hợp ràng buộc của mình:**
- **"Tăng dữ liệu theo tiền tố mục tiêu"** (Algorithm 1): cùng một màn hình là "đã xong" với mục tiêu ngắn nhưng là "làm tiếp" với mục tiêu dài.
  - Bản final đã có đúng cặp này, nhờ hai thứ:
    - **dòng nối chuỗi:** ở màn hình xong của bài A, mục tiêu "A rồi B" có đáp án `navigate_home`;
    - **cặp dừng của chính bài A:** cùng màn hình đó, mục tiêu "A" có đáp án `status`, và đã được giữ đủ 4.191/4.191.
- **Kiểm tra theo luật (AutoCheck):** phần lớn đã có trong harness (gate chọn trong danh sách, kiểm tra chữ đã vào ô). Mức tăng họ đo được nhỏ (+1,8).

**Không áp dụng được:**
- ReflectPlan: cần một planner sinh chữ.
- ExpSearch như họ làm: dùng kho truy xuất có cả dữ liệu đề thi.

## 9. Phân tích trace của cả 91 bài thất bại (v4 + V12) và quyết định cuối

Bốn agent đọc trace và ảnh màn hình độc lập, mỗi agent một nhóm. Bảng từng bài nằm ở `benchmarks/results/2026-11-12-failanalysis/*.tsv`. Mỗi bài đều được ghi bước sai đầu tiên, model đã làm gì, lẽ ra cần làm gì, và bằng chứng.

**Lỗi môi trường (không phải lỗi model).** Emulator **aw-emu3** chạy cả 6 bài Markor khi app đang ở màn hình giới thiệu. Đã sửa bằng `hostscripts/app_setup.py`: chỉ khi thấy màn hình giới thiệu thì mới chạy `MarkorApp.setup` chính thức. `run_arm_v2.sh` từ nay từ chối container còn kẹt ở màn hình đó.

VLC trên **mọi** emulator hiện "GO TO MEDIALIBRARY PREFERENCES" và danh sách thì "Loading." mãi. Vì giống nhau trên mọi máy nên đây không phải lỗi riêng một máy; còn phải điều tra.

**Kiểm tra sớm ở ckpt_250 của bản final lần 1** (dữ liệu v5 + cặp dừng): quay lại sai 44/798 (v4: 16/798), P(back | gold ≠ back) 0,077 (v4: 0,025). Mức này vượt quy tắc đã đăng ký trước (+0,02), nên run bị dừng. Đây là lần thứ hai dữ liệu quay lại gây hại.

| Nhóm (thất bại) | Nguyên nhân chính (số bài) | Sửa trong final (lần 2) | Bị chặn bởi ràng buộc |
|---|---|---|---|
| Hỏi thông tin (21) | Trả lời được nhưng không trả lời: 4 · định dạng: 1 · đi sai chỗ: 4 · suy luận ngày/giờ/màu: 4 · đếm qua nhiều màn hình: 2 · cơ chế trả lời không biểu diễn được: 6 | +350 bài trả lời; V13 `answer_format` (+1) | Ngày/giờ, màu, cộng/đổi đơn vị, đáp án nhiều mục |
| Dễ (27) | Cần nhấn giữ: 5 · bài Verify đã đúng sẵn: 3 · dừng sớm: 4 · chọn sai phần tử: 4 · đích ở ngoài màn hình: 3 · gõ chữ luôn xóa ô: 2 · clipboard bị lọc: 2 · môi trường/kiến thức app: 4 | Nhấn giữ ×4 · dòng "đúng sẵn thì báo xong" (154) · cặp dừng · chữ cùng hàng | Gợi ý trên bàn phím (AndroidControl không có nhãn), độ sáng không có trong cây a11y, ngày tương đối, gõ nối thêm |
| Vừa/khó (34) | Sai cách thao tác app: 7 · giá trị phải tính: 6 · làm xong mục 1 rồi dừng: 4 · lỗi môi trường Markor: 3 · ô form bị bàn phím che: 3 · trùng lặp: 3 · nội dung ghi chú: 2 · VLC: 2 · lẻ: 4 | Nhấn giữ · chữ cùng hàng · chuỗi nối bài + cặp dừng · sửa môi trường | Phép tính ngày/giờ, phát hiện trùng lặp qua nhiều màn hình, nội dung dài hơn 40 ký tự |
| Ảnh (9) | Đi lạc: 2 (SimpleDraw, SaveCopy) · Browser: 2 · chữ chỉ có trong ảnh: 4 · phép nhân: 1 | Nhấn giữ → "Copy to" | Câu hỏi `text`/`number`/`date` của d1 là con trỏ chỉ đọc được phần chữ của đầu vào (`pointer.py`), nên không đọc được chữ trong ảnh |

**Tổng:** khoảng 36/91 bài bị chặn bởi ràng buộc (một model, không planner, chỉ dữ liệu AndroidControl). Final lần 2 nhắm vào phần còn lại.

**Không đưa vào, kèm lý do:**
- **Câu hỏi riêng "đã xong chưa":** đã đo ngày 2026-11-10 và thấy trùng với câu hỏi hành động.
- **Mask lặp:** đã đo, thắng 2 thua 2 và dừng sớm tăng.
- **Gợi ý trên bàn phím:** id trong AndroidControl bị che nên không có dữ liệu học.
- **Ẩn bàn phím sau khi gõ:** là thay đổi harness, sẽ thử trên các bài form mà không cần train lại.
