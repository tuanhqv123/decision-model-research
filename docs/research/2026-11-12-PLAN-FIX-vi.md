# Kế hoạch sửa lỗi và dọn dẹp (2026-11-12, chiều)

Người vận hành yêu cầu:
- dừng mọi lượt chạy chưa có đúng bản sửa;
- lập plan cho tất cả;
- xóa đồ cũ, lỗi, dư thừa.

Hiện **không huấn luyện gì**.

## 0. Đã dừng

| Lượt | Lý do | Đã ghi |
|---|---|---|
| `aw116_v3o_s0{,_b}` (oracle_stop) | phân tích lỗi cho thấy dừng/lặp chỉ là 7% lỗi gốc; reviewer chỉ ra đường rò số chẩn đoán | 20 episode dở |
| `aw116_v4m_s0{,_b}` (v4 + V10 + mask) | harness V10 chưa có các bản sửa từ review | 0 episode |

## 1. Sửa code (TDD, mỗi mục một test đỏ → xanh)

Nguồn: review tiêu chuẩn (7 vi phạm) và review spec (bổ sung khi có).

| # | Việc | Lý do |
|---|---|---|
| C1 | **Xóa hẳn `oracle_stop`**: code trong agent/runner/run_bench, 3 test, 2 config v3o | Chẩn đoán đã bỏ. Hiện số ceiling vẫn mang `protocol: aw-main-v1` và header không ghi gì, nên có thể bị đọc nhầm thành SR thật (vi phạm AW_PROTOCOL). Xóa đi thì hết đường rò, hết lỗ hổng test (`_stop_ok` không có test), hết Feature Envy và hết việc dùng chung `mask_choice` |
| C2 | **Header episode ghi `policy`**: mask, deferred_effect, scroll_index, post_action_wait_s, effect. `aw_compare.py` in ra policy, và từ chối ghép cặp hai lượt khác policy mà không báo | Một con số phải mang theo cấu hình đã tạo ra nó. Mask là chính sách phía agent (xếp lại đầu ra của chính model), không đổi cách chấm, nên giữ `aw-main-v1` nhưng phải ghi rõ |
| C3 | `run_bench.agent_kwargs`: gom `harness = cfg.get("harness") or {}` về một chỗ | Message Chains / Duplicated Code |
| C4 | Phân tích lỗi: loại 3 episode không chấm được (#26, #28, #70) khỏi mẫu số → **96 episode thất bại có chấm**; xóa dòng #36 bị trùng; tính lại tỉ lệ | M5 (nhóm phải rời nhau, hạ tầng không phải lỗi model) |
| C5 | TRACKER/WORKLOG: mọi tỉ lệ ghi kèm n/N (v3h 15,04% = 17/113; v4 13,48% = tổng điểm 15,5/115) | M11 |
| C6 | TRACKER: số của paper khác ghi rõ **"khác harness, không so được"**; sửa mâu thuẫn "11–13%" với 5,2 | M4 |
| C7 | Doc: bỏ mọi đường dẫn thư mục home trên các máy, thay bằng "script `probe_swap.sh` trên máy học" | §7 |
| C8 | `rank_correct.py`: so khớp **chính xác** tên phần tử, không dùng chuỗi con. Dòng FilesDeleteFile hiện sai: file đích không có trên màn hình, model bấm vào file mồi `edited_…` | review spec (c1) |
| C9 | FIXES §2 viết lại đúng nguồn: Calendar có phần tử đúng xếp hạng **2**; thêm 3 bài Brightness (đích không có trong list, P(click) 0,03–0,04); kết quả mask cũ ghi đủ: **4 thắng (3 WiFi + RecipeDeleteMultipleRecipesWithConstraint), 1 thua (OpenAppTaskEval), p = 0,375** | review spec (c2, c3) |
| C10 | Mẫu số của mốc v4: tính lại bằng `official_sr` (14/112 có chấm, hay 14/113?), rồi dùng **một** con số đó ở mọi file | review spec (c4) |

## 2. Dọn dẹp

Chỉ xóa thứ đã hỏng, đã bị thay thế, hoặc đã có bản lưu. **Giữ** mọi thư mục lượt chạy hoàn chỉnh, vì đó là bằng chứng của các kết quả đã commit.

| Nơi | Xóa | Giữ |
|---|---|---|
| repo local | config `aw25_v4ir_s0_{a,b}` (chưa từng chạy), `aw116_v3o_s0_{a,b}`; scratch harness V7–V9 | V10 → V11 |
| máy AW | `awruns/aw116_v3o_s0{,_b}`, `aw116_v4m_s0{,_b}` (dở/rỗng); `aw25_v4ir*.yaml`, `aw116_v3o*.yaml`, `aw116_v4m*.yaml` (sẽ dựng lại với V11), `launch_v3o.sh`, `launch_v4ir.sh`, `wait_*.out`, `awhV9.tgz`; `probes2/v3_partial_crash.jsonl` | các lượt chạy hoàn chỉnh (v3, v3h, v3m, v4, r1, b0′…) |
| máy học | `runs/v4_s0_oom_attempt1` (OOM), `runs/v3i_s0` (không có checkpoint); checkpoint trung gian v3 (450/900/1350/1800) và v4 (600/1200/1800/2400); bản trộn không được chọn | checkpoint cuối v3 1824, v4 2482 (đã sao lưu local kèm md5), bản trộn được chọn |
| máy phục vụ | `runs/episodic-s0` (arm đã thất bại 11/10), `runs/v3_s0/ckpt_900` | v3 1824, v4 2482, các thư mục `serving_*` (script cũ còn trỏ tới) |

## 3. Chọn model (chỉ suy luận, theo bảng chỉ số)

Ứng viên: v3, v4, trộn 0,5, trộn 0,3. Đã đo:

| | validation | AC-High | dừng đúng | đổi goal (dừng sai) |
|---|---|---|---|---|
| v3 | 63,4% (524/827) | 67,1% | 77% | 28% |
| v4 | 62,9% (520/827) | 61,7% | 90% | 25% |
| trộn 0,5 | 63,4% | 66,5% | 82% | 20% |
| trộn 0,3 | 65,4% | (đang đo) | 81% | 22% |

- Chọn theo các chỉ số chặn: AC-High ≥ 66%, validation ≥ 63%, dừng đúng ≥ 85%, đổi goal ≤ 10%.
- Nếu không ứng viên nào đạt hết, chọn ứng viên đạt nhiều chỉ số nhất và ghi rõ chỉ số nào chưa đạt.
- Phải có `answer` (v3 không có) để lấy 25 bài hỏi đáp.

## 4. Chạy lại (sau khi xong 1 và 2)

1. Harness **V11** = V10 + C1–C3. Chạy cả bộ test. Kiểm chứng trên emulator một bài hỏi đáp (đúng → 1,0, sai → 0,0) và một header có `policy`.
2. AW 116 bài, seed 30, **hai lượt để tách đóng góp** (review spec: v4m đổi hai thứ cùng lúc):
   - **A** = model đã chọn + V11, không mask. A so với v4/V8 đo bản sửa `answer` (và hiệu ứng của việc đổi model, nếu có).
   - **B** = A + mask. B so với A đo riêng mask.
   - Ghép cặp theo **tên task**, không theo tên agent.
   - Báo SR có n/N, SR bỏ SMS, nhóm lỗi; bài hỏi đáp và các bài còn lại báo tách.
3. Báo cáo cuối (file local): số đo, bảng chỉ số, những gì còn lại (biểu diễn on/off, kiến thức app).
