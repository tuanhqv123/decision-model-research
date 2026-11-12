# Bảng theo dõi chỉ số trước khi huấn luyện (2026-11-12)

Người vận hành yêu cầu: phải hiểu vấn đề do học sai hay do gì, và đặt chỉ số theo dõi để chắc chắn số tăng.

**Quy tắc:** không chạy huấn luyện đầy đủ nào, và không chạy AW nào, cho đến khi ứng viên (dữ liệu mới qua một bản thử ≤ 300 bước, hoặc một checkpoint) đạt mọi chỉ số **chặn** dưới đây. Mỗi chỉ số đo bằng suy luận trên trạng thái cố định, chỉ mất vài phút, nên đo trước được.

## Học sai hay dữ liệu sai?

Model học **đúng những gì dữ liệu dạy**. Dữ liệu không dạy điều ta cần.

| Lỗi | Bằng chứng | Kết luận |
|---|---|---|
| Lặp sau "no change" | Người dùng lặp 3,7% sau "no change" và 5,1% sau "changed" (`acdata_v4/check.out`). Nhãn không dự đoán được hành động tiếp theo. Đổi nhãn: Δ −0,004 (v3), −0,011 (v4). | **Dữ liệu không có tương phản.** Đây là copycat (Wen et al. 2010.14876): CE không có lý do dùng nhãn. 4,4k dòng augment (11%) là quá ít; VeriGUI 2604.05477 Bảng 5: 10% dữ liệu lỗi gần như không tác dụng, 30% mới có. |
| Dừng sớm | Trong cùng một vị trí bước, AUROC của P(terminate) trên validation là 0,94 (v3 và v4). Thêm 10 dòng lịch sử: P(dừng) chỉ +0,018. | **Không phải do vị trí bước** (giả thuyết của báo cáo web bị số đo bác bỏ). Model nhận ra "màn hình trông như màn hình cuối của AndroidControl". |
| Dừng sớm (tiếp) | Thay goal bằng goal của app khác trên màn hình cuối: v3 vẫn dừng 28%, v4 25% (`goal_swap.out`). Goal còn bài thứ hai: v4 0,01 so với v3 0,41. | Model **có** đọc goal nhưng không đối chiếu đủ chặt. Dữ liệu chỉ có dòng "đã xong" ở bước cuối, không có dòng "màn hình trông xong nhưng goal chưa xong". |
| `answer` | v4 dùng `answer` ở 7/17 bài hỏi đáp, đúng 0 bài. | Chưa từng kiểm chứng trọn từ model tới checker trước khi huấn luyện. |
| Trần chung | Chỉ dùng AndroidControl hoặc chỉ dữ liệu offline, model 3B đạt 5–13% (AppVLM 11,4 trên tập 82 bài; VeriGUI 12,6, protocol không nêu; GUI-Libra SFT thường 5,2; số do các paper tự báo, **khác harness, không so trực tiếp được** với số của mình). Đạt 25% trở lên thì cần CoT + offline RL, hoặc rollout trên emulator. | Mục tiêu thực tế với ràng buộc hiện tại là **15–20%**. Phải nói thẳng điều này. |

## Chỉ số

Cột "chặn" = phải đạt trước khi huấn luyện đầy đủ.

| ID | Chỉ số (cách đo) | v3 | v4 | Mục tiêu | Chặn |
|---|---|---|---|---|---|
| S1 | Dừng sai trên AC-High: % bước dự đoán `status` (gold không có `status`), 8.444 bước | 6,4% | 13,9% | ≤ 5% | có |
| S2 | Dừng đúng: top-1 `status` trên 137 màn hình cuối validation | 77% | 90% | ≥ 85% | có |
| S3 | Đổi goal: top-1 `status` khi goal là của app khác (137 dòng) | 28% | 25% | ≤ 10% | có |
| S4 | Goal còn bài thứ hai (thăm dò P4): mean P(terminate) | 0,406 | 0,012 | ≤ 0,10 | có |
| F1 | Đổi nhãn "no change" ↔ "changed" (P1): \|Δ P(cùng loại)\| | 0,004 | 0,011 | ≥ 0,20 | có |
| F2 | Lặp dòng thất bại 4 lần (P2): Δ P(cùng phần tử) | −0,004 | −0,010 | ≤ −0,15 | có |
| F3 | AW online: lặp y hệt sau "no change" | 93% | 85% | ≤ 50% | sau AW |
| A1 | AC-High exact (InfiGUI 8.444 bước) | 67,1% | 61,7% | ≥ 66% | có |
| A2 | Validation step acc (827) | 63,4% | 62,9% | ≥ 63% | có |
| A3 | AC-Low exact | — | 89,3% | ≥ 88% | có |
| R1 | `answer`: chép đúng chữ trên màn hình cuối các bài hỏi đáp AW đã ghi (cần dựng) | — | — | ≥ 50% bài | có |
| W1 | AW seed 30, SR chính thức, ghép cặp với v3h | 15,04% (17/113) | 13,48% (15,5/115; 1 bài nửa điểm) | > 15%, thắng > thua | kết quả cuối |
| C1 | Trần dừng đúng (harness `oracle_stop`, v3), chỉ để chẩn đoán | đang chạy | — | quyết định hướng | — |

## Cách đo

- **S1, A1:** `benchmarks/results/2026-11-12-v4/actest/compare_actest.py` cùng phân bố `pred`.
- **S2, S3:** script `probe_swap.sh` trên máy học; tóm tắt ở `benchmarks/results/2026-11-12-tracker/goal_swap.out`.
- **S4, F1, F2:** `benchmarks/results/2026-11-12-probes2/probe_aw.py` + `summarize.py`, chạy trên trạng thái v3h đã ghi.
- **F3, W1:** `benchmarks/results/2026-11-12-v3h/aw_compare.py` (in protocol và policy của từng lượt, cảnh báo khi ghép cặp hai lượt khác policy).
- **C1:** cấu hình `aw116_v3o_s0_{a,b}` (harness V9).

## Thứ tự quyết định

1. Đọc C1.
   - Nếu trần dừng đúng ≤ 17%: thiếu kỹ năng, không phải lỗi dừng. Dừng hướng sửa dừng/lặp và báo người vận hành.
2. Với mỗi hướng sửa (VeriGUI 30% có cặp đối chứng; goal-swap âm ngay trước bước cuối; chỉ chép chữ cho `answer`), chạy thử ≤ 300 bước từ v3, đo mọi chỉ số chặn, ghi vào bảng này.
3. Chỉ hướng nào đạt mọi chỉ số chặn mới được học đầy đủ, rồi mới chạy AW.
