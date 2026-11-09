# PLAN — AndroidWorld 116-task standard run (chưa chạy, chờ user confirm)

*Viết 2026-11-08 sau khi user yêu cầu "mode chuẩn rồi chạy". Đây là plan để confirm, KHÔNG phải báo cáo kết quả.*

## Trạng thái hiện tại (đã kiểm bằng lệnh)

| Thứ | Trạng thái | Ghi chú |
|---|---|---|
| Emulator `aw-emu` | **up 6h, health OK** | Gate A 32 passed / 0 failed |
| Suite | **116 task** | `suite/task_list` trả về đúng 116 |
| Emulator thực thi `navigate_home`/`keyboard_enter`/`double_tap`/`swipe` | **✅ tất cả success** | tôi đã gửi thử từng cái |
| Checkpoint `ckpt-E2-final` | có trên host A | `~/acv2/ckpt/ckpt-E2-final` |
| Model server (port 8700) | **FAILED, chưa chạy** | `ModuleNotFoundError: train_d1` — tôi launch thiếu path |
| VRAM trống | **21.9 GB** | prod chỉ chiếm 2.1 GB (`masan2025`) |
| Container network | **`bridge`** | ⚠️ xem mục Rủi ro #1 |

## Việc tôi ĐÃ làm (cần biết)

1. Copy `serving/` + `benchmarks/train/ac_common.py` lên `~/d1-api` (84 KB) — vô hại.
2. **Launch model server → thất bại** (`ModuleNotFoundError: train_d1`). Không có process nào đang chạy.
3. Copy `~/awh` (harness test) lên từ trước — vô hại.
4. Dọn 11.4 GB tarball trong `/tmp` (lượt trước).

**Chưa chạy task nào.** Chưa có số AndroidWorld mới. Con số 1/8 vẫn là con số duy nhất hiện có.

## Rủi ro #1 — container dùng network `bridge`, không phải `host`

Trước đây (2026-11-06) tôi ghi trong notes là `--network host` vì container→host bị firewall. Hiện tại container là `bridge`, nên **model server ở 127.0.0.1:8700 trên host sẽ KHÔNG truy cập được từ trong container**.

Ba cách xử lý:
- **(a)** Chạy harness **trên host** (env in-process cần `android_world` + `adb` thấy emulator). Nhưng adb trên host **không thấy** emulator trong container (khác network namespace) — đã thử hôm nay, `emulator-5554 offline`.
- **(b)** Chạy harness **trong container**, model client trỏ tới `host.docker.internal:8700` (cần `--add-host` hoặc dùng IP gateway).
- **(c)** Restart container với `--network host` (như lần 2026-11-06), rồi model server ở `127.0.0.1:8700` truy cập được.

Tôi nghiêng **(c)** vì nó đã chứng minh chạy được, nhưng restart container mất ~5 phút boot + lifespan.

## Kế hoạch chạy chuẩn (chờ bạn duyệt)

### Bước 1 — Sửa model server (10 phút)
- `PYTHONPATH` phải gồm cả `benchmarks/train` (có `train_d1.py`) và `serving/`
- Launch lại, smoke 1 request, xác nhận `/health` = 200 và một quyết định thật trả về
- **Gate: nếu smoke fail → dừng, không đụng tới task**

### Bước 2 — Sửa đường container→model (10 phút)
- Chọn (b) hoặc (c) ở trên; tôi sẽ báo bạn chọn cái nào sau khi đo
- **Gate: một request model từ trong container phải trả 200**

### Bước 3 — Smoke 8 task (20 phút)
- `run_bench.py --config configs/gate_e8_pure.yaml --limit 8`
- **Gate: episode ghi đúng, không crash, timeout/invalid đếm được**
- Nếu fail → dừng và báo, không chạy full

### Bước 4 — Chạy chuẩn 116 task, điều kiện d1-2B (90 phút, background)
- `run_bench.py --config configs/main_116.yaml`
- Pre-register trước: `docs/research/PRE-REG-2026-11-08-androidworld-116.md`
- Ghi trước dự đoán để không đặt bar sau: **SR_all kỳ vọng 2–8%**
- Resumable nên đứt thì chạy tiếp

### Bước 5 — Control `wait` 116 task (30 phút)
- Cùng 116 task, policy `wait`
- **Kiểm tra harness**: nếu `wait` cũng >0 thì harness tự cho điểm → sai, phải dừng

### Bước 6 — Báo cáo
- Bảng: `SR_all` + Wilson, `SR_reachable` + coverage, invalid/crash/timeout, latency p50
- Sửa §5.7 paper: 8 task → 116 task
- Nêu rõ 2 action controller (`keyboard_enter`, `navigate_home`) là scaffolding nếu tôi thêm

## Điều tôi CẦN bạn quyết

1. **Xác nhận chạy** — 116 task × 2 điều kiện ≈ **2 giờ máy**, + retry.
2. **Có thêm 2 rule controller không?** (`type_text` → `keyboard_enter`; kẹt → `navigate_home`). Nếu có, phải khai báo `scaffolding` trong paper. Nếu không, SR sẽ thấp hơn nhưng "sạch".
3. **Cách nối container→model**: (b) hay (c)?

Tôi **không** làm gì thêm cho tới khi bạn xác nhận.
