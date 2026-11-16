# Hướng dẫn chạy DMAgent (LLM + DMA-2B) với giao diện web

Khi chạy xong, bạn mở trình duyệt và thấy một trang gồm:
- **Ô nhập prompt** ở trên cùng, kèm nút **Start** và **Stop**. Ví dụ: "Turn off Wi-Fi", "Create a note in Markor named groceries with the text: bread".
- **Màn hình điện thoại** bên trái: ảnh chụp từ thiết bị, cập nhật khoảng 1,5 giây một lần, có vòng đỏ đánh dấu chỗ agent vừa bấm.
- **Dòng thời gian** ở giữa: LLM nghĩ gì, ra mục tiêu nhỏ gì, DMA-2B làm gì và với xác suất bao nhiêu, các lần tự xử lý pop-up, các lần Reflection.
- **Khung "Needs you"** bên phải, chỉ hiện khi cần người: thao tác nhạy cảm cần duyệt, model không chắc chắn, hoặc agent hỏi thêm thông tin. Bạn bấm Approve hoặc Reject, hay gõ câu trả lời.
- **Summary** (kết quả, số bước, thời gian, số lần gọi LLM) và **Earlier tasks** (các task đã chạy).

Màn hình trong trang là ảnh chụp cập nhật liên tục, chỉ để xem, không bấm trực tiếp được. Muốn điều khiển điện thoại bằng tay song song thì dùng cửa sổ emulator, hoặc `scrcpy` với điện thoại thật.

## Kiến trúc khi chạy

```
Trình duyệt ──HTTP──> dma_dashboard (máy có ADB) ──ADB──> điện thoại / emulator
                               │
                               ├──HTTP──> DMA-2B API (máy có GPU, ~8 GB VRAM)
                               └──HTTP──> LLM (endpoint kiểu OpenAI, ví dụ Qwen3.5-27B)
```

Ba phần có thể nằm trên ba máy khác nhau, chỉ cần gọi được nhau qua HTTP. Dashboard phải chạy trên máy thấy thiết bị qua ADB (`adb devices`).

## Bước 1: Thiết bị

- **Emulator:** Android Studio → Device Manager → tạo máy ảo (đã thử trên Pixel 6, Android 13) và bật nó lên.
- **Điện thoại thật:** bật Developer options và USB debugging, rồi cắm cáp.
- Kiểm tra bằng `adb devices`; phải thấy thiết bị, ví dụ `emulator-5554  device`.

## Bước 2: DMA-2B API (máy có GPU)

Dùng mã nguồn trong thư mục `serving/` của repo này và trọng số từ Hugging Face. Cần GPU có khoảng 8 GB trống; `D1_MEM_FRAC` giới hạn phần bộ nhớ GPU mà API được dùng.

```bash
huggingface-cli download tuanhqv123/DMA-2B --local-dir DMA-2B
pip install "transformers>=5" torch pillow fastapi uvicorn
cd serving        # d1_api.py dùng ac_common.py và train_d1.py ở benchmarks/train
PYTHONPATH=../benchmarks/train D1_MODEL=../DMA-2B D1_IMG_W=720 D1_API_TOKEN=tok D1_APPS=../app_list.json \
  D1_MEM_FRAC=0.4 python -m uvicorn d1_api:app --host 0.0.0.0 --port 8700
curl -H "x-api-key: tok" http://localhost:8700/health      # phải thấy "ok": true
```

`app_list.json` là danh sách app đã cài trên thiết bị. DMA-2B chọn app cần mở từ danh sách này. Mỗi dòng có dạng:
`{"name": "settings", "aliases": ["settings"], "package": "com.android.settings"}`.
Với emulator ảnh AndroidWorld, dùng sẵn `serving/aw_apps.json` (33 app). Với điện thoại khác, sửa lại danh sách cho đúng app đã cài; tên package lấy bằng `adb shell pm list packages`.

## Bước 3: Cài mobile-use (bản có DMAgent)

```bash
cd integrations/mobile-use
pip install uv && uv venv .venv --python=3.12 && source .venv/bin/activate
uv pip install -e .
uv pip install uiautomator2 rapidocr-onnxruntime fastapi uvicorn requests
python -m pytest tests/test_dma_agent.py -q        # 17 passed
```

## Bước 4: Khai báo LLM

Tạo hai file, **không commit**:

```bash
echo "https://<llm-endpoint>/v1/chat/completions" > llm_url.txt
echo "<api-key>" > llm_key.txt
```

Endpoint nào theo chuẩn OpenAI chat completions cũng dùng được. Tên model truyền bằng `--llm-model` (mặc định `Qwen/Qwen3.5-27B`).

## Bước 5: Start dashboard

```bash
python -m mobile_use.dma_dashboard \
  --dma-url http://<máy-gpu>:8700 --llm-url-file llm_url.txt --llm-key-file llm_key.txt \
  --apps-json app_list.json --serial emulator-5554 --port 7870
```

Mở **http://localhost:7870**, gõ task vào ô trên cùng rồi bấm **Start**.

Các tuỳ chọn:
- `--confidence 0.79`: ngưỡng hỏi người. Đặt `0` để không hỏi khi model không chắc; thao tác nhạy cảm vẫn luôn phải duyệt.
- `--max-steps 40`: số bước tối đa mỗi task.
- `--ocr`: dùng OCR gắn nhãn cho phần tử không có nhãn (OCR thay thế khi không đọc được cây UI thì luôn bật).

## Chạy không cần giao diện

```bash
# một task, có hỏi duyệt ngay trên terminal
python examples/run_dma_task.py "Turn off Wi-Fi" --dma-url http://<máy-gpu>:8700 \
  --llm-url-file llm_url.txt --llm-key-file llm_key.txt --apps-json app_list.json --ask --out run_wifi
# bộ 12 kịch bản đo tỉ lệ hoàn thành (chỉ chạy trên emulator: các bước chuẩn bị sẽ đổi Wi-Fi, Bluetooth, ghi chú...)
python benchmark/dma_scenarios/run_eval.py --dma-url ... --llm-url-file ... --llm-key-file ... \
  --apps-json app_list.json --out results/
```

## Khi gặp lỗi

| Hiện tượng | Cách xử lý |
|---|---|
| Dashboard báo `error: ConnectionError` | Kiểm tra DMA-2B API (`/health`) và URL của LLM |
| Agent cứ chờ ("screen not readable") | Thử `adb shell uiautomator dump` và `python -c "import uiautomator2 as u; print(len(u.connect().dump_hierarchy()))"` |
| Mở sai app | Tên trong `app_list.json` phải khớp với app đã cài, và DMA-2B API phải được start với cùng file đó (`D1_APPS`) |
| Gõ chữ không ra (tiếng Việt, ký tự đặc biệt) | Cài ADBKeyboard theo hướng dẫn của mobile-use (README, mục 7) |
