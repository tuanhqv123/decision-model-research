# Kế hoạch: cổng pass@k cho việc khám phá top-k (2026-11-12)

> **ĐÃ BỎ (2026-11-12).** Thay bằng `2026-11-12-PLAN-v4-retrain-vi.md`: tối ưu khám phá trước khi biết model sai ở đâu là sai thứ tự.

Tài liệu này cụ thể hóa §5 của `2026-11-11-PLAN-after-selfplay-vi.md`. Mọi số liệu đều có file kèm theo trong
`benchmarks/results/2026-11-12-passk/` hoặc `benchmarks/results/2026-11-11-v3/postmortem/`. Số trong ngoặc là mẫu số.

## 0. Câu hỏi cổng này trả lời, và quyết định đi theo nó

**Câu hỏi:** cùng model v3 (ckpt_1824), cùng instance bài, nếu cho model thử các lựa chọn hạng 2–3 của chính nó thì có
tìm ra **thêm** đường đi tới đích mà chọn hạng 1 không tìm được không?

**Vì sao cần hỏi trước khi học lại:**
- Học lại kiểu RFT (AppVLM, UI-Voyager) chỉ có thứ để học khi việc khám phá tạo ra đường đi thành công **mới**.
- Self-play chọn hạng 1 chỉ cho 1 bài mới trong 9 bài thành công (`selfplay_value.out`).
- Nếu khám phá cũng không tìm thêm được gì thì mọi công sức học lại sau đó là vô ích. Cổng này chỉ tốn khoảng 3 giờ
  chạy máy giả lập và không cần học lại.

**Quyết định đi theo kết quả:**
- **ĐẠT:** thu dữ liệu đầy đủ trên các template luyện tập, rồi học v4. Việc này cần một bản đăng ký trước riêng.
- **KHÔNG ĐẠT:** bỏ hướng RFT. Chuyển sang sửa đầu vào: bỏ cắt 40, dòng Progress, học gõ chữ sau khi click ô nhập.

## 1. Thiết kế (mỗi lựa chọn có lý do đo được)

### 1.1 Tập bài: 30 bài, seed 1001, không bài nào chọn hạng 1 từng chạm đích
Nguồn là 54 tập self-play trên seed 1001 (`sp_tasks.txt`): 45 tập thất bại có chấm điểm.
- Bỏ 4 bài chỉ chấm qua `answer`. Model không có hành động này nên trần của các bài đó là 0.
- Còn 41 bài. Giữ các bài có ngân sách ≤ 24 bước, được 30 bài.
- Bỏ **CameraTakePhoto** và **OpenAppTaskEval**. Bộ chấm hỏi sau từng bước cho thấy chọn hạng 1 **đã từng chạm đích**
  rồi làm hỏng (probe = 1 ở bước 2 và bước 1–3). Hai bài này không còn là "chưa tìm ra".
- Thay bằng hai bài có ngân sách 34 bước: **ExpenseDeleteMultiple2** và **RecipeDeleteDuplicateRecipes3**.

Kết quả: **30 bài, cả 30 đều chưa chạm đích khi chọn hạng 1** ở instance này.
- Ngân sách: 10 (×5), 12 (×6), 14 (×3), 16, 18 (×3), 20 (×5), 22 (×2), 24 (×3), 34 (×2). Có 7 bài trên 20 bước.
- 7 bài nằm trong danh sách template để dành (`awenv/heldout_templates.json`): ClockTimerEntry, ExpenseDeleteDuplicates,
  MarkorDeleteNewestNote, RecipeDeleteDuplicateRecipes, BrowserMultiply, RecipeDeleteMultipleRecipes,
  ExpenseDeleteMultiple2. Cổng này chỉ đo, không học, nên được dùng; mọi kết quả sẽ báo tách hai nhóm.
- Seed 30 (seed đánh giá) không bị đụng tới.

### 1.2 Cách lấy mẫu: top-3, hai mức nhiệt độ, không bao giờ lấy mẫu VÀO `status`/`wait`
- **Đầu loại hành động:** lấy mẫu trong 3 loại có xác suất cao nhất, xác suất chuẩn hóa lại ở nhiệt độ T.
- **Phần tử:** nếu loại được chọn là click / long_press / type_text, lấy mẫu phần tử trong top-3 của `target_el`, cùng T.
- **Các đối số khác giữ theo hạng 1:** chữ cần gõ, hướng cuộn, tên app. Đây là giới hạn (§4, H11–H12).
- **Nhiệt độ:** lần thử a1 và a2 dùng T = 0,3; a3 và a4 dùng T = 0,7.

Số đo làm căn cứ: `sampling_policy.out` đo trên 8.444 dòng AC-High; `status_on_aw.out` đo trên 730 bước self-play AW.

| T | Độ chính xác loại, AC-High | Độ chính xác phần tử, AC-High | AW: tỉ lệ bước lệch loại | AW: tỉ lệ bước lệch phần tử |
|---|---|---|---|---|
| 0 (hạng 1) | 81,0% | 60,9% | 0% | 0% |
| 0,3 | 81,2% | 59,9% | 6,4% | 10,6% |
| 0,5 | 80,6% | 58,8% | 10,4% | 16,1% |
| 0,7 | 79,5% | 57,5% | 14,1% | 20,8% |
| 1,0 | 77,3% | 55,5% | 19,0% | 26,8% |

Cột độ chính xác tính theo chính sách "không lấy mẫu vào status". Lý do chọn như trên:

- **Top-3, không lấy mẫu tự do.**
  - Đáp án đúng nằm trong top-3: 97,8% với loại hành động, 73,9% với phần tử (`topk_actest.out`).
  - Lấy mẫu tự do ở T = 1 chỉ đạt 51,2% so với 60,4% của hạng 1.
- **Hai mức T, không một mức.**
  - T = 0,3 gần như không mất độ chính xác mỗi bước (loại 81,2%, phần tử 59,9%). Nó vẫn lệch khỏi hạng 1 ở khoảng 6% số
    bước theo loại và 11% theo phần tử.
  - T = 0,7 lệch nhiều gấp khoảng 2 lần, đổi lại mất khoảng 3 điểm độ chính xác phần tử.
  - Chưa biết trên AW mức nào tốt hơn. Chạy cả hai thì cho thêm thông tin để chọn T cho lần thu dữ liệu sau, mà không tốn
    thêm lần chạy nào. Chỉ báo mô tả; với 30 bài thì không đủ để kết luận.
- **Không lấy mẫu VÀO `status` (bộ chọn gọi là `terminate`), trừ khi `status` đã là hạng 1.** Đo trên AW:
  - Ở bước bài **chưa** xong, nếu cho phép lấy mẫu vào `status` thì xác suất kết thúc sai tăng từ 3,92% lên 4,71% (T 0,3)
    và 6,54% (T 0,7) **mỗi bước**. Chặn lại thì còn 3,51% và 3,07%.
  - Báo xong quá sớm vốn đã là lỗi lớn nhất của v3.
  - Cái giá: ở 15 bước bài **đã** xong, xác suất báo xong giảm từ 60,0% xuống 57,6% (T 0,3) và 52,9% (T 0,7).
  - Thiệt hại này không đáng kể: việc "đã chạm đích" được tính bằng bộ chấm hỏi sau mỗi bước (§1.4), không cần model tự
    báo xong.
- **Không lấy mẫu vào `wait`.** `wait` không làm thay đổi màn hình nên không khám phá được gì, chỉ tiêu tốn ngân sách bước.
- **Lấy mẫu trước, che sau.** Che no-effect (masking) vẫn bật như lúc chạy self-play làm mốc. Nếu mẫu lấy được rơi vào cặp
  (màn hình, hành động) đã được xác minh là không có tác dụng thì masking thay nó như cũ.
  - Trong 100 bước bị che lúc self-play có **0** lần lệch danh sách phần tử (`candidates_mismatch`). Cách dựng lại danh
    sách phần tử là đáng tin.
- **Tái lập được:** số ngẫu nhiên lấy từ `sha256(seed | episode_id | bước)`. Mỗi bước ghi lại hạng đã lấy, xác suất và lý
  do rơi về hạng 1 nếu có.

### 1.3 Năm lần chạy, mỗi lần 30 bài
| Lần chạy | Chính sách | Mục đích |
|---|---|---|
| a0 | hạng 1 (T = 0), chạy lại | **Đối chứng nhiễu:** chọn bài theo "đã thất bại" thì chạy lại cũng có thể thành công nhờ môi trường không tất định |
| a1, a2 | top-3, T = 0,3, seed khám phá 1 và 2 | khám phá nhẹ |
| a3, a4 | top-3, T = 0,7, seed khám phá 3 và 4 | khám phá mạnh |

- Cả 5 lần đều giữ nguyên các thứ sau: checkpoint `v3_s0_ckpt1824`, `cap40_reading_noime`, contract v3, track VA,
  history v3, masking bật, `record.state`, `probe_success_each_step`, `done_threshold` 0.
- Chỉ đường đi có lấy mẫu là thay đổi.

### 1.4 Chỉ số và luật cổng (đăng ký trước, chốt trước khi chạy)
**Chạm đích (reached):** ở một bước nào đó, bộ chấm hỏi sau bước ấy trả về `probe_score ≥ 1`, hoặc tập kết thúc thành
công. Đây là chỉ số liên quan tới dữ liệu: bước chạm đích cho một đường đi thành công, và bước ngay sau nó cho nhãn
`status` đúng.

**Thành công chặt (success):** model tự báo xong và bộ chấm xác nhận. Luôn báo kèm, nhưng không dùng làm cổng.

**Tính công cho khám phá:** một lần chạy a1–a4 chạm đích chỉ được tính nếu **trước bước chạm đích có ít nhất một bước**
mà hành động thực thi khác hạng 1 vì lấy mẫu. Nếu tới đích mà không lệch bước nào thì đó là nhiễu môi trường, không phải
nhờ khám phá.

Các đại lượng:
- `N_explore`: số bài (trên 30) có ít nhất 1 lần trong a1–a4 chạm đích **và** được tính công.
- `N_noise`: số bài a0 chạm đích.
- `N_new`: số bài được tính trong `N_explore` mà a0 **không** chạm đích.

**Luật cổng:**
- **ĐẠT** nếu `N_new ≥ 5`.
- **KHÔNG ĐẠT** nếu `N_new < 5`.

Ngưỡng 5 lấy từ kế hoạch 2026-11-11 và giữ nguyên, không chỉnh sau khi thấy số. Lý do: 5/30 = 17% số bài đang thất bại.
Vòng RFT đầu của AppVLM tăng 11,4 → 17,9 điểm, khoảng 7,5 bài trên 116. Con số 7,5 bài là **mình tự quy đổi** từ số của họ,
không phải số họ báo cáo. Nếu khám phá trên 30 bài không ra nổi 5 bài mới thì một vòng thu dữ liệu đầy đủ khó tạo được mức
tăng cỡ đó.

**Hiệu lực của lần chạy:**
- Một lần chạy không hợp lệ (phải chạy lại phần thiếu) nếu `setup_error` + `unscored` chiếm hơn 20% số tập.
- Cũng không hợp lệ nếu `candidates_mismatch` chiếm hơn 2% số bước không gian có lấy mẫu.
- Tập bị gạt sang bên vì emulator chết được chạy lại theo tên bài, đúng như các lần chạy trước.

**Dự đoán trước (để có thể sai):**
- P1: 7 bài có ngân sách trên 20 bước vẫn **0/7 chạm đích** ở mọi lần chạy.
- P2: `N_noise ≤ 1`. Lần chạy lại trước đây (Q10) có 0/20 kết quả bị lật.

**Báo cáo luôn có:**
- `N_new`, `N_explore`, `N_noise`, pass@1, pass@2 và pass@4 kèm khoảng Wilson;
- tách theo mức T, theo template để dành hay không, và theo ngân sách;
- số thành công chặt;
- tỉ lệ bước lệch thực tế;
- với mỗi bài mới, bước rẽ đầu tiên so với hạng 1, vì đó là đầu vào của kiểu học tại bước rẽ (GRSD).

## 2. Thực hiện (TDD, mỗi task một commit)

**Các file:**
- `benchmarks/aw/awenv/explore.py` (mới);
- `benchmarks/aw/awenv/agent.py`, `benchmarks/aw/run_bench.py` (sửa);
- `benchmarks/aw/passk_gate.py` (mới);
- `benchmarks/aw/tests/test_explore.py`, `test_passk_gate.py` (mới), thêm vào `test_agent.py`, `test_cli.py`;
- 5 config `benchmarks/aw/configs/aw30_passk_s1001_a{0..4}.yaml` và danh sách bài `benchmarks/aw/configs/passk30_tasks.txt`.

### Task 1: `awenv/explore.py`
**Giao diện tạo ra:**
- `explore_rng(seed: int, episode_id: str, step: int) -> random.Random`
- `sample_decision(dec, sent, width, height, *, top_k=3, temperature, rng, never_into=("status", "wait")) -> (dict, dict)`.
  Trả về (quyết định sẽ thực thi, bản ghi
  `{"T", "k", "type_rank", "element_rank", "p_type", "p_element", "fallback"}`).

**Dùng lại** `masking.action_ranking`, `masking.element_points` và `masking._argument_decision`, để khám phá và masking dựng
danh sách phần tử bằng đúng một hàm.

**Test (viết trước, xem chúng đỏ):**
- `test_temperature_zero_returns_the_argmax_decision_unchanged`
- `test_type_is_sampled_only_from_the_top_k`: 5 loại có xác suất, 3.000 lần rút, tập kết quả ⊆ top-3.
- `test_frequencies_follow_the_renormalised_tempered_probabilities`: xác suất 0,6/0,3/0,1, T = 1, 5.000 lần rút, sai số
  ±0,03; ở T = 0,5 thì ứng với bình phương rồi chuẩn hóa.
- `test_never_samples_into_status_or_wait_unless_it_is_the_argmax`: khi hạng 1 là `status` thì vẫn được giữ hoặc rời khỏi
  `status`.
- `test_spatial_sample_takes_an_element_from_the_top_k_of_target_el_at_its_centre`: dùng màn hình mẫu của
  `test_mask_v2.py` dưới contract v3.
- `test_candidates_mismatch_falls_back_to_the_argmax_element_and_says_so`
- `test_type_whose_argument_is_missing_falls_back_and_says_so`: lấy được `open_app` mà tên app rỗng.
- `test_same_seed_episode_and_step_give_the_same_draw`

### Task 2: nối vào `AgentCore`
**Giao diện:** tham số mới `explore: Optional[dict] = None`, gồm `{"top_k", "temperature", "seed"}`.
- Khi khám phá bật thì luôn xin readouts.
- Thứ tự: lấy mẫu → masking → thực thi.
- `history_line` sinh từ hành động **đã thực thi**, giống cách masking đang làm.
- `record["explore"]` lưu bản ghi lấy mẫu. `action_raw` vẫn là điều model đã nói.
- `episode_id` lấy từ runner. Runner cần truyền nó vào; nếu thiếu thì dùng goal + chỉ số bước, và ghi rõ trong bản ghi.

**Test:**
- `test_explore_is_off_by_default_and_changes_nothing`: bản ghi và quyết định giống hệt trước.
- `test_explore_executes_and_records_the_sampled_action_and_history_names_it`
- `test_mask_still_applies_to_a_sampled_action`

### Task 3: config
**Giao diện:** khóa `policy.explore: {top_k, temperature, seed}`.
- `ConfigError` nếu `temperature < 0`, `top_k < 1`, hoặc thiếu `suite.diagnostic: true`. Mục đích: một lần chạy khám phá
  không bao giờ bị đọc nhầm thành số benchmark.
- Bản in plan có một dòng `explore`.

**Test:**
- `test_explore_requires_a_diagnostic_suite`
- `test_plan_carries_the_explore_settings`
- `test_explore_rejects_negative_temperature`

### Task 4: `passk_gate.py`, phân tích đã đăng ký trước
**Giao diện:** `python passk_gate.py --tasks passk30_tasks.txt --budgets aw116_budgets.json --heldout heldout_templates.json --a0 DIR --explore DIR... > gate.out`.
In mọi đại lượng ở §1.4 và một dòng `VERDICT: PASS|FAIL`.

**Test (dữ liệu tập tổng hợp):**
- `test_reached_counts_a_probe_of_one_at_any_step`
- `test_a_success_with_no_sampled_deviation_before_the_goal_gets_no_credit`
- `test_n_new_excludes_tasks_the_argmax_rerun_also_reached`
- `test_verdict_is_pass_at_five_new_tasks_and_fail_at_four`
- `test_setup_errors_and_unscored_episodes_are_excluded_and_counted`
- `test_budget_prediction_is_reported_for_tasks_over_twenty_steps`

### Task 5: chạy
1. Chạy cả bộ test `pytest benchmarks/aw/tests -q`. Mốc hiện tại là 249 passed / 43 skipped; con số mới phải bằng mốc cộng
   số test mới.
2. Đóng gói harness thành `awhV7`, theo đúng cách đã làm với awhV5/V6, và ghi lại sha256.
3. Viết 5 config: bản sao của `aw116_selfplay_s1001.yaml` với `tasks` là 30 bài, `diagnostic: true`, và `policy.explore`.
4. Chạy thử 1 bài × 2 bước với T = 0,7. Kiểm tra các điều sau trước khi chạy dài:
   - bản ghi có `explore`;
   - có ít nhất một bước lệch;
   - `/health` báo đúng `v3_s0_ckpt1824`.
5. Chạy `stage2_chain_par2.sh` với TARGET = 30:
   - máy giả lập 1 chạy a0 → a1 → a3;
   - máy giả lập 2 chạy a2 → a4.
   Mỗi chuỗi có watchdog riêng, sweep sau mỗi cụm, và tự kiểm `/health`.
6. Phân tích bằng `passk_gate.py`, rồi commit kết quả vào `benchmarks/results/2026-11-12-passk/`. Báo cho bạn **chỉ khi
   xong hoặc khi hỏng**.

**Ước tính:**
- Mỗi lần chạy tối đa 518 bước; số bước thực tế ước khoảng 400 (bài thất bại thường chạy hết ngân sách).
- 5 lần chạy, khoảng 2.000 bước.
- Self-play chạy 730 bước trong 1,22 giờ. Với 2 máy giả lập, ước **2–3,5 giờ**, chưa tính sự cố máy giả lập.

**Không đụng tới:** container và tiến trình của khách, vLLM của khách, các file trên máy chủ ngoài `~/aw`, `~/awruns` và
`~/pm_tmp`.

## 3. Câu hỏi phản biện và cách thiết kế đã chặn chúng

| # | Phản biện | Trả lời / cách chặn |
|---|---|---|
| H1 | Chọn bài "đã thất bại" rồi chạy lại thì kiểu gì cũng có vài bài thành công (hồi quy về trung bình) | a0 đo đúng phần đó; `N_new` loại các bài a0 cũng làm được |
| H2 | Chạm đích nhờ may chứ không nhờ lấy mẫu | Chỉ tính công khi có bước lệch khỏi hạng 1 **trước** bước chạm đích |
| H3 | "Chạm đích" dễ hơn "thành công" | Đúng, và có chủ đích: đơn vị của dữ liệu học là bước. Thành công chặt luôn được báo kèm |
| H4 | Số T lấy từ AC-High, không phải AW | Tỉ lệ lệch cũng được đo trên AW (`status_on_aw.out`); AW có lệch phần tử 10,6% và 20,8% |
| H5 | Masking làm nhiễu phép so sánh | Masking bật ở mốc self-play và ở cả 5 lần chạy; khác biệt duy nhất là lấy mẫu |
| H6 | Rò rỉ sang bài đánh giá | Chỉ dùng seed 1001; seed 30 không đụng tới; cổng không học gì. 7 template để dành được báo tách |
| H7 | 30 bài quá ít để kiểm định | Không tuyên bố ý nghĩa thống kê. Đây là luật quyết định chốt trước, kèm khoảng Wilson |
| H8 | Môi trường không tất định nên "cùng instance" không còn giống nhau | Q10: 6/20 đường đi lệch nhau nhưng 0/20 kết quả bị lật; a0 đo lại điều này trên đúng 30 bài |
| H9 | Ngăn lấy mẫu vào `status` làm mất những lần báo xong đúng | Mất 2,4–7,1 điểm xác suất báo xong ở bước đã xong, đổi lại bớt kết thúc sai ở bước chưa xong. Chạm đích vẫn được bộ chấm ghi nhận |
| H10 | Bỏ 2 bài từng chạm đích là chọn mẫu theo ý mình | Luật bỏ được viết ra ("hạng 1 đã từng chạm đích") và áp như nhau cho cả 41 bài; chỉ 2 bài khớp |
| H11 | Không lấy mẫu chữ cần gõ | Giới hạn thật. Các bài hỏng vì gõ sai chữ sẽ không được khám phá. Nếu không đạt, hướng "học gõ chữ" xử lý phần này |
| H12 | Không lấy mẫu hướng cuộn | Giới hạn. Trên AC, lỗi cuộn chủ yếu là sai loại hành động (16/19), không phải sai hướng (3/19), theo bản ghi 2026-11-04 |
| H13 | Bản ghi lấy mẫu sai thì tính công sai | Task 1 và Task 4 có test cho đúng trường hợp này; chạy thử ở Task 5 bước 4 kiểm trước khi chạy dài |
| H14 | Hai máy giả lập khác nhau | pass@k gộp các lần chạy nên lệch máy chỉ làm loãng kết quả; a0 chạy trên máy 1 |

## 4. Ngoài phạm vi của cổng này
- Không học lại. Không chạm seed 30. Không đổi `cap40`.
- Hai thử nghiệm rẻ (dòng Progress, bỏ cắt 40) sẽ có kế hoạch riêng. Chúng cần GPU suy luận trên máy học, không cần máy giả
  lập, nên chạy song song được và không ảnh hưởng tới cổng này.
- Nếu ĐẠT, kế hoạch thu dữ liệu đầy đủ cần một bản đăng ký trước riêng. Bản đó phải nêu: tỉ lệ template để dành (một phần
  ba), T, k, và cách ghép dữ liệu với dữ liệu v3.
