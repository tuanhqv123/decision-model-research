# DMA-2B + an LLM planner inside mobile-use

[mobile-use](https://github.com/MadeAgents/mobile-use) (MIT) controls an Android device over ADB and lets new agents be
registered. `dma_agent.py` adds one, `DMAgent`:

| Part | Who does it |
|---|---|
| Plan the task, write the next local goal (1–3 steps), answer questions, decide when the task is finished | an OpenAI-compatible LLM (tested: Qwen3.5-27B, no thinking) |
| Execute a local goal one action per step: action, element, scroll direction, app, text span; say when it is done | DMA-2B over HTTP (`serving/d1_api.py`, one forward pass per decision) |
| Screenshot, actions over ADB | mobile-use's `Environment` |
| Accessibility element list for DMA-2B | `uiautomator dump`, parsed into the element shape DMA-2B was trained on |

DMA-2B's own previous actions, with whether each changed the screen, are its history, exactly as in training
(`ac_common.history_entry_with_label`, `effect_changed_v4`). A local goal ends when DMA-2B answers "done"; it is reported
BLOCKED to the LLM after 6 actions or 2 actions that change nothing.

## Run

```bash
# in a mobile-use checkout (pip install -e .), with the device on adb
cp dma_agent.py mobile_use/agents/ && cp ../../benchmarks/train/ac_common.py mobile_use/agents/dma_ac_common.py
python run_dma_task.py "Turn off Wi-Fi" --serial emulator-5554 --dma-url http://<dma-api>:8700 \
    --llm-url-file llm_url.txt --llm-key-file llm_key.txt --apps-json app_list.json --out run_wifi
```

`--apps-json` is the list of installed apps (name, aliases, package) that DMA-2B chooses from for "open app"; the API
must be started with the same list (`D1_APPS`).

## Tested on an Android 13 emulator (2026-11-16)

| Task | Result (checked with adb) | Actions | LLM calls | Time |
|---|---|---|---|---|
| Turn off Wi-Fi | done: `settings get global wifi_on` = 0 | 13 | 8 | 182 s |
| Create a new note in Markor named shopping with the text: milk and eggs | done: `Documents/markor/shopping.md` = "milk and eggs" | 23 | 13 | 282 s |

Traces and screenshots: `runs/`.
- In the Wi-Fi run, the freshly booted emulator showed an "app not responding" dialog. DMA-2B closed it, and the LLM had
  Settings opened again.
- In the Markor run, the pair got through Markor's first-run screens and the "manage all files" permission page.

Two runs on one emulator are a demonstration, not an evaluation.
