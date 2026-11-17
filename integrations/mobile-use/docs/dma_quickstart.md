# Quickstart: run DMAgent (LLM + DMA-2B) with the web dashboard

When it runs, a browser page shows a chat on the left and the phone on the right:
- **the chat**: you type a task at the bottom (e.g. "Turn off Wi-Fi", or "Create a new note in Markor named groceries
  with the text: bread and butter") and press Enter. The agent's work appears under it as it happens: each planner
  thought and local goal, each DMA-2B action with its probability drawn against the ask threshold, pop-up recoveries
  and the reflection check before it finishes, then a result line (outcome, steps, time, planner calls). Click a
  DMA-2B action to open its decision card: the screen DMA-2B read with the chosen element boxed and its confidence,
  the question it was given (the local goal), its answer (the element, the text to type, the scroll direction or the
  app) and the probability of each action against the ask threshold;
- **questions inline**: when the agent needs a person (a sensitive action, an action DMA-2B is unsure of, or a
  question), the chat shows Allow / Don't allow, or a box for your answer, with an optional note for the planner;
- **the phone**, streamed live (about 2 frames per second), with the agent's last tap circled;
- **earlier tasks** in the left column (click one to read its whole run again), and the setup at its foot: the device,
  where DMA-2B runs (GPU or CPU), the planner model and the ask threshold. Click it to open **Settings**.

While no task runs, clicking the screen taps the phone at that point, and the **Back** and **Home** buttons press
those keys. During a task, manual control is off.

## How the parts connect

```
browser --HTTP--> dma_dashboard (a machine with ADB) --ADB--> phone / emulator
                         |
                         +--HTTP--> DMA-2B API (a machine with a GPU, ~5 GB of GPU memory)
                         +--HTTP--> LLM (any OpenAI-compatible chat endpoint, e.g. Qwen3.5-27B)
```

The three parts can run on three machines if they reach each other over HTTP. The dashboard must run where
`adb devices` lists the phone.

## Step 1: the device

- **Emulator:** Android Studio → Device Manager → create a virtual device and start it. Tested on a Pixel 6 with
  Android 13.
- **Phone:** enable Developer options and USB debugging, then connect the cable and accept the prompt on the phone.
- Check with `adb devices`; the device must be listed, e.g. `emulator-5554  device`.

## Step 2: the app list

DMA-2B opens an app by choosing its name from a list, and the agent maps that name to a package. Build the list for
your device from its installed apps (step 4 installs the code first; this command needs only `adbutils`, and
`apkutils` for real app labels):

```bash
python -m mobile_use.agents.dma.make_app_list --serial emulator-5554 --out app_list.json
```

Edit the file to add the names people use ("play store"). Each entry looks like
`{"name": "settings", "aliases": ["settings"], "package": "com.android.settings"}`.

## Step 3: the DMA-2B API (machine with a GPU)

The API is `serving/d1_api.py` in this repository. It imports `ac_common.py` and `train_d1.py` from
`benchmarks/train`. The weights are on Hugging Face.

```bash
git clone https://github.com/tuanhqv123/decision-model-research.git && cd decision-model-research
git clone https://huggingface.co/tuanhqv123/DMA-2B models/DMA-2B          # needs git-lfs
pip install "transformers>=5" torch torchvision pillow fastapi uvicorn
cd serving
PYTHONPATH=../benchmarks/train D1_MODEL=../models/DMA-2B D1_IMG_W=720 D1_API_TOKEN=tok \
  D1_APPS=/path/to/app_list.json D1_MEM_FRAC=0.4 \
  python -m uvicorn d1_api:app --host 0.0.0.0 --port 8700
curl -H "x-api-key: tok" http://localhost:8700/health      # "ok": true
```

- `D1_APPS` must be the same app list as in step 2. The API refuses to start without it.
- `D1_MEM_FRAC` caps the share of GPU memory the API may take.
- **No free GPU:** add `D1_DEVICE=cpu` (and `OMP_NUM_THREADS=16`; install the CPU builds of `torch` and
  `torchvision`). The model then runs in float32 and gives the same decision as the GPU on the released example
  (`examples/ac_test_ep27_step1.json`: click on element 13, p = 0.9995 on CPU, 0.9994 on GPU). It is slower: about
  11 s per decision on 16 cores of a Xeon Platinum 8444H, against 0.14 s on an RTX 4090, so a short task takes minutes.
- Use `--host 127.0.0.1` and an SSH tunnel if the machine is shared.

## Step 4: install MobileUse with DMAgent

```bash
cd integrations/mobile-use
pip install uv && uv venv .venv --python=3.12 && source .venv/bin/activate
uv pip install -e .
uv pip install uiautomator2 rapidocr-onnxruntime fastapi uvicorn requests httpx apkutils
python -m pytest tests/test_dma_agent.py tests/test_dma_dashboard.py -q
```

The first time the agent connects, uiautomator2 installs its helper app on the phone. The agent types through
uiautomator2's input method, so any text works, Vietnamese included. When a task ends, it switches back to the
phone's own keyboard.

## Step 5: the LLM

Create two files and **do not commit them**:

```bash
echo "https://<llm-endpoint>/v1/chat/completions" > llm_url.txt
echo "<api-key>" > llm_key.txt
```

Any OpenAI-compatible chat-completions endpoint works. Set the model name with `--llm-model`; the default is
`Qwen/Qwen3.5-27B`.

## Step 6: start the dashboard and set the endpoints

```bash
python -m mobile_use.dma_dashboard --apps-json app_list.json --port 7870
```

Open **http://localhost:7870**. The first time, **Settings** opens by itself. Fill in the three endpoints and press
**Test**, which reaches each one, then **Save**:

| Group | Field | Example |
|---|---|---|
| Phone | ADB device | `emulator-5554`, or `host:5555` for an emulator on another machine (connected with `adb connect`) |
| DMA-2B | API URL, token | `http://<gpu-machine>:8700`, the API's `D1_API_TOKEN` (`tok` if unset) |
| Planner | chat URL, API key, model | `https://<llm-endpoint>/v1/chat/completions`, its key, `Qwen/Qwen3.5-27B` |
| Agent | ask below, step budget | `0.79` (`0` turns these questions off; sensitive actions are always asked), `40` |

The settings are saved to `~/.dmagent.json` (change it with `--settings`). The file is readable by its owner only,
because it holds the keys; the page never receives a key back. They are loaded at the next start. Settings cannot
change while a task runs.

The endpoints can also be given at start, which suits a container. Values saved from the page take precedence.

```bash
DEVICE=emulator-5554 DMA_URL=http://<gpu-machine>:8700 DMA_API_TOKEN=tok \
LLM_URL=https://<llm-endpoint>/v1/chat/completions LLM_KEY=<key> LLM_MODEL=Qwen/Qwen3.5-27B \
python -m mobile_use.dma_dashboard --apps-json app_list.json --port 7870
```

The older flags (`--dma-url`, `--llm-url-file`, `--llm-key-file`, `--llm-model`, `--serial`) still work. `--ocr`
labels elements that have no label from the OCR text on them. OCR is always used when the screen cannot be read at
all.

The app list (step 2) belongs to the phone. If you point the dashboard at another phone, build its list again. The
DMA-2B API must use the same list (its `D1_APPS` file).

## Without the dashboard

```bash
# one task; approvals and questions are asked in the terminal
python examples/run_dma_task.py "Turn off Wi-Fi" --dma-url http://<gpu-machine>:8700 \
  --llm-url-file llm_url.txt --llm-key-file llm_key.txt --apps-json app_list.json --ask --out run_wifi
# the 20-task suite (emulator only: the setups change Wi-Fi, Bluetooth, notes and contacts)
python benchmark/dma_scenarios/check_setups.py emulator-5554
python benchmark/dma_scenarios/run_eval.py --dma-url http://<gpu-machine>:8700 --llm-url-file llm_url.txt \
  --llm-key-file llm_key.txt --apps-json app_list.json --repeat 3 --out results/
```

## Troubleshooting

| Symptom | What to do |
|---|---|
| The API exits with "app list ... is missing" | Set `D1_APPS` to the file from step 2 |
| The timeline shows `error: ConnectionError` | Check the API (`/health`) and the LLM URL |
| The agent keeps waiting ("screen not readable") | Run `python -c "import uiautomator2 as u; print(len(u.connect().dump_hierarchy()))"`; a number above a few hundred means the tree is readable |
| The wrong app opens | The names in the app list must match the installed apps, and the API and the agent must use the same file |
| The phone's keyboard stays hidden after a crash | `adb shell ime list -s`, then `adb shell ime set <your keyboard>` |
| The live screen is blank | Reload the page; the stream reconnects. `/api/screen.png` gives a single frame |
