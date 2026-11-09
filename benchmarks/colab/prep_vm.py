"""VM-side data prep: pull both files from public sources (no uploads needed).
Run: colab exec -s d12b --timeout 5400 -f /tmp/prep_vm.py
"""
import json, os, zipfile, urllib.request, time

os.makedirs("/content/data/results", exist_ok=True)

# 1) 398-step leaderboard JSON (public, GUI-Libra repo)
json_url = ("https://raw.githubusercontent.com/GUI-Libra/GUI-Libra/main/"
            "evaluation/offline_evaluation/AndroidControl/data/500_steps_filtered.json")
jp = "/content/data/500_steps_filtered.json"
if not os.path.exists(jp):
    t0 = time.time()
    urllib.request.urlretrieve(json_url, jp)
    print(f"json {os.path.getsize(jp)/1e6:.1f}MB in {time.time()-t0:.0f}s", flush=True)
rows = json.load(open(jp))
need = sorted({r["screenshot"] for r in rows})
print(f"rows={len(rows)} distinct_shots={len(need)}", flush=True)

# 2) images zip (public HF dataset, 6.6GB)
zp = "/content/AndroidControl_images.zip"
if not os.path.exists(zp):
    zurl = ("https://huggingface.co/datasets/demisama/UGround-Offline-Evaluation/"
            "resolve/main/AndroidControl_images.zip")
    t0 = time.time()
    urllib.request.urlretrieve(zurl, zp)
    print(f"zip {os.path.getsize(zp)/1e9:.2f}GB in {time.time()-t0:.0f}s", flush=True)

# 3) extract only the needed shots -> /content/data/images/<ep>/screenshot_<n>.png
zf = zipfile.ZipFile(zp)
names = zf.namelist()
print("zip entries:", len(names), "| sample:", names[:3], flush=True)
endmap = {}
for n in names:
    parts = n.split("/")
    if len(parts) >= 2:
        endmap[f"{parts[-2]}/{parts[-1]}"] = n
os.makedirs("/content/data/images", exist_ok=True)
extracted, missing = 0, []
for s in need:
    dst = f"/content/data/images/{s}"
    if os.path.exists(dst):
        extracted += 1
        continue
    src = endmap.get(s)
    if src is None:
        missing.append(s)
        continue
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with zf.open(src) as fsrc, open(dst, "wb") as fdst:
        fdst.write(fsrc.read())
    extracted += 1
print(f"extracted={extracted}/{len(need)} missing={len(missing)}", flush=True)
if missing:
    print("missing sample:", missing[:5], flush=True)

# 4) integrity: every row resolves
bad = 0
for r in rows:
    if not os.path.exists(f"/content/data/images/{r['screenshot']}"):
        bad += 1
print(f"PREP_DONE rows={len(rows)} unresolved={bad}", flush=True)
import torch
print("cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "", flush=True)