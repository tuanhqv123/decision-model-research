"""At hand-annotated first-error steps (v3h), the rank of the CORRECT action type and element in the model's own
recorded readouts (no model call). Rank 2-3 = a light fix can flip it; deep rank = missing knowledge."""
import ast, glob, json, sys
sys.path.insert(0, "../../train")
import ac_common as A

RUNS = "../2026-11-12-v3h/androidworld/aw116_v3h_s0*/episodes/*.jsonl"
# (task, step, correct action, substring of the correct element label or None)
GOLD = [
    ("SystemWifiTurnOffVerify", 2, "click", "Internet"), ("SystemWifiTurnOn", 2, "click", "Internet"),
    ("SystemWifiTurnOnVerify", 2, "click", "Internet"), ("TurnOffWifiAndTurnOnBluetooth", 2, "click", "Internet"),
    ("TurnOnWifiAndOpenApp", 2, "click", "Internet"),
    ("SystemBrightnessMax", 1, "click", "Display"), ("SystemBrightnessMin", 1, "click", "Display"),
    ("SystemBrightnessMinVerify", 1, "click", "Display"),
    ("MarkorDeleteAllNotes", 1, "long_press", None), ("MarkorDeleteNewestNote", 1, "long_press", None),
    ("MarkorDeleteNote", 1, "long_press", "File bold_king_edited"),
    ("SimpleCalendarAddOneEvent", 11, "click", "OK"),
    ("SystemBluetoothTurnOffVerify", 4, "status", None),
    ("ClockTimerEntry", 3, "click", "6"),
    ("CameraTakePhoto", 2, "status", None),
    ("RecipeDeleteMultipleRecipesWithNoise", 2, "click", "Grilled Cheese with Tomato and Basil"),
    ("FilesDeleteFile", 7, "click", "q2a8_fancy_banana.mp3"),
    ("ExpenseAddSingle", 5, "click", "Category"),
    ("MarkorChangeNoteContent", 4, "click", "Save"),
]
OPT = {"status": "terminate", "type_text": "write"}

def load(task):
    for f in glob.glob(RUNS):
        if ".abandoned." in f: continue
        rows = [json.loads(l) for l in open(f)]
        if rows[0]["task_name"] == task:
            return [r for r in rows if r.get("type") == "step"]

for task, i, act, el in GOLD:
    st = load(task); s = st[i]
    si = s["state_in"]; si = ast.literal_eval(si) if isinstance(si, str) else si
    ro = si.get("readouts") or {}
    ap = (ro.get("action") or {}).get("probabilities") or {}
    ranked = sorted(ap, key=lambda k: -ap[k]); want = OPT.get(act, act)
    arank = ranked.index(want) + 1 if want in ranked else None
    erank, elabel, present = None, None, None
    if el:
        kept, _ = A.select_elements(si["elements"], 1080, 2400, "cap40_reading_noime")
        # EXACT label match (review 2026-11-12): a substring match counted the distractor `edited_q2a8_fancy_banana.mp3`
        # as the target `q2a8_fancy_banana.mp3`. Labels are cut at 40 chars, so compare on that prefix.
        want_l = el.lower()[:A.LABEL_CHARS]
        hits = [k for k, e in enumerate(kept, 1) if (e["label"] or "").lower() == want_l]
        allhits = [e for e in si["elements"] if want_l in {str(e.get(f) or "").lower()[:A.LABEL_CHARS] for f in ("text", "content_description")}]
        present = f"in list:{bool(hits)} on screen:{bool(allhits)}"
        tp = (ro.get("target_el") or {}).get("probabilities") or {}
        tr = sorted(tp, key=lambda k: -tp[k])
        if hits:
            erank = min(tr.index(str(h)) + 1 for h in hits if str(h) in tr)
    chosen = (s.get("action_raw") or {}).get("action")
    print(f"{task:38s} step {i:2d}  want {act}:{el or '-':18s} model {chosen:10s} action rank {arank} P={ap.get(want, 0):.2f}  element rank {erank}  {present or ''}")
