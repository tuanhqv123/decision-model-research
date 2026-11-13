"""AndroidWorld v3 run: rebuild each screen's element list from the trace (`elements_compact`), apply the server's
rule, and check the rebuild against what the model actually chose (its element idx and label)."""
import json, glob, sys, collections as C
sys.path.insert(0, "benchmarks/train"); import ac_common as A
FL = {"c": "clickable", "e": "editable", "s": "scrollable", "l": "long_clickable"}
def rebuild(comp):
    out = []
    for lab, cls, b, fl in comp:
        d = {"text": lab, "class_name": cls, "bbox": b, "package_name": ""}
        for ch, f in FL.items(): d["is_" + f] = ch in fl
        if not fl and b[3] <= 130: d["package_name"] = A.SYSTEM_UI   # status bar row
        out.append(d)
    return out
def nocap(els, W, H): return A._reading(els, [i for i in A.rule_interactive_or_labelled(els, W, H) if not A.is_keyboard(els[i])])
c = C.Counter(); cut_tasks = C.Counter(); ne_tasks = set()
for f in glob.glob(sys.argv[1] + "/aw116_v3_s0*/episodes/*.jsonl"):
    if ".abandoned." in f: continue
    for l in open(f):
        s = json.loads(l)
        if s.get("type") != "step" or not s.get("elements_compact"): continue
        els = rebuild(s["elements_compact"]); canon = [A.canonical_element(e) for e in els]
        full = nocap(canon, 1080, 2400); kept = A.ELEMENT_RULES["cap40_reading_noime"](canon, 1080, 2400)
        c["steps"] += 1; c["steps > 40 candidates"] += len(full) > 40
        if len(full) > 40: cut_tasks[s["episode_id"]] += 1
        labs = {canon[i]["label"] for i in full}; keptl = {canon[i]["label"] for i in kept}
        if "New Event" in labs:
            c["steps with New Event on screen"] += 1
            if "New Event" not in keptl: c["  ... New Event cut by the cap"] += 1; ne_tasks.add(s["episode_id"].split("_s30")[0])
        el = (s.get("action_raw") or {}).get("element") or {}
        if s.get("action_raw", {}).get("action") in ("click", "long_press", "type_text") and el.get("idx"):
            c["check: element choices"] += 1
            k = int(el["idx"]) - 1
            got = canon[kept[k]]["label"] if k < len(kept) else None
            c["check: rebuilt list gives the same label"] += (got or "(no label)") == el.get("label")
print(dict(c)); print("episodes with any step > 40:", len(cut_tasks)); print("tasks where New Event was cut:", sorted(ne_tasks))
