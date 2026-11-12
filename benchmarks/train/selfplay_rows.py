#!/usr/bin/env python3
"""Self-play AndroidWorld episodes -> contract-v3 training rows (2026-11-11). No external model: labels come from the
task's own checker (`probe_score`, asked after every step) and from whether an action changed the screen.

Per step i (state = what the model was given, `state_in`; done_before = probe after step i-1):
* done_before >= 1                      -> label "done":    gold `status` (the task was already complete)
* a click/long-press that changed nothing -> label "avoid": push down that element (`avoid.target_el_idx`)
* `status` while the checker said not done (failed episode) -> label "avoid": push down `status`
* otherwise, in an episode that SUCCEEDED -> label "imitate": gold = the executed action

    python selfplay_rows.py RUN_DIR RUN_NAME APPS.json OUT.jsonl
"""
import glob
import json
import os
import sys

SPATIAL_EXEC = ("click", "long_press")


def element_at(elements, x, y):
    """Index of the element the harness tapped: it taps an element's bbox centre; else the smallest box containing."""
    best, area = None, None
    for i, e in enumerate(elements):
        b = e.get("bbox_pixels") or {}
        c = b.get("center") or [(b.get("x_min", 0) + b.get("x_max", 0)) / 2, (b.get("y_min", 0) + b.get("y_max", 0)) / 2]
        if abs(c[0] - x) <= 1 and abs(c[1] - y) <= 1:
            return i
        if b and b["x_min"] <= x <= b["x_max"] and b["y_min"] <= y <= b["y_max"]:
            a = (b["x_max"] - b["x_min"]) * (b["y_max"] - b["y_min"])
            if area is None or a < area:
                best, area = i, a
    return best


def gold_from_executed(ex, elements):
    """(compact action, target bbox, element index) for one executed AndroidWorld action, or None."""
    t = (ex or {}).get("action_type")
    if t in SPATIAL_EXEC or (t == "input_text" and ex.get("x") is not None):
        idx = element_at(elements, ex["x"], ex["y"])
        if idx is None:
            return None
        act = ({"action_type": "type_text", "text": ex.get("text") or "", "x": ex["x"], "y": ex["y"]}
               if t == "input_text" else {"action_type": t, "x": ex["x"], "y": ex["y"]})
        return act, elements[idx]["bbox_pixels"], idx
    if t == "open_app":
        return {"action_type": "open_app", "app_name": ex.get("app_name") or ""}, None, None
    if t == "scroll":
        return {"action_type": "scroll", "direction": ex.get("direction")}, None, None
    if t in ("navigate_back", "navigate_home", "wait"):
        return {"action_type": t}, None, None
    return None


def _screen_size(elements):
    W = max([(e.get("bbox_pixels") or {}).get("x_max", 0) for e in elements] or [1080])
    H = max([(e.get("bbox_pixels") or {}).get("y_max", 0) for e in elements] or [2400])
    return int(W), int(H)


def episode_rows(lines, run, apps):
    steps = sorted((x for x in lines if x.get("type") == "step" and x.get("state_in")), key=lambda s: s["i"])
    res = [x for x in lines if x.get("type") == "result"]
    if not steps or not res:
        return []
    r = res[-1]
    success = bool(r.get("success")) or (r.get("score") or 0) >= 1.0
    ep = "%s:%s" % (run, r.get("episode_id"))
    rows = []
    for k, s in enumerate(steps):
        st = s["state_in"]
        els = st.get("elements") or []
        W, H = _screen_size(els)
        base = {"ep": ep, "step": s["i"], "source": "selfplay", "goal": st.get("goal") or "",
                "history_v3": list(st.get("history") or []), "elements": els, "W": W, "H": H, "apps": list(apps),
                "img": "selfplay/%s/screens/%s" % (run, os.path.basename(s.get("screenshot") or "")),
                "action": None, "target": None, "target_el_idx": None}
        done_before = steps[k - 1].get("probe_score") if k > 0 else None
        ex = s.get("action_executed")
        said_done = bool(s.get("done")) and ex is None
        if done_before is not None and done_before >= 1.0:
            rows.append(dict(base, label="done", action={"action_type": "status", "goal_status": "successful"}))
            continue
        if ex and ex.get("action_type") in SPATIAL_EXEC and s.get("ui_changed") is False:
            idx = element_at(els, ex["x"], ex["y"])
            if idx is not None:
                rows.append(dict(base, label="avoid", avoid={"target_el_idx": idx}))
            continue
        if said_done and not success and done_before is not None and done_before < 1.0:
            rows.append(dict(base, label="avoid", avoid={"action": "status"}))
            continue
        if success and ex:
            g = gold_from_executed(ex, els)
            if g is not None:
                act, target, idx = g
                rows.append(dict(base, label="imitate", action=act, target=target, target_el_idx=idx))
    return rows


def main(run_dir, run, apps_path, out):
    apps = [a["name"] for a in json.load(open(apps_path))]
    n = 0
    with open(out, "w") as fo:
        for f in sorted(glob.glob(os.path.join(run_dir, "episodes", "*.jsonl"))):
            if ".abandoned." in f:
                continue
            for row in episode_rows([json.loads(l) for l in open(f) if l.strip()], run, apps):
                fo.write(json.dumps(row, separators=(",", ":")) + "\n")
                n += 1
    print("rows", n)


if __name__ == "__main__":
    main(*sys.argv[1:5])
