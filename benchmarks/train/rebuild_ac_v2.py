#!/usr/bin/env python3
"""R2 corpus: the v1 rows, re-extracted the AndroidWorld way, with effect and local-goal labels.

Plan 2026-11-10 §5 Stage 2 "R2"; audit 2026-11-10 §2 D2/D3, §5.1, §5.3 R4.

Input  : AndroidControl raw TFRecords (all shards + nothing else) and the v1 compact files
         (`compact_cur15k_v1.jsonl`, `compact_validation_v1.jsonl`).
Output : `compact_cur15k_v2.jsonl`, `compact_validation_v2.jsonl` — EXACTLY the v1 (ep, step) rows in v1
         order, every v1 field kept unchanged (their `img` paths stay valid), plus:

  elements            AndroidWorld's served list for this screen: `forest_to_ui_elements(forest,
                      exclude_invisible_elements=True)` vendored verbatim from android_world@55ddaf5
                      (`aw_extract.py`), one dict per element with text/content_description/hint_text/
                      class_name/bbox_pixels/state flags/resource_name/package_name. List order =
                      AndroidWorld's index.
  target_el_idx       index in `elements` of the SMALLEST element whose bbox contains the gold point
                      (click / long_press / type_text), else null. Zero-area boxes are never chosen.
  target_el_exact     that element's bbox equals the v1 gold target bbox (null if no element)
  target_el_hits_gold that element's centre lies inside the v1 gold target bbox (null if no element)
  target_el_centre_idx smallest element whose CENTRE lies inside the v1 gold target bbox, else null
                      (diagnostic for audit D2: the tap that would still reach a dropped container)
  effect              t < last: the multiset of (class_name, text, content_description, bbox, flags) of the
                      extracted elements differs between screen t and t+1; null on the terminal status row
  effect_app          same, ignoring `com.android.systemui` elements (status-bar clock / battery /
                      notification icons change without any action); null on the terminal status row
  pixel_change        t < last: fraction of pixels (grey, 270x600, top 30 rows excluded) differing by > 24
                      between screenshots t and t+1 (the B0-diagnosis definition); null on the status row
  local_goal          the step's AndroidControl `step_instruction` (null on the status row)
  local_done          next step's instruction differs (whitespace/case-normalised), or t is the last action
                      before the terminal status; null on the status row or if an instruction is missing
  n_steps             number of actions in the source episode (the status row has step == n_steps)
  elements_source     extractor provenance string

Whole source episodes are read (effect needs screen t+1; local_done needs instruction t+1).

The differential check recomputes, from the raw records and with the ORIGINAL v1 code
(`preprocess_ac.flat_nodes/smallest/convert_action`, `ac_common.pick_candidates/render_history_line`),
every v1 field it can: goal, action, gold target bbox, W/H, history, candidate list, and per-episode step
count; mismatches are counted, never repaired.

Usage (the extract phase reads ~50 GB; run it niced, it is resumable per shard):
  python rebuild_ac_v2.py extract --raw RAW --rows V1_TRAIN V1_VAL --work WORK --workers 6
  python rebuild_ac_v2.py merge --work WORK --rows V1_TRAIN V1_VAL --outs V2_TRAIN V2_VAL
"""
import argparse
import glob
import hashlib
import json
import os
import statistics
import sys
import time
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aw_extract as AX  # noqa: E402

SPATIAL = ("click", "long_press", "type_text")
NEEDED = {}  # ep -> {step: {"pt": [x, y] | None, "tgt": bbox | None}}; set per worker


# ----------------------------------------------------------------------------- v1 rows
def read_jsonl(path):
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def needed_from_rows(rows):
    """{ep: {step: {"pt", "tgt"}}}: which (ep, step) rows to rebuild and their v1 gold point / target."""
    need = defaultdict(dict)
    for r in rows:
        a = r["action"]
        pt = [a["x"], a["y"]] if a["action_type"] in SPATIAL and "x" in a and "y" in a else None
        need[int(r["ep"])][int(r["step"])] = {"pt": pt, "tgt": r.get("target")}
    return dict(need)


# ----------------------------------------------------------------------------- the v1 oracle
def keep_bbox(b):
    """compact_ac.keep_bbox, verbatim semantics."""
    return {k: b[k] for k in ("x_min", "x_max", "y_min", "y_max", "center")}


def cands_sha(cands):
    return hashlib.sha1(json.dumps(cands, sort_keys=True).encode()).hexdigest()


def v1_oracle(acts, old_nodes, W, H, steps):
    """Recompute, with the ORIGINAL v1 code, action / target / candidate-hash for the requested steps.

    Mirrors preprocess_ac.do_shard lines 129-147 (type_text located at the focused editable node, else at
    the previous click) and compact_ac (pick_candidates + keep_bbox). old_nodes: {step: flat_nodes list}.
    """
    from ac_common import pick_candidates, render_history_line
    from preprocess_ac import convert_action, smallest
    T = len(acts)
    out, last_click = {}, None
    for i in range(T + 1):
        if i == T:
            action = {"action_type": "status", "goal_status": "successful"}
        else:
            action = convert_action(acts[i])
        nodes = old_nodes.get(i)
        target, target_pos = None, None
        if action["action_type"] in ("click", "long_press"):
            if nodes is not None:
                target = smallest(nodes, action["x"], action["y"])
            last_click = (action["x"], action["y"])
        elif action["action_type"] == "type_text" and nodes is not None:
            foc = [n for n in nodes if n["is_focused"] and n["is_editable"]]
            if foc:
                target = min(foc, key=lambda n: n["bbox_pixels"]["area"])
            elif last_click:
                target = smallest(nodes, *last_click)
            if target:
                action["x"], action["y"] = target["bbox_pixels"]["center"]
        if i in steps:
            if target is not None:
                target_pos = next(k for k, n in enumerate(nodes) if n is target)
            cands = pick_candidates(nodes, W[i], H[i]) if nodes is not None else []
            for c in cands:
                c["bbox"] = keep_bbox(c["bbox"])
            out[i] = {"action": json.loads(json.dumps(action)),
                      "history": [render_history_line(convert_action(acts[j])) for j in range(min(i, T))],
                      "target": keep_bbox(target["bbox_pixels"]) if target else None,
                      "target_pos": target_pos, "cands_sha": cands_sha(json.loads(json.dumps(cands))),
                      "n_nodes": len(nodes) if nodes is not None else None}
    return out


def raw_node_list(forest):
    """Raw nodes in flat_nodes order (windows, then nodes) — the index space of preprocess_ac.flat_nodes."""
    return [n for w in forest.windows for n in w.tree.nodes]


def d2_reason(forest, tgt):
    """Why AndroidWorld's list has no element with the gold bbox: inspect every raw node with that bbox."""
    if tgt is None:
        return None
    same = [n for n in raw_node_list(forest)
            if (n.bounds_in_screen.left, n.bounds_in_screen.top, n.bounds_in_screen.right,
                n.bounds_in_screen.bottom) == (tgt["x_min"], tgt["y_min"], tgt["x_max"], tgt["y_max"])]
    if not same:
        return "no_node_with_bbox"
    kinds = set()
    for n in same:
        structural = (not n.child_ids) or bool(n.content_description) or n.is_scrollable
        if not n.is_visible_to_user:
            kinds.add("invisible")
        elif not structural:
            kinds.add("container")
        else:
            kinds.add("kept?")  # should not happen if no AW element has this bbox
    return "+".join(sorted(kinds))


# ----------------------------------------------------------------------------- one episode
def build_episode(ep, goal, acts, instrs, forests, W, H, need, grey=None, old_nodes=None):
    """New fields (and differential-check fields) for the requested steps of ONE episode.

    forests  : list of T+1 parsed forests (objects with .windows[].tree.nodes[])
    need     : {step: {"pt", "tgt"}} from the v1 rows
    grey     : callable i -> grey_small array (None skips pixel_change)
    old_nodes: {step: preprocess_ac.flat_nodes list} for the v1 oracle (None skips it)
    """
    T = len(acts)
    steps = set(need)
    cache = {}

    def elems(i):
        if i not in cache:
            cache[i] = AX.extract_elements(forests[i])
        return cache[i]

    gcache = {}

    def g(i):
        if i not in gcache:
            gcache[i] = grey(i)
        return gcache[i]

    done = AX.local_done_labels(instrs, T)
    oracle = v1_oracle(acts, old_nodes, W, H, steps) if old_nodes is not None else {}
    out = {}
    for t in sorted(steps):
        if t > T:
            out[t] = {"error": f"step {t} > n_steps {T}"}
            continue
        el = elems(t)
        row = {"elements": el}
        row.update(AX.target_fields(el, need[t]["pt"], need[t]["tgt"]))
        tgt = need[t]["tgt"]
        cidx = None
        if tgt is not None:
            best = float("inf")
            for i, e in enumerate(el):
                b = e["bbox_pixels"]
                if b and b["x_max"] > b["x_min"] and b["y_max"] > b["y_min"] and AX.centre_in(b, tgt):
                    a = AX._area(b)
                    if a < best:
                        cidx, best = i, a
        row["target_el_centre_idx"] = cidx
        if t < T:
            row["effect"] = AX.ui_changed(el, elems(t + 1))
            row["effect_app"] = AX.ui_changed(el, elems(t + 1), exclude_packages=(AX.SYSTEM_UI,))
            row["pixel_change"] = round(AX.pixel_change(g(t), g(t + 1)), 6) if grey is not None else None
            row["local_goal"] = instrs[t] if t < len(instrs) else None
            row["local_done"] = done[t]
        else:
            row.update({"effect": None, "effect_app": None, "pixel_change": None, "local_goal": None,
                        "local_done": None})
        row["n_steps"] = T
        row["elements_source"] = AX.EXTRACTOR
        diff = {"goal": goal, "n_steps": T, "W": W[t], "H": H[t], "n_instr": len(instrs),
                "n_aw_elements": len(el)}
        if t in oracle:
            diff.update(oracle[t])
            # D2: the gold bbox exists among ALL nodes (by construction) but no AW element carries it
            if tgt is not None and not any(AX.same_bbox(e["bbox_pixels"], tgt) for e in el):
                diff["d2_reason"] = d2_reason(forests[t], tgt)
        out[t] = {"v2": row, "diff": diff}
    return out


# ----------------------------------------------------------------------------- shard worker
def _mem_available_gb():
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024 / 1024
    except OSError:
        pass
    return float("inf")


def _wait_for_ram(min_gb):
    waited = 0
    while _mem_available_gb() < min_gb:
        if waited % 300 == 0:
            print(f"[ram] MemAvailable < {min_gb} GB, pausing", flush=True)
        time.sleep(30)
        waited += 30


def _init(need, min_gb):
    global NEEDED, MIN_GB
    NEEDED, MIN_GB = need, min_gb


def do_shard(args):
    path, work = args
    from preprocess_ac import flat_nodes, records
    from tfrecord import example_pb2
    name = os.path.basename(path)
    out_path = os.path.join(work, "shards", name + ".jsonl")
    if os.path.exists(out_path + ".done"):
        return name, "skip", 0
    t0, n_ep, n_rows, n_seen = time.time(), 0, 0, 0
    with open(out_path + ".tmp", "w") as fo:
        for raw in records(path):
            n_seen += 1
            ex = example_pb2.Example()
            ex.ParseFromString(raw)
            f = ex.features.feature
            ep = int(f["episode_id"].int64_list.value[0])
            need = NEEDED.get(ep)
            if not need:
                continue
            _wait_for_ram(MIN_GB)
            goal = f["goal"].bytes_list.value[0].decode()
            shots = list(f["screenshots"].bytes_list.value)
            trees = list(f["accessibility_trees"].bytes_list.value)
            W = list(f["screenshot_widths"].int64_list.value)
            H = list(f["screenshot_heights"].int64_list.value)
            acts = [json.loads(x) for x in f["actions"].bytes_list.value]
            instrs = [x.decode() for x in f["step_instructions"].bytes_list.value]
            T = len(acts)
            shape = {"shots": len(shots), "trees": len(trees), "W": len(W), "H": len(H)}
            if any(v != T + 1 for v in shape.values()):
                fo.write(json.dumps({"ep": ep, "episode_error": {"n_actions": T, **shape}}) + "\n")
                continue
            forests = [AX.parse_forest(b) for b in trees]
            old = {i: flat_nodes(trees[i]) for i in need if i <= T}
            res = build_episode(ep, goal, acts, instrs, forests, W, H, need,
                                grey=lambda i: AX.grey_small(shots[i]), old_nodes=old)
            for step, r in sorted(res.items()):
                fo.write(json.dumps({"ep": ep, "step": step, **r}, separators=(",", ":")) + "\n")
                n_rows += 1
            n_ep += 1
    os.replace(out_path + ".tmp", out_path)
    open(out_path + ".done", "w").close()
    return name, f"episodes={n_ep}/{n_seen} rows={n_rows} {time.time() - t0:.0f}s", n_rows


def cmd_extract(a):
    from multiprocessing import Pool
    rows = [r for p in a.rows for r in read_jsonl(p)]
    need = needed_from_rows(rows)
    os.makedirs(os.path.join(a.work, "shards"), exist_ok=True)
    shards = sorted(p for p in glob.glob(os.path.join(a.raw, "android_control-*-of-*"))
                    if not p.endswith(".done") and os.path.exists(p + ".done"))
    if a.limit_shards:
        shards = shards[: a.limit_shards]
    print(f"[extract] rows={len(rows)} episodes={len(need)} shards={len(shards)} workers={a.workers}",
          flush=True)
    total = 0
    with Pool(a.workers, initializer=_init, initargs=(need, a.min_avail_gb)) as pool:
        for name, msg, n in pool.imap_unordered(do_shard, [(p, a.work) for p in shards]):
            total += n
            print(f"[extract] {name}: {msg}", flush=True)
    print(f"EXTRACT_DONE rows_written_this_run={total}", flush=True)


# ----------------------------------------------------------------------------- merge + validation
def load_shard_outputs(work):
    got, ep_errors = {}, []
    for p in sorted(glob.glob(os.path.join(work, "shards", "*.jsonl"))):
        if not os.path.exists(p + ".done"):
            raise SystemExit(f"shard not finished: {p}")
        for line in open(p):
            d = json.loads(line)
            if "episode_error" in d:
                ep_errors.append(d)
                continue
            key = (d["ep"], d["step"])
            if key in got:
                raise SystemExit(f"duplicate output for {key}")
            got[key] = d
    return got, ep_errors


def _q(xs, ps=(0.05, 0.25, 0.5, 0.75, 0.95)):
    if not xs:
        return {}
    s = sorted(xs)
    out = {f"p{int(p * 100)}": s[min(len(s) - 1, int(p * (len(s) - 1) + 0.5))] for p in ps}
    out.update({"min": s[0], "max": s[-1], "mean": round(statistics.mean(s), 2), "n": len(s)})
    return out


def _rate(num, den):
    return {"n": num, "of": den, "pct": round(100 * num / den, 2) if den else None}


def differential(rows, got):
    """v1 field vs the same field recomputed from raw by the v1 code. Returns (counts, examples)."""
    checks = ("missing", "goal", "action_type", "action", "target", "W_H", "history", "cands",
              "status_step_ne_n_steps", "step_beyond_n_steps")
    bad = Counter()
    ex = defaultdict(list)

    def note(k, r, **kw):
        bad[k] += 1
        if len(ex[k]) < 5:
            ex[k].append({"ep": r["ep"], "step": r["step"], **kw})

    for r in rows:
        d = got.get((r["ep"], r["step"]))
        if d is None or "diff" not in d:
            note("missing", r)
            continue
        df = d["diff"]
        if df["goal"] != r["goal"]:
            note("goal", r, v1=r["goal"][:80], raw=df["goal"][:80])
        ra = df.get("action") or {}
        if ra.get("action_type") != r["action"]["action_type"]:
            note("action_type", r, v1=r["action"]["action_type"], raw=ra.get("action_type"))
        elif ra != r["action"]:
            note("action", r, v1=r["action"], raw=ra)
        if df.get("target") != r.get("target"):
            note("target", r, v1=r.get("target"), raw=df.get("target"))
        if (df["W"], df["H"]) != (r["W"], r["H"]):
            note("W_H", r, v1=[r["W"], r["H"]], raw=[df["W"], df["H"]])
        if df.get("cands_sha") != cands_sha(r.get("cands") or []):
            note("cands", r, v1_n=len(r.get("cands") or []))
        if r["action"]["action_type"] == "status" and r["step"] != df["n_steps"]:
            note("status_step_ne_n_steps", r, n_steps=df["n_steps"])
        if r["step"] > df["n_steps"] or (r["action"]["action_type"] != "status" and r["step"] >= df["n_steps"]):
            note("step_beyond_n_steps", r, n_steps=df["n_steps"])
        if df.get("history") != r.get("history"):
            note("history", r, v1=r.get("history"), raw=df.get("history"))
    return {k: bad.get(k, 0) for k in checks}, dict(ex)


def validate(rows, got):
    """Coverage, D2, effect, local_done, effect-vs-pixel agreement on the rebuilt rows."""
    v = [got[(r["ep"], r["step"])]["v2"] for r in rows]
    df = [got[(r["ep"], r["step"])]["diff"] for r in rows]
    at = [r["action"]["action_type"] for r in rows]
    sp = [i for i, a in enumerate(at) if a in SPATIAL]
    res = {"rows": len(rows), "episodes": len({r["ep"] for r in rows})}
    # --- coverage
    with_idx = [i for i in sp if v[i]["target_el_idx"] is not None]
    res["spatial_rows"] = len(sp)
    res["target_el_idx_present"] = _rate(len(with_idx), len(sp))
    res["target_el_exact"] = _rate(sum(1 for i in sp if v[i]["target_el_exact"]), len(sp))
    res["target_el_hits_gold"] = _rate(sum(1 for i in sp if v[i]["target_el_hits_gold"]), len(sp))
    res["target_el_centre_idx_present"] = _rate(sum(1 for i in sp if v[i]["target_el_centre_idx"] is not None),
                                                len(sp))
    res["coverage_by_action"] = {
        a: {"rows": sum(1 for i in sp if at[i] == a),
            "idx": sum(1 for i in sp if at[i] == a and v[i]["target_el_idx"] is not None),
            "exact": sum(1 for i in sp if at[i] == a and v[i]["target_el_exact"]),
            "hits_gold": sum(1 for i in sp if at[i] == a and v[i]["target_el_hits_gold"])}
        for a in SPATIAL}
    res["elements_per_screen"] = _q([len(x["elements"]) for x in v])
    res["empty_element_lists"] = sum(1 for x in v if not x["elements"])
    res["v1_cands_per_screen"] = _q([len(r.get("cands") or []) for r in rows])
    res["all_nodes_per_screen"] = _q([d["n_nodes"] for d in df if d.get("n_nodes") is not None])
    # --- D2: the gold target the OLD extraction had, and the AW extraction drops
    from ac_common import gold_candidate
    old_exact = old_ctr = aw_drop_of_old_exact = 0
    reasons = Counter()
    drop_any_centre = drop_hits = 0
    for i in sp:
        r = rows[i]
        tgt = r.get("target")
        a = r["action"]
        gc = gold_candidate(r.get("cands") or [], tgt, a.get("x"), a.get("y"))
        is_exact_old = gc is not None and AX.same_bbox(gc["bbox"], tgt)
        old_exact += is_exact_old
        old_ctr += gc is not None and not is_exact_old
        aw_has = any(AX.same_bbox(e["bbox_pixels"], tgt) for e in v[i]["elements"])
        if is_exact_old and not aw_has:
            aw_drop_of_old_exact += 1
        if not aw_has:
            reasons[df[i].get("d2_reason") or "unknown"] += 1
            drop_any_centre += v[i]["target_el_centre_idx"] is not None
            drop_hits += bool(v[i]["target_el_hits_gold"])
    aw_has_n = sum(1 for i in sp if any(AX.same_bbox(e["bbox_pixels"], rows[i].get("target"))
                                        for e in v[i]["elements"]))
    res["d2"] = {
        "gold_bbox_in_v1_cands_exact": _rate(old_exact, len(sp)),
        "gold_in_v1_cands_centre_only": _rate(old_ctr, len(sp)),
        "gold_bbox_in_aw_elements": _rate(aw_has_n, len(sp)),
        "dropped_by_aw_of_those_v1_cands_had": _rate(aw_drop_of_old_exact, old_exact),
        "aw_drop_reasons": dict(reasons),
        "dropped_rows_with_an_aw_element_centred_in_gold": _rate(drop_any_centre, len(sp) - aw_has_n),
        "dropped_rows_where_smallest_containing_still_hits_gold": _rate(drop_hits, len(sp) - aw_has_n),
    }
    # --- effect / effect_app / pixel_change
    thr = AX.PIX_SAME
    has = [i for i in range(len(v)) if v[i]["effect"] is not None]
    pc = [v[i]["pixel_change"] for i in has if v[i]["pixel_change"] is not None]
    res["pixel_change"] = _q(pc)
    res["pixel_changed_pct"] = round(100 * sum(1 for x in pc if x > thr) / len(pc), 2) if pc else None
    res["effect_true_but_effect_app_false"] = sum(1 for i in has if v[i]["effect"] and not v[i].get("effect_app"))
    res["effect_false_but_effect_app_true"] = sum(1 for i in has if not v[i]["effect"] and v[i].get("effect_app"))
    for lab in ("effect", "effect_app"):
        eff = [i for i in has if v[i].get(lab) is not None]
        k = sum(1 for i in eff if v[i][lab])
        res[lab] = {"true": k, "false": len(eff) - k, "null": len(v) - len(eff),
                    "pct_true": round(100 * k / len(eff), 2) if eff else None}
        res[f"{lab}_by_action"] = {}
        for a in sorted(set(at)):
            ii = [i for i in eff if at[i] == a]
            if ii:
                kk = sum(1 for i in ii if v[i][lab])
                res[f"{lab}_by_action"][a] = {"rows": len(ii), "true": kk, "pct_true": round(100 * kk / len(ii), 1)}
        eff = [i for i in eff if v[i]["pixel_change"] is not None]  # agreement needs both labels
        tt = sum(1 for i in eff if v[i][lab] and v[i]["pixel_change"] > thr)
        tf = sum(1 for i in eff if v[i][lab] and not v[i]["pixel_change"] > thr)
        ft = sum(1 for i in eff if not v[i][lab] and v[i]["pixel_change"] > thr)
        ff = sum(1 for i in eff if not v[i][lab] and not v[i]["pixel_change"] > thr)
        n = tt + tf + ft + ff
        po = (tt + ff) / n if n else None
        pe = (((tt + tf) * (tt + ft) + (ft + ff) * (tf + ff)) / n ** 2) if n else None
        res[f"{lab}_vs_pixel"] = {
            "threshold": thr, f"{lab}&pix": tt, f"{lab}&!pix": tf, f"!{lab}&pix": ft, f"!{lab}&!pix": ff,
            "agreement_pct": round(100 * po, 2) if n else None,
            "cohen_kappa": round((po - pe) / (1 - pe), 3) if n and pe < 1 else None}
        res[f"{lab}_vs_pixel_by_action"] = {}
        for a in sorted(set(at)):
            ii = [i for i in eff if at[i] == a]
            if ii:
                agree = sum(1 for i in ii if bool(v[i][lab]) == (v[i]["pixel_change"] > thr))
                res[f"{lab}_vs_pixel_by_action"][a] = {
                    "rows": len(ii), "agreement_pct": round(100 * agree / len(ii), 1),
                    "pix_changed_pct": round(100 * sum(1 for i in ii if v[i]["pixel_change"] > thr) / len(ii), 1)}
    # --- local_done
    ld = Counter("null" if x["local_done"] is None else str(x["local_done"]).lower() for x in v)
    res["local_done"] = dict(ld)
    nonnull = [i for i in range(len(v)) if v[i]["local_done"] is not None]
    res["local_done_pct_true"] = round(100 * sum(1 for i in nonnull if v[i]["local_done"]) / len(nonnull), 2) \
        if nonnull else None
    res["local_done_by_action"] = {}
    for a in sorted(set(at)):
        ii = [i for i in nonnull if at[i] == a]
        if ii:
            k = sum(1 for i in ii if v[i]["local_done"])
            res["local_done_by_action"][a] = {"rows": len(ii), "true": k, "pct_true": round(100 * k / len(ii), 1)}
    res["local_done_null_non_status"] = sum(1 for i in range(len(v)) if v[i]["local_done"] is None and at[i] != "status")
    res["local_goal_null_non_status"] = sum(1 for i in range(len(v)) if v[i]["local_goal"] is None and at[i] != "status")
    return res


def cmd_merge(a):
    got, ep_errors = load_shard_outputs(a.work)
    summary = {"extractor": AX.EXTRACTOR, "aw_source": AX.AW_SOURCE, "episode_errors": ep_errors, "files": {}}
    for src, dst in zip(a.rows, a.outs):
        rows = read_jsonl(src)
        counts, examples = differential(rows, got)
        missing = [r for r in rows if (r["ep"], r["step"]) not in got or "v2" not in got[(r["ep"], r["step"])]]
        if missing:
            raise SystemExit(f"{len(missing)} rows of {src} have no rebuilt output (e.g. {missing[0]['ep']}, "
                             f"{missing[0]['step']}); refusing to write a partial corpus")
        tmp = dst + ".tmp"
        with open(tmp, "w") as fo:
            for r in rows:
                new = dict(r)
                for k, val in got[(r["ep"], r["step"])]["v2"].items():
                    if k in new:
                        raise SystemExit(f"v2 field {k} collides with a v1 field")
                    new[k] = val
                fo.write(json.dumps(new, separators=(",", ":")) + "\n")
        os.replace(tmp, dst)
        sha = hashlib.sha256(open(dst, "rb").read()).hexdigest()
        summary["files"][os.path.basename(dst)] = {
            "source": os.path.basename(src), "rows": len(rows), "sha256": sha, "bytes": os.path.getsize(dst),
            "differential": counts, "differential_examples": examples, "validation": validate(rows, got)}
        print(f"[merge] {dst}: rows={len(rows)} sha256={sha}", flush=True)
        print(f"[merge] differential {json.dumps(counts)}", flush=True)
    with open(os.path.join(a.work, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=str)
    print("MERGE_DONE", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--raw", required=True)
    e.add_argument("--rows", nargs="+", required=True)
    e.add_argument("--work", required=True)
    e.add_argument("--workers", type=int, default=6)
    e.add_argument("--min-avail-gb", type=float, default=8.0)
    e.add_argument("--limit-shards", type=int, default=0)
    m = sub.add_parser("merge")
    m.add_argument("--work", required=True)
    m.add_argument("--rows", nargs="+", required=True)
    m.add_argument("--outs", nargs="+", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "extract":
        cmd_extract(a)
    else:
        if len(a.rows) != len(a.outs):
            raise SystemExit("--rows and --outs must pair up")
        cmd_merge(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
