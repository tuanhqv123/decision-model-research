#!/usr/bin/env python3
"""Structural label audit for compact_cur15k.jsonl / compact_validation.jsonl (A4, issue I1).

Stdlib only (json, csv, os, math, statistics, random, argparse). No numpy, no torch, no pixels read.
Runs in well under 3 minutes on an M-series Mac.

  python3 benchmarks/train/audit_labels.py \
      --data backup-2026-10-29/GPU_HOST_C/acwork/acdata \
      --out  docs/research/a4-audit

Emits:
  <out>/audit_summary.json   every measured number
  <out>/flagged_rows.csv      one row per flagged training row, with the flag names

What it measures is exactly what A4 asks for and NOTHING that requires human ground truth:
a structural audit can only find labels that are *self-inconsistent* with the screen's own
accessibility tree. It cannot tell you the human annotator clicked the wrong widget. See the
"cannot rule out" section of docs/research/A4-label-audit.md.

Design note: the a11y candidate list in the compact files is already filtered (`pick_candidates`),
so the checks below are consistency checks against that filtered list, not against the raw tree.
Where `records/` is available we additionally join `type_text` targets back to the raw node
attributes (is_editable / is_focused / class_name), because the compact files do not carry them.
"""
import argparse
import csv
import json
import math
import os
import random
import statistics
import sys
import time

SPATIAL = ("click", "long_press", "type_text")


# ----------------------------------------------------------------- helpers
def load_jsonl(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def area(b):
    return (b["x_max"] - b["x_min"]) * (b["y_max"] - b["y_min"])


def contains_point(b, x, y):
    return b["x_min"] <= x <= b["x_max"] and b["y_min"] <= y <= b["y_max"]


def contains_box(outer, inner):
    return (outer["x_min"] <= inner["x_min"] and outer["x_max"] >= inner["x_max"]
            and outer["y_min"] <= inner["y_min"] and outer["y_max"] >= inner["y_max"])


def same_box(a, b):
    return (a["x_min"], a["y_min"], a["x_max"], a["y_max"]) == \
           (b["x_min"], b["y_min"], b["x_max"], b["y_max"])


def gold_candidate(cands, target, gx, gy):
    """Verbatim port of benchmarks/train/ac_common.py:gold_candidate (2026-11-07 version).

    Returns the candidate the trainer would use as the element target, or None when the row
    carries no scoreable element supervision. Duplicated here rather than imported so the audit
    stays importable-free and cannot drift silently if ac_common changes: if ac_common's version
    changes, update this one and the diff is the audit trail.
    """
    if target is None or gx is None:
        return None
    for c in cands:
        if same_box(c["bbox"], target):
            return c

    def centre_in(b):
        cx, cy = (b["x_min"] + b["x_max"]) / 2, (b["y_min"] + b["y_max"]) / 2
        return (target["x_min"] <= cx <= target["x_max"]) and (target["y_min"] <= cy <= target["y_max"])

    usable = [c for c in cands if centre_in(c["bbox"])]
    if usable:
        return min(usable, key=lambda c: area(c["bbox"]))
    return None


def pct(n, d):
    return round(100.0 * n / d, 2) if d else None


def mean_ci(xs, z=1.959963985):
    """Normal-approx 95% CI for a mean. Used only for area stats; proportions use wilson()."""
    if not xs:
        return None
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return [round(m, 2), round(m, 2)]
    sd = statistics.stdev(xs)
    half = z * sd / math.sqrt(len(xs))
    return [round(m - half, 2), round(m + half, 2)]


def wilson(k, n, z=1.959963985):
    """Wilson score 95% interval for a proportion, as a percentage pair."""
    if not n:
        return None
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [round(100 * max(0.0, centre - half), 2), round(100 * min(1.0, centre + half), 2)]


def quantiles(vals):
    if not vals:
        return None
    vals = sorted(vals)
    n = len(vals)

    def q(f):
        # linear interpolation between order statistics (numpy default)
        if n == 1:
            return float(vals[0])
        pos = f * (n - 1)
        lo, hi = int(math.floor(pos)), int(math.ceil(pos))
        if lo == hi:
            return float(vals[lo])
        return float(vals[lo] + (vals[hi] - vals[lo]) * (pos - lo))

    return {
        "n": n,
        "p10": round(q(0.10), 1),
        "median": round(q(0.50), 1),
        "p90": round(q(0.90), 1),
        "mean": round(sum(vals) / n, 1),
        "min": round(float(vals[0]), 1),
        "max": round(float(vals[-1]), 1),
    }


# ----------------------------------------------------------------- stages
def stage_coordinate_sanity(rows):
    """Item 3: gold point in bounds, degenerate target boxes."""
    by_type = {}
    global_flags = {"gold_point_out_of_bounds": 0, "target_zero_area": 0,
                    "target_inverted_x": 0, "target_inverted_y": 0,
                    "gold_equals_zero_zero": 0}
    areas = {}
    examples = []
    for r in rows:
        at = r["action"]["action_type"]
        st = by_type.setdefault(at, {"rows": 0, "gold_out_of_bounds": 0, "zero_area_target": 0,
                                     "inverted": 0, "missing_target": 0})
        st["rows"] += 1
        W, H = r.get("W"), r.get("H")
        t = r.get("target")
        if at in SPATIAL:
            gx, gy = r["action"].get("x"), r["action"].get("y")
            if gx is not None and W and H and not (0 <= gx <= W and 0 <= gy <= H):
                st["gold_out_of_bounds"] += 1
                global_flags["gold_point_out_of_bounds"] += 1
                if len(examples) < 20:
                    examples.append({"ep": r["ep"], "step": r["step"], "type": at,
                                     "flag": "gold_out_of_bounds", "x": gx, "y": gy, "W": W, "H": H})
            if gx == 0 and gy == 0:
                global_flags["gold_equals_zero_zero"] += 1
        if t is None:
            st["missing_target"] += 1
            continue
        a = area(t)
        areas.setdefault(at, []).append(a)
        if a <= 0:
            st["zero_area_target"] += 1
            global_flags["target_zero_area"] += 1
            if len(examples) < 40:
                examples.append({"ep": r["ep"], "step": r["step"], "type": at,
                                 "flag": "target_zero_area"})
        if t["x_min"] > t["x_max"]:
            global_flags["target_inverted_x"] += 1
            st["inverted"] += 1
        if t["y_min"] > t["y_max"]:
            global_flags["target_inverted_y"] += 1
            st["inverted"] += 1
        if (t["x_min"] < 0 or t["y_min"] < 0) and len(examples) < 60:
            examples.append({"ep": r["ep"], "step": r["step"], "type": at,
                             "flag": "target_negative_origin"})
    return {"per_action_type": by_type, "totals": global_flags,
            "target_bbox_area_by_type": {k: quantiles(v) for k, v in areas.items()},
            "examples": examples}


def stage_gold_in_candidate(rows):
    """Item 1: is the gold point inside >=1 candidate, and is gold_candidate() scoreable."""
    out = {}
    for at in ("click", "long_press", "type_text"):
        sub = [r for r in rows if r["action"]["action_type"] == at]
        in_some = 0
        gc_none = 0
        exact = 0
        centre_ok = 0
        no_cands = 0
        flagged = []
        for r in sub:
            gx, gy = r["action"].get("x"), r["action"].get("y")
            cands = r.get("cands") or []
            if not cands:
                no_cands += 1
            hit = [c for c in cands if gx is not None and contains_point(c["bbox"], gx, gy)]
            if hit:
                in_some += 1
            elif len(flagged) < 15:
                flagged.append({"ep": r["ep"], "step": r["step"], "x": gx, "y": gy,
                                "n_cands": len(cands)})
            g = gold_candidate(cands, r.get("target"), gx, gy)
            if g is None:
                gc_none += 1
            else:
                t = r["target"]
                if same_box(g["bbox"], t):
                    exact += 1
                else:
                    cx = (g["bbox"]["x_min"] + g["bbox"]["x_max"]) / 2
                    cy = (g["bbox"]["y_min"] + g["bbox"]["y_max"]) / 2
                    if contains_point(t, cx, cy):
                        centre_ok += 1
        out[at] = {
            "rows": len(sub),
            "rows_with_no_candidates": no_cands,
            "gold_point_inside_some_candidate": in_some,
            "gold_point_inside_some_candidate_pct": pct(in_some, len(sub)),
            "gold_point_inside_some_candidate_ci95": wilson(in_some, len(sub)),
            "unscoreable_gold_point": len(sub) - in_some,
            "gold_candidate_returns_None": gc_none,
            "gold_candidate_returns_None_pct": pct(gc_none, len(sub)),
            "gold_candidate_exact_bbox_match": exact,
            "gold_candidate_smallest_centre_in_target": centre_ok,
            "examples_unscoreable_point": flagged,
        }
    return out


def stage_type_text(rows, type_text_nodes):
    """Item 2: type_text label consistency (compact-only part)."""
    sub = [r for r in rows if r["action"]["action_type"] == "type_text"]
    res = {"rows": len(sub), "target_present": 0, "target_missing": 0,
           "gold_point_inside_target": 0, "gold_point_outside_target": 0,
           "target_equals_no_candidate_bbox": 0,
           "text_empty": 0, "text_empty_pct": None,
           "text_copyable_from_goal_or_history": 0,
           "text_not_copyable_from_goal_or_history": 0,
           "editable_join": None,
           "examples": []}
    for r in sub:
        a = r["action"]
        t = r.get("target")
        gx, gy = a.get("x"), a.get("y")
        if t is None:
            res["target_missing"] += 1
        else:
            res["target_present"] += 1
            if gx is not None and contains_point(t, gx, gy):
                res["gold_point_inside_target"] += 1
                if len(res["examples"]) < 15:
                    res["examples"].append({"ep": r["ep"], "step": r["step"], "x": gx, "y": gy})
            else:
                res["gold_point_outside_target"] += 1
            if not any(same_box(c["bbox"], t) for c in (r.get("cands") or [])):
                res["target_equals_no_candidate_bbox"] += 1
        txt = (a.get("text") or "").strip()
        if not txt:
            res["text_empty"] += 1
        hay = (r.get("goal", "") + " " + " ".join(r.get("history") or [])).lower()
        if txt.lower() in hay:
            res["text_copyable_from_goal_or_history"] += 1
        else:
            res["text_not_copyable_from_goal_or_history"] += 1
    res["text_empty_pct"] = pct(res["text_empty"], res["rows"])
    res["gold_point_inside_target_pct"] = pct(res["gold_point_inside_target"],
                                              res["target_present"])
    res["editable_join"] = type_text_nodes
    return res


def join_type_text_nodes(data_dir, rows):
    """Optional: resolve each type_text target bbox back to its raw a11y node to read
    is_editable / is_focused / class_name. These flags exist only in records/, not in the
    compact files. Streams records/ once, parsing only lines that mention type_text."""
    rdir = os.path.join(data_dir, "records")
    want = {(r["ep"], r["step"]) for r in rows if r["action"]["action_type"] == "type_text"}
    if not os.path.isdir(rdir) or not want:
        return {"available": False, "reason": "records/ not present or no type_text rows"}
    found, parsed = {}, 0
    t0 = time.time()
    for name in sorted(os.listdir(rdir)):
        if not name.endswith(".jsonl"):
            continue
        with open(os.path.join(rdir, name), "rb") as f:
            for line in f:
                if b'"type_text"' not in line:   # cheap prefilter, avoids parsing 89k big rows
                    continue
                d = json.loads(line)
                parsed += 1
                k = (d["ep"], d["step"])
                if k in want:
                    found[k] = d
    stats = {"available": True, "type_text_records_in_corpus": parsed,
             "compact_type_text_rows_matched": len(found),
             "compact_type_text_rows_missing": len(want - set(found)),
             "join_seconds": round(time.time() - t0, 1),
             # Bbox is NOT a unique node identity: 105/450 type_text targets match >1 node.
             # So we report both the naive first-match reading and the any-match reading.
             "target_bbox_matches_multiple_nodes": 0,
             "target_bbox_matches_no_node": 0,
             "any_matching_node_is_editable": 0,
             "no_matching_node_is_editable": 0,
             "first_matching_node_is_editable": 0,
             "any_matching_node_is_focused": 0,
             "no_matching_node_is_focused": 0,
             "any_matching_node_class_contains_EditText": 0,
             "preprocess_path_focused_editable": 0,
             "preprocess_path_last_click_fallback": 0,
             "class_name_histogram_editable_preferred": {}}
    hist = {}
    bad_keys = set()
    for k, d in found.items():
        t = d.get("target")
        nodes = d.get("nodes") or []
        # preprocess_ac.do_shard: type_text target = smallest focused+editable node if any,
        # else the smallest node containing the previous click point. Reproduce which branch ran.
        foc = [n for n in nodes if n.get("is_focused") and n.get("is_editable")]
        if foc:
            stats["preprocess_path_focused_editable"] += 1
        else:
            stats["preprocess_path_last_click_fallback"] += 1
        if t is None:
            stats["target_bbox_matches_no_node"] += 1
            continue
        m = [n for n in nodes if same_box(n["bbox_pixels"], t)]
        if not m:
            stats["target_bbox_matches_no_node"] += 1
            continue
        if len(m) > 1:
            stats["target_bbox_matches_multiple_nodes"] += 1
        if any(n.get("is_editable") for n in m):
            stats["any_matching_node_is_editable"] += 1
        else:
            stats["no_matching_node_is_editable"] += 1
            bad_keys.add(k)
        if m[0].get("is_editable"):
            stats["first_matching_node_is_editable"] += 1
        if any(n.get("is_focused") for n in m):
            stats["any_matching_node_is_focused"] += 1
        else:
            stats["no_matching_node_is_focused"] += 1
        # prefer the editable node when several share the bbox, else the first
        n = next((x for x in m if x.get("is_editable")), m[0])
        cn = n.get("class_name") or "null"
        hist[cn] = hist.get(cn, 0) + 1
        if "EditText" in " ".join(x.get("class_name") or "" for x in m):
            stats["any_matching_node_class_contains_EditText"] += 1
    stats["class_name_histogram_editable_preferred"] = \
        dict(sorted(hist.items(), key=lambda kv: -kv[1]))
    stats["_not_editable_keys"] = sorted(k for k in bad_keys)
    return stats


def stage_duplicates(rows, val_rows):
    """Item 4: exact dup, conflicting-type dup, train/val leakage."""
    sig_seen, dups = {}, []
    for i, r in enumerate(rows):
        sig = (r["img"], json.dumps(r["action"], sort_keys=True))
        if sig in sig_seen:
            dups.append({"ep": r["ep"], "step": r["step"], "first_index": sig_seen[sig]})
        else:
            sig_seen[sig] = i
    by_img = {}
    for r in rows:
        by_img.setdefault(r["img"], set()).add(r["action"]["action_type"])
    mono = sum(1 for v in by_img.values() if len(v) == 1)
    conflicting = {k: sorted(v) for k, v in by_img.items() if len(v) > 1}
    img_cur = {r["img"] for r in rows}
    img_val = {r["img"] for r in val_rows}
    leak_imgs = sorted(img_cur & img_val)
    ep_cur = {r["ep"] for r in rows}
    ep_val = {r["ep"] for r in val_rows}
    val_dup = 0
    seen = set()
    for r in val_rows:
        sig = (r["img"], json.dumps(r["action"], sort_keys=True))
        if sig in seen:
            val_dup += 1
        seen.add(sig)
    return {
        "train_exact_duplicate_rows": len(dups),
        "train_exact_duplicate_examples": dups[:20],
        "train_distinct_images": len(by_img),
        "train_images_with_single_action_type": mono,
        "train_images_with_conflicting_action_types": len(conflicting),
        "train_images_with_conflicting_action_types_pct": pct(len(conflicting), len(by_img)),
        "conflicting_image_examples": [{"img": k, "types": v} for k, v in
                                       list(conflicting.items())[:20]],
        "val_exact_duplicate_rows": val_dup,
        "val_distinct_images": len(img_val),
        "train_val_shared_episodes": len(ep_cur & ep_val),
        "train_val_shared_images_LEAKAGE": len(leak_imgs),
        "train_val_shared_image_examples": leak_imgs[:20],
    }


def stage_history(rows):
    """Item 5: history length vs step, empty history, per-episode contiguity."""
    bad_len = bad_step0 = empty_hist = 0
    empty_slots = empty_slot_rows = 0
    for r in rows:
        h, s = r.get("history") or [], r["step"]
        if len(h) > s:
            bad_len += 1
        if s == 0 and h:
            bad_step0 += 1
        if len(h) == 0:
            empty_hist += 1
        blanks = sum(1 for x in h if not (x or "").strip())
        if blanks:
            empty_slot_rows += 1
            empty_slots += blanks
    ep_steps = {}
    for r in rows:
        ep_steps.setdefault(r["ep"], set()).add(r["step"])
    complete_like = 0
    gaps = {}
    for ep, steps in ep_steps.items():
        mx = max(steps)
        if len(steps) == mx + 1:
            complete_like += 1
        else:
            gaps[ep] = {"max_step": mx, "missing": sorted(set(range(mx + 1)) - steps)[:10]}
    return {
        "rows": len(rows),
        "history_longer_than_step_impossible": bad_len,
        "step0_with_nonempty_history": bad_step0,
        "rows_with_empty_history": empty_hist,
        "rows_with_empty_history_pct": pct(empty_hist, len(rows)),
        "rows_with_blank_history_slots": empty_slot_rows,
        "blank_history_slots_total": empty_slots,
        "episodes": len(ep_steps),
        "episodes_with_contiguous_steps_0_to_max": complete_like,
        "episodes_with_step_gaps_after_subsampling": len(gaps),
        "gap_examples": [{"ep": k, **v} for k, v in list(gaps.items())[:10]],
        "_ep_steps": ep_steps,
    }


def stage_status(rows, ep_steps, records_nsteps):
    """Item 6: status rows should only ever sit on the terminal step of their episode."""
    sub = [r for r in rows if r["action"]["action_type"] == "status"]
    at_max_in_cur15k = 0
    mid_cur15k = []
    at_true_end = 0
    mid_true = []
    no_nsteps = 0
    for r in sub:
        mx = max(ep_steps.get(r["ep"], {r["step"]}))
        if r["step"] == mx:
            at_max_in_cur15k += 1
        else:
            mid_cur15k.append({"ep": r["ep"], "step": r["step"], "max_step_in_cur15k": mx})
        ns = records_nsteps.get(r["ep"])
        if ns is None:
            no_nsteps += 1
            continue
        if r["step"] == ns:      # preprocess_ac writes status at i == T == n_steps
            at_true_end += 1
        else:
            mid_true.append({"ep": r["ep"], "step": r["step"], "true_n_steps": ns})
    return {
        "status_rows": len(sub),
        "at_last_step_present_in_cur15k": at_max_in_cur15k,
        "mid_episode_within_cur15k": len(mid_cur15k),
        "mid_episode_examples": mid_cur15k[:20],
        "episode_n_steps_known": len(sub) - no_nsteps,
        "at_true_terminal_step_n_steps": at_true_end,
        "not_at_true_terminal_step": len(mid_true),
        "not_at_true_terminal_examples": mid_true[:20],
        "goal_status_values": sorted({r["action"].get("goal_status") for r in sub}),
    }


def episode_nsteps(data_dir, rows):
    """n_steps only exists in records/; read it from any record per episode (cheap prefilter)."""
    rdir = os.path.join(data_dir, "records")
    want = {r["ep"] for r in rows if r["action"]["action_type"] == "status"}
    out = {}
    if not os.path.isdir(rdir) or not want:
        return out
    for name in sorted(os.listdir(rdir)):
        if not name.endswith(".jsonl"):
            continue
        with open(os.path.join(rdir, name), "rb") as f:
            for line in f:
                if b'"n_steps"' not in line:
                    continue
                i = line.find(b'"ep":')
                if i < 0:
                    continue
                j = line.find(b",", i)
                try:
                    ep = int(line[i + 5:j])
                except ValueError:
                    continue
                if ep in want and ep not in out:
                    k = line.find(b'"n_steps":')
                    if k >= 0:
                        out[ep] = int(line[k + 10:line.find(b",", k)])
        if len(out) == len(want):
            break
    return out


def stage_images(rows, data_dir, sample_n, seed):
    """Item 7: does the img path resolve. Existence only, no pixels."""
    rng = random.Random(seed)
    idx = list(range(len(rows)))
    sample = rng.sample(idx, min(sample_n, len(rows)))
    miss_sample = [rows[i] for i in sample
                   if not os.path.exists(os.path.join(data_dir, rows[i]["img"]))]
    miss_all = [r for r in rows if not os.path.exists(os.path.join(data_dir, r["img"]))]
    return {
        "sample_size": len(sample),
        "seed": seed,
        "sample_missing": len(miss_sample),
        "sample_missing_pct": pct(len(miss_sample), len(sample)),
        "sample_missing_ci95": wilson(len(miss_sample), len(sample)),
        "full_row_scan_missing": len(miss_all),
        "full_row_scan_missing_pct": pct(len(miss_all), len(rows)),
        "full_row_scan_distinct_paths": len({r["img"] for r in rows}),
        "missing_examples": [r["img"] for r in miss_all[:20]],
    }


def stage_ambiguity(rows):
    """Item 4b: a usable proxy for 'multiple valid actions'.

    Warning on interpretation: a per-image conflict test is VACUOUS on cur15k, because curation
    kept at most one row per (ep, step), so every img appears exactly once. The proxy below is
    instead about the *screen*: how many distinct candidates carry the same visible label as the
    gold element, i.e. how many other widgets a reasonable annotator could have picked instead.
    """
    out = {}
    for at in ("click", "long_press", "type_text"):
        sub = [r for r in rows if r["action"]["action_type"] == at]
        g = gold_candidate_rows = 0
        ambiguous = 0
        amb_examples = []
        for r in sub:
            gc = gold_candidate(r.get("cands") or [], r.get("target"),
                                r["action"].get("x"), r["action"].get("y"))
            if gc is None:
                continue
            g += 1
            gold_candidate_rows += 1
            lab = (gc.get("label") or "").strip()
            if not lab:
                continue      # unlabelled icons are not a visible-label ambiguity
            same = [c for c in r["cands"]
                    if (c.get("label") or "").strip() == lab and c["idx"] != gc["idx"]]
            if same:
                ambiguous += 1
                if len(amb_examples) < 10:
                    amb_examples.append({"ep": r["ep"], "step": r["step"], "label": lab,
                                         "n_same_label_on_screen": len(same) + 1})
        out[at] = {"rows": len(sub), "rows_with_element_supervision": g,
                   "rows_with_duplicate_gold_label_on_screen": ambiguous,
                   "rows_with_duplicate_gold_label_on_screen_pct": pct(ambiguous, g),
                   "examples": amb_examples}
    return out


# ----------------------------------------------------------------- sample size math
def sample_size_proportion(p, half_width_pct, z=1.959963985):
    """n for a proportion estimate with a target half-width (all in percentage points)."""
    e = half_width_pct / 100.0
    p = p  # proportion, not percent
    return int(math.ceil(z * z * p * (1 - p) / (e * e)))


def sample_size_plan():
    """Item 8: the smallest protocol that could actually measure the true label error rate.

    Finite-population corrected against N = 15,000 rows (fpc matters little at these n).
    """
    N = 15000
    plan = {}
    for target_pct in (10.0, 37.63):
        p = target_pct / 100.0
        for hw in (4.0, 3.0, 2.0):
            n0 = sample_size_proportion(p, hw)
            n = int(math.ceil(n0 / (1 + (n0 - 1) / N)))   # finite population correction
            plan[f"p={target_pct}% +/-{hw}pp 95%"] = {
                "n_uncorrected": n0, "n_fpc_15000": n,
                "worst_case_z2_0.25": sample_size_proportion(0.5, hw),
            }
    plan["_notes"] = (
        "The textbook worst case p=0.5 dominates: 601 rows buys +/-4pp at 95% regardless of the "
        "true rate. Using p=0.5 is the safe planning choice because the true rate is what we are "
        "trying to discover."
    )
    return plan


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True, help="directory holding compact_*.jsonl and imgs/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--train", default="compact_cur15k.jsonl")
    ap.add_argument("--val", default="compact_validation.jsonl")
    ap.add_argument("--image-sample", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-records-join", action="store_true",
                    help="skip the records/ join for type_text node flags and n_steps")
    args = ap.parse_args()

    t0 = time.time()
    train_path = os.path.join(args.data, args.train)
    val_path = os.path.join(args.data, args.val)
    for p in (train_path, val_path):
        if not os.path.exists(p):
            sys.exit(f"FATAL: missing {p}\nRefusing to guess. Point --data at the acdata directory.")
    os.makedirs(args.out, exist_ok=True)

    rows = load_jsonl(train_path)
    val_rows = load_jsonl(val_path)

    mix = {}
    for r in rows:
        mix[r["action"]["action_type"]] = mix.get(r["action"]["action_type"], 0) + 1
    val_mix = {}
    for r in val_rows:
        val_mix[r["action"]["action_type"]] = val_mix.get(r["action"]["action_type"], 0) + 1

    summary = {
        "audit": "A4 structural label audit",
        "train_file": train_path,
        "val_file": val_path,
        "rows": len(rows),
        "episodes": len({r["ep"] for r in rows}),
        "val_rows": len(val_rows),
        "val_episodes": len({r["ep"] for r in val_rows}),
        "action_mix": dict(sorted(mix.items(), key=lambda kv: -kv[1])),
        "action_mix_pct": {k: pct(v, len(rows)) for k, v in
                           sorted(mix.items(), key=lambda kv: -kv[1])},
        "val_action_mix": dict(sorted(val_mix.items(), key=lambda kv: -kv[1])),
        "rows_missing_target": sum(1 for r in rows if r.get("target") is None),
        "rows_with_zero_candidates": sum(1 for r in rows if not r.get("cands")),
    }
    summary["rows_with_zero_candidates_by_type"] = {}
    for r in rows:
        if not r.get("cands"):
            k = r["action"]["action_type"]
            summary["rows_with_zero_candidates_by_type"][k] = \
                summary["rows_with_zero_candidates_by_type"].get(k, 0) + 1

    print("[1/8] coordinate sanity ...", flush=True)
    summary["coordinate_sanity"] = stage_coordinate_sanity(rows)

    print("[2/8] gold point in a11y candidate ...", flush=True)
    summary["gold_in_candidate"] = stage_gold_in_candidate(rows)

    print("[3/8] type_text consistency ...", flush=True)
    tt_nodes = {"available": False, "reason": "skipped by --skip-records-join"}
    nsteps = {}
    if not args.skip_records_join:
        print("      joining records/ for type_text node flags (streams ~8 GB, ~30 s) ...", flush=True)
        tt_nodes = join_type_text_nodes(args.data, rows)
        print("      reading n_steps per episode ...", flush=True)
        nsteps = episode_nsteps(args.data, rows)
    tt = stage_type_text(rows, tt_nodes)
    not_editable_keys = set(tt["editable_join"].pop("_not_editable_keys", []) or [])
    summary["type_text"] = tt

    print("[4/8] duplicates and leakage ...", flush=True)
    summary["duplicates_leakage"] = stage_duplicates(rows, val_rows)
    summary["ambiguity_proxy"] = stage_ambiguity(rows)

    print("[5/8] history validity ...", flush=True)
    hist = stage_history(rows)
    ep_steps = hist.pop("_ep_steps")
    summary["history"] = hist

    print("[6/8] status label placement ...", flush=True)
    summary["status"] = stage_status(rows, ep_steps, nsteps)

    print("[7/8] image existence ...", flush=True)
    summary["images"] = stage_images(rows, args.data, args.image_sample, args.seed)

    # ------------------------------------------------- flagged rows -> CSV
    print("[8/8] building flagged-row CSV ...", flush=True)
    flags = {}

    def flag(r, name):
        flags.setdefault((r["ep"], r["step"]), {"row": r, "flags": []})["flags"].append(name)

    for r in rows:
        at = r["action"]["action_type"]
        t = r.get("target")
        gx, gy = r["action"].get("x"), r["action"].get("y")
        W, H = r.get("W"), r.get("H")
        if at in SPATIAL:
            if gx is None:
                flag(r, "spatial_missing_gold_point")
            elif W and H and not (0 <= gx <= W and 0 <= gy <= H):
                flag(r, "gold_point_out_of_bounds")
            cands = r.get("cands") or []
            if not cands:
                flag(r, "no_candidates")
            elif gx is not None and not any(contains_point(c["bbox"], gx, gy) for c in cands):
                flag(r, "gold_point_in_no_candidate")
            if gold_candidate(cands, t, gx, gy) is None:
                flag(r, "no_scoreable_element_target")
        if t is not None:
            if area(t) <= 0:
                flag(r, "target_zero_area")
            if t["x_min"] > t["x_max"]:
                flag(r, "target_x_min_gt_x_max")
            if t["y_min"] > t["y_max"]:
                flag(r, "target_y_min_gt_y_max")
        if at == "type_text":
            if not (r["action"].get("text") or "").strip():
                flag(r, "type_text_empty_string")
            if t is None:
                flag(r, "type_text_no_target")
            elif gx is not None and not contains_point(t, gx, gy):
                flag(r, "type_text_gold_outside_target")
            if not os.path.exists(os.path.join(args.data, r["img"])):
                flag(r, "image_missing")
        elif at in ("click", "long_press"):
            if not os.path.exists(os.path.join(args.data, r["img"])):
                flag(r, "image_missing")
        if (r["ep"], r["step"]) in not_editable_keys:
            # type_text whose target bbox resolves to no is_editable a11y node: either the
            # preprocess last-click fallback fired, or the focused field was not in the tree.
            flag(r, "type_text_target_not_editable")
        h = r.get("history") or []
        if len(h) > r["step"]:
            flag(r, "history_longer_than_step")
        if r["step"] == 0 and h:
            flag(r, "step0_with_history")
        if any(not (x or "").strip() for x in h):
            flag(r, "blank_history_slot")

    # exact duplicates: flag every member of a duplicated signature
    sigs = {}
    for r in rows:
        sig = (r["img"], json.dumps(r["action"], sort_keys=True))
        sigs.setdefault(sig, []).append(r)
    for sig, group in sigs.items():
        if len(group) > 1:
            for r in group:
                flag(r, "exact_duplicate_row")
    for r in rows:
        ns = nsteps.get(r["ep"])
        if r["action"]["action_type"] == "status" and ns is not None and r["step"] != ns:
            flag(r, "status_not_at_terminal_step")

    csv_path = os.path.join(args.out, "flagged_rows.csv")
    flag_counts = {}
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ep", "step", "action_type", "episode_max_step_in_file", "flags",
                    "gold_x", "gold_y", "target_x_min", "target_x_max", "target_y_min",
                    "target_y_max", "n_cands", "history_len", "img"])
        for (ep, step), v in sorted(flags.items()):
            r = v["row"]
            t = r.get("target") or {}
            for fl in v["flags"]:
                flag_counts[fl] = flag_counts.get(fl, 0) + 1
            w.writerow([ep, step, r["action"]["action_type"],
                        max(ep_steps.get(ep, {step})),
                        ";".join(sorted(v["flags"])),
                        r["action"].get("x"), r["action"].get("y"),
                        t.get("x_min"), t.get("x_max"), t.get("y_min"), t.get("y_max"),
                        len(r.get("cands") or []), len(r.get("history") or []), r["img"]])
    summary["flagged_rows"] = {
        "total_flagged_rows": len(flags),
        "total_flagged_rows_pct": pct(len(flags), len(rows)),
        "flag_counts": dict(sorted(flag_counts.items(), key=lambda kv: -kv[1])),
        "csv": csv_path,
    }
    clean = len(rows) - len(flags)
    summary["flagged_rows"]["clean_rows"] = clean
    summary["flagged_rows"]["clean_rows_pct"] = pct(clean, len(rows))
    summary["flagged_rows"]["clean_rows_ci95"] = wilson(clean, len(rows))

    # per-action-type clean rate: the number the decision actually turns on
    per_type = {}
    for at in sorted(mix):
        n = sum(1 for r in rows if r["action"]["action_type"] == at)
        bad = sum(1 for (ep, step), v in flags.items()
                  if v["row"]["action"]["action_type"] == at)
        per_type[at] = {"rows": n, "flagged": bad, "flagged_pct": pct(bad, n),
                        "clean_pct": pct(n - bad, n)}
    summary["flagged_rows"]["per_action_type"] = per_type
    summary["sampling_protocol"] = sample_size_plan()
    summary["runtime_seconds"] = round(time.time() - t0, 1)

    with open(os.path.join(args.out, "audit_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, sort_keys=False)

    print(json.dumps({
        "rows": summary["rows"], "episodes": summary["episodes"],
        "flagged_rows": len(flags), "clean_pct": summary["flagged_rows"]["clean_rows_pct"],
        "gold_in_cand": {k: v["gold_point_inside_some_candidate_pct"]
                         for k, v in summary["gold_in_candidate"].items()},
        "per_type_clean_pct": {k: v["clean_pct"] for k, v in per_type.items()},
        "runtime_s": summary["runtime_seconds"],
    }, indent=2))
    print(f"\nwrote {os.path.join(args.out, 'audit_summary.json')}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
