"""Evaluate a (fine-tuned) d1-2B checkpoint on the AC-v2 398-step leaderboard set.

Serving path only (model.system_one_batch, zero generated tokens). State = goal + the harness's
'Interaction History' (row['previous_actions']) [+ app memory] [+ a11y candidates for track VA].
Scoring = run_acv2.score_row (port of the official scorer, differential-tested 398/398; mirrors eval_rl.py:
no point is read for a non-spatial prediction).

`run_eval` is also called in-process by train_d1 every --eval-every steps.

Usage: python eval_d1.py --ckpt runs/x/ckpt --track V --tag x [--limit 50] [--val]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ac_common import (CONTRACTS, DEFAULT_CONTRACT, FROM_OPTION, SPATIAL, build_questions,  # noqa: E402
                       build_state, chosen_element, element_centre, element_inputs, pick_candidates,
                       read_contract)
from run_acv2 import score_row  # noqa: E402


def ece(conf, y, bins=10):
    conf, y = np.asarray(conf), np.asarray(y)
    e = 0.0
    for lo in np.linspace(0, 1, bins, endpoint=False):
        m = (conf > lo) & (conf <= lo + 1 / bins)
        if m.any():
            e += m.mean() * abs(conf[m].mean() - y[m].mean())
    return float(e)


def val_row(rec, data):
    """A validation decision in the eval-row shape: the a11y 'tree' is just the gold element (score_row's
    smallest-node lookup then returns it), candidates are the stored ones."""
    tree = [{"bbox_pixels": rec["target"], "text": None, "content_description": None}] if rec["target"] else []
    row = {"episode_id": rec["ep"], "step": rec["step"], "goal": rec["goal"], "previous_actions": rec["history"],
           "screenshot": os.path.join(data, rec["img"]), "screenshot_width": rec["W"],
           "screenshot_height": rec["H"], "action": dict(rec["action"]), "accessibility_tree": tree,
           "cands": rec["cands"]}
    if "elements" in rec:  # R2 corpus (compact_*_v2.jsonl): the AndroidWorld element list contract v2 reads
        row["elements"] = rec["elements"]
    return row


def resolve_track(contract, track):
    """The track to evaluate: an element contract (v2) exists on VA only, so VA is its default and V is refused;
    every other contract keeps the old default V."""
    if CONTRACTS[contract].get("elements"):
        if track not in (None, "VA"):
            raise SystemExit(f"contract {contract!r} is defined on the VA track only, not {track!r}")
        return "VA"
    return track or "V"


def element_rule_for(ckpt, contract):
    """The element rule the checkpoint was TRAINED with (its contract.json), for an element contract; else None.
    No contract.json, or one naming another contract -> refused: the element list depends on the rule."""
    if not CONTRACTS[contract].get("elements"):
        return None
    try:
        doc = read_contract(ckpt)
    except FileNotFoundError:
        raise SystemExit(f"contract {contract!r} needs the checkpoint's contract.json for its element_rule ({ckpt})")
    if doc["name"] != contract:
        raise SystemExit(f"--contract {contract} but {ckpt} was trained on contract {doc['name']!r}")
    return doc["element_rule"]


def load_rows(path, shots, data, val=False, limit=0, val_set="validation"):
    if val:
        rows = [val_row(json.loads(l), data) for l in open(f"{data}/compact_{val_set}.jsonl")]
    else:
        rows = json.load(open(path))
        for r in rows:
            r["screenshot"] = os.path.join(shots, r["screenshot"])
    return rows[:limit] if limit else rows


@torch.no_grad()
def run_eval(model, rows, track, memory, batch=8, save_readouts=False, contract=DEFAULT_CONTRACT, element_rule=None):
    """Serve every row through model.system_one_batch and score it. Returns (records, summary).

    Element contract (v2): every row must carry `elements` (the R2 validation file); the state and questions are
    built by `ac_common.element_inputs` / `build_questions` exactly as in training, and the point is the chosen
    element's centre. `element_rule` = the checkpoint's stamped rule (default: the contract's)."""
    from train_d1 import load_image
    elements_contract = bool(CONTRACTS[contract].get("elements"))
    if elements_contract:
        if track != "VA":
            raise ValueError(f"contract {contract!r} is evaluated on the VA track only, not {track!r}")
        element_rule = element_rule or CONTRACTS[contract]["element_rule"]
        return _run_eval_elements(model, rows, batch, save_readouts, contract, element_rule, load_image)
    out, t_start = [], time.time()
    for b0 in range(0, len(rows), batch):
        chunk = rows[b0:b0 + batch]
        reqs, metas = [], []
        for r in chunk:
            W, H = r["screenshot_width"], r["screenshot_height"]
            tr, cands = track, None
            if tr == "VA":
                cands = r["cands"] if "cands" in r else pick_candidates(r.get("accessibility_tree") or [], W, H)
                if not cands:
                    tr, cands = "V", None
            apps = memory.query(r["goal"]) if memory else None
            from train_d1 import TRAJ
            trajs = TRAJ["mem"].query_trajs(r["goal"], TRAJ["k"]) if TRAJ["k"] else None
            state = build_state(r["goal"], tr, cands, apps, r.get("previous_actions") or [], trajs)
            reqs.append((state, build_questions(tr, cands, contract=contract), [load_image(r["screenshot"])]))
            metas.append((r, cands))
        t0 = time.time()
        res = model.system_one_batch(reqs)
        lat = (time.time() - t0) * 1000 / len(chunk)
        for (r, cands), rs in zip(metas, res):
            ans = rs["answers"]
            at = FROM_OPTION[ans["action"]["choice"]]
            probs = ans["action"]["probabilities"]
            conf = float(ans["action"].get("confidence") or max(probs.values()))
            value = None
            if at == "scroll":
                value = ans["scroll_dir"]["choice"]
            elif at == "open_app":
                value = ans["app_name"].get("text") or ""
            elif at == "type_text":
                value = ans["type_text"].get("text") or ""
            elif at == "status":
                value = "successful"
            W, H = r["screenshot_width"], r["screenshot_height"]
            box = ans["target_box"].get("box")
            box_pt = (((box["left"] + box["right"]) / 2) * W, ((box["top"] + box["bottom"]) / 2) * H) if box else None
            pt, el_pt, el_idx = box_pt, None, None
            if cands is not None:
                c = next((c for c in cands if str(c["idx"]) == str(ans["target_el"].get("choice") or "")), None)
                if c is not None:
                    el_pt = tuple(c["bbox"]["center"])
                    el_idx = c["idx"]          # log the CHOICE, not just its centre (R4 defect 5)
                    pt = el_pt
            if at not in SPATIAL:  # the leaderboard scorer (eval_rl.py) reads no point for non-spatial actions
                pt = None
            correct, grounding = score_row(r, at, value, pt)
            # is_done: the independent p(task complete). Recorded for every row so the AUROC can be
            # computed, and so the non-terminal AUROC can be separated from the terminal one.
            is_done_p = None
            try:
                d = ans.get("is_done")
                if isinstance(d, dict):
                    is_done_p = float(d.get("noul"))
                elif d is not None:
                    is_done_p = float(d)
            except (TypeError, ValueError):
                is_done_p = None
            rec = {"episode_id": r["episode_id"], "step": r["step"], "gold": r["action"], "pred": at,
                   "value": value, "pt": pt, "box_pt": box_pt, "el_pt": el_pt, "conf": conf, "probs": probs,
                   "found": ans["target_box"].get("found"), "box_conf": ans["target_box"].get("confidence"),
                   "el_conf": ans["target_el"].get("confidence") if cands is not None else None,
                   "el_idx": el_idx,
                   "box_only_pt": box_pt,
                   "is_done": is_done_p,
                   "correct": bool(correct), "grounding": grounding, "latency_ms": lat}
            if save_readouts:
                # stage 2 needs every slot, including the one a post-processor switches TO
                rec["readouts"] = ans
            if cands is not None:  # the box alone on the same rows: ablation of the a11y choice
                rec["correct_box_only"] = bool(score_row(r, at, value, box_pt if at in SPATIAL else None)[0])
            out.append(rec)
    return out, summarize(out, track, time.time() - t_start)


def _value(at, ans):
    if at == "scroll":
        return ans["scroll_dir"]["choice"]
    if at == "open_app":
        return ans["app_name"].get("text") or ""
    if at == "type_text":
        return ans["type_text"].get("text") or ""
    if at == "status":
        return "successful"
    return None


def _run_eval_elements(model, rows, batch, save_readouts, contract, element_rule, load_image):
    out, t_start = [], time.time()
    for b0 in range(0, len(rows), batch):
        chunk = rows[b0:b0 + batch]
        reqs, metas = [], []
        for r in chunk:
            if not r.get("elements"):
                raise ValueError(f"contract {contract!r} evaluates rows that carry the AndroidWorld element list "
                                 "(--val --val-set validation_v2); this row has none (the AC-v2 398 test rows carry "
                                 f"none): episode {r.get('episode_id')} step {r.get('step')}")
            W, H = r["screenshot_width"], r["screenshot_height"]
            state, kept, keep_idx = element_inputs(r["goal"], r.get("previous_actions") or [], r["elements"], W, H,
                                                   element_rule)
            reqs.append((state, build_questions("VA", kept, contract=contract), [load_image(r["screenshot"])]))
            metas.append((r, kept, keep_idx))
        t0 = time.time()
        res = model.system_one_batch(reqs)
        lat = (time.time() - t0) * 1000 / len(chunk)
        for (r, kept, keep_idx), rs in zip(metas, res):
            ans = rs["answers"]
            at = FROM_OPTION[ans["action"]["choice"]]
            probs = ans["action"]["probabilities"]
            conf = float(ans["action"].get("confidence") or max(probs.values()))
            value = _value(at, ans)
            el_id, e = chosen_element(ans, kept)
            el_pt = element_centre(e) if e is not None else None
            pt = el_pt if at in SPATIAL else None  # the leaderboard scorer reads no point for non-spatial actions
            correct, grounding = score_row(r, at, value, pt)
            rec = {"episode_id": r["episode_id"], "step": r["step"], "gold": r["action"], "pred": at,
                   "value": value, "pt": pt, "box_pt": None, "el_pt": el_pt, "conf": conf, "probs": probs,
                   "found": None, "box_conf": None,
                   "el_conf": (ans.get("target_el") or {}).get("confidence"),
                   "el_idx": el_id, "aw_index": keep_idx[el_id - 1] if el_id is not None else None,
                   "n_elements": len(kept), "element_rule": element_rule,
                   "box_only_pt": None, "is_done": None,
                   "correct": bool(correct), "grounding": grounding, "latency_ms": lat}
            if save_readouts:
                rec["readouts"] = ans
            out.append(rec)
    summ = summarize(out, "VA", time.time() - t_start)
    summ["contract"], summ["element_rule"] = contract, element_rule
    return out, summ


def summarize(out, track, secs):
    n = len(out)
    types = sorted(set(o["gold"]["action_type"] for o in out))
    g = [o for o in out if o["grounding"] is not None]
    s = {"track": track, "n": n, "step_acc": round(sum(o["correct"] for o in out) / n, 4),
         "type_acc": round(sum(o["pred"] == o["gold"]["action_type"] for o in out) / n, 4),
         "grounding": round(sum(1 for o in g if o["grounding"]) / max(1, len(g)), 4),
         "per_type": {t: f"{sum(o['correct'] for o in out if o['gold']['action_type'] == t)}/"
                         f"{sum(1 for o in out if o['gold']['action_type'] == t)}" for t in types},
         "ece_action": round(ece([o["conf"] for o in out], [o["correct"] for o in out]), 4),
         "always_status": round(sum(o["gold"]["action_type"] == "status" for o in out) / n, 4),
         "eval_sec": round(secs)}
    if any("correct_box_only" in o for o in out):
        s["step_acc_box_only"] = round(sum(o.get("correct_box_only", False) for o in out) / n, 4)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="LiquidAI/d1-2B")
    ap.add_argument("--rows", default="evalset/500_steps_filtered.json")
    ap.add_argument("--shots", default="evalset/images")
    ap.add_argument("--data", default="acdata", help="acdata dir (app memory from the TRAIN split)")
    ap.add_argument("--track", default=None, choices=["V", "VA"],
                    help="default V; an element contract (v2) is VA only and defaults to it")
    ap.add_argument("--no-memory", action="store_true")
    ap.add_argument("--traj-k", type=int, default=0)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", default="results")
    ap.add_argument("--val", action="store_true", help="the train-time validation split (calibration), not test")
    ap.add_argument("--val-set", default="validation", help="compact_<name>.jsonl used with --val")
    ap.add_argument("--contract", default="", help="question set; default = the checkpoint's contract.json")
    ap.add_argument("--save-readouts", action="store_true", help="keep every slot readout per row (stage 2)")
    a = ap.parse_args()

    from transformers import AutoModel
    from train_d1 import BASE_REV, build_memory
    rev = None if os.path.isdir(a.ckpt) else BASE_REV
    model = AutoModel.from_pretrained(a.ckpt, trust_remote_code=True, dtype=torch.bfloat16, revision=rev).cuda()
    model.engine.calibration = None  # raw readouts, as trained
    rows = load_rows(a.rows, a.shots, a.data, a.val, a.limit, a.val_set)
    contract = a.contract or (read_contract(a.ckpt)["name"] if os.path.isdir(a.ckpt) else DEFAULT_CONTRACT)
    a.track = resolve_track(contract, a.track)
    rule = element_rule_for(a.ckpt, contract)
    print(f"[contract] {contract}" + (f" element_rule={rule}" if rule else ""), flush=True)
    # An element contract's state has no memory line (training and serving add none).
    memory = None if (a.no_memory or rule) else build_memory(a.data)
    if a.traj_k:
        from train_d1 import TRAJ, build_traj_memory
        TRAJ["mem"], TRAJ["k"] = build_traj_memory(a.data), a.traj_k
    out, summ = run_eval(model, rows, a.track, memory, a.batch, a.save_readouts, contract=contract,
                         element_rule=rule)
    summ["tag"] = a.tag
    summ["contract"] = contract
    os.makedirs(a.out, exist_ok=True)
    path = f"{a.out}/eval-{a.tag}-{a.track}{'-val' if a.val else ''}.jsonl"
    with open(path, "w") as f:
        for o in out:
            f.write(json.dumps(o) + "\n")
    json.dump(summ, open(path.replace(".jsonl", ".summary.json"), "w"), indent=1)
    print("SUMMARY " + json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
