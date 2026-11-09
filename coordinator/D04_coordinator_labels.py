"""D04 — coordinator targets (Step 9-10 of the guide), built from legal inputs only, with an audit.

The coordinator's output is exactly CONTINUE_PLAN or CREATE_OR_UPDATE_PLAN. Labels here come from the
**step-instruction sequence** of each AndroidControl episode, which is the observable plan-progress evidence
available at decision time. No evaluator outcome, no future branch, no gold action is used as a feature.

Deterministic cases (guide §34), all observable in this data:
  UPDATE(no_plan)        no prior instruction  -> a plan must be created
                         ** EXCLUDED from the learned task (see NOTE): it is a deterministic rule, and its label
                         is identical to the feature hist_len==0. Training on it teaches the model nothing and
                         inflates accuracy to 98%. It stays in the audit as a forced branch, not as a target. 
  UPDATE(app_switch)     the next action launches an app -> new app, new plan
  UPDATE(recovery_back)  system Back after >=2 steps -> the plan's premise is contradicted
  UPDATE(repeat_subgoal) the same subgoal seen before, with no progress -> replan
  CONTINUE               otherwise: the plan is valid and the subgoal pointer advances

Output: coordinator/labels_train.jsonl with {ep, step, goal, hist_len, active_subgoal, prev_subgoal,
        action_type, label, rule}
        coordinator/D04_audit.json with the class balance, per-track strata, and the heuristic baseline.

Run: python3 coordinator/D04_coordinator_labels.py <compact_train.jsonl> <outdir>
"""
import json
import os
import re
import sys
import collections

TOK = re.compile(r"[a-z0-9]+")


def toks(s):
    return set(TOK.findall((s or "").lower()))


LEARNABLE_RULES = ("plan_valid", "repeat_subgoal", "recovery_back", "app_switch")


def label_step(rec, seen, include_no_plan=False):
    """Pure function of (record, set of subgoals already visited). Returns (label, rule).

    Rows with no history are reported as rule `no_plan` but are NOT emitted as learnable targets unless
    ``include_no_plan`` is set: the label equals the feature (hist_len == 0), so it is a rule, not a decision.
    """
    hist = rec.get("history") or []
    at = rec["action"]["action_type"]
    if not hist:
        return "CREATE_OR_UPDATE_PLAN", "no_plan"
    cur = hist[-1]
    if at == "open_app":
        return "CREATE_OR_UPDATE_PLAN", "app_switch"
    if at == "navigate_back" and len(hist) > 2:
        return "CREATE_OR_UPDATE_PLAN", "recovery_back"
    if cur in seen and at in ("click", "wait", "scroll"):
        return "CREATE_OR_UPDATE_PLAN", "repeat_subgoal"
    return "CONTINUE_PLAN", "plan_valid"


def main():
    src = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "coordinator"
    os.makedirs(out, exist_ok=True)
    eps = collections.defaultdict(list)
    for line in open(src, encoding="utf-8"):
        r = json.loads(line)
        eps[r["ep"]].append(r)

    n = 0
    cls = collections.Counter()
    rules = collections.Counter()
    by_type = collections.defaultdict(collections.Counter)
    by_len = collections.defaultdict(collections.Counter)
    path = os.path.join(out, "labels_train.jsonl")
    with open(path, "w", encoding="utf-8") as fp:
        for ep, rows in eps.items():
            rows.sort(key=lambda r: r["step"])
            seen = set()
            for r in rows:
                lab, rule = label_step(r, seen)
                if rule == "no_plan":
                    continue  # deterministic: skip as a learning target, kept in the audit below
                hist = r.get("history") or []
                fp.write(json.dumps({
                    "ep": ep, "step": r["step"], "goal": r.get("goal", "")[:200],
                    "hist_len": len(hist),
                    "active_subgoal": (hist[-1][:160] if hist else ""),
                    "prev_subgoal": (hist[-2][:160] if len(hist) > 1 else ""),
                    "action_type": r["action"]["action_type"],
                    "label": lab, "rule": rule,
                }) + "\n")
                n += 1
                cls[lab] += 1
                rules[rule] += 1
                by_type[r["action"]["action_type"]][lab] += 1
                bucket = "0" if not hist else "1-2" if len(hist) <= 2 else "3-5" if len(hist) <= 5 else "6+"
                by_len[bucket][lab] += 1
                seen.add(hist[-1]) if hist else None

    cont, upd = cls["CONTINUE_PLAN"], cls["CREATE_OR_UPDATE_PLAN"]
    # Guide §43: a trivial heuristic on the same legal inputs must be beaten, not merely matched.
    # The strongest trivial policy is "always CONTINUE"; report its accuracy.
    audit = {
        "n": n, "labels": dict(cls),
        "continue_pct": round(100 * cont / n, 2), "update_pct": round(100 * upd / n, 2),
        "imbalance_ratio": round(cont / max(upd, 1), 2),
        "rules": dict(rules),
        "trivial_baseline_always_continue_acc": round(100 * cont / n, 2),
        "trivial_baseline_always_update_acc": round(100 * upd / n, 2),
        "by_action_type": {k: dict(v) for k, v in sorted(by_type.items(), key=lambda x: -sum(x[1].values()))},
        "by_hist_len": {k: dict(v) for k, v in by_len.items()},
        "leak_check": ("features are goal, hist_len, active_subgoal, prev_subgoal, action_type; "
                       "no evaluator status, no future action, no gold label of a later step"),
    }
    json.dump(audit, open(os.path.join(out, "D04_audit.json"), "w"), indent=1)
    print(json.dumps({k: audit[k] for k in
                      ("n", "labels", "continue_pct", "update_pct", "imbalance_ratio",
                       "trivial_baseline_always_continue_acc")}, indent=1))
    print(f"wrote {path} and D04_audit.json")


if __name__ == "__main__":
    main()
