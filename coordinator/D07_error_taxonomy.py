"""D07 — error taxonomy (guide Step 17), applied to whatever episode records exist.

Every failure gets exactly one first-cause category, chosen in the guide's order, so a failure is never attributed
to the coordinator when it was an adapter or infrastructure problem.

Run: python3 coordinator/D07_error_taxonomy.py runs/coord_v1/episodes.jsonl
"""
import collections
import json
import sys

CATEGORIES = (
    "technical_infrastructure",     # env crash, connection, OOM
    "action_schema_or_adapter",     # invalid/unrepresentable action
    "planner_output_failure",       # planner returned nothing usable
    "wrong_coordinator_choice",     # policy chose CONTINUE/UPDATE against evidence
    "stale_plan",                   # plan no longer matches the screen
    "incorrect_progress_update",    # subgoal advanced without evidence
    "perception_or_grounding",      # right intent, wrong element
    "user_or_tool_interaction",     # ASK_USER / MCP path
    "timeout",                      # step budget exhausted
    "unknown",                      # could not be diagnosed from the record
)


def classify(rec):
    """First identifiable cause, in the guide's order."""
    if rec.get("terminal_reason") == "technical_error" or rec.get("error"):
        return "technical_infrastructure"
    if rec.get("invalid_actions", 0) > 0:
        return "action_schema_or_adapter"
    if rec.get("terminal_reason") == "planner_failed":
        return "planner_output_failure"
    if rec.get("evaluator_status") is True:
        return None
    if rec.get("terminal_reason") == "step_budget":
        # Only blame the policy when the plan was revised repeatedly and never progressed.
        revs = sum(1 for d in rec.get("decisions", []) if d.get("choice") == "CREATE_OR_UPDATE_PLAN")
        if rec.get("actions", 0) == 0:
            return "wrong_coordinator_choice"
        if revs >= 3:
            return "stale_plan"
        return "timeout"
    if rec.get("evaluator_status") is False:
        return "perception_or_grounding"
    return "unknown"


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if not path:
        print(__doc__)
        print("No record file given. Categories and the classifier are defined; frequencies require a run.")
        return
    recs = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    c = collections.Counter(classify(r) or "success" for r in recs)
    by_pol = collections.defaultdict(collections.Counter)
    for r in recs:
        by_pol[r["policy"]][classify(r) or "success"] += 1
    print(f"{len(recs)} episodes")
    for k, v in c.most_common():
        print(f"  {k:28s} {v:4d}  {100*v/len(recs):5.1f}%")
    print("\nby policy:")
    for pol, cc in by_pol.items():
        print(f"  {pol}: {dict(cc)}")


if __name__ == "__main__":
    main()
