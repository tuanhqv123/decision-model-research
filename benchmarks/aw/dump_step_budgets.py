#!/usr/bin/env python3
"""Read android_world's per-task step budget the way upstream allocates it.

Upstream (`suite_utils.py`):
    run_episode(task) -> max_n_steps=_allocate_step_budget(task.complexity)
    _allocate_step_budget(c) -> int(10 * c)

The FastAPI server does NOT expose complexity or max_steps over HTTP (only
`/task_length`, which is the number of task instances). So the only correct
source is the registry itself, imported here and dumped to JSON.

Writes {"<TaskName>": {"complexity": float, "max_steps": int}} for every task in
the android_world family. Run this *inside* the container.
"""
import json
import sys


def main(out_path="/tmp/step_budgets.json"):
    from android_world import registry, suite_utils

    reg = registry.TaskRegistry()
    tasks = reg.get_registry(registry.TaskRegistry.ANDROID_WORLD_FAMILY)

    table, missing = {}, []
    for name, cls in tasks.items():
        # TaskEval subclasses carry `complexity` as a class attribute; that is the
        # exact value `_allocate_step_budget` consumes.
        complexity = getattr(cls, "complexity", None)
        if complexity is None:
            missing.append(name)
            continue
        table[name] = {
            "complexity": float(complexity),
            "max_steps": int(suite_utils._allocate_step_budget(complexity)),
        }

    with open(out_path, "w") as fh:
        json.dump(table, fh, indent=1, sort_keys=True)

    print("TASKS=%d  WITH_BUDGET=%d  MISSING=%d" % (len(tasks), len(table), len(missing)))
    if missing:
        print("missing:", missing[:10])
    for name in list(table)[:8]:
        print("  %-42s complexity=%-4s -> max_steps=%d"
              % (name, table[name]["complexity"], table[name]["max_steps"]))
    return 0 if table else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:2]))
