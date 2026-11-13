#!/usr/bin/env python3
"""Write a config that runs exactly the suite tasks a run directory has not scored yet.

    make_completion_cfg.py <parent.yaml> <run_dir> <tasks.txt> <out.yaml>

Why it exists (2026-11-10): `_slug` cut task names at 24 characters, so 12 of the 116 tasks shared an
episode id with a shorter-named task and every 116-task run skipped them. The fixed harness gives
those tasks new ids, but it also gives new ids to every OTHER long-named task, so re-running a
finished arm with `tasks: all` would re-run work already on disk. This lists only the tasks with no
scored (`status = 'ok'`) row in the run's index, by TASK NAME, so the shadowed 12 and any task an
infra abort left unscored are both covered and nothing finished is run twice.

The output is the parent config byte-for-byte except a header and the `tasks:` list. Exit 3 and no
file when there is nothing to run.
"""
import datetime
import os
import sqlite3
import sys


def main(argv):
    parent, run_dir, tasks_file, out = argv[1:5]
    with open(tasks_file, "r", encoding="utf-8") as fp:
        names = [line.strip() for line in fp if line.strip()]
    db = sqlite3.connect(os.path.join(run_dir, "index.db"))
    scored = {row[0] for row in db.execute("SELECT task_name FROM episodes WHERE status = 'ok'")}
    todo = [n for n in names if n not in scored]
    print("%d of %d suite tasks scored; %d to run: %s"
          % (len(set(names) & scored), len(names), len(todo), " ".join(todo) or "-"))
    if not todo:
        return 3

    with open(parent, "r", encoding="utf-8") as fp:
        src = fp.read()
    if src.count("  tasks: all\n") != 1:
        raise SystemExit("parent config must contain exactly one `  tasks: all` line: %s" % parent)
    header = ("# COMPLETION RUN generated %s by make_completion_cfg.py from %s for %s:\n"
              "# the %d suite task(s) with no scored episode there. Identical to the parent except this\n"
              "# header and `tasks`.\n"
              % (datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%MZ"), os.path.basename(parent),
                 os.path.basename(os.path.normpath(run_dir)), len(todo)))
    body = src.replace("  tasks: all\n", "  tasks:\n" + "".join("  - %s\n" % n for n in todo))
    with open(out, "w", encoding="utf-8") as fp:
        fp.write(header + body)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
