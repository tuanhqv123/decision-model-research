#!/usr/bin/env python3
"""Number of scored episodes in a run directory, read from its index (the recorder's own record).

    python3 scored_count.py <run_dir>

Counting `episodes/*.jsonl` files is wrong twice over: `<id>.abandoned.N.jsonl` files match the glob, and a
partial episode killed with its container keeps the plain `<id>.jsonl` name (it is set aside as
`.abandoned.` only when the same task restarts). On 2026-11-10 R1 s0 had 105 such files and 104 scored
episodes. A scored episode is a row with status 'ok' — the same rule resumption uses (`Recorder.has_result`).
The index is opened read-only, so a run directory that has none stays untouched.
"""
import os
import sqlite3
import sys


def scored_count(run_dir: str) -> int:
    db_path = os.path.join(run_dir, "index.db")
    if not os.path.exists(db_path):
        return 0
    db = sqlite3.connect("file:%s?mode=ro" % db_path, uri=True)
    try:
        return db.execute("SELECT COUNT(DISTINCT episode_id) FROM episodes WHERE status='ok'").fetchone()[0]
    finally:
        db.close()


if __name__ == "__main__":
    print(scored_count(sys.argv[1]))
