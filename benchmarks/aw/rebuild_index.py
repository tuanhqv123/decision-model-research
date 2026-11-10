#!/usr/bin/env python3
"""Rebuild a run's index.db from the episode jsonl files on disk.

Why this exists. `Recorder.has_result()` consults SQLite, not the filesystem, so the index is
the only thing that decides whether an episode is skipped. On 2026-11-10 the run directory was
accidentally nested (`<root>/<id>/<id>`); the episode files were moved back into place by hand
but the index was left empty, so every completed episode would have been re-run silently while
the summary printed normally.

Rebuilding from the episode files also makes the index reconstructible in general: results are
the artifacts, the database is a cache of them, and a cache must never be the only copy.

Usage (inside the container, or anywhere the run directory is visible):
    python3 rebuild_index.py /out            # /out is the run root holding episodes/ and screens/
    python3 rebuild_index.py /out --dry-run
"""
import argparse
import json
import os
import sqlite3
import sys


def read_episode(path):
    """Return (episode_id, run_id, status) from an episode jsonl, or None if unreadable."""
    episode_id = run_id = None
    has_result = False
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    # A truncated final line is normal for an episode that died mid-write.
                    continue
                if rec.get("type") == "episode":
                    episode_id = rec.get("episode_id") or episode_id
                    run_id = rec.get("run_id") or run_id
                elif rec.get("type") == "result":
                    has_result = True
    except OSError:
        return None
    if not episode_id:
        # Fall back to the filename: "<episode_id>.jsonl", with optional ".abandoned.N" suffix.
        base = os.path.basename(path)
        episode_id = base.split(".abandoned.")[0][:-len(".jsonl")]
    status = "ok" if has_result else "abandoned"
    return episode_id, run_id or "unknown", status


def _schema():
    """Use the recorder's own schema, never a copy.

    A hand-written schema drifted from `Recorder`'s within one edit (missing `finished_at`), and
    the next `finish_episode` raised `no such column`. The recorder owns the schema; this script
    only fills it.
    """
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from awenv.recorder import _SCHEMA

    return _SCHEMA


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_root")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    ep_dir = os.path.join(a.run_root, "episodes")
    if not os.path.isdir(ep_dir):
        raise SystemExit("no episodes/ under %s" % a.run_root)

    files = sorted(f for f in os.listdir(ep_dir) if f.endswith(".jsonl"))
    rows, skipped_abandoned = [], 0
    for name in files:
        path = os.path.join(ep_dir, name)
        got = read_episode(path)
        if got is None:
            continue
        episode_id, run_id, status = got
        if ".abandoned." in name:
            skipped_abandoned += 1
            continue
        rows.append((episode_id, run_id, status, path))

    ok = sum(1 for r in rows if r[2] == "ok")
    print("files=%d  indexed=%d  ok=%d  abandoned_files_ignored=%d"
          % (len(files), len(rows), ok, skipped_abandoned))
    for r in rows[:5]:
        print("  %-46s %s" % (r[0][:46], r[2]))
    if len(rows) > 5:
        print("  ... %d more" % (len(rows) - 5))

    if a.dry_run:
        print("dry run: nothing written")
        return 0

    db_path = os.path.join(a.run_root, "index.db")
    db = sqlite3.connect(db_path)
    db.executescript(_schema())
    # Delete rows whose episode file no longer exists, BEFORE inserting. The rebuild is the
    # authority on what is on disk, so a stale row is a lie: on 2026-11-10 an episode file was
    # deleted by hand and the index kept claiming it was `ok`, so preflight read "14 ok" against
    # "13 files" forever and refused to start a chunk. INSERT OR REPLACE cannot remove rows.
    live = {r[0] for r in rows}
    stale = [e for (e,) in db.execute("select episode_id from episodes").fetchall()
             if e not in live]
    for e in stale:
        db.execute("delete from episodes where episode_id=?", (e,))
    if stale:
        print("removed %d stale index rows (file no longer on disk): %s"
              % (len(stale), ", ".join(s[:40] for s in stale[:3])))
    for episode_id, run_id, status, path in rows:
        db.execute(
            "INSERT OR REPLACE INTO episodes (episode_id, run_id, status, jsonl_path) "
            "VALUES (?,?,?,?)", (episode_id, run_id, status, path))
        db.execute(
            "INSERT OR IGNORE INTO runs (run_id, config_json, git_sha, started_at) "
            "VALUES (?,?,?,?)", (run_id, "{}", "", ""))
    db.commit()
    total = db.execute("select count(*) from episodes where status='ok'").fetchone()[0]
    db.close()
    print("wrote %s: %d ok episodes" % (db_path, total))
    return 0


if __name__ == "__main__":
    sys.exit(main())
