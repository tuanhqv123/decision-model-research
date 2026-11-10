#!/usr/bin/env python3
"""Verify the run directory is writable by the uid the container actually uses.

Why this exists. On 2026-11-10 a chunk of the 116-task suite crashed at the 14th step of an
episode with `sqlite3.OperationalError: attempt to write a readonly database`. The cause was
ownership, not logic: `index.db` had been rebuilt by the HOST user (`aivision`, mode 644) while
the container runs as `root`. SQLite needs to create journal/WAL files next to the database, so
a database the container can read is not enough — the DIRECTORY must be writable too, by the
right uid. File mode `644` plus a different owner is the failure, and it only shows up when the
recorder commits, i.e. after a whole episode has already run.

This script creates, commits and deletes a probe row, then reports the uid the container will
actually write as, so the fix is mechanical instead of guessed.
"""
import argparse
import os
import sqlite3
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", default="/out")
    ap.add_argument("--fix", action="store_true",
                    help="create the run dirs and chown them to the container uid")
    a = ap.parse_args()
    root = a.run_root
    uid, gid = os.getuid(), os.getgid()
    print("container uid=%d gid=%d  run root=%s" % (uid, gid, root))

    if a.fix:
        for sub in ("", "episodes", "screens"):
            os.makedirs(os.path.join(root, sub), exist_ok=True)
        for dirpath, dirnames, filenames in os.walk(root):
            os.chown(dirpath, uid, gid)
            os.chmod(dirpath, 0o777)
            for name in filenames:
                p = os.path.join(dirpath, name)
                try:
                    os.chown(p, uid, gid)
                    os.chmod(p, 0o666)
                except OSError as exc:
                    print("  could not adjust %s: %s" % (p, exc))
        print("fixed ownership/mode under", root)

    ok = True

    # 1. the directory must accept a new file (SQLite writes journals next to the db)
    probe = os.path.join(root, ".write_probe")
    try:
        with open(probe, "w") as fh:
            fh.write("x")
        os.remove(probe)
        print("dir writable            PASS")
    except OSError as exc:
        print("dir writable            FAIL  %s" % exc)
        ok = False

    # 2. a real sqlite write, committed, then rolled back — this is the exact operation the
    #    recorder performs at the end of an episode
    db_path = os.path.join(root, "index.db")
    try:
        db = sqlite3.connect(db_path)
        db.execute("create table if not exists __probe(a)")
        db.execute("insert into __probe values (1)")
        db.commit()
        db.execute("delete from __probe")
        db.commit()
        db.close()
        print("sqlite write+commit     PASS  (%s)" % db_path)
    except Exception as exc:  # noqa: BLE001
        print("sqlite write+commit     FAIL  %s: %s" % (type(exc).__name__, exc))
        ok = False

    # 3. the episodes dir must accept a jsonl, since every episode writes one at the end
    try:
        p = os.path.join(root, "episodes", ".probe.jsonl")
        with open(p, "w") as fh:
            fh.write("{}\n")
        os.remove(p)
        print("episodes dir writable   PASS")
    except OSError as exc:
        print("episodes dir writable   FAIL  %s" % exc)
        ok = False

    print()
    print("RUN DIR %s" % ("OK" if ok else "NOT WRITABLE — the recorder will fail at commit"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
