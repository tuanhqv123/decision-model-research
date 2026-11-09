"""Trajectory recording: JSONL per episode (source of truth) + SQLite index (derived).

Design decisions that matter:

* JSONL is append-only and one file per episode, so a run is resumable and two
  episodes never contend for a lock.
* Screenshots are referenced by path, never inlined. The HTTP server's habit of
  returning ~7.8M ints per frame is exactly what this avoids.
* A partial episode (process killed mid-episode) is preserved as ``.abandoned`` and
  indexed as such. It is never silently overwritten, and it never counts as a result.
"""
import json
import os
import shutil
import sqlite3
import time
from typing import Any, Dict, List, Optional

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY, config_json TEXT, git_sha TEXT,
  started_at TEXT, finished_at TEXT, n_tasks INTEGER, n_episodes INTEGER
);
CREATE TABLE IF NOT EXISTS episodes (
  episode_id TEXT PRIMARY KEY, run_id TEXT, task_name TEXT, task_idx INTEGER, seed INTEGER,
  agent TEXT, checkpoint TEXT, status TEXT, success INTEGER, declared_done INTEGER,
  n_steps INTEGER, n_invalid_actions INTEGER, n_action_errors INTEGER,
  env_crash INTEGER, timeout INTEGER, model_p50 REAL, e2e_total_ms REAL, jsonl_path TEXT
);
CREATE TABLE IF NOT EXISTS steps (
  episode_id TEXT, i INTEGER, action TEXT, confidence REAL, gate_score REAL,
  invalid_action INTEGER, action_ok INTEGER, screen_changed INTEGER,
  model_ms REAL, env_ms REAL, PRIMARY KEY (episode_id, i)
);
"""


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class Recorder:
    def __init__(self, root: str, run_id: str, config: Optional[Dict[str, Any]] = None,
                 git_sha: str = ""):
        self.root = os.path.abspath(root)
        self.run_id = run_id
        self.ep_dir = os.path.join(self.root, "episodes")
        self.step_dir = os.path.join(self.root, "screens")
        os.makedirs(self.ep_dir, exist_ok=True)
        os.makedirs(self.step_dir, exist_ok=True)
        self.db_path = os.path.join(self.root, "index.db")
        self._db = sqlite3.connect(self.db_path)
        self._db.executescript(_SCHEMA)
        self._db.execute(
            "INSERT OR REPLACE INTO runs (run_id, config_json, git_sha, started_at) VALUES (?,?,?,?)",
            (run_id, json.dumps(config or {}), git_sha, _now()),
        )
        self._db.commit()
        self._current: Optional[str] = None
        self._steps: List[Dict[str, Any]] = []
        self._fp = None

    # -------------------------------------------------------------- paths
    def episode_path(self, episode_id: str) -> str:
        return os.path.join(self.ep_dir, "%s.jsonl" % episode_id)

    def step_image_path(self, episode_id: str, i: int) -> str:
        return os.path.join(self.step_dir, "%s_%04d.png" % (episode_id, i))

    def episode_header(self, episode_id: str) -> Dict[str, Any]:
        """The header line, read back from disk.

        Pairing is verified from this, not from the config that was supposed to produce it.
        """
        path = self.episode_path(episode_id)
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as fp:
            for line in fp:
                record = json.loads(line)
                if record.get("type") == "episode":
                    return record
        return {}

    # -------------------------------------------------------------- lifecycle
    def has_result(self, episode_id: str) -> bool:
        row = self._db.execute(
            "SELECT status FROM episodes WHERE episode_id=?", (episode_id,)
        ).fetchone()
        return bool(row) and row[0] == "ok"

    def _abandon_previous(self, episode_id: str) -> None:
        path = self.episode_path(episode_id)
        if not os.path.exists(path):
            return
        n = 0
        while True:
            dest = os.path.join(self.ep_dir, "%s.abandoned.%d.jsonl" % (episode_id, n))
            if not os.path.exists(dest):
                break
            n += 1
        shutil.move(path, dest)
        self._db.execute(
            "INSERT OR REPLACE INTO episodes (episode_id, run_id, status, jsonl_path) VALUES (?,?,?,?)",
            (episode_id, self.run_id, "abandoned", dest),
        )
        self._db.commit()

    def start_episode(self, header: Dict[str, Any]) -> str:
        episode_id = header["episode_id"]
        self._abandon_previous(episode_id)
        self._current = episode_id
        self._steps = []
        self._fp = open(self.episode_path(episode_id), "w", encoding="utf-8")
        record = {"type": "episode", "run_id": self.run_id, "started_at": _now()}
        record.update(header)
        self._fp.write(json.dumps(record, sort_keys=True) + "\n")
        self._fp.flush()
        return episode_id

    def add_step(self, step: Dict[str, Any]) -> None:
        if self._fp is None:
            raise RuntimeError("start_episode() first")
        record = dict(step)
        record["type"] = "step"
        if "episode_id" not in record:
            record["episode_id"] = self._current
        self._fp.write(json.dumps(record, sort_keys=True) + "\n")
        self._fp.flush()
        self._steps.append(record)

    def finish_episode(self, result: Dict[str, Any]) -> None:
        if self._fp is None:
            raise RuntimeError("start_episode() first")
        episode_id = self._current
        record = dict(result)
        record["type"] = "result"
        record["episode_id"] = episode_id
        record["ended_at"] = _now()
        self._fp.write(json.dumps(record, sort_keys=True) + "\n")
        self._fp.close()
        self._fp = None

        lat = record.get("latency") or {}
        self._db.execute(
            "INSERT OR REPLACE INTO episodes (episode_id, run_id, task_name, task_idx, seed, agent,"
            " checkpoint, status, success, declared_done, n_steps, n_invalid_actions,"
            " n_action_errors, env_crash, timeout, model_p50, e2e_total_ms, jsonl_path)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                episode_id,
                self.run_id,
                record.get("task_name"),
                record.get("task_idx"),
                record.get("seed"),
                record.get("agent"),
                record.get("checkpoint"),
                "ok",
                int(bool(record.get("success"))),
                int(bool(record.get("declared_done"))),
                record.get("n_steps"),
                record.get("n_invalid_actions"),
                record.get("n_action_errors"),
                int(bool(record.get("env_crash"))),
                int(bool(record.get("timeout"))),
                lat.get("model_p50"),
                lat.get("e2e_total_ms"),
                self.episode_path(episode_id),
            ),
        )
        self._db.execute("DELETE FROM steps WHERE episode_id=?", (episode_id,))
        for s in self._steps:
            self._db.execute(
                "INSERT OR REPLACE INTO steps (episode_id, i, action, confidence, gate_score,"
                " invalid_action, action_ok, screen_changed, model_ms, env_ms)"
                " VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    episode_id,
                    s.get("i"),
                    (s.get("action_raw") or {}).get("action"),
                    s.get("confidence"),
                    s.get("gate_score"),
                    int(bool(s.get("invalid_action"))),
                    int(bool(s.get("action_ok"))),
                    None if s.get("screen_changed") is None else int(bool(s.get("screen_changed"))),
                    s.get("model_ms"),
                    s.get("env_ms"),
                ),
            )
        self._db.commit()
        self._steps = []
        self._current = None

    # -------------------------------------------------------------- reporting
    def episodes(self) -> List[Dict[str, Any]]:
        """Header merged with result: one dict per finished episode.

        Fields set at task setup (coverage, step_budget, pairing key) live in the
        header; fields known at the end live in the result line. Consumers want both.
        """
        # Scope is the RUN DIRECTORY, not this process's run_id. Every resume is a new run_id
        # (the id embeds a timestamp), and `has_result` already skips episodes finished by an
        # earlier attempt — so counting only this run_id made a 116-task suite report
        # "Completeness: 0/116" after a restart even though 104 episodes were on disk
        # (2026-11-08). The directory is what a resumable run means.
        rows = self._db.execute(
            "SELECT episode_id, jsonl_path FROM episodes WHERE status='ok' ORDER BY episode_id"
        ).fetchall()
        out = []
        for episode_id, path in rows:
            if not path or not os.path.exists(path):
                continue
            if not os.path.abspath(path).startswith(self.ep_dir):
                continue
            header: Dict[str, Any] = {}
            result: Optional[Dict[str, Any]] = None
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    if rec.get("type") == "episode":
                        header = rec
                    elif rec.get("type") == "result":
                        result = rec
            if result is None:
                continue
            merged = dict(header)
            merged.update(result)
            merged.setdefault("episode_id", episode_id)
            out.append(merged)
        return out

    def steps(self) -> List[Dict[str, Any]]:
        # Same scoping rule as episodes(): the run directory, not this process's run_id.
        rows = self._db.execute(
            "SELECT episode_id, jsonl_path FROM episodes WHERE status='ok' ORDER BY episode_id"
        ).fetchall()
        out = []
        for episode_id, path in rows:
            if not path or not os.path.exists(path):
                continue
            if not os.path.abspath(path).startswith(self.ep_dir):
                continue
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    if rec.get("type") == "step":
                        rec.setdefault("episode_id", episode_id)
                        out.append(rec)
        return out

    def close(self) -> None:
        if self._fp is not None:
            self._fp.close()
            self._fp = None
        self._db.execute(
            "UPDATE runs SET finished_at=?, n_episodes="
            "(SELECT COUNT(*) FROM episodes WHERE run_id=? AND status='ok') WHERE run_id=?",
            (_now(), self.run_id, self.run_id),
        )
        self._db.commit()
        self._db.close()
