"""SQLite log of runs and the errors seen during them. See CLAUDE.md."""

import os
import sqlite3
import time

import config

SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA user_version = 2;

CREATE TABLE IF NOT EXISTS run (
  id          INTEGER PRIMARY KEY,
  code        TEXT    NOT NULL,
  sku         TEXT    NOT NULL,
  day         TEXT    NOT NULL,
  outcome     TEXT    NOT NULL,
  started_at  REAL    NOT NULL,
  ended_at    REAL    NOT NULL,
  duration_ms INTEGER NOT NULL,
  step_count  INTEGER NOT NULL,
  reached_idx INTEGER NOT NULL,
  event_count INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS run_step (
  run_id   INTEGER NOT NULL REFERENCES run(id) ON DELETE CASCADE,
  idx      INTEGER NOT NULL,
  title    TEXT    NOT NULL,
  done_at  REAL    NOT NULL,
  split_ms INTEGER NOT NULL,
  clip     TEXT,
  PRIMARY KEY (run_id, idx)
);

CREATE TABLE IF NOT EXISTS run_event (
  id          INTEGER PRIMARY KEY,
  run_id      INTEGER NOT NULL REFERENCES run(id) ON DELETE CASCADE,
  step_idx    INTEGER NOT NULL,
  title       TEXT    NOT NULL,
  kind        TEXT    NOT NULL,
  expected    TEXT,
  got         TEXT,
  at          REAL    NOT NULL,
  cleared_at  REAL,
  duration_ms INTEGER
);

CREATE INDEX IF NOT EXISTS run_day_idx    ON run(day, started_at);
CREATE INDEX IF NOT EXISTS run_sku_idx    ON run(sku, started_at);
CREATE INDEX IF NOT EXISTS event_run_idx  ON run_event(run_id);
CREATE INDEX IF NOT EXISTS event_kind_idx ON run_event(kind, step_idx);
"""

_conn = None


class EventLog:
    def __init__(self):
        self._open = {}
        self._done = []

    def observe(self, state, now=None):
        now = time.time() if now is None else now
        live = {}
        err = state.get("error") or {}
        if err.get("active"):
            live["wrong_item"] = {"expected": err.get("expected"), "got": err.get("got")}
        if state.get("misplaced"):
            live["misplaced"] = {"expected": state["misplaced"], "got": None}
        idx = state.get("current", 0)
        steps = state.get("steps") or []
        title = steps[idx]["title"] if 0 <= idx < len(steps) else ""

        for kind in list(self._open):
            if self._open[kind]["what"] != live.get(kind):
                self._close(kind, now)
        for kind, what in live.items():
            if kind not in self._open:
                self._open[kind] = {"what": what, "at": now,
                                    "step_idx": idx, "title": title}

    def _close(self, kind, now):
        ev = self._open.pop(kind)
        self._done.append({
            "kind": kind, "step_idx": ev["step_idx"], "title": ev["title"],
            "expected": ev["what"]["expected"], "got": ev["what"]["got"],
            "at": ev["at"], "cleared_at": now,
            "duration_ms": round((now - ev["at"]) * 1000)})

    def flush(self, now=None):
        now = time.time() if now is None else now
        out = list(self._done)
        for kind, ev in self._open.items():
            out.append({
                "kind": kind, "step_idx": ev["step_idx"], "title": ev["title"],
                "expected": ev["what"]["expected"], "got": ev["what"]["got"],
                "at": ev["at"], "cleared_at": None, "duration_ms": None})
        out.sort(key=lambda e: e["at"])
        return out

    def clear(self):
        self._open.clear()
        self._done.clear()


def init(log=print):
    global _conn
    if not config.RUNS_ENABLE:
        return
    try:
        folder = os.path.dirname(config.RUNS_DB)
        if folder:
            os.makedirs(folder, exist_ok=True)
        conn = sqlite3.connect(config.RUNS_DB, check_same_thread=False)
        conn.executescript(SCHEMA)
        _conn = conn
        log(f"[runs] logging to {config.RUNS_DB}")
    except Exception as e:
        log(f"[runs] disabled, init failed: {e!r}")


def save(run, proofs, events, outcome, log=print):
    if _conn is None or not run:
        return
    started, steps = run["started_at"], run["steps"]
    events = events or []
    ended = steps[-1]["done_at"] if outcome == "complete" and steps else time.time()
    try:
        with _conn:
            cur = _conn.execute(
                "INSERT INTO run (code, sku, day, outcome, started_at, ended_at,"
                " duration_ms, step_count, reached_idx, event_count)"
                " VALUES (?,?,?,?,?,?,?,?,?,?)",
                (run["code"] or "", run["sku"] or "",
                 time.strftime("%Y-%m-%d", time.localtime(started)), outcome,
                 started, ended, round((ended - started) * 1000),
                 run["step_count"], run["reached_idx"], len(events)))
            rid = cur.lastrowid
            rows, prev = [], started
            for s in steps:
                rows.append((rid, s["idx"], s["title"], s["done_at"],
                             round((s["done_at"] - prev) * 1000),
                             (proofs or {}).get(s["idx"])))
                prev = s["done_at"]
            _conn.executemany(
                "INSERT INTO run_step (run_id, idx, title, done_at,"
                " split_ms, clip) VALUES (?,?,?,?,?,?)", rows)
            _conn.executemany(
                "INSERT INTO run_event (run_id, step_idx, title, kind, expected,"
                " got, at, cleared_at, duration_ms) VALUES (?,?,?,?,?,?,?,?,?)",
                [(rid, e["step_idx"], e["title"], e["kind"], e["expected"],
                  e["got"], e["at"], e["cleared_at"], e["duration_ms"])
                 for e in events])
        log(f"[runs] saved #{rid} {run['sku']} {outcome} "
            f"{run['reached_idx']}/{run['step_count']} steps, "
            f"{len(events)} events, {ended - started:.1f}s")
    except Exception as e:
        log(f"[runs] save failed: {e!r}")


def close():
    global _conn
    if _conn is not None:
        try:
            _conn.close()
        except Exception:
            pass
        _conn = None
