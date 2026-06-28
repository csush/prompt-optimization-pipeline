"""SQLite-backed run store. stdlib sqlite3; one connection per call (safe + simple).

Tracer-bullet scope: single-process, in-process DB. No migrations —
`init_db()` creates the table if absent.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

_lock = threading.Lock()
_DB_PATH = Path(__file__).resolve().parents[3] / "data" / "runs.sqlite"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  status TEXT NOT NULL,
  config_json TEXT NOT NULL,
  seed_prompt TEXT,
  best_prompt TEXT,
  baseline_accuracy REAL,
  optimized_accuracy REAL,
  n_test INTEGER,
  iterations_json TEXT NOT NULL DEFAULT '[]',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  error TEXT
)
"""


@dataclass
class Run:
    id: str
    name: str
    status: str  # queued | running | completed | failed
    config: dict[str, Any]
    seed_prompt: str | None = None
    best_prompt: str | None = None
    baseline_accuracy: float | None = None
    optimized_accuracy: float | None = None
    n_test: int | None = None
    iterations: list[dict[str, Any]] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""
    error: str | None = None


def _conn() -> sqlite3.Connection:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(_DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _lock, _conn() as c:
        c.executescript(_SCHEMA)


def _row_to_run(row: sqlite3.Row) -> Run:
    return Run(
        id=row["id"],
        name=row["name"],
        status=row["status"],
        config=json.loads(row["config_json"]),
        seed_prompt=row["seed_prompt"],
        best_prompt=row["best_prompt"],
        baseline_accuracy=row["baseline_accuracy"],
        optimized_accuracy=row["optimized_accuracy"],
        n_test=row["n_test"],
        iterations=json.loads(row["iterations_json"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        error=row["error"],
    )


def create_run(run: Run) -> Run:
    with _lock, _conn() as c:
        c.execute(
            """INSERT INTO runs
               (id, name, status, config_json, iterations_json, created_at, updated_at)
               VALUES (?, ?, ?, ?, '[]', ?, ?)""",
            (
                run.id, run.name, run.status,
                json.dumps(run.config), run.created_at, run.updated_at,
            ),
        )
    return run


def list_runs() -> list[Run]:
    with _lock, _conn() as c:
        rows = c.execute("SELECT * FROM runs ORDER BY created_at DESC").fetchall()
    return [_row_to_run(r) for r in rows]


def get_run(run_id: str) -> Run | None:
    with _lock, _conn() as c:
        row = c.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    return _row_to_run(row) if row else None


def _touch(updated_at: str, conn: sqlite3.Connection, run_id: str) -> None:
    conn.execute("UPDATE runs SET updated_at = ? WHERE id = ?", (updated_at, run_id))


def set_status(run_id: str, status: str, *, updated_at: str, **fields: Any) -> None:
    cols = ["status = ?", "updated_at = ?"]
    vals: list[Any] = [status, updated_at]
    for k, v in fields.items():
        cols.append(f"{k} = ?")
        vals.append(v)
    vals.append(run_id)
    with _lock, _conn() as c:
        c.execute(f"UPDATE runs SET {', '.join(cols)} WHERE id = ?", vals)


def append_iteration(run_id: str, score: float, accepted: bool, *, updated_at: str) -> None:
    with _lock, _conn() as c:
        row = c.execute("SELECT iterations_json FROM runs WHERE id = ?", (run_id,)).fetchone()
        iters = json.loads(row["iterations_json"]) if row else []
        iters.append({"score": score, "accepted": accepted})
        c.execute(
            "UPDATE runs SET iterations_json = ?, updated_at = ? WHERE id = ?",
            (json.dumps(iters), updated_at, run_id),
        )


def finish_run(run_id: str, *, report: dict[str, Any], updated_at: str) -> None:
    """Mark completed and persist best_prompt + accuracies + test count."""
    with _lock, _conn() as c:
        row = c.execute("SELECT iterations_json FROM runs WHERE id = ?", (run_id,)).fetchone()
        iters = json.loads(row["iterations_json"]) if row else []
        c.execute(
            """UPDATE runs SET
                 status = 'completed', updated_at = ?,
                 seed_prompt = ?, best_prompt = ?,
                 baseline_accuracy = ?, optimized_accuracy = ?, n_test = ?
               WHERE id = ?""",
            (
                updated_at,
                report.get("seed_prompt"), report.get("best_prompt"),
                report.get("baseline_accuracy"), report.get("optimized_accuracy"),
                report.get("n_test"),
                run_id,
            ),
        )


def fail_run(run_id: str, *, error: str, updated_at: str) -> None:
    with _lock, _conn() as c:
        c.execute(
            "UPDATE runs SET status = 'failed', error = ?, updated_at = ? WHERE id = ?",
            (error, updated_at, run_id),
        )


def to_dict(run: Run) -> dict[str, Any]:
    return asdict(run)