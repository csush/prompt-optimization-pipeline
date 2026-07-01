"""Persist per-rollout traces to disk for offline review and error analysis.

Each run writes two files under ``data/runs/<run_id>/``:

- ``rollouts.jsonl`` — one record per student+judge rollout with the full
  trace (question, gold, pred, correct, student output, judge feedback,
  candidate prompt, phase). This is the raw material for ``error-analysis``
  and ``validate-evaluator``: reviewers judge real rollouts, not aggregate
  scores.
- ``meta.json`` — seed/best prompt, baseline/optimized accuracy, n_test,
  config snapshot, timestamps. Lets the review interface group and filter
  rollouts by run without re-deriving them.

The GEPA evaluator runs across worker threads, so appends are serialized with
a module-level lock. Records are written incrementally so a crashed run still
leaves a partial trace for diagnosis.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

_BASE = Path(__file__).resolve().parents[2] / "data" / "runs"
_LOCK = threading.Lock()


def _run_dir(run_id: str) -> Path:
    d = _BASE / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def append_rollout(run_id: str, record: dict[str, Any]) -> None:
    """Append one rollout record as a JSON line. Thread-safe.

    ``record`` is written verbatim; callers should include ``rollout_id``,
    ``phase``, ``prompt``, ``question``, ``gold``, ``pred``, ``correct``,
    ``student_out`` and ``feedback`` so the review interface has the full
    trace without re-deriving anything.
    """
    if "ts" not in record:
        record = {**record, "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
    line = json.dumps(record, ensure_ascii=False)
    with _LOCK:
        with (_run_dir(run_id) / "rollouts.jsonl").open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def write_meta(run_id: str, meta: dict[str, Any]) -> None:
    """Atomically write (overwrite) a run's metadata JSON."""
    payload = {**meta, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    tmp = _run_dir(run_id) / "meta.json.tmp"
    dest = _run_dir(run_id) / "meta.json"
    with _LOCK:
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(dest)