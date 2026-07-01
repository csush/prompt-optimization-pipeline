"""Run execution + live-event broadcast for the webapp.

Tracer-bullet scope: at most one run RUNNING at a time (enforced). The run
executes `pipeline.run(cfg)` on a background thread; `log.event()` events are
captured via the single global subscriber and fanned out to every SSE client
subscribed to that run. Iterations are best-effort parsed from GEPA log lines.

Per-run in-memory state (event log + iteration list + subscriber queues) lives
for the process lifetime; the durable record (prompts, accuracies, status) is
persisted to SQLite via `store`.
"""

from __future__ import annotations

import dataclasses
import json
import queue
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from ..config import Config
from ..log import set_subscriber, unset_subscriber, SubscriberToken
from ..optimize import run as run_pipeline, Report
from . import store

_GEPA_ITER = re.compile(
    r"(?:Best score on valset|Base program full valset score|"
    r"Selected program \d+ score|Found a better program on the valset with score)"
    r":?\s+([0-9.]+)"
)


class ConflictError(RuntimeError):
    """Raised when a new run is started while another is already active."""


@dataclass
class RunState:
    run_id: str
    events: list[dict[str, Any]] = field(default_factory=list)  # full event log
    iterations: list[dict[str, Any]] = field(default_factory=list)
    queues: list[queue.Queue] = field(default_factory=list)  # SSE subscriber queues
    lock: threading.Lock = field(default_factory=threading.Lock)


class RunManager:
    def __init__(self) -> None:
        self._states: dict[str, RunState] = {}
        self._lock = threading.Lock()
        self._active_run_id: str | None = None

    # ── public API ──────────────────────────────────────────
    def start_run(self, name: str, overrides: dict[str, Any]) -> store.Run:
        with self._lock:
            if self._active_run_id is not None:
                raise ConflictError(f"run {self._active_run_id} is already active")
            cfg = self._build_config(overrides)
            now = time.strftime("%Y-%m-%d %H:%M:%S")
            run_id = f"run-{uuid.uuid4().hex[:8]}"
            run = store.Run(
                id=run_id,
                name=name or f"run {run_id}",
                status="queued",
                config=self._config_dict(cfg),
                created_at=now,
                updated_at=now,
            )
            store.create_run(run)
            st = RunState(run_id=run_id)
            self._states[run_id] = st
            self._active_run_id = run_id
            threading.Thread(target=self._worker, args=(run_id, cfg, st), daemon=True).start()
            return store.get_run(run_id) or run

    def subscribe(self, run_id: str) -> tuple[queue.Queue, list[dict[str, Any]], list[dict[str, Any]]] | None:
        """Register an SSE queue for `run_id`. Returns (queue, replay_events, replay_iterations)."""
        st = self._states.get(run_id)
        if st is None:
            return None
        q: queue.Queue = queue.Queue()
        with st.lock:
            st.queues.append(q)
            replay_events = list(st.events)
            replay_iterations = list(st.iterations)
        return q, replay_events, replay_iterations

    def unsubscribe(self, run_id: str, q: queue.Queue) -> None:
        st = self._states.get(run_id)
        if st is None:
            return
        with st.lock:
            if q in st.queues:
                st.queues.remove(q)

    # ── worker ───────────────────────────────────────────────
    def _worker(self, run_id: str, cfg: Config, st: RunState) -> None:
        store.set_status(run_id, "running", updated_at=time.strftime("%Y-%m-%d %H:%M:%S"))
        self._emit(st, {"type": "status", "status": "running"})

        token = set_subscriber(lambda msg, tag: self._on_event(st, msg, tag))
        try:
            report = run_pipeline(cfg)
            self._on_report(run_id, report)
            done = {"type": "done", "status": "completed"}
        except Exception as e:  # noqa: BLE001
            store.fail_run(run_id, error=f"{type(e).__name__}: {e}",
                           updated_at=time.strftime("%Y-%m-%d %H:%M:%S"))
            done = {"type": "done", "status": "failed", "error": str(e)}
        finally:
            unset_subscriber(token)
            with self._lock:
                if self._active_run_id == run_id:
                    self._active_run_id = None
            self._emit(st, done)

    def _on_event(self, st: RunState, msg: str, tag: str) -> None:
        evt = {"type": "log", "tag": tag, "msg": msg, "ts": time.strftime("%H:%M:%S")}
        with st.lock:
            st.events.append(evt)
            m = _GEPA_ITER.search(msg) if tag == "GEPA" else None
            if m:
                score = float(m.group(1))
                # "better program" / "Found a better" => accepted; anything else
                # (Base program, Selected program, New subsample ... not better) => skip
                # but count as iteration only when score relates to a candidate outcome.
                accepted = "better" in msg.lower()
                st.iterations.append({"score": score, "accepted": accepted})
                store.append_iteration(st.run_id, score, accepted,
                                       updated_at=time.strftime("%Y-%m-%d %H:%M:%S"))
                self._emit_unlocked(st, {"type": "iteration", "score": score, "accepted": accepted})
            self._emit_unlocked(st, evt)

    def _on_report(self, run_id: str, report: Report) -> None:
        store.finish_run(
            run_id,
            report=dataclasses.asdict(report),
            updated_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        )

    def _emit(self, st: RunState, evt: dict[str, Any]) -> None:
        with st.lock:
            self._emit_unlocked(st, evt)

    def _emit_unlocked(self, st: RunState, evt: dict[str, Any]) -> None:
        dead: list[queue.Queue] = []
        for q in st.queues:
            try:
                q.put_nowait(evt)
            except queue.Full:
                dead.append(q)
        if dead:
            st.queues = [q for q in st.queues if q not in dead]

    # ── config helpers ──────────────────────────────────────
    @staticmethod
    def _build_config(overrides: dict[str, Any]) -> Config:
        cfg = Config()
        allowed = {f.name for f in dataclasses.fields(Config)} | {"train_size", "val_size", "test_size",
                                                                  "max_metric_calls", "max_workers",
                                                                  "reflection_minibatch_size", "seed",
                                                                  "student_model", "better_model"}
        clean: dict[str, Any] = {}
        for k, v in overrides.items():
            if k not in allowed:
                continue
            field_type = {f.name: f.type for f in dataclasses.fields(Config)}.get(k)
            if field_type is int and v is not None:
                try:
                    v = int(v)
                except (TypeError, ValueError):
                    continue
            clean[k] = v
        if clean:
            cfg = dataclasses.replace(cfg, **clean)
        return cfg

    @staticmethod
    def _config_dict(cfg: Config) -> dict[str, Any]:
        return dataclasses.asdict(cfg)


# Module-level singleton; app.py imports this.
manager = RunManager()