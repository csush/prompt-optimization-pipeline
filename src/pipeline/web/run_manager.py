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
import typing
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

# Per-iteration decision signals from GEPA's log. Matching one of these means
# a candidate was either accepted (better program found) or rejected (subsample
# not better). Other "score" lines (base program, selected program, best-on-valset)
# are aggregates, not decisions — excluded so the chart shows one point per iteration.
_GEPA_ITER = re.compile(
    r"(?:Found a better program on the valset with score"
    r"|New subsample score)\D*([0-9.]+)"
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
        # Bounded so a stalled SSE client gets dropped (via the Full path in
        # _emit_unlocked) rather than accumulating every event forever.
        q: queue.Queue = queue.Queue(maxsize=1000)
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
        # Unblock any orphaned executor thread still blocked in q.get on this
        # queue (e.g. SSE client disconnected mid-30s wait) so it returns
        # promptly instead of pinning a pool thread.
        try:
            q.put_nowait({"type": "__unsubscribe__"})
        except queue.Full:
            pass

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
                # ACCEPT = "Found a better program"; REJECT = "New subsample score
                # ... is not better". Both contain "better" -> match by prefix.
                accepted = msg.lower().startswith("found a better program")
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
        # `from __future__ import annotations` makes dataclass field types strings,
        # so resolve real hints to detect int fields and coerce DOM-string overrides.
        hints = typing.get_type_hints(Config)
        for k, v in overrides.items():
            if k not in allowed:
                continue
            if hints.get(k) is int and v is not None and v != "":
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