"""Lightweight, timestamped logging for visibility into each optimization step.

Two surfaces:
- `event()` — timestamped stdout line for our own pipeline events (load,
  rollout, accept/reject, test results). Flushes immediately.
- `GepaLogger` — a LoggerProtocol passed to GEPA so its engine's per-iteration
  events (candidate proposals, val scores, accept/reject) are routed the same
  way.
- `rollout()` — a thread-safe counter so each student+judge evaluation is
  numbered as it happens.
"""

from __future__ import annotations

import itertools
import threading
import time

_lock = threading.Lock()
_counter = itertools.count(1)


def event(msg: str, tag: str = "PIPE") -> None:
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] [{tag}] {msg}"
    with _lock:
        print(line, flush=True)


def rollout() -> int:
    """Return the next global rollout index (1-based)."""
    return next(_counter)


class GepaLogger:
    """Adapts GEPA's `log(message)` calls to our timestamped stream."""

    def log(self, message: str) -> None:
        event(str(message).rstrip(), tag="GEPA")