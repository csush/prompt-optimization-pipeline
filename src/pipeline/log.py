"""Lightweight, timestamped logging for visibility into each optimization step.

Two surfaces:
- `event()` — timestamped stdout line for our own pipeline events.
- `GepaLogger` — a LoggerProtocol passed to GEPA so its engine's per-iteration
  events (candidate proposals, val scores, accept/reject) are printed too.
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
    with _lock:
        print(f"[{ts}] [{tag}] {msg}", flush=True)


def rollout() -> int:
    """Return the next global rollout index (1-based)."""
    return next(_counter)


class GepaLogger:
    """Adapts GEPA's `log(message)` calls to our timestamped stream."""

    def log(self, message: str) -> None:
        event(str(message).rstrip(), tag="GEPA")
