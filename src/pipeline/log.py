"""Lightweight, timestamped logging for visibility into each optimization step.

Three surfaces:
- `event()` — timestamped stdout line for our own pipeline events. Also fans out
  to the currently-registered subscriber (if any) so the webapp can stream
  per-iteration events live over SSE.
- `GepaLogger` — a LoggerProtocol passed to GEPA so its engine's per-iteration
  events (candidate proposals, val scores, accept/reject) are routed the same way.
- `rollout()` — a thread-safe counter so each student+judge evaluation is
  numbered as it happens.

Subscriber model: a single module-level subscriber at a time (tracer-bullet
scope — one active run's stream). `set_subscriber(fn)` returns a token; pass it
to `unset_subscriber` to deregister. GEPA's parallel evaluator worker threads
all hit the same global, so events fan out regardless of which thread emits.
"""

from __future__ import annotations

import itertools
import threading
import time
from typing import Callable

_lock = threading.Lock()
_counter = itertools.count(1)

_subscriber_lock = threading.Lock()
_subscriber: Callable[[str, str], None] | None = None
_subscriber_token = 0


class SubscriberToken:
    """Opaque handle returned by `set_subscriber` for later deregistration."""

    __slots__ = ("_id",)

    def __init__(self, id_: int) -> None:
        self._id = id_


def set_subscriber(fn: Callable[[str, str], None]) -> SubscriberToken:
    """Register `fn(msg, tag)` to receive every event. Replaces any prior one.

    Returns a token for `unset_subscriber`. Only one subscriber active at a time
    (tracer-bullet scope).
    """
    global _subscriber, _subscriber_token
    with _subscriber_lock:
        _subscriber_token += 1
        _subscriber = fn
        return SubscriberToken(_subscriber_token)


def unset_subscriber(token: SubscriberToken) -> None:
    """Deregister the subscriber if `token` is still the active one."""
    global _subscriber
    with _subscriber_lock:
        if token._id == _subscriber_token:
            _subscriber = None


def event(msg: str, tag: str = "PIPE") -> None:
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] [{tag}] {msg}"
    with _lock:
        print(line, flush=True)
    with _subscriber_lock:
        sub = _subscriber
    if sub is not None:
        try:
            sub(msg, tag)
        except Exception:
            # Subscriber must never break the pipeline.
            pass


def rollout() -> int:
    """Return the next global rollout index (1-based)."""
    return next(_counter)


class GepaLogger:
    """Adapts GEPA's `log(message)` calls to our timestamped stream."""

    def log(self, message: str) -> None:
        event(str(message).rstrip(), tag="GEPA")