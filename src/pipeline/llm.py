"""Minimal client for a local OpenAI-compatible chat endpoint (LM Studio).

One function, `chat`, talks to any model the endpoint exposes. The GEPA
reflection LM also needs a plain `(prompt) -> str` callable, which
`make_reflection_lm` adapts from this.
"""

from __future__ import annotations

import time
from typing import Any

import requests

from .config import Config

Messages = list[dict[str, Any]]


def chat(
    cfg: Config,
    model: str,
    messages: Messages,
    *,
    temperature: float = 0.0,
    max_retries: int = 3,
) -> str:
    """Send a chat completion request, returning the assistant text.

    Retries transient failures with linear backoff. Raises on persistent
    failure so the optimizer surfaces the problem instead of silently scoring 0.
    """
    url = f"{cfg.api_base}/chat/completions"
    headers = {"Authorization": f"Bearer {cfg.api_key}"}
    payload = {"model": model, "messages": messages, "temperature": temperature}

    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                url, json=payload, headers=headers, timeout=cfg.request_timeout
            )
            if resp.status_code >= 400:
                # Surface the endpoint's error body — LM Studio explains 400s here.
                raise RuntimeError(
                    f"HTTP {resp.status_code} for model={model}: {resp.text[:500]}"
                )
            return resp.json()["choices"][0]["message"]["content"] or ""
        except Exception as e:  # noqa: BLE001 - retry any transient error
            last_err = e
            time.sleep(1.0 * (attempt + 1))
    raise RuntimeError(f"chat() failed after {max_retries} attempts: {last_err}")


def make_reflection_lm(cfg: Config):
    """Build the `(prompt) -> str` callable GEPA uses as its reflection LM."""

    def _lm(prompt: str | Messages) -> str:
        messages: Messages = (
            [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
        )
        return chat(
            cfg, cfg.better_model, messages, temperature=cfg.better_temperature
        )

    return _lm
