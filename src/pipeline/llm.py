"""Clients for OpenAI-compatible chat endpoints.

The student model and the better (judge/reflection) model live on different
endpoints, so the low-level `chat` takes explicit base/key/model. Convenience
wrappers `student_chat` / `better_chat` pull those from Config.
"""

from __future__ import annotations

import time
from typing import Any

import requests

from .config import Config

Messages = list[dict[str, Any]]


def chat(
    api_base: str,
    api_key: str,
    model: str,
    messages: Messages,
    *,
    temperature: float = 0.0,
    timeout: int = 300,
    max_retries: int = 3,
) -> str:
    """Send a chat completion request, returning the assistant text.

    Retries transient failures with linear backoff. Surfaces the endpoint's
    error body on 4xx/5xx so failures are diagnosable.
    """
    url = f"{api_base}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {"model": model, "messages": messages, "temperature": temperature}

    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
            if resp.status_code >= 400:
                raise RuntimeError(
                    f"HTTP {resp.status_code} for model={model}: {resp.text[:500]}"
                )
            return resp.json()["choices"][0]["message"]["content"] or ""
        except Exception as e:  # noqa: BLE001 - retry any transient error
            last_err = e
            time.sleep(1.0 * (attempt + 1))
    raise RuntimeError(f"chat() failed after {max_retries} attempts: {last_err}")


def student_chat(cfg: Config, messages: Messages) -> str:
    return chat(
        cfg.student_api_base,
        cfg.student_api_key,
        cfg.student_model,
        messages,
        temperature=cfg.student_temperature,
        timeout=cfg.request_timeout,
    )


def better_chat(cfg: Config, messages: Messages) -> str:
    return chat(
        cfg.better_api_base,
        cfg.better_api_key,
        cfg.better_model,
        messages,
        temperature=cfg.better_temperature,
        timeout=cfg.request_timeout,
    )


def make_reflection_lm(cfg: Config):
    """Build the `(prompt) -> str` callable GEPA uses as its reflection LM."""

    def _lm(prompt: str | Messages) -> str:
        messages: Messages = (
            [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
        )
        return better_chat(cfg, messages)

    return _lm
