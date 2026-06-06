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


class ContextExceededError(RuntimeError):
    """Raised when the prompt overflows the model's context window.

    Deterministic, so it is not retried; callers can turn it into optimization
    feedback (e.g. "make the prompt shorter").
    """


def _is_context_exceeded(body: str) -> bool:
    b = body.lower()
    return "context" in b and ("exceed" in b or "too long" in b or "maximum" in b)


def chat(
    api_base: str,
    api_key: str,
    model: str,
    messages: Messages,
    *,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    timeout: int = 300,
    max_retries: int = 3,
) -> str:
    """Send a chat completion request, returning the assistant text.

    Retries transient failures with linear backoff. Surfaces the endpoint's
    error body on 4xx/5xx so failures are diagnosable. A context-overflow 400
    is raised immediately as ContextExceededError (retrying would not help).
    """
    url = f"{api_base}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
            if resp.status_code >= 400:
                if resp.status_code == 400 and _is_context_exceeded(resp.text):
                    raise ContextExceededError(
                        f"context exceeded for model={model}: {resp.text[:300]}"
                    )
                raise RuntimeError(
                    f"HTTP {resp.status_code} for model={model}: {resp.text[:500]}"
                )
            return resp.json()["choices"][0]["message"]["content"] or ""
        except ContextExceededError:
            raise  # deterministic — do not retry
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
        max_tokens=cfg.student_max_tokens,
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
