"""Central config for the GEPA prompt-optimization POC.

Two independent endpoints:
- student: the cheap model being optimized (local LM Studio by default)
- better:  the strong model used as judge + reflection (OpenRouter by default)

All values overridable via environment variables (loaded from .env) so the
same code runs as a quick smoke test or a fuller run without edits.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


@dataclass(frozen=True)
class Config:
    # --- Student endpoint (local LM Studio) ---
    student_api_base: str = os.environ.get("STUDENT_API_BASE", "http://localhost:1234/v1")
    student_api_key: str = os.environ.get("STUDENT_API_KEY", "lm-studio")
    student_model: str = os.environ.get("STUDENT_MODEL", "google/gemma-4-12b-qat")

    # --- Better endpoint (OpenRouter): judge + reflection ---
    better_api_base: str = os.environ.get("BETTER_API_BASE", "https://openrouter.ai/api/v1")
    better_api_key: str = os.environ.get(
        "BETTER_API_KEY", os.environ.get("OPENROUTER_API_KEY", "")
    )
    better_model: str = os.environ.get(
        "BETTER_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free"
    )

    # Generation knobs.
    student_temperature: float = float(os.environ.get("STUDENT_TEMPERATURE", "0.0"))
    better_temperature: float = float(os.environ.get("BETTER_TEMPERATURE", "0.3"))
    # Bound the student's completion so prompt + output stays within its context
    # window (GSM8K chain-of-thought fits comfortably in ~1024 tokens).
    student_max_tokens: int = _int("STUDENT_MAX_TOKENS", 1024)
    request_timeout: int = _int("LLM_TIMEOUT", 300)

    # Dataset slice sizes (GSM8K). Kept tiny by default for local inference.
    train_size: int = _int("TRAIN_SIZE", 15)
    val_size: int = _int("VAL_SIZE", 8)
    test_size: int = _int("TEST_SIZE", 15)

    # GEPA budget.
    max_metric_calls: int = _int("MAX_METRIC_CALLS", 40)
    reflection_minibatch_size: int = _int("REFLECTION_MINIBATCH_SIZE", 3)
    # Student is local (serial); better is a remote API. Workers parallelize the
    # GEPA evaluation loop — safe to raise now the judge no longer thrashes a
    # single local endpoint.
    max_workers: int = _int("MAX_WORKERS", 2)

    seed: int = _int("SEED", 0)


SEED_PROMPT = "You are a helpful assistant. Solve the math problem."

OBJECTIVE = (
    "Optimize the system prompt so the student model solves grade-school math "
    "word problems correctly and ends its response with the final answer on a "
    "line of the form '#### <number>'."
)
