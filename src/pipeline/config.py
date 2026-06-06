"""Central config for the GEPA prompt-optimization POC.

All values overridable via environment variables so the same code runs as a
quick smoke test or a fuller optimization run without edits.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


@dataclass(frozen=True)
class Config:
    # Local OpenAI-compatible endpoint (LM Studio).
    api_base: str = os.environ.get("LLM_API_BASE", "http://localhost:1234/v1")
    api_key: str = os.environ.get("LLM_API_KEY", "lm-studio")  # ignored by LM Studio

    # Models exposed by the endpoint.
    student_model: str = os.environ.get("STUDENT_MODEL", "google/gemma-4-12b-qat")
    better_model: str = os.environ.get("BETTER_MODEL", "qwen/qwen3.6-27b")

    # Generation knobs.
    student_temperature: float = float(os.environ.get("STUDENT_TEMPERATURE", "0.0"))
    better_temperature: float = float(os.environ.get("BETTER_TEMPERATURE", "0.3"))
    request_timeout: int = _int("LLM_TIMEOUT", 300)

    # Dataset slice sizes (GSM8K). Kept tiny by default for local inference.
    train_size: int = _int("TRAIN_SIZE", 15)
    val_size: int = _int("VAL_SIZE", 8)
    test_size: int = _int("TEST_SIZE", 15)

    # GEPA budget.
    max_metric_calls: int = _int("MAX_METRIC_CALLS", 40)
    reflection_minibatch_size: int = _int("REFLECTION_MINIBATCH_SIZE", 3)
    # Default 1: a single local endpoint serializes anyway, and concurrent
    # requests trigger model-swap thrash / 400s in LM Studio.
    max_workers: int = _int("MAX_WORKERS", 1)

    seed: int = _int("SEED", 0)


SEED_PROMPT = "You are a helpful assistant. Solve the math problem."

OBJECTIVE = (
    "Optimize the system prompt so the student model solves grade-school math "
    "word problems correctly and ends its response with the final answer on a "
    "line of the form '#### <number>'."
)
