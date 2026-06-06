"""Tie the pieces together: baseline -> GEPA optimize -> report."""

from __future__ import annotations

from dataclasses import dataclass

from gepa.optimize_anything import (
    EngineConfig,
    GEPAConfig,
    ReflectionConfig,
    optimize_anything,
)

from .config import OBJECTIVE, SEED_PROMPT, Config
from .data import Example, load_datasets
from .llm import make_reflection_lm
from .metric import extract_answer, is_correct, make_evaluator, run_student


@dataclass
class Report:
    seed_prompt: str
    best_prompt: str
    baseline_accuracy: float
    optimized_accuracy: float
    n_test: int


def _accuracy(cfg: Config, prompt: str, test: list[Example]) -> float:
    correct = 0
    for ex in test:
        out = run_student(cfg, prompt, ex.question)
        if is_correct(extract_answer(out), ex.gold):
            correct += 1
    return correct / len(test) if test else 0.0


def _as_prompt(candidate: str | dict[str, str]) -> str:
    return candidate["prompt"] if isinstance(candidate, dict) else candidate


def run(cfg: Config) -> Report:
    train, val, test = load_datasets(cfg)
    evaluate = make_evaluator(cfg)

    print(f"Baseline: evaluating seed prompt on {len(test)} held-out test items...")
    baseline_acc = _accuracy(cfg, SEED_PROMPT, test)
    print(f"Baseline accuracy: {baseline_acc:.1%}")

    print(f"Optimizing (max_metric_calls={cfg.max_metric_calls})...")
    result = optimize_anything(
        seed_candidate=SEED_PROMPT,
        evaluator=evaluate,
        dataset=train,
        valset=val,
        objective=OBJECTIVE,
        config=GEPAConfig(
            engine=EngineConfig(
                max_metric_calls=cfg.max_metric_calls,
                max_workers=cfg.max_workers,
                parallel=cfg.max_workers > 1,
                seed=cfg.seed,
                display_progress_bar=True,
                track_best_outputs=True,
            ),
            reflection=ReflectionConfig(
                reflection_lm=make_reflection_lm(cfg),
                reflection_minibatch_size=cfg.reflection_minibatch_size,
            ),
        ),
    )

    best_prompt = _as_prompt(result.best_candidate)
    print("Optimized: evaluating best prompt on held-out test items...")
    optimized_acc = _accuracy(cfg, best_prompt, test)
    print(f"Optimized accuracy: {optimized_acc:.1%}")

    return Report(
        seed_prompt=SEED_PROMPT,
        best_prompt=best_prompt,
        baseline_accuracy=baseline_acc,
        optimized_accuracy=optimized_acc,
        n_test=len(test),
    )
