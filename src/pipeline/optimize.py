"""Tie the pieces together: baseline -> GEPA optimize -> report."""

from __future__ import annotations

from dataclasses import dataclass

from gepa.optimize_anything import (
    EngineConfig,
    GEPAConfig,
    ReflectionConfig,
    TrackingConfig,
    optimize_anything,
)

from .config import OBJECTIVE, SEED_PROMPT, Config
from .data import Example, load_datasets
from .llm import make_reflection_lm
from .log import GepaLogger, event
from .metric import extract_answer, is_correct, make_evaluator, run_student


@dataclass
class Report:
    seed_prompt: str
    best_prompt: str
    baseline_accuracy: float
    optimized_accuracy: float
    n_test: int


def _accuracy(cfg: Config, prompt: str, test: list[Example], label: str) -> float:
    correct = 0
    for i, ex in enumerate(test, 1):
        out = run_student(cfg, prompt, ex.question)
        pred = extract_answer(out)
        ok = is_correct(pred, ex.gold)
        correct += ok
        event(
            f"{label} {i}/{len(test)}: {'✓' if ok else '✗'} "
            f"gold={ex.gold} pred={pred} (running {correct}/{i})",
            tag="TEST",
        )
    return correct / len(test) if test else 0.0


def _as_prompt(candidate: str | dict[str, str]) -> str:
    return candidate["prompt"] if isinstance(candidate, dict) else candidate


def run(cfg: Config) -> Report:
    event(f"loading GSM8K (train={cfg.train_size} val={cfg.val_size} test={cfg.test_size})")
    train, val, test = load_datasets(cfg)
    event(
        f"models: student={cfg.student_model} better/judge/reflection={cfg.better_model} "
        f"workers={cfg.max_workers}"
    )
    evaluate = make_evaluator(cfg)

    event(f"baseline: evaluating seed prompt on {len(test)} held-out items")
    baseline_acc = _accuracy(cfg, SEED_PROMPT, test, label="baseline")
    event(f"baseline accuracy: {baseline_acc:.1%}")

    event(f"optimizing (max_metric_calls={cfg.max_metric_calls})")
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
                display_progress_bar=False,
                track_best_outputs=True,
            ),
            reflection=ReflectionConfig(
                reflection_lm=make_reflection_lm(cfg),
                reflection_minibatch_size=cfg.reflection_minibatch_size,
            ),
            tracking=TrackingConfig(logger=GepaLogger()),
        ),
    )

    best_prompt = _as_prompt(result.best_candidate)
    event("optimization done; evaluating best prompt on held-out items")
    optimized_acc = _accuracy(cfg, best_prompt, test, label="optimized")
    event(f"optimized accuracy: {optimized_acc:.1%}")

    return Report(
        seed_prompt=SEED_PROMPT,
        best_prompt=best_prompt,
        baseline_accuracy=baseline_acc,
        optimized_accuracy=optimized_acc,
        n_test=len(test),
    )
