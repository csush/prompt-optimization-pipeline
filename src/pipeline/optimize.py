"""Tie the pieces together: baseline -> GEPA optimize -> report.

Each run gets a ``run_id`` so its rollouts and metadata can be located on disk
under ``data/runs/<run_id>/`` for offline review and error analysis — the
report's aggregate accuracies alone are not enough to diagnose *why* a prompt
did or didn't improve.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass

from gepa.optimize_anything import (
    EngineConfig,
    GEPAConfig,
    ReflectionConfig,
    TrackingConfig,
    optimize_anything,
)

from . import traces
from .config import OBJECTIVE, SEED_PROMPT, Config
from .data import Example, load_datasets
from .llm import make_reflection_lm
from .log import GepaLogger, event
from .metric import extract_answer, is_correct, make_evaluator, run_student


@dataclass
class Report:
    run_id: str
    seed_prompt: str
    best_prompt: str
    baseline_accuracy: float
    optimized_accuracy: float
    n_test: int


def _accuracy(
    cfg: Config,
    run_id: str,
    prompt: str,
    test: list[Example],
    label: str,
) -> float:
    """Run the student on the held-out test set and persist each rollout.

    Test rollouts are saved with the given ``label`` as phase so reviewers can
    tell baseline and final-optimized traces apart. No judge is invoked here:
    fairness on the held-out set is measured purely by the deterministic
    ``is_correct`` check.
    """
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
        traces.append_rollout(run_id, {
            "rollout_id": None,
            "phase": label,
            "prompt": prompt,
            "question": ex.question,
            "gold": ex.gold,
            "pred": pred,
            "correct": ok,
            "student_out": out,
            "feedback": "",
        })
    return correct / len(test) if test else 0.0


def _as_prompt(candidate: str | dict[str, str]) -> str:
    return candidate["prompt"] if isinstance(candidate, dict) else candidate


def _new_run_id() -> str:
    return f"run-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:4]}"


def run(cfg: Config) -> Report:
    run_id = _new_run_id()
    event(f"run_id={run_id}")
    traces.write_meta(run_id, {
        "run_id": run_id,
        "seed_prompt": SEED_PROMPT,
        "status": "running",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {
            "student_model": cfg.student_model,
            "better_model": cfg.better_model,
            "train_size": cfg.train_size,
            "val_size": cfg.val_size,
            "test_size": cfg.test_size,
            "max_metric_calls": cfg.max_metric_calls,
            "seed": cfg.seed,
        },
    })

    event(f"loading GSM8K (train={cfg.train_size} val={cfg.val_size} test={cfg.test_size})")
    train, val, test = load_datasets(cfg)
    event(
        f"models: student={cfg.student_model} better/judge/reflection={cfg.better_model} "
        f"workers={cfg.max_workers}"
    )
    evaluate = make_evaluator(cfg, run_id)

    event(f"baseline: evaluating seed prompt on {len(test)} held-out items")
    baseline_acc = _accuracy(cfg, run_id, SEED_PROMPT, test, label="baseline")
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
    optimized_acc = _accuracy(cfg, run_id, best_prompt, test, label="optimized")
    event(f"optimized accuracy: {optimized_acc:.1%}")

    traces.write_meta(run_id, {
        "run_id": run_id,
        "seed_prompt": SEED_PROMPT,
        "status": "completed",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "baseline_accuracy": baseline_acc,
        "optimized_accuracy": optimized_acc,
        "n_test": len(test),
        "best_prompt": best_prompt,
        "config": {
            "student_model": cfg.student_model,
            "better_model": cfg.better_model,
            "train_size": cfg.train_size,
            "val_size": cfg.val_size,
            "test_size": cfg.test_size,
            "max_metric_calls": cfg.max_metric_calls,
            "seed": cfg.seed,
        },
    })

    return Report(
        run_id=run_id,
        seed_prompt=SEED_PROMPT,
        best_prompt=best_prompt,
        baseline_accuracy=baseline_acc,
        optimized_accuracy=optimized_acc,
        n_test=len(test),
    )