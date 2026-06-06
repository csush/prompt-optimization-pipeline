"""Evaluation: run the student prompt, then have the better model judge it.

The "better model as judge" produces both the optimization score and the
textual feedback (Actionable Side Information) that drives GEPA's reflection.
A deterministic exact-match correctness flag is computed alongside for honest
accuracy reporting, independent of judge noise.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .config import Config
from .data import Example
from .llm import chat

_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def extract_answer(text: str) -> str | None:
    """Pull the final numeric answer from a model response.

    Prefers the '#### <number>' convention; otherwise falls back to the last
    number mentioned. Returns a normalized string (commas stripped) or None.
    """
    if "####" in text:
        tail = text.split("####")[-1]
        m = _NUM.search(tail)
        if m:
            return m.group().replace(",", "").rstrip(".")
    nums = _NUM.findall(text)
    return nums[-1].replace(",", "").rstrip(".") if nums else None


def is_correct(pred: str | None, gold: str) -> bool:
    if pred is None:
        return False
    try:
        return abs(float(pred) - float(gold)) < 1e-6
    except ValueError:
        return pred.strip() == gold.strip()


def run_student(cfg: Config, prompt: str, question: str) -> str:
    return chat(
        cfg,
        cfg.student_model,
        [
            {"role": "system", "content": prompt},
            {"role": "user", "content": question},
        ],
        temperature=cfg.student_temperature,
    )


_JUDGE_INSTRUCTIONS = """You are a strict grader for grade-school math answers.

Question:
{question}

Reference solution (authoritative):
{solution}

Correct final answer: {gold}

The student model produced this response:
---
{student_out}
---
The student's extracted final answer was: {pred}

Grade the student. Respond with ONLY a JSON object:
{{"score": <float 0.0-1.0>, "feedback": "<concise critique>"}}

Scoring:
- 1.0 only if the final answer equals {gold}.
- 0.3-0.6 if the method is sound but the final answer is wrong or missing.
- 0.0 if the reasoning is wrong or absent.
In feedback, explain the key mistake and what the prompt should instruct the
student to do differently (e.g. show steps, state the answer as '#### <number>').
"""


def judge(cfg: Config, ex: Example, student_out: str, pred: str | None) -> tuple[float, str]:
    """Better model grades the student output. Returns (score, feedback)."""
    msg = _JUDGE_INSTRUCTIONS.format(
        question=ex.question,
        solution=ex.solution,
        gold=ex.gold,
        student_out=student_out[:4000],
        pred=pred,
    )
    raw = chat(
        cfg,
        cfg.better_model,
        [{"role": "user", "content": msg}],
        temperature=cfg.better_temperature,
    )
    score, feedback = _parse_judge(raw)
    # Anchor the judge to ground truth so a noisy judge can't reward wrong answers.
    if is_correct(pred, ex.gold):
        score = 1.0
    elif score >= 1.0:
        score = 0.6
    return score, feedback


def _parse_judge(raw: str) -> tuple[float, str]:
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group())
            return float(obj.get("score", 0.0)), str(obj.get("feedback", "")).strip()
        except (ValueError, TypeError):
            pass
    return 0.0, raw.strip()[:500]


def make_evaluator(cfg: Config):
    """Build the GEPA evaluator: (candidate_prompt, example) -> (score, side_info)."""
    from gepa.optimize_anything import log

    from .log import event, rollout

    def evaluate(candidate: str | dict[str, str], example: Example) -> tuple[float, dict[str, Any]]:
        prompt = candidate["prompt"] if isinstance(candidate, dict) else candidate
        n = rollout()
        event(f"rollout #{n}: student solving | Q: {example.question[:80]}...", tag="EVAL")
        student_out = run_student(cfg, prompt, example.question)
        pred = extract_answer(student_out)
        event(f"rollout #{n}: judging (gold={example.gold} pred={pred})", tag="EVAL")
        score, feedback = judge(cfg, example, student_out, pred)
        correct = is_correct(pred, example.gold)
        mark = "✓" if correct else "✗"
        event(
            f"rollout #{n}: {mark} correct={correct} score={score:.2f} | "
            f"feedback: {feedback[:140]}",
            tag="EVAL",
        )
        # Feedback is the optimization signal the reflection LM reads.
        log(f"Q: {example.question[:120]}")
        log(f"gold={example.gold} pred={pred} correct={correct} score={score:.2f}")
        log(f"judge: {feedback}")
        return score, {
            "question": example.question,
            "gold": example.gold,
            "pred": pred,
            "correct": correct,
            "student_out": student_out,
            "feedback": feedback,
        }

    return evaluate
