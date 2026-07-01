"""Evaluation: run the student prompt, then capture a deterministic
correctness signal and judge feedback separately.

The optimization score is the binary ``is_correct`` result — an objective,
code-based check against the gold numeric answer. There is no judge-derived
score in the objective: Likert partial credit and judge noise are not allowed
to steer GEPA toward wrong answers.

The better model still participates, but only as a reflection aid: it emits a
concise textual critique of *what went wrong and what the prompt should do
differently* (Actionable Side Information for GEPA's reflection LM). Scoring is
code; interpretation is the judge.

Every rollout is appended to ``data/runs/<run_id>/rollouts.jsonl`` so the
trace (question, gold, pred, student output, feedback, candidate prompt) is
available for error analysis and judge validation after the run.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .config import Config
from .data import Example
from .llm import ContextExceededError, better_chat, student_chat

_CONTEXT_FEEDBACK = (
    "The system prompt was too long: combined with the question it exceeded the "
    "student model's context window, so it could not answer. Make the prompt "
    "substantially shorter and more concise while keeping the key instructions."
)

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
    return student_chat(
        cfg,
        [
            {"role": "system", "content": prompt},
            {"role": "user", "content": question},
        ],
    )


_JUDGE_INSTRUCTIONS = """You are reviewing a grade-school math answer to help
improve the system prompt that produced it. You do NOT assign a score;
scoring is handled deterministically by exact numeric match against the gold
answer. Your job is to describe the mistake concisely so the reflection step
can change the prompt.

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

Respond with ONLY a JSON object:
{{"feedback": "<one or two sentences: the key mistake and what the system prompt should instruct the student to do differently>"}}

If the final answer is correct, say so briefly and suggest no change. If it is
wrong or missing, name the concrete error (wrong operation, arithmetic slip,
misread question, answer not on a '#### <number>' line) and what the prompt
should tell the student to do differently.
"""


def judge(cfg: Config, ex: Example, student_out: str, pred: str | None) -> str:
    """Better model critiques the student output. Returns feedback text only.

    No score is returned: the optimization objective comes from the
    deterministic ``is_correct`` check. The judge contributes only the
    textual side information GEPA's reflection LM reads.
    """
    msg = _JUDGE_INSTRUCTIONS.format(
        question=ex.question,
        solution=ex.solution,
        gold=ex.gold,
        student_out=student_out[:4000],
        pred=pred,
    )
    raw = better_chat(cfg, [{"role": "user", "content": msg}])
    return _parse_judge(raw)


def _parse_judge(raw: str) -> str:
    """Extract the feedback string from the judge's JSON response.

    Falls back to the trimmed raw text (capped) if the model doesn't emit
    valid JSON, so a malformed judge response still produces reflection
    signal rather than crashing the rollout.
    """
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group())
            return str(obj.get("feedback", "")).strip()
        except (ValueError, TypeError):
            pass
    return raw.strip()[:500]


def make_evaluator(cfg: Config, run_id: str):
    """Build the GEPA evaluator: (candidate_prompt, example) -> (score, side_info).

    The returned score is binary — ``1.0`` if the extracted answer matches the
    gold exactly, else ``0.0``. This keeps GEPA's optimization target honest:
    judge partial credit cannot reward wrong answers. The judge still runs and
    its feedback flows into ``side_info`` for GEPA's reflection LM and into the
    persisted trace for review.
    """
    from gepa.optimize_anything import log

    from .log import event, rollout
    from . import traces

    def _persist(rollout_id: int, phase: str, prompt: str, example: Example,
                 pred: str | None, correct: bool, student_out: str, feedback: str) -> None:
        traces.append_rollout(run_id, {
            "run_id": run_id,
            "rollout_id": rollout_id,
            "phase": phase,
            "prompt": prompt,
            "question": example.question,
            "gold": example.gold,
            "pred": pred,
            "correct": correct,
            "student_out": student_out,
            "feedback": feedback,
        })

    def evaluate(candidate: str | dict[str, str], example: Example) -> tuple[float, dict[str, Any]]:
        prompt = candidate["prompt"] if isinstance(candidate, dict) else candidate
        n = rollout()
        event(f"rollout #{n}: student solving | Q: {example.question[:80]}...", tag="EVAL")
        try:
            student_out = run_student(cfg, prompt, example.question)
        except ContextExceededError:
            # Turn the overflow into optimization signal: score 0 and tell the
            # reflection LM to shorten the prompt, instead of aborting the run.
            event(f"rollout #{n}: ✗ context exceeded — prompt too long", tag="EVAL")
            log(f"context exceeded: {_CONTEXT_FEEDBACK}")
            _persist(n, "optimize", prompt, example, None, False, "", _CONTEXT_FEEDBACK)
            return 0.0, {
                "question": example.question,
                "gold": example.gold,
                "pred": None,
                "correct": False,
                "student_out": "",
                "feedback": _CONTEXT_FEEDBACK,
            }
        pred = extract_answer(student_out)
        event(f"rollout #{n}: judging (gold={example.gold} pred={pred})", tag="EVAL")
        feedback = judge(cfg, example, student_out, pred)
        correct = is_correct(pred, example.gold)
        mark = "✓" if correct else "✗"
        event(
            f"rollout #{n}: {mark} correct={correct} | feedback: {feedback[:140]}",
            tag="EVAL",
        )
        # Feedback is the optimization signal the reflection LM reads.
        log(f"Q: {example.question[:120]}")
        log(f"gold={example.gold} pred={pred} correct={correct}")
        log(f"judge: {feedback}")
        _persist(n, "optimize", prompt, example, pred, correct, student_out, feedback)
        # Binary objective: code-check only, judge noise cannot steer GEPA.
        return 1.0 if correct else 0.0, {
            "question": example.question,
            "gold": example.gold,
            "pred": pred,
            "correct": correct,
            "student_out": student_out,
            "feedback": feedback,
        }

    return evaluate
