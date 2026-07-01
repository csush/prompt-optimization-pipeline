# prompt-optimization-pipeline

Reflective prompt optimization with [GEPA](https://github.com/gepa-ai/gepa)'s
`optimize_anything`. A cheap **student** model's system prompt is optimized by a
stronger **better** model that acts as both judge and reflection LM. Demo task:
GSM8K math.

## Setup

```
uv sync
echo "OPENROUTER_API_KEY=sk-or-..." > .env
```

Defaults: student `google/gemma-4-12b-qat` on a local endpoint
(`http://localhost:1234/v1`), better `nvidia/nemotron-3-ultra-550b-a55b:free`
on OpenRouter. Override via env vars (see `src/pipeline/config.py`).

## Run

```
uv run run.py --smoke                      # tiny sanity run
uv run run.py --save optimized_prompt.txt  # full run
```

## Reviewing traces

Every run persists its rollouts to `data/runs/<run_id>/rollouts.jsonl` — one
JSON record per student+judge rollout with the question, gold answer, extracted
prediction, correctness flag, full student output, judge feedback, and the
candidate system prompt. Run metadata (seed/best prompt, accuracies, config) is
written alongside to `meta.json`. These traces are the raw material for error
analysis and judge validation: aggregate accuracy alone cannot diagnose *why* a
prompt did or didn't improve.

Load them in the trace-review interface:

```
open review/index.html
```

Pick a `rollouts.jsonl` (or a `meta.json`) file. The interface renders one
trace at a time with Pass/Fail/Defer buttons, a notes field, next/previous
navigation, and keyboard shortcuts. Labels are auto-saved to your browser and
can be exported as a `labels.jsonl` for `validate-evaluator`.
