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
