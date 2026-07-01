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

## Web app

A built-in web app for creating, running, and visualizing optimization runs
live in the browser. Configure a run in the UI; watch the per-iteration GEPA
log stream over SSE; inspect config, iteration chart, and seed/best prompt
diff for any run in the rail.

```
uv run uvicorn pipeline.web.app:app --port 8000
```

Then open <http://localhost:8000>. Add `--reload` during development.

- Runs persist across restarts in `data/runs.sqlite` (gitignored).
- One run executes at a time; starting another while a run is active returns
  409 and is surfaced in the form.
- Iterations are parsed from GEPA's accept/reject decision log lines and
  plotted live as they stream.
