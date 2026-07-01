"""FastAPI app: run the pipeline + serve the Explorer UI + SSE event stream."""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import store
from .run_manager import ConflictError, manager

_STATIC = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(_: FastAPI):
    store.init_db()
    yield


app = FastAPI(title="Optimization Runs", version="0.1.0", lifespan=lifespan)


class CreateRunRequest(BaseModel):
    name: str | None = None
    overrides: dict[str, Any] = {}


@app.get("/api/runs")
def list_runs() -> list[dict[str, Any]]:
    return [store.to_dict(r) for r in store.list_runs()]


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    r = store.get_run(run_id)
    if r is None:
        raise HTTPException(status_code=404, detail="run not found")
    return store.to_dict(r)


@app.post("/api/runs")
def create_run(req: CreateRunRequest) -> dict[str, Any]:
    try:
        run = manager.start_run(req.name or "", req.overrides)
    except ConflictError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return store.to_dict(run)


@app.get("/api/runs/{run_id}/events")
async def run_events(run_id: str):
    """SSE stream of a run's events. Replays buffered log + iterations, then live."""
    r = store.get_run(run_id)
    if r is None:
        raise HTTPException(status_code=404, detail="run not found")

    async def stream():
        sub = manager.subscribe(run_id)
        if sub is None:
            # Run not in memory — it's already completed/failed. Emit snapshot + done.
            yield _sse({"type": "status", "status": r.status})
            for it in r.iterations:
                yield _sse({"type": "iteration", "score": it["score"], "accepted": it["accepted"]})
            yield _sse({"type": "done", "status": r.status})
            return
        q, replay_events, replay_iterations = sub
        try:
            for it in replay_iterations:
                yield _sse({"type": "iteration", "score": it["score"], "accepted": it["accepted"]})
            for evt in replay_events:
                yield _sse(evt)
            while True:
                try:
                    evt = await asyncio.get_event_loop().run_in_executor(None, q.get, True, 30)
                except Exception:
                    # queue.Empty on timeout — send keepalive comment.
                    yield ": keepalive\n\n"
                    continue
                if evt.get("type") == "__unsubscribe__":
                    break
                yield _sse(evt)
                if evt.get("type") == "done":
                    break
        finally:
            manager.unsubscribe(run_id, q)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload)}\n\n"


app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(_STATIC / "index.html")