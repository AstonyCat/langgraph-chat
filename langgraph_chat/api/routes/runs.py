"""Runs endpoints: execute graphs, stream results, manage run lifecycle."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from langgraph_chat.api.models import Run, RunCreate
from langgraph_chat.api.storage import storage

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Runs"])


async def _execute_run(
    thread_id: str, run: Run, input_data: Any, assistant_id: str
) -> dict[str, Any]:
    """Execute the graph and return the result."""
    from langgraph_chat.api.graph_registry import execute_graph

    storage.update_run_status(run.run_id, "running")
    storage.set_thread_status(thread_id, "busy")
    try:
        result = await asyncio.to_thread(
            execute_graph, assistant_id, thread_id, input_data
        )
        storage.update_run_status(run.run_id, "success")
        storage.set_thread_status(thread_id, "idle")
        return result
    except Exception as e:
        logger.exception("Run %s failed", run.run_id)
        storage.update_run_status(run.run_id, "error")
        storage.set_thread_status(thread_id, "error")
        raise HTTPException(status_code=500, detail=str(e)) from e


# ── Thread-scoped runs ──────────────────────────────────────


@router.post("/threads/{thread_id}/runs/wait")
async def create_run_wait(thread_id: str, body: RunCreate) -> dict[str, Any]:
    """Create a run and wait for the output."""
    try:
        storage.get_thread(thread_id)
    except KeyError:
        storage.create_thread(thread_id=thread_id)

    run = storage.create_run(
        thread_id=thread_id,
        assistant_id=body.assistant_id,
        metadata=body.metadata,
        kwargs={"input": body.input},
        multitask_strategy=body.multitask_strategy,
    )
    result = await _execute_run(thread_id, run, body.input, body.assistant_id)
    return result


@router.post("/threads/{thread_id}/runs/stream")
async def create_run_stream(thread_id: str, body: RunCreate) -> StreamingResponse:
    """Create a run and stream the output as SSE."""
    try:
        storage.get_thread(thread_id)
    except KeyError:
        storage.create_thread(thread_id=thread_id)

    run = storage.create_run(
        thread_id=thread_id,
        assistant_id=body.assistant_id,
        metadata=body.metadata,
        kwargs={"input": body.input},
        multitask_strategy=body.multitask_strategy,
    )

    async def event_stream() -> Any:
        yield _sse_event("metadata", {"run_id": run.run_id})
        try:
            result = await _execute_run(
                thread_id, run, body.input, body.assistant_id
            )
            yield _sse_event("values", result)
            yield _sse_event("end", None)
        except Exception as e:
            yield _sse_event("error", {"message": str(e)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/threads/{thread_id}/runs", response_model=Run)
async def create_background_run(thread_id: str, body: RunCreate) -> Run:
    """Create a background run (non-blocking)."""
    try:
        storage.get_thread(thread_id)
    except KeyError:
        storage.create_thread(thread_id=thread_id)

    run = storage.create_run(
        thread_id=thread_id,
        assistant_id=body.assistant_id,
        metadata=body.metadata,
        kwargs={"input": body.input},
        multitask_strategy=body.multitask_strategy,
    )

    asyncio.create_task(
        _execute_run(thread_id, run, body.input, body.assistant_id)
    )
    return run


@router.get("/threads/{thread_id}/runs", response_model=list[Run])
async def list_runs(thread_id: str, limit: int = 10, offset: int = 0) -> list[Run]:
    return storage.list_runs(thread_id, limit=limit, offset=offset)


@router.get("/threads/{thread_id}/runs/{run_id}", response_model=Run)
async def get_run(thread_id: str, run_id: str) -> Run:
    try:
        return storage.get_run(run_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@router.delete("/threads/{thread_id}/runs/{run_id}")
async def delete_run(thread_id: str, run_id: str) -> dict:
    return {"ok": True}


@router.post("/threads/{thread_id}/runs/{run_id}/cancel")
async def cancel_run(thread_id: str, run_id: str) -> dict:
    storage.update_run_status(run_id, "interrupted")
    return {"ok": True}


@router.get("/threads/{thread_id}/runs/{run_id}/join")
async def join_run(thread_id: str, run_id: str) -> dict[str, Any]:
    run = storage.get_run(run_id)
    while run.status in ("pending", "running"):
        await asyncio.sleep(0.5)
        run = storage.get_run(run_id)
    state = storage.get_thread_state(thread_id)
    return state.values if state else {}


# ── Stateless runs ──────────────────────────────────────────


@router.post("/runs/wait")
async def create_stateless_run_wait(body: RunCreate) -> dict[str, Any]:
    """Stateless run - create temporary thread, execute, return result."""
    thread = storage.create_thread()
    run = storage.create_run(
        thread_id=thread.thread_id,
        assistant_id=body.assistant_id,
        metadata=body.metadata,
    )
    result = await _execute_run(
        thread.thread_id, run, body.input, body.assistant_id
    )
    return result


@router.post("/runs/stream")
async def create_stateless_run_stream(body: RunCreate) -> StreamingResponse:
    """Stateless streaming run."""
    thread = storage.create_thread()
    run = storage.create_run(
        thread_id=thread.thread_id,
        assistant_id=body.assistant_id,
        metadata=body.metadata,
    )

    async def event_stream() -> Any:
        yield _sse_event("metadata", {"run_id": run.run_id})
        try:
            result = await _execute_run(
                thread.thread_id, run, body.input, body.assistant_id
            )
            yield _sse_event("values", result)
            yield _sse_event("end", None)
        except Exception as e:
            yield _sse_event("error", {"message": str(e)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
    )


@router.post("/runs", response_model=Run)
async def create_stateless_background_run(body: RunCreate) -> Run:
    thread = storage.create_thread()
    run = storage.create_run(
        thread_id=thread.thread_id,
        assistant_id=body.assistant_id,
        metadata=body.metadata,
    )
    asyncio.create_task(
        _execute_run(thread.thread_id, run, body.input, body.assistant_id)
    )
    return run


@router.post("/runs/batch")
async def create_run_batch(body: list[RunCreate]) -> list[Run]:
    runs = []
    for item in body:
        thread = storage.create_thread()
        run = storage.create_run(
            thread_id=thread.thread_id,
            assistant_id=item.assistant_id,
            metadata=item.metadata,
        )
        asyncio.create_task(
            _execute_run(thread.thread_id, run, item.input, item.assistant_id)
        )
        runs.append(run)
    return runs


def _sse_event(event: str, data: Any) -> str:
    """Format a Server-Sent Event."""
    payload = json.dumps(data) if data is not None else ""
    return f"event: {event}\ndata: {payload}\n\n"
