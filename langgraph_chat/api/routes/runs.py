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


async def _stream_run(
    thread_id: str,
    run: Run,
    input_data: Any,
    assistant_id: str,
    stream_mode: list[str] | str = "values",
) -> Any:
    """Stream graph execution using real graph.stream()."""
    from langgraph_chat.api.graph_registry import stream_graph

    modes = (
        [stream_mode] if isinstance(stream_mode, str) else stream_mode
    )

    yield _sse("metadata", {"run_id": run.run_id, "attempt": 1})

    storage.update_run_status(run.run_id, "running")
    storage.set_thread_status(thread_id, "busy")

    try:

        def _do_stream() -> list[tuple[str, Any]]:
            events = []
            for event_type, data in stream_graph(
                assistant_id, thread_id, input_data, modes
            ):
                events.append((event_type, data))
            return events

        events = await asyncio.to_thread(_do_stream)

        for event_type, data in events:
            yield _sse(event_type, data)

        storage.update_run_status(run.run_id, "success")
        storage.set_thread_status(thread_id, "idle")

        yield _sse("end", None)

    except Exception as e:
        logger.exception("Stream run %s failed", run.run_id)
        storage.update_run_status(run.run_id, "error")
        storage.set_thread_status(thread_id, "error")
        yield _sse("error", {"message": str(e)})


# ── Thread-scoped runs ──────────────────────────────────────


@router.post("/threads/{thread_id}/runs/wait")
async def create_run_wait(thread_id: str, body: RunCreate) -> dict[str, Any]:
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
    return await _execute_run(thread_id, run, body.input, body.assistant_id)


@router.post("/threads/{thread_id}/runs/stream")
async def create_run_stream(
    thread_id: str, body: RunCreate
) -> StreamingResponse:
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

    return StreamingResponse(
        _stream_run(
            thread_id, run, body.input, body.assistant_id, body.stream_mode
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/threads/{thread_id}/runs", response_model=Run)
async def create_background_run(thread_id: str, body: RunCreate) -> Run:
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
async def list_runs(
    thread_id: str, limit: int = 10, offset: int = 0
) -> list[Run]:
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
    thread = storage.create_thread()
    run = storage.create_run(
        thread_id=thread.thread_id,
        assistant_id=body.assistant_id,
        metadata=body.metadata,
    )
    return await _execute_run(
        thread.thread_id, run, body.input, body.assistant_id
    )


@router.post("/runs/stream")
async def create_stateless_run_stream(
    body: RunCreate,
) -> StreamingResponse:
    thread = storage.create_thread()
    run = storage.create_run(
        thread_id=thread.thread_id,
        assistant_id=body.assistant_id,
        metadata=body.metadata,
    )
    return StreamingResponse(
        _stream_run(
            thread.thread_id,
            run,
            body.input,
            body.assistant_id,
            body.stream_mode,
        ),
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
            _execute_run(
                thread.thread_id, run, item.input, item.assistant_id
            )
        )
        runs.append(run)
    return runs


def _sse(event: str, data: Any) -> str:
    payload = json.dumps(data, ensure_ascii=False) if data is not None else ""
    return f"event: {event}\ndata: {payload}\n\n"
