"""Threads endpoints: CRUD, state management, history."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from langgraph_chat.api.models import (
    Thread,
    ThreadCreate,
    ThreadPatch,
    ThreadSearch,
    ThreadState,
    ThreadStateUpdate,
)
from langgraph_chat.api.storage import storage

router = APIRouter(prefix="/threads", tags=["Threads"])


@router.post("", response_model=Thread)
async def create_thread(body: ThreadCreate | None = None) -> Thread:
    b = body or ThreadCreate()
    try:
        return storage.create_thread(
            thread_id=b.thread_id,
            metadata=b.metadata,
            if_exists=b.if_exists,
        )
    except KeyError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e


@router.post("/search", response_model=list[Thread])
async def search_threads(
    body: ThreadSearch | None = None,
) -> list[Thread]:
    b = body or ThreadSearch()
    return storage.search_threads(
        metadata=b.metadata,
        status=b.status,
        limit=b.limit,
        offset=b.offset,
    )


@router.post("/count")
async def count_threads() -> int:
    return storage.count_threads()


@router.get("/{thread_id}", response_model=Thread)
async def get_thread(thread_id: str) -> Thread:
    try:
        return storage.get_thread(thread_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@router.patch("/{thread_id}", response_model=Thread)
async def patch_thread(thread_id: str, body: ThreadPatch) -> Thread:
    try:
        return storage.patch_thread(
            thread_id, **body.model_dump(exclude_none=True)
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@router.delete("/{thread_id}")
async def delete_thread(thread_id: str) -> dict:
    storage.delete_thread(thread_id)
    return {"ok": True}


@router.get("/{thread_id}/state", response_model=ThreadState | None)
async def get_thread_state(thread_id: str) -> Any:
    try:
        storage.get_thread(thread_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    return storage.get_thread_state(thread_id)


@router.post("/{thread_id}/state")
async def update_thread_state(
    thread_id: str, body: ThreadStateUpdate
) -> dict[str, str]:
    try:
        storage.get_thread(thread_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    from langgraph_chat.api.graph_registry import apply_state_update

    apply_state_update(thread_id, body)
    return {"ok": "true"}


@router.get("/{thread_id}/history", response_model=list[ThreadState])
async def get_thread_history(thread_id: str, limit: int = 10) -> list[ThreadState]:
    try:
        storage.get_thread(thread_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    return storage.get_thread_history(thread_id, limit=limit)


@router.post("/{thread_id}/history", response_model=list[ThreadState])
async def get_thread_history_post(
    thread_id: str, body: dict[str, Any] | None = None
) -> list[ThreadState]:
    limit = body.get("limit", 10) if body else 10
    return storage.get_thread_history(thread_id, limit=limit)


@router.post("/{thread_id}/copy", response_model=Thread)
async def copy_thread(thread_id: str) -> Thread:
    try:
        original = storage.get_thread(thread_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    new_thread = storage.create_thread(metadata=original.metadata)
    state = storage.get_thread_state(thread_id)
    if state:
        storage.set_thread_state(new_thread.thread_id, state)
    return new_thread
