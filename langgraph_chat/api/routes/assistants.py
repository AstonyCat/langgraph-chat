"""Assistants endpoints: CRUD, graph visualization, schemas."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from langgraph_chat.api.models import (
    Assistant,
    AssistantCreate,
    AssistantPatch,
    AssistantSearch,
)
from langgraph_chat.api.storage import storage

router = APIRouter(prefix="/assistants", tags=["Assistants"])


@router.post("", response_model=Assistant)
async def create_assistant(body: AssistantCreate) -> Assistant:
    try:
        return storage.create_assistant(
            assistant_id=body.assistant_id,
            graph_id=body.graph_id,
            config=body.config,
            metadata=body.metadata,
            name=body.name,
            description=body.description,
            if_exists=body.if_exists,
        )
    except KeyError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e


@router.post("/search", response_model=list[Assistant])
async def search_assistants(body: AssistantSearch) -> list[Assistant]:
    return storage.search_assistants(
        graph_id=body.graph_id,
        metadata=body.metadata,
        limit=body.limit,
        offset=body.offset,
    )


@router.post("/count")
async def count_assistants(body: dict[str, Any] | None = None) -> int:
    graph_id = body.get("graph_id") if body else None
    return storage.count_assistants(graph_id=graph_id)


@router.get("/{assistant_id}", response_model=Assistant)
async def get_assistant(assistant_id: str) -> Assistant:
    try:
        return storage.get_assistant(assistant_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@router.patch("/{assistant_id}", response_model=Assistant)
async def patch_assistant(assistant_id: str, body: AssistantPatch) -> Assistant:
    try:
        return storage.patch_assistant(
            assistant_id, **body.model_dump(exclude_none=True)
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@router.delete("/{assistant_id}")
async def delete_assistant(assistant_id: str) -> dict:
    storage.delete_assistant(assistant_id)
    return {"ok": True}


@router.get("/{assistant_id}/graph")
async def get_assistant_graph(assistant_id: str) -> dict[str, Any]:
    """Return the graph structure for visualization."""
    try:
        assistant = storage.get_assistant(assistant_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    from langgraph_chat.api.graph_registry import get_graph_structure

    return get_graph_structure(assistant.graph_id)


@router.get("/{assistant_id}/schemas")
async def get_assistant_schemas(assistant_id: str) -> dict[str, Any]:
    """Return input/output schemas for the assistant's graph."""
    try:
        assistant = storage.get_assistant(assistant_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    from langgraph_chat.api.graph_registry import get_graph_schemas

    return get_graph_schemas(assistant.graph_id)


@router.get("/{assistant_id}/subgraphs")
async def get_assistant_subgraphs(assistant_id: str) -> dict[str, Any]:
    return {}


@router.post("/{assistant_id}/versions")
async def get_assistant_versions(assistant_id: str) -> list[Assistant]:
    try:
        a = storage.get_assistant(assistant_id)
        return [a]
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
