"""Store endpoints: persistent key-value store for long-term memory."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from langgraph_chat.api.models import (
    NamespaceList,
    StoreItem,
    StoreItemDelete,
    StoreItemPut,
    StoreItemSearch,
)
from langgraph_chat.api.storage import storage

router = APIRouter(prefix="/store", tags=["Store"])


@router.put("/items")
async def put_item(body: StoreItemPut) -> dict:
    storage.store_put(body.namespace, body.key, body.value)
    return {"ok": True}


@router.get("/items")
async def get_item(namespace: str, key: str) -> Any:
    """Get a single item. Namespace passed as dot-separated string."""
    ns = namespace.split(".")
    item = storage.store_get(ns, key)
    if item is None:
        raise HTTPException(status_code=404, detail="Item not found")
    return item


@router.delete("/items")
async def delete_item(body: StoreItemDelete) -> dict:
    storage.store_delete(body.namespace, body.key)
    return {"ok": True}


@router.post("/items/search", response_model=list[StoreItem])
async def search_items(body: StoreItemSearch) -> list[StoreItem]:
    return storage.store_search(
        namespace_prefix=body.namespace_prefix,
        limit=body.limit,
        offset=body.offset,
    )


@router.post("/namespaces")
async def list_namespaces(body: NamespaceList) -> list[list[str]]:
    return storage.store_list_namespaces(
        match_prefix=body.match_prefix,
        max_depth=body.max_depth,
        limit=body.limit,
        offset=body.offset,
    )
