"""Pydantic v2 models matching the LangGraph Platform API schemas."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


def _now() -> datetime:
    return datetime.now(UTC)


def _uuid() -> str:
    return str(uuid4())


# ── Assistants ──────────────────────────────────────────────


class AssistantCreate(BaseModel):
    assistant_id: str = Field(default_factory=_uuid)
    graph_id: str
    config: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    name: str = "Untitled"
    description: str | None = None
    if_exists: Literal["do_nothing", "raise", "update"] = "raise"


class AssistantPatch(BaseModel):
    graph_id: str | None = None
    config: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    name: str | None = None
    description: str | None = None


class Assistant(BaseModel):
    assistant_id: str
    graph_id: str
    config: dict[str, Any]
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, Any]
    version: int = 1
    name: str = "Untitled"
    description: str | None = None


class AssistantSearch(BaseModel):
    graph_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    limit: int = 10
    offset: int = 0


# ── Threads ─────────────────────────────────────────────────


class ThreadCreate(BaseModel):
    thread_id: str = Field(default_factory=_uuid)
    metadata: dict[str, Any] = Field(default_factory=dict)
    if_exists: Literal["do_nothing", "raise"] = "raise"


class ThreadPatch(BaseModel):
    metadata: dict[str, Any] | None = None


class Thread(BaseModel):
    thread_id: str
    created_at: datetime
    updated_at: datetime
    state_updated_at: datetime | None = None
    metadata: dict[str, Any]
    status: Literal["idle", "busy", "interrupted", "error"] = "idle"
    config: dict[str, Any] = Field(default_factory=dict)
    values: dict[str, Any] | None = None
    interrupts: dict[str, Any] = Field(default_factory=dict)


class ThreadSearch(BaseModel):
    metadata: dict[str, Any] = Field(default_factory=dict)
    status: str | None = None
    limit: int = 10
    offset: int = 0


class ThreadState(BaseModel):
    values: dict[str, Any]
    next: list[str]
    tasks: list[dict[str, Any]] = Field(default_factory=list)
    checkpoint: dict[str, Any]
    metadata: dict[str, Any]
    created_at: datetime
    parent_checkpoint: dict[str, Any] | None = None


class ThreadStateUpdate(BaseModel):
    values: dict[str, Any] | None = None
    as_node: str | None = None
    checkpoint: dict[str, Any] | None = None


# ── Runs ────────────────────────────────────────────────────


class RunCreate(BaseModel):
    assistant_id: str
    input: dict[str, Any] | list | str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)
    stream_mode: list[str] | str = "values"
    interrupt_before: list[str] | str | None = None
    interrupt_after: list[str] | str | None = None
    webhook: str | None = None
    multitask_strategy: Literal["reject", "rollback", "interrupt", "enqueue"] = "reject"
    on_disconnect: Literal["cancel", "continue"] = "cancel"
    feedback_keys: list[str] | None = None
    if_not_exists: Literal["create", "reject"] = "create"


class Run(BaseModel):
    run_id: str
    thread_id: str
    assistant_id: str
    created_at: datetime
    updated_at: datetime
    status: Literal[
        "pending", "running", "error", "success", "timeout", "interrupted"
    ] = "pending"
    metadata: dict[str, Any]
    kwargs: dict[str, Any] = Field(default_factory=dict)
    multitask_strategy: str = "reject"


# ── Store ───────────────────────────────────────────────────


class StoreItemPut(BaseModel):
    namespace: list[str]
    key: str
    value: dict[str, Any]


class StoreItemGet(BaseModel):
    namespace: list[str]
    key: str


class StoreItemDelete(BaseModel):
    namespace: list[str]
    key: str


class StoreItemSearch(BaseModel):
    namespace_prefix: list[str]
    filter: dict[str, Any] | None = None
    limit: int = 10
    offset: int = 0


class StoreItem(BaseModel):
    namespace: list[str]
    key: str
    value: dict[str, Any]
    created_at: datetime
    updated_at: datetime


class NamespaceList(BaseModel):
    match_prefix: list[str] | None = None
    match_suffix: list[str] | None = None
    max_depth: int | None = None
    limit: int = 100
    offset: int = 0
