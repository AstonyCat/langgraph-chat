"""In-memory storage backend for threads, runs, assistants, and key-value store."""

from __future__ import annotations

import threading
from copy import deepcopy
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from langgraph_chat.api.models import (
    Assistant,
    Run,
    StoreItem,
    Thread,
    ThreadState,
)


def _now() -> datetime:
    return datetime.now(UTC)


class InMemoryStorage:
    """Thread-safe in-memory storage for the LangGraph API server."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._assistants: dict[str, Assistant] = {}
        self._threads: dict[str, Thread] = {}
        self._thread_states: dict[str, ThreadState] = {}
        self._thread_history: dict[str, list[ThreadState]] = {}
        self._runs: dict[str, Run] = {}
        self._thread_runs: dict[str, list[str]] = {}
        self._store: dict[str, StoreItem] = {}

    # ── Assistants ──────────────────────────────────────────

    def create_assistant(
        self,
        *,
        assistant_id: str | None = None,
        graph_id: str,
        config: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        name: str = "Untitled",
        description: str | None = None,
        if_exists: str = "raise",
    ) -> Assistant:
        aid = assistant_id or str(uuid4())
        with self._lock:
            if aid in self._assistants:
                if if_exists == "do_nothing":
                    return self._assistants[aid]
                if if_exists == "raise":
                    raise KeyError(f"Assistant {aid} already exists")
                # update
                existing = self._assistants[aid]
                existing.graph_id = graph_id
                existing.config = config or {}
                existing.metadata = metadata or {}
                existing.name = name
                existing.description = description
                existing.updated_at = _now()
                existing.version += 1
                return deepcopy(existing)
            a = Assistant(
                assistant_id=aid,
                graph_id=graph_id,
                config=config or {},
                metadata=metadata or {},
                name=name,
                description=description,
                created_at=_now(),
                updated_at=_now(),
            )
            self._assistants[aid] = a
            return deepcopy(a)

    def get_assistant(self, assistant_id: str) -> Assistant:
        with self._lock:
            if assistant_id not in self._assistants:
                raise KeyError(f"Assistant {assistant_id} not found")
            return deepcopy(self._assistants[assistant_id])

    def search_assistants(
        self,
        graph_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        limit: int = 10,
        offset: int = 0,
    ) -> list[Assistant]:
        with self._lock:
            results = list(self._assistants.values())
        if graph_id:
            results = [a for a in results if a.graph_id == graph_id]
        if metadata:
            results = [
                a for a in results
                if all(a.metadata.get(k) == v for k, v in metadata.items())
            ]
        return [deepcopy(a) for a in results[offset : offset + limit]]

    def count_assistants(self, graph_id: str | None = None) -> int:
        with self._lock:
            if graph_id:
                return sum(
                    1 for a in self._assistants.values()
                    if a.graph_id == graph_id
                )
            return len(self._assistants)

    def patch_assistant(self, assistant_id: str, **kwargs: Any) -> Assistant:
        with self._lock:
            if assistant_id not in self._assistants:
                raise KeyError(f"Assistant {assistant_id} not found")
            a = self._assistants[assistant_id]
            for k, v in kwargs.items():
                if v is not None and hasattr(a, k):
                    setattr(a, k, v)
            a.updated_at = _now()
            a.version += 1
            return deepcopy(a)

    def delete_assistant(self, assistant_id: str) -> None:
        with self._lock:
            self._assistants.pop(assistant_id, None)

    # ── Threads ─────────────────────────────────────────────

    def create_thread(
        self,
        *,
        thread_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        if_exists: str = "raise",
    ) -> Thread:
        tid = thread_id or str(uuid4())
        with self._lock:
            if tid in self._threads:
                if if_exists == "do_nothing":
                    return deepcopy(self._threads[tid])
                raise KeyError(f"Thread {tid} already exists")
            t = Thread(
                thread_id=tid,
                created_at=_now(),
                updated_at=_now(),
                metadata=metadata or {},
            )
            self._threads[tid] = t
            self._thread_runs[tid] = []
            self._thread_history[tid] = []
            return deepcopy(t)

    def get_thread(self, thread_id: str) -> Thread:
        with self._lock:
            if thread_id not in self._threads:
                raise KeyError(f"Thread {thread_id} not found")
            return deepcopy(self._threads[thread_id])

    def search_threads(
        self,
        metadata: dict[str, Any] | None = None,
        status: str | None = None,
        limit: int = 10,
        offset: int = 0,
    ) -> list[Thread]:
        with self._lock:
            results = list(self._threads.values())
        if status:
            results = [t for t in results if t.status == status]
        if metadata:
            results = [
                t for t in results
                if all(t.metadata.get(k) == v for k, v in metadata.items())
            ]
        return [deepcopy(t) for t in results[offset : offset + limit]]

    def count_threads(self) -> int:
        with self._lock:
            return len(self._threads)

    def patch_thread(self, thread_id: str, **kwargs: Any) -> Thread:
        with self._lock:
            if thread_id not in self._threads:
                raise KeyError(f"Thread {thread_id} not found")
            t = self._threads[thread_id]
            for k, v in kwargs.items():
                if v is not None and hasattr(t, k):
                    setattr(t, k, v)
            t.updated_at = _now()
            return deepcopy(t)

    def delete_thread(self, thread_id: str) -> None:
        with self._lock:
            self._threads.pop(thread_id, None)
            self._thread_states.pop(thread_id, None)
            self._thread_history.pop(thread_id, None)
            run_ids = self._thread_runs.pop(thread_id, [])
            for rid in run_ids:
                self._runs.pop(rid, None)

    def get_thread_state(self, thread_id: str) -> ThreadState | None:
        with self._lock:
            return deepcopy(self._thread_states.get(thread_id))

    def set_thread_state(self, thread_id: str, state: ThreadState) -> None:
        with self._lock:
            self._thread_states[thread_id] = deepcopy(state)
            if thread_id not in self._thread_history:
                self._thread_history[thread_id] = []
            self._thread_history[thread_id].append(deepcopy(state))
            if thread_id in self._threads:
                self._threads[thread_id].state_updated_at = _now()
                self._threads[thread_id].updated_at = _now()
                self._threads[thread_id].values = state.values

    def get_thread_history(self, thread_id: str, limit: int = 10) -> list[ThreadState]:
        with self._lock:
            history = self._thread_history.get(thread_id, [])
            return [deepcopy(s) for s in reversed(history[-limit:])]

    # ── Runs ────────────────────────────────────────────────

    def create_run(
        self,
        *,
        thread_id: str,
        assistant_id: str,
        metadata: dict[str, Any] | None = None,
        kwargs: dict[str, Any] | None = None,
        multitask_strategy: str = "reject",
    ) -> Run:
        run_id = str(uuid4())
        with self._lock:
            r = Run(
                run_id=run_id,
                thread_id=thread_id,
                assistant_id=assistant_id,
                created_at=_now(),
                updated_at=_now(),
                status="pending",
                metadata=metadata or {},
                kwargs=kwargs or {},
                multitask_strategy=multitask_strategy,
            )
            self._runs[run_id] = r
            if thread_id in self._thread_runs:
                self._thread_runs[thread_id].append(run_id)
            return deepcopy(r)

    def get_run(self, run_id: str) -> Run:
        with self._lock:
            if run_id not in self._runs:
                raise KeyError(f"Run {run_id} not found")
            return deepcopy(self._runs[run_id])

    def update_run_status(
        self, run_id: str, status: str
    ) -> None:
        with self._lock:
            if run_id in self._runs:
                self._runs[run_id].status = status  # type: ignore[assignment]
                self._runs[run_id].updated_at = _now()

    def list_runs(self, thread_id: str, limit: int = 10, offset: int = 0) -> list[Run]:
        with self._lock:
            run_ids = self._thread_runs.get(thread_id, [])
            runs = [deepcopy(self._runs[rid]) for rid in run_ids if rid in self._runs]
        return list(reversed(runs))[offset : offset + limit]

    def set_thread_status(self, thread_id: str, status: str) -> None:
        with self._lock:
            if thread_id in self._threads:
                self._threads[thread_id].status = status  # type: ignore[assignment]
                self._threads[thread_id].updated_at = _now()

    # ── Store ───────────────────────────────────────────────

    def store_put(self, namespace: list[str], key: str, value: dict[str, Any]) -> None:
        store_key = ":".join(namespace) + ":" + key
        with self._lock:
            now = _now()
            existing = self._store.get(store_key)
            self._store[store_key] = StoreItem(
                namespace=namespace,
                key=key,
                value=value,
                created_at=existing.created_at if existing else now,
                updated_at=now,
            )

    def store_get(self, namespace: list[str], key: str) -> StoreItem | None:
        store_key = ":".join(namespace) + ":" + key
        with self._lock:
            item = self._store.get(store_key)
            return deepcopy(item) if item else None

    def store_delete(self, namespace: list[str], key: str) -> None:
        store_key = ":".join(namespace) + ":" + key
        with self._lock:
            self._store.pop(store_key, None)

    def store_search(
        self,
        namespace_prefix: list[str],
        limit: int = 10,
        offset: int = 0,
    ) -> list[StoreItem]:
        prefix = ":".join(namespace_prefix)
        with self._lock:
            results = [
                deepcopy(item)
                for key, item in self._store.items()
                if key.startswith(prefix)
            ]
        return results[offset : offset + limit]

    def store_list_namespaces(
        self,
        match_prefix: list[str] | None = None,
        max_depth: int | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[list[str]]:
        namespaces: set[tuple[str, ...]] = set()
        prefix = ":".join(match_prefix) if match_prefix else ""
        with self._lock:
            for item in self._store.values():
                ns = item.namespace
                if match_prefix and not ":".join(ns).startswith(prefix):
                    continue
                if max_depth:
                    ns = ns[:max_depth]
                namespaces.add(tuple(ns))
        result = [list(ns) for ns in sorted(namespaces)]
        return result[offset : offset + limit]


storage = InMemoryStorage()
