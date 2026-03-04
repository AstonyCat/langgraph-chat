"""Graph registry: bridges LangGraph graphs with the API server.

Manages graph loading, execution, state serialization, and schema extraction.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from langgraph_chat.api.models import ThreadState, ThreadStateUpdate
from langgraph_chat.api.storage import storage
from langgraph_chat.graph import build_graph

logger = logging.getLogger(__name__)

_graphs: dict[str, Any] = {}


def _now() -> datetime:
    return datetime.now(UTC)


def _get_or_build_graph(graph_id: str) -> Any:
    """Get a compiled graph, building it if necessary."""
    if graph_id not in _graphs:
        _graphs[graph_id] = build_graph()
    return _graphs[graph_id]


def _serialize_messages(messages: list) -> list[dict[str, Any]]:
    """Convert LangChain message objects to serializable dicts."""
    result = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            result.append({
                "type": "human",
                "content": str(msg.content),
                "id": getattr(msg, "id", None),
            })
        elif isinstance(msg, AIMessage):
            result.append({
                "type": "ai",
                "content": str(msg.content),
                "id": getattr(msg, "id", None),
                "response_metadata": getattr(msg, "response_metadata", {}),
            })
        elif isinstance(msg, dict):
            result.append(msg)
        else:
            result.append({
                "type": getattr(msg, "type", "unknown"),
                "content": str(getattr(msg, "content", msg)),
                "id": getattr(msg, "id", None),
            })
    return result


def _deserialize_input(input_data: Any) -> dict[str, Any]:
    """Convert API input into graph-compatible input."""
    if input_data is None:
        return {"messages": []}
    if isinstance(input_data, dict):
        messages = input_data.get("messages", [])
        converted = []
        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get("role", msg.get("type", "human"))
                content = msg.get("content", "")
                if role in ("human", "user"):
                    converted.append(HumanMessage(content=content))
                elif role in ("ai", "assistant"):
                    converted.append(AIMessage(content=content))
                else:
                    converted.append(HumanMessage(content=content))
            elif isinstance(msg, (HumanMessage, AIMessage)):
                converted.append(msg)
        return {"messages": converted}
    return {"messages": []}


def _resolve_graph_id(assistant_id: str) -> str:
    """Resolve assistant_id to a graph_id, falling back to direct use."""
    try:
        assistant = storage.get_assistant(assistant_id)
        return assistant.graph_id
    except KeyError:
        return assistant_id


def execute_graph(
    assistant_id: str, thread_id: str, input_data: Any
) -> dict[str, Any]:
    """Execute the graph for an assistant on a thread.

    Loads existing thread state, merges new input, executes graph,
    and persists the resulting state.
    """
    graph_id = _resolve_graph_id(assistant_id)
    graph = _get_or_build_graph(graph_id)
    graph_input = _deserialize_input(input_data)

    existing_state = storage.get_thread_state(thread_id)
    if existing_state and existing_state.values.get("messages"):
        existing_msgs = existing_state.values["messages"]
        deserialized_existing = []
        for m in existing_msgs:
            if isinstance(m, dict):
                mtype = m.get("type", "human")
                if mtype in ("human", "user"):
                    deserialized_existing.append(HumanMessage(content=m["content"]))
                else:
                    deserialized_existing.append(AIMessage(content=m["content"]))
            else:
                deserialized_existing.append(m)
        graph_input["messages"] = deserialized_existing + graph_input["messages"]

    result = graph.invoke(graph_input)

    serialized = {
        "messages": _serialize_messages(result.get("messages", []))
    }

    checkpoint_id = f"checkpoint-{thread_id}-{_now().isoformat()}"
    state = ThreadState(
        values=serialized,
        next=[],
        tasks=[],
        checkpoint={"thread_id": thread_id, "checkpoint_id": checkpoint_id},
        metadata={"assistant_id": assistant_id},
        created_at=_now(),
    )
    storage.set_thread_state(thread_id, state)

    return serialized


def apply_state_update(thread_id: str, update: ThreadStateUpdate) -> None:
    """Apply a manual state update to a thread."""
    existing = storage.get_thread_state(thread_id)
    if existing and update.values:
        merged = {**existing.values, **update.values}
    elif update.values:
        merged = update.values
    else:
        merged = existing.values if existing else {}

    checkpoint_id = f"checkpoint-{thread_id}-{_now().isoformat()}"
    state = ThreadState(
        values=merged,
        next=[],
        tasks=[],
        checkpoint={"thread_id": thread_id, "checkpoint_id": checkpoint_id},
        metadata={"source": "update", "as_node": update.as_node or "__manual__"},
        created_at=_now(),
    )
    storage.set_thread_state(thread_id, state)


def get_graph_structure(graph_id: str) -> dict[str, Any]:
    """Return the graph structure for visualization."""
    graph = _get_or_build_graph(graph_id)
    try:
        drawn = graph.get_graph()
        nodes = [
            {"id": n.id, "name": n.name}
            for n in drawn.nodes.values()
        ]
        edges = [
            {"source": e.source, "target": e.target}
            for e in drawn.edges
        ]
        return {"nodes": nodes, "edges": edges}
    except Exception:
        return {
            "nodes": [
                {"id": "__start__", "name": "__start__"},
                {"id": "chatbot", "name": "chatbot"},
                {"id": "__end__", "name": "__end__"},
            ],
            "edges": [
                {"source": "__start__", "target": "chatbot"},
                {"source": "chatbot", "target": "__end__"},
            ],
        }


def get_graph_schemas(graph_id: str) -> dict[str, Any]:
    """Return input/output schemas for a graph."""
    return {
        "graph_id": graph_id,
        "input_schema": {
            "type": "object",
            "properties": {
                "messages": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "role": {"type": "string"},
                            "content": {"type": "string"},
                        },
                    },
                    "title": "Messages",
                }
            },
            "required": ["messages"],
            "title": "ChatState",
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "messages": {
                    "type": "array",
                    "title": "Messages",
                }
            },
            "title": "ChatState",
        },
        "state_schema": {
            "type": "object",
            "properties": {
                "messages": {
                    "type": "array",
                    "title": "Messages",
                }
            },
            "title": "ChatState",
        },
        "config_schema": {
            "type": "object",
            "properties": {},
        },
    }
