"""Graph registry: bridges LangGraph graphs with the API server.

Manages graph loading, execution, state serialization, and schema extraction.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

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


def _serialize_tool_calls(tool_calls: list) -> list[dict[str, Any]]:
    """Serialize tool_calls to match official format."""
    result = []
    for tc in tool_calls:
        if isinstance(tc, dict):
            result.append({
                "id": tc.get("id", ""),
                "name": tc.get("name", ""),
                "args": tc.get("args", {}),
                "type": tc.get("type", "tool_call"),
            })
        else:
            result.append({
                "id": getattr(tc, "id", ""),
                "name": getattr(tc, "name", ""),
                "args": getattr(tc, "args", {}),
                "type": "tool_call",
            })
    return result


def _serialize_usage(msg: Any) -> dict[str, Any] | None:
    """Extract usage_metadata matching official format."""
    um = getattr(msg, "usage_metadata", None)
    if not um:
        return None
    if isinstance(um, dict):
        return {
            "input_tokens": um.get("input_tokens", 0),
            "output_tokens": um.get("output_tokens", 0),
            "total_tokens": um.get("total_tokens", 0),
        }
    return {
        "input_tokens": getattr(um, "input_tokens", 0),
        "output_tokens": getattr(um, "output_tokens", 0),
        "total_tokens": getattr(um, "total_tokens", 0),
    }


def _serialize_messages(messages: list) -> list[dict[str, Any]]:
    """Convert LangChain messages to official LangGraph format."""
    result = []
    for msg in messages:
        if isinstance(msg, dict):
            result.append(msg)
            continue

        base: dict[str, Any] = {
            "type": getattr(msg, "type", "unknown"),
            "content": str(getattr(msg, "content", "")),
            "id": getattr(msg, "id", None),
            "name": getattr(msg, "name", None),
            "additional_kwargs": getattr(
                msg, "additional_kwargs", {}
            ),
            "response_metadata": getattr(
                msg, "response_metadata", {}
            ),
        }

        if isinstance(msg, AIMessage):
            tc = getattr(msg, "tool_calls", [])
            base["tool_calls"] = _serialize_tool_calls(tc)
            base["invalid_tool_calls"] = getattr(
                msg, "invalid_tool_calls", []
            )
            um = _serialize_usage(msg)
            if um:
                base["usage_metadata"] = um

        if isinstance(msg, ToolMessage):
            base["tool_call_id"] = getattr(
                msg, "tool_call_id", ""
            )
            base["artifact"] = getattr(msg, "artifact", None)
            base["status"] = getattr(msg, "status", "success")

        result.append(base)
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
    _save_state(thread_id, assistant_id, serialized)
    return serialized


def stream_graph(
    assistant_id: str,
    thread_id: str,
    input_data: Any,
    stream_mode: list[str] | None = None,
) -> Any:
    """Stream graph execution using real graph.stream().

    Yields (event_type, data) tuples matching LangGraph Server SSE.
    """

    graph_id = _resolve_graph_id(assistant_id)
    graph = _get_or_build_graph(graph_id)
    graph_input = _deserialize_input(input_data)

    modes = stream_mode or ["values"]
    use_multi = len(modes) > 1

    accumulated_content: dict[str, str] = {}

    if use_multi:
        for mode, chunk in graph.stream(
            graph_input, stream_mode=modes
        ):
            yield from _process_stream_chunk(
                mode, chunk, accumulated_content
            )
    else:
        single_mode = modes[0]
        for chunk in graph.stream(
            graph_input, stream_mode=single_mode
        ):
            yield from _process_stream_chunk(
                single_mode, chunk, accumulated_content
            )

    final_result = graph.invoke(graph_input)
    serialized = {
        "messages": _serialize_messages(
            final_result.get("messages", [])
        )
    }
    _save_state(thread_id, assistant_id, serialized)


def _process_stream_chunk(
    mode: str, chunk: Any, accumulated: dict[str, str]
) -> Any:
    """Convert a graph.stream() chunk to SSE event tuples."""
    if mode == "values":
        if isinstance(chunk, dict) and "messages" in chunk:
            yield (
                "values",
                {
                    "messages": _serialize_messages(
                        chunk["messages"]
                    )
                },
            )
        else:
            yield ("values", chunk)

    elif mode == "updates":
        if isinstance(chunk, dict):
            serialized_update = {}
            for node, data in chunk.items():
                if (
                    isinstance(data, dict)
                    and "messages" in data
                ):
                    serialized_update[node] = {
                        "messages": _serialize_messages(
                            data["messages"]
                        )
                    }
                else:
                    serialized_update[node] = data
            yield ("updates", serialized_update)

    elif mode == "messages":
        msg_chunk, metadata = chunk
        node = metadata.get("langgraph_node", "agent")
        run_id = metadata.get("run_id", "")
        msg_id = getattr(msg_chunk, "id", "") or run_id

        if msg_id not in accumulated:
            accumulated[msg_id] = ""
            yield (
                "messages/metadata",
                {
                    msg_id: {
                        "metadata": {
                            "langgraph_node": node,
                            **{
                                k: v
                                for k, v in metadata.items()
                                if k.startswith("ls_")
                            },
                        }
                    }
                },
            )

        content = str(getattr(msg_chunk, "content", ""))
        if content:
            accumulated[msg_id] += content

        rm = getattr(msg_chunk, "response_metadata", {}) or {}
        partial: dict[str, Any] = {
            "content": accumulated[msg_id],
            "id": msg_id,
            "type": "ai",
            "response_metadata": (
                {
                    "model_provider": rm.get(
                        "model_provider", ""
                    )
                }
                if not rm.get("finish_reason")
                else rm
            ),
        }

        um = _serialize_usage(msg_chunk)
        if um and um.get("total_tokens"):
            partial["usage"] = um

        yield ("messages/partial", [partial])


def _save_state(
    thread_id: str, assistant_id: str, serialized: dict
) -> None:
    """Persist graph result as thread state."""
    checkpoint_id = (
        f"checkpoint-{thread_id}-{_now().isoformat()}"
    )
    state = ThreadState(
        values=serialized,
        next=[],
        tasks=[],
        checkpoint={
            "thread_id": thread_id,
            "checkpoint_id": checkpoint_id,
        },
        metadata={"assistant_id": assistant_id},
        created_at=_now(),
    )
    storage.set_thread_state(thread_id, state)


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
