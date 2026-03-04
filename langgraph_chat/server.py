"""FastAPI server for the LangGraph chat application."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from langgraph_chat.graph import chat, graph

app = FastAPI(title="LangGraph Chat", version="0.1.0")

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
if STATIC_DIR.exists():
    app.mount(
        "/static", StaticFiles(directory=str(STATIC_DIR)), name="static"
    )


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    """Serve the chat UI."""
    index_file = STATIC_DIR / "index.html"
    return HTMLResponse(content=index_file.read_text())


@app.post("/api/chat", response_model=ChatResponse)
async def api_chat(request: ChatRequest) -> ChatResponse:
    """Non-streaming: return the full reply at once."""
    response = await asyncio.to_thread(chat, request.message)
    return ChatResponse(reply=str(response.content))


@app.post("/api/chat/stream")
async def api_chat_stream(request: ChatRequest) -> StreamingResponse:
    """SSE streaming using real graph.stream(stream_mode='messages')."""

    async def event_generator() -> AsyncIterator[str]:
        run_id = str(uuid4())
        yield _sse("metadata", {"run_id": run_id, "attempt": 1})

        try:
            from langchain_core.messages import HumanMessage

            inp = {
                "messages": [HumanMessage(content=request.message)]
            }

            def _do_stream() -> (
                list[tuple[str, dict[str, Any]]]
            ):
                events: list[tuple[str, dict[str, Any]]] = []
                accumulated = ""
                msg_id = ""
                metadata_sent = False

                for msg_chunk, meta in graph.stream(
                    inp, stream_mode="messages"
                ):
                    node = meta.get("langgraph_node", "agent")
                    chunk_id = getattr(msg_chunk, "id", "") or ""
                    if chunk_id and not msg_id:
                        msg_id = chunk_id

                    if not metadata_sent and msg_id:
                        events.append((
                            "messages/metadata",
                            {
                                msg_id: {
                                    "metadata": {
                                        "langgraph_node": node,
                                    }
                                }
                            },
                        ))
                        metadata_sent = True

                    content = str(
                        getattr(msg_chunk, "content", "")
                    )
                    if content:
                        accumulated += content

                    rm = (
                        getattr(
                            msg_chunk, "response_metadata", {}
                        )
                        or {}
                    )
                    partial: dict[str, Any] = {
                        "content": accumulated,
                        "id": msg_id,
                        "type": "ai",
                        "response_metadata": rm if rm else {},
                    }

                    um = getattr(
                        msg_chunk, "usage_metadata", None
                    )
                    if um:
                        u = um if isinstance(um, dict) else {}
                        if u.get("total_tokens"):
                            partial["usage"] = u

                    events.append(("messages/partial", [partial]))
                return events

            events = await asyncio.to_thread(_do_stream)
            for event_type, data in events:
                yield _sse(event_type, data)

            yield _sse("end", None)
        except Exception as e:
            yield _sse("error", {"message": str(e)})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def _sse(event: str, data: Any) -> str:
    payload = (
        json.dumps(data, ensure_ascii=False) if data is not None else ""
    )
    return f"event: {event}\ndata: {payload}\n\n"


@app.get("/api/health")
async def health() -> dict:
    """Health check endpoint."""
    return {"status": "ok"}
