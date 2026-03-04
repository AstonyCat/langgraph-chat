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

from langgraph_chat.graph import chat

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
    """SSE streaming in official LangGraph messages/partial format."""

    async def event_generator() -> AsyncIterator[str]:
        run_id = str(uuid4())
        lc_run_id = f"lc_run--{uuid4()}"

        yield _sse("metadata", {"run_id": run_id, "attempt": 1})

        try:
            response = await asyncio.to_thread(chat, request.message)
            content = str(response.content)
            rm = getattr(response, "response_metadata", None) or {}

            yield _sse(
                "messages/metadata",
                {
                    lc_run_id: {
                        "metadata": {
                            "langgraph_node": "chatbot",
                            "ls_model_name": rm.get("model_name", ""),
                            "ls_provider": rm.get(
                                "model_provider", ""
                            ),
                        }
                    }
                },
            )

            for i in range(1, len(content) + 1):
                partial: dict[str, Any] = {
                    "content": content[:i],
                    "id": lc_run_id,
                    "type": "ai",
                    "response_metadata": {},
                }
                if i == len(content):
                    partial["response_metadata"] = {
                        "finish_reason": rm.get(
                            "finish_reason", "stop"
                        ),
                        "model_name": rm.get("model_name", ""),
                        "model_provider": rm.get(
                            "model_provider", ""
                        ),
                    }
                    usage = rm.get("token_usage", {})
                    if usage:
                        partial["usage"] = {
                            "input_tokens": usage.get(
                                "prompt_tokens", 0
                            ),
                            "output_tokens": usage.get(
                                "completion_tokens", 0
                            ),
                            "total_tokens": usage.get(
                                "total_tokens", 0
                            ),
                        }
                yield _sse("messages/partial", [partial])
                await asyncio.sleep(0.02)

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
