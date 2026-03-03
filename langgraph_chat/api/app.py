"""Main FastAPI application implementing the LangGraph Platform API.

This is a custom implementation of the LangGraph API server, providing
the same REST endpoints as `langgraph dev` but built from scratch.

Usage:
    uv run uvicorn langgraph_chat.api.app:app --host 0.0.0.0 --port 2024 --reload
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from langgraph_chat.api.routes.assistants import router as assistants_router
from langgraph_chat.api.routes.runs import router as runs_router
from langgraph_chat.api.routes.store import router as store_router
from langgraph_chat.api.routes.system import router as system_router
from langgraph_chat.api.routes.threads import router as threads_router
from langgraph_chat.api.storage import storage

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Register the default 'chat' assistant on startup."""
    try:
        storage.create_assistant(
            assistant_id="chat",
            graph_id="chat",
            name="LangGraph Chat",
            description="Chat assistant powered by LangGraph + GLM-5",
            if_exists="update",
        )
        logger.info("Registered default assistant: chat")
    except Exception:
        logger.exception("Failed to register default assistant")
    yield


app = FastAPI(
    title="LangGraph Chat API Server",
    description="Custom implementation of the LangGraph Platform API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(system_router)
app.include_router(assistants_router)
app.include_router(threads_router)
app.include_router(runs_router)
app.include_router(store_router)

STATIC_DIR = Path(__file__).resolve().parent.parent.parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


_INDEX_HTML = (
    "<!DOCTYPE html><html><head><title>LangGraph Chat API</title>"
    "</head><body><h1>LangGraph Chat API Server</h1>"
    '<p>Custom implementation of the '
    '<a href="https://docs.langchain.com/langgraph-platform/'
    'server-api-ref">LangGraph Platform API</a>.</p>'
    "<ul>"
    '<li><a href="/docs">OpenAPI Docs</a></li>'
    '<li><a href="/ok">Health Check</a></li>'
    '<li><a href="/info">Server Info</a></li>'
    "</ul></body></html>"
)


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    """Serve a simple landing page with links to docs."""
    return HTMLResponse(content=_INDEX_HTML)
