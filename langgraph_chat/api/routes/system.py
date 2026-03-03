"""System endpoints: health check, server info, API docs."""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()

SERVER_VERSION = "0.1.0"


@router.get("/ok")
async def health_check() -> dict:
    return {"ok": True}


@router.get("/info")
async def server_info() -> dict:
    return {
        "version": SERVER_VERSION,
        "langgraph_chat_version": "0.1.0",
        "flags": {
            "assistants": True,
            "crons": False,
            "store": True,
            "langsmith": True,
        },
        "host": {
            "kind": "self-hosted",
            "project_id": None,
        },
    }
