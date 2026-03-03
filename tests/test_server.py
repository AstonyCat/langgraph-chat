"""Tests for the FastAPI server."""

import os

import pytest
from httpx import ASGITransport, AsyncClient

from langgraph_chat.server import app

_has_api_key = bool(os.environ.get("OPENAI_API_KEY"))


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_health(client: AsyncClient):
    """Health endpoint should return ok."""
    response = await client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_chat_endpoint(client: AsyncClient):
    """Chat endpoint should return a reply."""
    response = await client.post(
        "/api/chat", json={"message": "Hello from tests"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "reply" in data
    assert len(data["reply"]) > 0


@pytest.mark.asyncio
@pytest.mark.skipif(_has_api_key, reason="Real LLM active; echo mode not used")
async def test_chat_endpoint_echo(client: AsyncClient):
    """In echo mode, reply should contain the user message."""
    response = await client.post(
        "/api/chat", json={"message": "ping"}
    )
    assert response.status_code == 200
    assert "ping" in response.json()["reply"]


@pytest.mark.asyncio
async def test_index_page(client: AsyncClient):
    """Index page should return HTML."""
    response = await client.get("/")
    assert response.status_code == 200
    assert "LangGraph Chat" in response.text
