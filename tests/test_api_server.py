"""Tests for the custom LangGraph API server."""

import pytest
from httpx import ASGITransport, AsyncClient

from langgraph_chat.api.app import app


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        await ac.post(
            "/assistants",
            json={
                "assistant_id": "chat",
                "graph_id": "chat",
                "name": "Test Chat",
                "if_exists": "update",
            },
        )
        yield ac


@pytest.mark.asyncio
async def test_health(client: AsyncClient):
    r = await client.get("/ok")
    assert r.status_code == 200
    assert r.json() == {"ok": True}


@pytest.mark.asyncio
async def test_info(client: AsyncClient):
    r = await client.get("/info")
    assert r.status_code == 200
    data = r.json()
    assert "version" in data
    assert data["flags"]["assistants"] is True


@pytest.mark.asyncio
async def test_assistant_lifecycle(client: AsyncClient):
    r = await client.post(
        "/assistants",
        json={"graph_id": "chat", "name": "Test Bot"},
    )
    assert r.status_code == 200
    a = r.json()
    aid = a["assistant_id"]
    assert a["name"] == "Test Bot"

    r = await client.get(f"/assistants/{aid}")
    assert r.status_code == 200
    assert r.json()["name"] == "Test Bot"

    r = await client.patch(
        f"/assistants/{aid}", json={"name": "Updated Bot"}
    )
    assert r.status_code == 200
    assert r.json()["name"] == "Updated Bot"

    r = await client.post("/assistants/search", json={})
    assert r.status_code == 200
    assert any(x["assistant_id"] == aid for x in r.json())

    r = await client.delete(f"/assistants/{aid}")
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_thread_lifecycle(client: AsyncClient):
    r = await client.post("/threads", json={})
    assert r.status_code == 200
    t = r.json()
    tid = t["thread_id"]
    assert t["status"] == "idle"

    r = await client.get(f"/threads/{tid}")
    assert r.status_code == 200

    r = await client.post("/threads/search", json={})
    assert r.status_code == 200

    r = await client.delete(f"/threads/{tid}")
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_run_wait(client: AsyncClient):
    """Execute a run and wait for the result."""
    r = await client.post("/threads", json={})
    tid = r.json()["thread_id"]

    r = await client.post(
        f"/threads/{tid}/runs/wait",
        json={
            "assistant_id": "chat",
            "input": {
                "messages": [
                    {"role": "human", "content": "Hello from API test!"}
                ]
            },
        },
        timeout=30.0,
    )
    assert r.status_code == 200
    data = r.json()
    assert "messages" in data
    assert len(data["messages"]) >= 2

    r = await client.get(f"/threads/{tid}/state")
    assert r.status_code == 200
    state = r.json()
    assert state is not None
    assert "values" in state

    r = await client.get(f"/threads/{tid}/history")
    assert r.status_code == 200
    assert len(r.json()) > 0


@pytest.mark.asyncio
async def test_run_list(client: AsyncClient):
    r = await client.post("/threads", json={})
    tid = r.json()["thread_id"]

    await client.post(
        f"/threads/{tid}/runs/wait",
        json={
            "assistant_id": "chat",
            "input": {
                "messages": [{"role": "human", "content": "test"}]
            },
        },
        timeout=30.0,
    )

    r = await client.get(f"/threads/{tid}/runs")
    assert r.status_code == 200
    runs = r.json()
    assert len(runs) > 0
    assert runs[0]["status"] == "success"


@pytest.mark.asyncio
async def test_store_lifecycle(client: AsyncClient):
    await client.put(
        "/store/items",
        json={
            "namespace": ["test", "ns"],
            "key": "k1",
            "value": {"data": "hello"},
        },
    )

    r = await client.get("/store/items", params={"namespace": "test.ns", "key": "k1"})
    assert r.status_code == 200
    assert r.json()["value"] == {"data": "hello"}

    r = await client.post(
        "/store/items/search",
        json={"namespace_prefix": ["test"]},
    )
    assert r.status_code == 200
    assert len(r.json()) >= 1

    await client.request(
        "DELETE",
        "/store/items",
        json={"namespace": ["test", "ns"], "key": "k1"},
    )

    r = await client.get("/store/items", params={"namespace": "test.ns", "key": "k1"})
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_graph_endpoint(client: AsyncClient):
    r = await client.get("/assistants/chat/graph")
    assert r.status_code == 200
    data = r.json()
    assert "nodes" in data
    assert "edges" in data


@pytest.mark.asyncio
async def test_schemas_endpoint(client: AsyncClient):
    r = await client.get("/assistants/chat/schemas")
    assert r.status_code == 200
    data = r.json()
    assert "input_schema" in data
    assert "output_schema" in data
