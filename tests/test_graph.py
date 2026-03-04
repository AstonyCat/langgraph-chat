"""Tests for the LangGraph chat graph."""

import os

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from langgraph_chat.graph import build_graph, chat

_has_api_key = bool(os.environ.get("OPENAI_API_KEY"))


def test_build_graph_compiles():
    """Graph should compile without errors."""
    compiled = build_graph()
    assert compiled is not None


def test_chat_returns_ai_message():
    """chat() should return an AIMessage."""
    result = chat("Hello")
    assert isinstance(result, AIMessage)
    assert len(str(result.content)) > 0


@pytest.mark.skipif(_has_api_key, reason="Real LLM active; echo mode not used")
def test_chat_echo_mode():
    """In echo mode (no API key), response should contain the user's text."""
    result = chat("testing echo")
    assert "testing echo" in str(result.content)


def test_graph_invoke_with_history():
    """Graph should accept messages with history."""
    compiled = build_graph()
    history = [
        HumanMessage(content="Hi"),
        AIMessage(content="Hello!"),
    ]
    result = compiled.invoke(
        {"messages": history + [HumanMessage(content="How are you?")]}
    )
    assert len(result["messages"]) > 0
    last = result["messages"][-1]
    assert isinstance(last, AIMessage)
