"""Tests for the EchoChatModel."""

from langchain_core.messages import AIMessage, HumanMessage

from langgraph_chat.fake_llm import EchoChatModel


def test_echo_model_responds():
    """Echo model should return an AIMessage."""
    model = EchoChatModel()
    result = model.invoke([HumanMessage(content="test")])
    assert isinstance(result, AIMessage)
    assert "test" in result.content


def test_echo_model_empty_messages():
    """Echo model should handle empty message list."""
    model = EchoChatModel()
    result = model.invoke([])
    assert isinstance(result, AIMessage)
    assert len(result.content) > 0


def test_echo_model_type():
    """Echo model should report its type."""
    model = EchoChatModel()
    assert model._llm_type == "echo-chat-model"
