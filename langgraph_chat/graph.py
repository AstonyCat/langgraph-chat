"""LangGraph chat graph definition."""

from __future__ import annotations

import os
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph

from langgraph_chat.state import ChatState


def _get_llm() -> BaseChatModel:
    """Return a chat model based on environment configuration.

    Uses OpenAI if OPENAI_API_KEY is set, otherwise falls back to a
    deterministic echo model for development/testing.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0.7,
        )

    from langgraph_chat.fake_llm import EchoChatModel

    return EchoChatModel()


def chatbot_node(state: ChatState) -> dict[str, Any]:
    """Process the conversation and generate a response."""
    llm = _get_llm()
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


def build_graph() -> Any:
    """Build and return the compiled chat graph."""
    graph_builder = StateGraph(ChatState)
    graph_builder.add_node("chatbot", chatbot_node)
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_edge("chatbot", END)
    return graph_builder.compile()


graph = build_graph()


def chat(user_message: str, history: list[Any] | None = None) -> AIMessage:
    """Send a message and get a response.

    Args:
        user_message: The user's message text.
        history: Optional list of previous messages for context.

    Returns:
        The AI's response message.
    """
    from langchain_core.messages import HumanMessage

    messages = list(history) if history else []
    messages.append(HumanMessage(content=user_message))

    result: dict[str, Any] = graph.invoke({"messages": messages})
    last_msg: AIMessage = result["messages"][-1]
    return last_msg
