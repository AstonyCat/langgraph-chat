"""LangGraph ReAct agent with tool calling."""

from __future__ import annotations

import os
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from langgraph_chat.state import ChatState
from langgraph_chat.tools import all_tools


def _get_llm() -> BaseChatModel:
    """Return a chat model with tools bound."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            base_url=os.environ.get(
                "OPENAI_API_BASE",
                "https://open.bigmodel.cn/api/paas/v4",
            ),
            model=os.environ.get("OPENAI_MODEL", "glm-5"),
            temperature=0.7,
        )

    from langgraph_chat.fake_llm import EchoChatModel

    return EchoChatModel()


def agent_node(state: ChatState) -> dict[str, Any]:
    """LLM decides whether to call tools or respond directly."""
    llm = _get_llm()
    llm_with_tools = llm.bind_tools(all_tools)
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def build_graph() -> Any:
    """Build ReAct agent: agent → tools_condition → tools ↔ agent."""
    builder = StateGraph(ChatState)

    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(tools=all_tools))

    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")

    return builder.compile()


graph = build_graph()


def chat(
    user_message: str, history: list[Any] | None = None
) -> AIMessage:
    """Send a message and get a response."""
    messages = list(history) if history else []
    messages.append(HumanMessage(content=user_message))

    result: dict[str, Any] = graph.invoke({"messages": messages})
    last_msg: AIMessage = result["messages"][-1]
    return last_msg
