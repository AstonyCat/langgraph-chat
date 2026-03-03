"""Chat state definitions for the LangGraph graph."""

from typing import Annotated

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class ChatState(TypedDict):
    """State schema for the chat graph.

    The messages field uses the add_messages reducer so that new messages
    are appended rather than replacing the list.
    """

    messages: Annotated[list, add_messages]
