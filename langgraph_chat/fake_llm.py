"""A simple echo chat model for development and testing without API keys."""

from typing import Any

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class EchoChatModel(BaseChatModel):
    """A deterministic chat model that echoes back user input.

    Useful for development and testing when no LLM API key is available.
    """

    model_name: str = "echo-dev"

    @property
    def _llm_type(self) -> str:
        return "echo-chat-model"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        last_message = messages[-1] if messages else None
        if last_message is None:
            content = "Hello! I'm the LangGraph Chat echo bot. Send me a message!"
        else:
            user_text = str(last_message.content)
            content = (
                f"[Echo Bot] You said: \"{user_text}\"\n\n"
                f"I'm running in demo mode (no OPENAI_API_KEY set). "
                f"Set the OPENAI_API_KEY environment variable to use a real LLM."
            )

        message = AIMessage(content=content)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])
