# langgraph-chat

A chat application built with [LangGraph](https://github.com/langchain-ai/langgraph) and FastAPI. Managed with [uv](https://docs.astral.sh/uv/). Default LLM: Zhipu AI GLM-5.

## Quick Start

```bash
# Install dependencies
uv sync

# Run the FastAPI chat UI
uv run uvicorn langgraph_chat.server:app --host 0.0.0.0 --port 8000 --reload

# Or run the LangGraph dev server (Platform API + Studio)
uv run langgraph dev --host 0.0.0.0 --port 2024 --no-browser
```

- Chat UI: http://localhost:8000
- LangGraph API: http://localhost:2024
- LangGraph Studio: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024

## LLM Configuration

By default the app runs in **echo/demo mode**. To use a real LLM (默认接入智谱 AI GLM-5), set:

```bash
export OPENAI_API_KEY=your-zhipu-api-key
# Base URL defaults to Zhipu AI, override for other OpenAI-compatible APIs:
# export OPENAI_API_BASE=https://open.bigmodel.cn/api/paas/v4
# Model defaults to glm-5:
# export OPENAI_MODEL=glm-5
```

## LangSmith Tracing

```bash
export LANGSMITH_API_KEY=lsv2_pt_...
export LANGSMITH_TRACING=true
export LANGSMITH_PROJECT=langgraph-chat
```

## Development

```bash
# Lint
uv run ruff check .

# Type check
uv run mypy langgraph_chat/ --ignore-missing-imports

# Run tests
uv run pytest -v
```