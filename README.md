# langgraph-chat

A chat application built with [LangGraph](https://github.com/langchain-ai/langgraph) and FastAPI. Managed with [uv](https://docs.astral.sh/uv/).

## Quick Start

```bash
# Install dependencies (including dev group)
uv sync

# Run the development server
uv run uvicorn langgraph_chat.server:app --host 0.0.0.0 --port 8000 --reload
```

Open http://localhost:8000 in your browser.

## LLM Configuration

By default the app runs in **echo/demo mode**. To use a real LLM (默认接入智谱 AI), set:

```bash
export OPENAI_API_KEY=your-zhipu-api-key
# Base URL defaults to Zhipu AI, override for other OpenAI-compatible APIs:
# export OPENAI_API_BASE=https://open.bigmodel.cn/api/paas/v4
# Model defaults to glm-4-flash:
# export OPENAI_MODEL=glm-4-flash
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