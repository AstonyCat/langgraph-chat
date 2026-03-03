# langgraph-chat

A chat application built with [LangGraph](https://github.com/langchain-ai/langgraph) and FastAPI.

## Quick Start

```bash
# Install dependencies (with dev tools)
pip install -e ".[dev]"

# Run the development server
python3 -m uvicorn langgraph_chat.server:app --host 0.0.0.0 --port 8000 --reload
```

Open http://localhost:8000 in your browser.

## LLM Configuration

By default the app runs in **echo/demo mode**. To use a real LLM, set:

```bash
export OPENAI_API_KEY=sk-...
export OPENAI_MODEL=gpt-4o-mini  # optional, defaults to gpt-4o-mini
```

## Development

```bash
# Lint
python3 -m ruff check .

# Type check
python3 -m mypy langgraph_chat/ --ignore-missing-imports

# Run tests
python3 -m pytest -v
```