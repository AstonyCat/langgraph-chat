# AGENTS.md

## Project overview

LangGraph Chat is a Python chat application built with LangGraph and FastAPI, managed with [uv](https://docs.astral.sh/uv/).

## Cursor Cloud specific instructions

### Running the app

```bash
uv run uvicorn langgraph_chat.server:app --host 0.0.0.0 --port 8000 --reload
```

The app runs on `http://localhost:8000`. Without `OPENAI_API_KEY`, it uses a built-in echo model for development/testing.

### Key commands

| Action | Command |
|--------|---------|
| Install/sync deps | `uv sync` |
| Lint | `uv run ruff check .` |
| Type check | `uv run mypy langgraph_chat/ --ignore-missing-imports` |
| Tests | `uv run pytest -v` |
| Dev server | `uv run uvicorn langgraph_chat.server:app --host 0.0.0.0 --port 8000 --reload` |

### Non-obvious caveats

- **uv must be installed first.** If not on PATH, install with `curl -LsSf https://astral.sh/uv/install.sh | sh` and ensure `$HOME/.local/bin` is on PATH.
- The app gracefully falls back to `EchoChatModel` when `OPENAI_API_KEY` is not set, so all tests and the dev server work without any API keys.
- The default LLM backend is **智谱 AI (Zhipu AI)** with `base_url=https://open.bigmodel.cn/api/paas/v4` and `model=glm-4-flash`. Override via `OPENAI_API_BASE` and `OPENAI_MODEL` env vars for other OpenAI-compatible providers.
- `pytest-asyncio` is configured with `asyncio_mode = "auto"` in `pyproject.toml`, so async test functions don't need the `@pytest.mark.asyncio` decorator individually but it doesn't hurt to include it.
- `uv.lock` is committed and should be kept up to date. Run `uv lock` after changing dependencies in `pyproject.toml`.
