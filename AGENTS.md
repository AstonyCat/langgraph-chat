# AGENTS.md

## Project overview

LangGraph Chat is a Python chat application built with LangGraph and FastAPI. It features a LangGraph state graph for conversation management and a web UI served via FastAPI.

## Cursor Cloud specific instructions

### Running the app

```bash
python3 -m uvicorn langgraph_chat.server:app --host 0.0.0.0 --port 8000 --reload
```

The app runs on `http://localhost:8000`. Without `OPENAI_API_KEY`, it uses a built-in echo model for development/testing.

### Key commands

| Action | Command |
|--------|---------|
| Install deps | `pip install -e ".[dev]"` |
| Lint | `python3 -m ruff check .` |
| Type check | `python3 -m mypy langgraph_chat/ --ignore-missing-imports` |
| Tests | `python3 -m pytest -v` |
| Dev server | `python3 -m uvicorn langgraph_chat.server:app --host 0.0.0.0 --port 8000 --reload` |

### Non-obvious caveats

- Tools are invoked via `python3 -m <tool>` rather than bare commands (e.g. `ruff`, `pytest`) because pip installs into the system Python and does not always update `PATH` for script entry points in this environment.
- The app gracefully falls back to `EchoChatModel` when `OPENAI_API_KEY` is not set, so all tests and the dev server work without any API keys.
- `pytest-asyncio` is configured with `asyncio_mode = "auto"` in `pyproject.toml`, so async test functions don't need the `@pytest.mark.asyncio` decorator individually but it doesn't hurt to include it.
