# AGENTS.md

## Project overview

LangGraph Chat is a Python chat application built with LangGraph and FastAPI, managed with [uv](https://docs.astral.sh/uv/). Default LLM backend is Zhipu AI GLM-5.

## Cursor Cloud specific instructions

### Running the app

Two ways to run:

**1. FastAPI dev server (Web Chat UI):**
```bash
uv run uvicorn langgraph_chat.server:app --host 0.0.0.0 --port 8000 --reload
```
Opens at `http://localhost:8000`.

**2. LangGraph dev server (LangGraph Platform API + Studio):**
```bash
uv run langgraph dev --host 0.0.0.0 --port 2024 --no-browser
```
- API: `http://localhost:2024`
- Studio UI: `https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024`
- API Docs: `http://localhost:2024/docs`

### Key commands

| Action | Command |
|--------|---------|
| Install/sync deps | `uv sync` |
| Lint | `uv run ruff check .` |
| Type check | `uv run mypy langgraph_chat/ --ignore-missing-imports` |
| Tests | `uv run pytest -v` |
| FastAPI dev server | `uv run uvicorn langgraph_chat.server:app --host 0.0.0.0 --port 8000 --reload` |
| LangGraph dev server | `uv run langgraph dev --host 0.0.0.0 --port 2024 --no-browser` |

### Non-obvious caveats

- **uv must be installed first.** If not on PATH, install with `curl -LsSf https://astral.sh/uv/install.sh | sh` and ensure `$HOME/.local/bin` is on PATH.
- The default LLM is **Zhipu AI GLM-5** (`base_url=https://open.bigmodel.cn/api/paas/v4`, `model=glm-5`). Override via `OPENAI_API_BASE` / `OPENAI_MODEL` env vars.
- The app falls back to `EchoChatModel` when `OPENAI_API_KEY` is not set, so tests and dev server work without API keys. When an API key is present, two echo-specific tests are automatically skipped.
- `langgraph.json` defines the graph entry point for `langgraph dev`. The graph variable is `langgraph_chat/graph.py:graph`.
- **LangSmith tracing** is enabled via `LANGSMITH_API_KEY`, `LANGSMITH_TRACING=true`, and `LANGSMITH_PROJECT` in `.env`. Traces appear in the LangSmith dashboard.
- `uv.lock` is committed. Run `uv lock` after changing dependencies in `pyproject.toml`.
