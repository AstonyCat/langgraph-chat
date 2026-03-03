# langgraph-chat

基于 [LangGraph](https://github.com/langchain-ai/langgraph) + FastAPI 构建的聊天应用，使用 [uv](https://docs.astral.sh/uv/) 管理项目，默认接入智谱 AI GLM-5 大模型。

## 项目结构

```
langgraph-chat/
├── langgraph_chat/
│   ├── __init__.py
│   ├── state.py              # LangGraph 状态定义 (ChatState)
│   ├── graph.py              # LangGraph 图定义 (chatbot node)
│   ├── fake_llm.py           # Echo 模型 (无 API Key 时的回退)
│   ├── server.py             # FastAPI 聊天服务器 (普通 + SSE 流式)
│   └── api/                  # 自定义 LangGraph Platform API 实现
│       ├── app.py            # API 主应用 (40+ 端点)
│       ├── models.py         # Pydantic v2 数据模型
│       ├── storage.py        # 内存存储后端
│       ├── graph_registry.py # 图执行引擎
│       └── routes/           # 路由模块
│           ├── assistants.py # Assistants CRUD
│           ├── threads.py    # Threads CRUD + State + History
│           ├── runs.py       # Runs: wait/stream/background
│           ├── store.py      # KV Store
│           └── system.py     # /ok, /info
├── static/
│   └── index.html            # 聊天前端 (支持普通/SSE 切换)
├── tests/                    # 测试 (20 个)
├── langgraph.json            # LangGraph 平台配置
├── pyproject.toml            # 项目配置 + 依赖
├── uv.lock                   # 锁文件
├── .env.example              # 环境变量示例
└── README.md
```

## 本地启动详细步骤

### 前置条件

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/) 包管理器

### 第一步：安装 uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env    # 将 uv 加入 PATH
```

### 第二步：克隆仓库

```bash
git clone https://github.com/AstonyCat/langgraph-chat.git
cd langgraph-chat
```

### 第三步：安装依赖

```bash
uv sync
```

这会自动创建 `.venv` 虚拟环境并安装所有依赖（包括 dev 工具）。

### 第四步：配置环境变量

```bash
cp .env.example .env
```

编辑 `.env` 文件，填入你的 API Key：

```env
# 智谱 AI API Key (必填，用于接入 GLM-5 大模型)
OPENAI_API_KEY=your-zhipu-api-key

# 以下为默认值，可不修改
OPENAI_API_BASE=https://open.bigmodel.cn/api/paas/v4
OPENAI_MODEL=glm-5

# LangSmith 追踪 (可选)
LANGSMITH_API_KEY=lsv2_pt_...
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=langgraph-chat
```

> **注意：** 不设置 `OPENAI_API_KEY` 也可以启动，会使用内置的 Echo 模式。

### 第五步：启动服务

项目提供三种启动方式：

#### 方式一：聊天 Web UI（推荐上手体验）

```bash
uv run uvicorn langgraph_chat.server:app --host 0.0.0.0 --port 8000 --reload
```

打开浏览器访问 http://localhost:8000，即可看到聊天界面。
- 支持 **普通模式**（一次性返回）和 **流式 SSE 模式**（逐字符推送）
- 页面右上角按钮切换两种模式

#### 方式二：LangGraph Platform API（官方 CLI）

```bash
uv run langgraph dev --host 0.0.0.0 --port 2024 --no-browser
```

- API 地址：http://localhost:2024
- Studio 可视化：https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
- API 文档：http://localhost:2024/docs

#### 方式三：自定义 LangGraph API Server

```bash
uv run uvicorn langgraph_chat.api.app:app --host 0.0.0.0 --port 2024 --reload
```

这是从零实现的 LangGraph Platform API 兼容服务器，提供 40+ 端点。
- API 地址：http://localhost:2024
- API 文档：http://localhost:2024/docs

### 第六步：验证

```bash
# 健康检查
curl http://localhost:8000/api/health

# 发送非流式消息
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "你好"}'

# 发送流式 SSE 消息
curl -N -X POST http://localhost:8000/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "你好"}'
```

## API 端点一览

### 聊天服务器 (port 8000)

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 聊天 Web UI |
| POST | `/api/chat` | 非流式聊天 |
| POST | `/api/chat/stream` | SSE 流式聊天 |
| GET | `/api/health` | 健康检查 |

### LangGraph API Server (port 2024)

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/ok` | 健康检查 |
| GET | `/info` | 服务器信息 |
| POST | `/assistants` | 创建 Assistant |
| POST | `/threads` | 创建 Thread |
| POST | `/threads/{id}/runs/wait` | 执行图 (等待结果) |
| POST | `/threads/{id}/runs/stream` | 执行图 (SSE 流式) |
| GET | `/threads/{id}/state` | 获取线程状态 |
| GET | `/threads/{id}/history` | 获取对话历史 |
| PUT | `/store/items` | 存储 KV 数据 |
| GET | `/docs` | OpenAPI 文档 |

## 开发

```bash
# 代码检查
uv run ruff check .

# 类型检查
uv run mypy langgraph_chat/ --ignore-missing-imports

# 运行测试
uv run pytest -v
```

## 技术栈

- **LLM**: 智谱 AI GLM-5 (通过 OpenAI 兼容 API)
- **框架**: LangGraph + FastAPI
- **前端**: 原生 HTML/JS (支持 SSE)
- **包管理**: uv
- **追踪**: LangSmith
- **测试**: pytest + httpx
- **代码质量**: ruff + mypy
