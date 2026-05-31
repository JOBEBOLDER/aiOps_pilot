# Day 1 — 项目骨架 · FastAPI 启动 · `/health` 接口

> **日期**：2026-05-23  
> **目标**：补全缺失的基础文件，成功启动 FastAPI 服务，打通第一个接口 `GET /api/health`  
> **完成标准**：`curl http://localhost:9900/api/health` 有响应（即使是 503 也算成功）

---

## 一、项目是什么

**AIOps-Pilot** — 一个智能运维诊断系统。

核心思想：当线上出现告警时，不需要人工去查日志、查指标、写报告，而是让 AI Agent 自动完成这一切。

### 系统三条主线

```
HTTP 请求入口 (FastAPI)
      │
      ├── /chat, /chat_stream  ────→ RAG 对话 Agent（问答 + 检索知识库）
      │
      ├── /aiops               ────→ Plan-Execute-Replan（自动诊断告警）
      │       ├── Planner   （把告警拆成执行步骤）
      │       ├── Executor  （逐步执行，调用工具/MCP）
      │       └── Replanner （评估结果，继续还是生成报告）
      │
      └── /upload, /index_directory ──→ 文档索引管道（运维手册 → 向量 → Milvus）
```

### 技术栈

| 层 | 技术 |
|---|---|
| Web 框架 | FastAPI + Uvicorn |
| LLM | 阿里云 Qwen（通过 DashScope / langchain-qwq）|
| Agent 编排 | LangGraph（状态机） + LangChain |
| 向量数据库 | Milvus |
| 工具协议 | MCP（Model Context Protocol）|
| 日志 | Loguru |

---

## 二、今天做了什么

### 发现问题：仓库缺少关键文件

接手项目时发现这些文件不存在，导致项目根本跑不起来：

| 缺失文件 | 作用 |
|---|---|
| `requirements.txt` | 依赖包清单 |
| `app/config.py` | 全局配置（API Key、Milvus 地址等）|
| `app/models/` | Pydantic 请求/响应数据结构 |
| `app/tools/__init__.py` | 工具函数（检索知识库、获取时间）|
| `main.py` | FastAPI 启动入口 |

### 今天创建的文件（按依赖顺序）

```
requirements.txt
app/config.py
app/models/__init__.py
app/models/request.py
app/models/response.py
app/models/aiops.py
app/tools/__init__.py
main.py
```

---

## 三、关键文件逐一讲解

### `app/config.py` — 全局配置

**职责**：所有模块的配置从这里统一读取，是整个项目的"配置单一数据源"。

**关键设计**：用 `pydantic-settings` 的 `BaseSettings`，自动从环境变量 / `.env` 文件读取配置。

```python
class Config(BaseSettings):
    app_name: str = "AIOps-Pilot"
    dashscope_api_key: str = Field(default="", alias="DASHSCOPE_API_KEY")
    milvus_host: str = Field(default="localhost", alias="MILVUS_HOST")
    milvus_port: int = Field(default=19530, alias="MILVUS_PORT")
    chunk_max_size: int = Field(default=1000, alias="CHUNK_MAX_SIZE")
    # ... 等等

config = Config()   # ← 全局单例，其他文件 from app.config import config
```

**为什么这么设计**：
- 不把 API Key 硬编码在代码里（安全）
- 一处改配置，全局生效
- 换生产环境只需要改 `.env`，不需要动代码

---

### `app/models/` — 请求/响应数据结构

**职责**：定义接口的"合同"——规定客户端必须发什么、服务器会返回什么。

FastAPI 用这些 Pydantic 模型自动做：
1. 请求校验（字段缺失或类型错误 → 自动返回 422）
2. 生成 Swagger 文档（访问 `/docs` 可视化）

```python
# request.py
class ChatRequest(BaseModel):
    id: str = "default"       # 会话 ID（有默认值，可选）
    question: str             # 用户问题（必填，无默认值）

# response.py
class ApiResponse(BaseModel):
    status: str               # "success" | "error"
    message: str
    data: Optional[Any] = None
```

---

### `app/tools/__init__.py` — 工具函数

**职责**：提供给 AI Agent 可以"调用"的能力。Agent 会根据任务需要，自主决定是否调用工具。

**两个工具**：
- `get_current_time`：返回当前时间（简单，无外部依赖）
- `retrieve_knowledge`：在 Milvus 里检索相关文档（复杂，依赖向量数据库）

**关键设计：懒加载（Lazy Import）**

```python
@tool(response_format="content_and_artifact")
def retrieve_knowledge(query: str) -> Tuple[str, List]:
    # 注意：import 在函数体内部，不在文件顶层
    from app.services.vector_store_manager import vector_store_manager
    docs = vector_store_manager.similarity_search(query, k=3)
    ...
```

**为什么这么做**：`vector_store_manager` 在被 import 时会自动连接 Milvus。  
如果放在文件顶层，启动服务器时 Milvus 必须在线，否则直接崩溃。  
放在函数体内，只有工具被真正调用时才连接，启动和数据库解耦了。

---

### `main.py` — FastAPI 启动入口

**职责**：
1. 创建 FastAPI 实例
2. 注册路由（今天只注册了 health）
3. 配置 lifespan（开机连 Milvus，关机断开）
4. 启动 Uvicorn 服务器

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── 开机 ──
    milvus_manager.connect()      # 建立数据库连接
    yield                          # ← 服务器在这里运行，处理所有请求
    # ── 关机 ──
    milvus_manager.close()        # 释放连接

app = FastAPI(lifespan=lifespan)

app.include_router(health_router, prefix="/api")   # /health → /api/health
```

**Day 1 的策略**：其他路由暂时注释掉，只开放 `/health`。  
原因：chat/aiops 路由依赖 Milvus 和 API Key，还没就绪；分阶段开放便于定位问题。

---

## 四、一次请求的完整主链

> **场景**：`curl http://localhost:9900/api/health`

```
curl 发出 GET /api/health
        │
        ▼
  Uvicorn（ASGI 服务器）接收 HTTP 请求
        │
        ▼
  FastAPI 路由匹配
        │  main.py: app.include_router(health_router, prefix="/api")
        │  health.py: @router.get("/health")
        │  匹配到 → 调用 health_check() 函数
        ▼
  app/api/health.py — health_check()
        │
        ├── 读取 config.app_name, config.app_version
        │        └── app/config.py
        │
        └── 调用 milvus_manager.health_check()
                 └── app/core/milvus_client.py
                          └── self._client is None → return False
        │
        ▼
  health.py 判断：milvus status = "disconnected"
        └── overall_status = "unhealthy"
        └── status_code = 503
        │
        ▼
  return JSONResponse(503, {...})
        │
        ▼
  curl 收到 503 响应
```

**实际返回结果**：
```json
{
    "code": 503,
    "message": "服务不可用",
    "data": {
        "service": "AIOps-Pilot",
        "version": "1.0.0",
        "status": "unhealthy",
        "milvus": {
            "status": "disconnected",
            "message": "Milvus 连接异常"
        }
    }
}
```

503 不是错误，是**正确的**预期行为——服务器在跑，如实报告了依赖组件的状态。

---

## 五、Day 1 复盘 Q&A

### Q1：`lifespan` 什么时候执行？`yield` 前后分别代表什么？

**执行时机**：
- `yield` 之前 → 服务器启动后、接受第一个请求之前（"开机初始化"）
- `yield` 期间 → 服务器正常运行，处理所有请求
- `yield` 之后 → 收到关闭信号后（Ctrl+C）执行（"关机清理"）

**为什么用 lifespan 而不是 `@app.on_event`**：  
FastAPI 已废弃 `on_event`，lifespan 把开机/关机逻辑放在一起，更清晰，也保证关机清理一定会执行。

---

### Q2：为什么 URL 是 `/api/health` 而不是 `/health`？

两段代码叠加决定的：

```python
# app/api/health.py
@router.get("/health")             # 路由本身是 /health

# main.py
app.include_router(health_router, prefix="/api")   # 加了 /api 前缀
```

最终 = `/api` + `/health` = `/api/health`

**设计价值**：所有 API 统一在 `/api/` 下，和静态资源、前端页面分开，方便 Nginx 反向代理配置。

---

### Q3：`milvus_manager = MilvusClientManager()` 在哪里执行？启动时还是调用时？

```python
# app/core/milvus_client.py 最后一行
milvus_manager = MilvusClientManager()
```

**执行时机**：这行是模块级代码，当这个文件第一次被 `import` 时就执行。

**关键区分**：
- `MilvusClientManager()` — 只是**创建 Python 对象**（`__init__` 仅设置 `self._client = None`，极快）
- `milvus_manager.connect()` — 才是**建立网络连接**（在 lifespan 里调用，可能失败）

创建对象 ≠ 连接数据库。这就是服务器没有 Milvus 也能启动的原因。

---

### Q4：为什么现在 503？Milvus 连上后返回什么？

**现在是 503 的原因链**：

```
lifespan 里 milvus_manager.connect() 失败（Milvus 没运行）
    └── self._client 依然是 None

health_check() 里：
    if self._client is None: return False

health.py 里：
    milvus_healthy = False
    → status = "disconnected"
    → status_code = 503
```

**Milvus 正常后**：

```
lifespan 连接成功 → self._client = MilvusClient(...)

health_check()：
    self._client is not None
    → connections.list_connections() 成功
    → return True

health.py：
    → status = "connected"
    → status_code = 200
    → 返回 {"code": 200, "message": "服务运行正常", ...}
```

---

## 六、关键词汇卡片

| 词 | 人话解释 | 在项目里的体现 |
|---|---|---|
| **FastAPI Router** | 路由表，收到什么 URL 就调哪个函数 | `router = APIRouter()` + `@router.get(...)` |
| **prefix** | 给一批路由批量加 URL 前缀 | `app.include_router(health_router, prefix="/api")` |
| **lifespan** | 服务器开机/关机的回调 | `@asynccontextmanager async def lifespan(app)` |
| **BaseSettings** | 自动从环境变量读配置的 Pydantic 类 | `class Config(BaseSettings)` in `config.py` |
| **模块单例** | 模块级变量第一次 import 时创建，全局共享 | `milvus_manager = MilvusClientManager()` |
| **懒加载** | 把 import 放进函数体，用到时才执行 | `retrieve_knowledge` 里的 `from app.services...` |
| **HTTP 503** | Service Unavailable，服务在线但依赖组件不可用 | Milvus 没连上时 `/health` 返回 503 |
| **Pydantic BaseModel** | 定义数据结构并自动校验类型 | `ChatRequest`, `ApiResponse` 等 model 类 |

---

## 七、面试时可以这样讲

> **"说说你这个项目的技术选型和架构？"**

"这是一个基于 FastAPI + LangGraph 实现的 AIOps 自动诊断系统。整体分三层：
- API 层用 FastAPI，通过 SSE 支持流式输出；
- Agent 层用 LangGraph 实现 Plan-Execute-Replan 状态机，Planner 把告警拆成步骤，Executor 调工具执行，Replanner 评估并决定继续还是生成报告；
- 知识层用 Milvus 存向量，文档上传后经过分块、向量化存入，诊断时通过语义检索找相关经验。"

> **"为什么 FastAPI 要用 lifespan 而不是 on_event？"**

"FastAPI 已经废弃了 `on_event`，推荐用 `lifespan` 上下文管理器。好处是把开机和关机逻辑放在一起，代码更清晰，而且能保证服务关闭时清理代码一定会执行，不会因为异常跳过。"

> **"你的服务启动时没有 Milvus 会崩溃吗？"**

"不会。lifespan 里的连接失败被 try/except 包住了，只记录 warning 日志，不抛异常。服务器正常启动，但 `/health` 接口会返回 503，告诉调用方数据库当前不可用。这样运维团队能通过健康检查接口监控依赖状态，而不是通过服务崩溃来发现问题。"

---

## 八、明天的目标（Day 2）

- [ ] 用 Docker 启动 Milvus standalone
- [ ] 让 `/health` 从 503 变成 200
- [ ] 创建 `.env` 文件，配置 `DASHSCOPE_API_KEY`
- [ ] 上传一个 `.md` 文档，验证"切分 → 向量化 → 存入 Milvus"全链路
- [ ] 打通 `/upload` 和 `/index_directory` 接口
