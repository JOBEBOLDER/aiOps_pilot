# Day 1 — 补全基础骨架，打通第一个接口

**日期**：2026-05-23  
**今日目标**：补全项目缺失的基础文件，启动 FastAPI 服务，成功访问 `/api/health` 接口  
**完成标准**：`python main.py` 启动不报错，`curl http://localhost:9900/api/health` 返回 JSON 响应

---

## 一、今天做了什么

### 发现的问题
接手项目时发现仓库里**缺少五类关键文件**，导致项目根本跑不起来：

| 缺失文件 | 作用 |
|---|---|
| `requirements.txt` | Python 依赖包清单，没有它 pip 不知道装什么 |
| `app/config.py` | 全局配置，所有其他模块都从这里读 API Key、地址等 |
| `app/models/` | Pydantic 请求/响应模型，FastAPI 用它做参数校验 |
| `app/tools/__init__.py` | LangChain 工具函数（知识库检索、获取时间） |
| `main.py` | FastAPI 启动入口，注册路由、管理生命周期 |

### 今天新建的文件

```
requirements.txt                  ← 所有依赖包（fastapi, langgraph, pymilvus 等）
app/config.py                     ← pydantic-settings 读取环境变量
app/models/__init__.py            ← 空文件，标识这是一个 Python 包
app/models/request.py             ← ChatRequest, ClearRequest
app/models/response.py            ← ApiResponse, SessionInfoResponse
app/models/aiops.py               ← AIOpsRequest
app/tools/__init__.py             ← get_current_time, retrieve_knowledge 两个工具
main.py                           ← FastAPI app 入口，Day 1 只挂了 health 路由
```

---

## 二、关键代码解析

### 2.1 `app/config.py` — 全局配置单例

```python
from pydantic_settings import BaseSettings

class Config(BaseSettings):
    app_name: str = "AIOps-Pilot"
    dashscope_api_key: str = Field(default="", alias="DASHSCOPE_API_KEY")
    milvus_host: str = Field(default="localhost", alias="MILVUS_HOST")
    milvus_port: int = Field(default=19530, alias="MILVUS_PORT")
    # ...

config = Config()   # ← 模块底部的单例，其他地方 from app.config import config 就能用
```

**设计要点**：
- `pydantic-settings` 自动从环境变量和 `.env` 文件读取配置，字段名和环境变量名通过 `alias` 对应
- 模块底部只创建一个 `config` 实例，全项目共享（单例模式）
- 优先级：环境变量 > `.env` 文件 > 代码里的默认值

### 2.2 `main.py` — FastAPI 入口

**关键结构**：

```python
# 1. lifespan：管理数据库连接的生命周期
@asynccontextmanager
async def lifespan(app):
    milvus_manager.connect()   # 开机时连 Milvus
    yield                      # ← 服务器在这里运行，等待请求
    milvus_manager.close()     # 关机时断开连接

# 2. 创建 FastAPI app，绑定 lifespan
app = FastAPI(lifespan=lifespan)

# 3. 注册路由（Day 1 只开放 health）
from app.api.health import router as health_router
app.include_router(health_router, prefix="/api")
# 其他路由 Day 2+ 再逐步放开

# 4. 启动 uvicorn
uvicorn.run("main:app", host="0.0.0.0", port=9900, reload=True)
```

### 2.3 `app/tools/__init__.py` — LangChain 工具函数

**关键设计：懒加载（lazy import）**

```python
@tool(response_format="content_and_artifact")
def retrieve_knowledge(query: str):
    # ← 注意：import 放在函数体里，而不是文件顶部
    from app.services.vector_store_manager import vector_store_manager
    docs = vector_store_manager.similarity_search(query, k=3)
    ...
```

为什么要懒加载？`vector_store_manager` 在 import 时会立即连接 Milvus。如果放在文件顶部，导入 `tools` 模块就会触发 Milvus 连接，服务器启动就会报错（Milvus 还没起）。

---

## 三、一次 `/health` 请求的完整链路

```
curl GET http://localhost:9900/api/health
        │
        ▼
main.py — app.include_router(health_router, prefix="/api")
        │  → 匹配到 /api + /health = /api/health
        ▼
app/api/health.py — @router.get("/health") async def health_check()
        │  → 读取 config.app_name, config.app_version
        │  → 调用 milvus_manager.health_check()
        ▼
app/core/milvus_client.py — health_check()
        │  → self._client is None → return False（没连上）
        ▼
health.py — milvus_status = "disconnected"
        │  → overall_status = "unhealthy", status_code = 503
        ▼
返回 JSONResponse(503, {"code": 503, "data": {...}})
```

---

## 四、Day 1 问题复盘（Q&A）

### Q1：`lifespan` 函数在什么时候执行？`yield` 前后分别是什么？

**答**：`lifespan` 是 FastAPI 的生命周期管理器，通过 `@asynccontextmanager` 装饰。

- **`yield` 之前**：服务器启动阶段，在接受第一个请求之前运行（做初始化，比如连接数据库）
- **`yield` 本身**：服务器运行中，一直"暂停"在这里，等待请求
- **`yield` 之后**：服务器关机阶段，收到 Ctrl+C 后运行（做清理，比如断开连接）

绑定方式：`app = FastAPI(lifespan=lifespan)`

---

### Q2：为什么 URL 是 `/api/health` 而不是 `/health`？

**答**：由两段代码叠加决定：

```python
# app/api/health.py
@router.get("/health")                              # 路由定义为 /health

# main.py
app.include_router(health_router, prefix="/api")   # 加了 /api 前缀
```

最终 URL = `/api`（前缀）+ `/health`（路由）= `/api/health`

**设计意义**：所有 API 接口统一收纳在 `/api/` 命名空间，方便 Nginx 等反向代理只代理 `/api/*`，与前端静态资源分开。

---

### Q3：`milvus_manager = MilvusClientManager()` 在哪里执行？是启动时就运行还是被调用时才运行？

**答**：这行在 `app/core/milvus_client.py` 的**最后一行**，属于模块级代码。

**Python 规则**：模块在第一次被 `import` 时，文件里的所有模块级代码都会执行一次。

执行时机链路：
```
main.py 启动
  → from app.api.health import router
    → health.py: from app.core.milvus_client import milvus_manager
      → 此刻执行 milvus_manager = MilvusClientManager()
        → __init__ 只设置 self._client = None（不连接数据库）
  → lifespan 启动
    → milvus_manager.connect()  ← 这才是真正建立网络连接的地方
```

**关键区分**：
- `MilvusClientManager()` = 创建 Python 对象（瞬间，不需要数据库在线）
- `milvus_manager.connect()` = 建立 TCP 网络连接（耗时，Milvus 必须在线）

---

### Q4：现在 503 是因为什么？Milvus 连上了会返回什么？

**现在 503 的原因**：

```python
# milvus_client.py — health_check() 内部
if self._client is None:
    return False   # ← connect() 失败了，_client 还是 None
```

```python
# health.py
if health_data["milvus"]["status"] != "connected":
    status_code = 503   # ← 所以是 503
```

**Milvus 连上后的响应**：

```json
{
  "code": 200,
  "message": "服务运行正常",
  "data": {
    "service": "AIOps-Pilot",
    "version": "1.0.0",
    "status": "healthy",
    "milvus": {
      "status": "connected",
      "message": "Milvus 连接正常"
    }
  }
}
```

---

## 五、今天的关键词卡片

| 关键词 | 用人话解释 | 项目中的真实例子 |
|---|---|---|
| `lifespan` | FastAPI 的开关机回调，`yield` 是服务运行的"中间态" | `main.py` 里连接/断开 Milvus |
| `prefix="/api"` | 给一批路由批量加 URL 命名空间前缀 | `include_router(health_router, prefix="/api")` |
| 模块单例 | Python 文件被 import 时模块级代码运行，全局共享一个实例 | `config = Config()` / `milvus_manager = MilvusClientManager()` |
| 懒加载（lazy import） | 把 import 放进函数体，延迟到真正调用时才执行 | `retrieve_knowledge` 里才 import `vector_store_manager` |
| HTTP 503 | Service Unavailable，服务在线但依赖挂了 | Milvus 未连接时 health 接口返回 503 |
| Pydantic BaseSettings | 自动从环境变量 / .env 文件读配置的设置类 | `app/config.py` 里的 `Config` 类 |

---

## 六、今天踩的坑 / 注意事项

1. **`python main.py 2>&1 &` 后台启动后输出不可见**：改用前台运行或 `uvicorn main:app --port 9900` 看完整日志
2. **503 不是 bug，是正确行为**：服务器本身健康，只是诚实地报告了 Milvus 没连上
3. **模块级代码 vs 函数调用**：`MilvusClientManager()` 和 `connect()` 是两回事，前者在 import 时执行，后者在 lifespan 里执行

---

## 七、明天（Day 2）要做什么

**目标**：用 Docker 启动 Milvus，让 `/health` 返回 200，打通文档上传→切分→向量化→存入 Milvus 的完整链路

**需要准备**：
- [ ] 确认本地有 Docker：`docker --version`
- [ ] 准备好 DashScope API Key（向量化调用阿里云嵌入模型）
- [ ] 在项目根目录创建 `.env` 文件，填入 `DASHSCOPE_API_KEY=sk-xxx`

**Day 2 会解锁的代码**：
```python
# main.py 里取消注释：
from app.api.file import router as file_router
app.include_router(file_router, prefix="/api")
```

涉及的链路：`POST /upload` → `vector_index_service` → `document_splitter_service` → `vector_embedding_service` → `vector_store_manager` → Milvus
