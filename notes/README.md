# AIOps-Pilot 学习笔记

每天完成后整理一份，记录：今天做了什么、关键代码链路、踩的坑、Q&A 复盘。  
面试前翻这个文件夹，能快速还原整个项目的建设过程。

---

## 目录

| 文件 | 日期 | 核心内容 |
|---|---|---|
| [day1.md](./day1.md) | 2026-05-23 | 补全基础骨架（config / models / tools / main.py），打通 `/health` 接口，理解 FastAPI 分层架构 |
| day2.md | - | Milvus 启动 + 文档上传/向量化/存储链路 |
| day3.md | - | RAG 对话 Agent，`/chat` 接口 |
| day4.md | - | Plan-Execute-Replan AIOps 诊断工作流 |
| day5.md | - | SSE 流式输出 + 完整联调 |

---

## 项目整体架构（一句话版）

```
HTTP 入口 (FastAPI)
  ├── /health          → 健康检查（Milvus 是否在线）
  ├── /upload          → 文档上传 → 切分 → 向量化 → 存 Milvus
  ├── /chat            → RAG 对话 Agent（检索知识库 + Qwen LLM）
  └── /aiops           → 自动诊断（Plan → Execute → Replan 循环）
```

## 面试时能说的亮点

- **为什么用 LangGraph 而不是普通链**：LangGraph 是有状态的图，支持循环（Replan），普通 Chain 只能线性执行
- **为什么用 SSE 而不是 WebSocket**：诊断流程是单向推送，SSE 更轻量，HTTP 原生支持，无需握手
- **向量维度为什么是 1024**：DashScope `text-embedding-v4` 模型输出维度是 1024，需要和 Milvus Collection Schema 保持一致
- **为什么 Planner 用 `with_structured_output(Plan)`**：强制 LLM 输出 JSON 格式的步骤列表，避免自由文本导致 Executor 解析失败
