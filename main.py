"""
Application entry point.

Responsibilities:
  1. Create the FastAPI app instance
  2. Register all API routers under the /api prefix
  3. Connect to Milvus on startup and disconnect on shutdown (lifespan)
  4. Start the Uvicorn ASGI server

Why a lifespan handler instead of @app.on_event?
-------------------------------------------------
FastAPI deprecated @app.on_event in favour of the `lifespan` context manager
(introduced in Starlette 0.20).  Using lifespan keeps startup and shutdown logic
in one place and guarantees that cleanup runs even when the server crashes.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.config import config


# ---------------------------------------------------------------------------
# Lifespan: connect / disconnect Milvus around the server lifetime
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Code before `yield`  → runs once on startup
    Code after  `yield`  → runs once on shutdown
    """
    # ---- STARTUP ----
    logger.info(f"Starting {config.app_name} v{config.app_version}")

    try:
        from app.core.milvus_client import milvus_manager
        milvus_manager.connect()
        logger.info("Milvus connected successfully")
    except Exception as e:
        # Log the error but do NOT crash the server.
        # The /health endpoint will report Milvus as disconnected,
        # which is the correct observable behaviour for ops teams.
        logger.warning(f"Milvus connection failed at startup: {e}")

    logger.info("Server is ready to accept requests")
    yield  # ← server is running while we wait here

    # ---- SHUTDOWN ----
    logger.info("Shutting down...")
    try:
        from app.core.milvus_client import milvus_manager
        milvus_manager.close()
        logger.info("Milvus connection closed")
    except Exception as e:
        logger.warning(f"Error closing Milvus connection: {e}")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title=config.app_name,
    version=config.app_version,
    description="Autonomous AIOps agent — Plan · Execute · Replan",
    lifespan=lifespan,
)

# Allow any origin during development.
# In production, restrict `allow_origins` to your frontend's domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Router registration
#
# Day 1: only the health router is active.
# We add the others one day at a time so we can verify each layer works
# before building on top of it.
#
# Day 2 → uncomment: file router  (upload + index pipeline)
# Day 3 → uncomment: chat router  (RAG conversational agent)
# Day 4 → uncomment: aiops router (Plan-Execute-Replan diagnostic)
# ---------------------------------------------------------------------------

from app.api.health import router as health_router
app.include_router(health_router, prefix="/api")

# Day 2: file upload & indexing
from app.api.file import router as file_router
app.include_router(file_router, prefix="/api")

# Day 3: RAG chat agent
# from app.api.chat import router as chat_router
# app.include_router(chat_router, prefix="/api")

# Day 4: AIOps diagnostic workflow
# from app.api.aiops import router as aiops_router
# app.include_router(aiops_router, prefix="/api")


# ---------------------------------------------------------------------------
# Dev server
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=9900,
        reload=True,   # auto-restart when source files change
        log_level="info",
    )
