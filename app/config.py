"""
Global configuration module.

All settings are read from environment variables (or a .env file).
Every other module imports `config` from here — this is the single source of truth.
"""

import os
from typing import Any, Dict
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    """
    Application configuration.

    Pydantic-settings automatically reads values from:
    1. Environment variables (highest priority)
    2. A .env file in the working directory
    3. The default values defined below
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",       # silently ignore unknown env vars
        case_sensitive=False, # MILVUS_HOST and milvus_host are the same
    )

    # ------------------------------------------------------------------
    # Application metadata
    # ------------------------------------------------------------------
    app_name: str = "AIOps-Pilot"
    app_version: str = "1.0.0"

    # ------------------------------------------------------------------
    # Alibaba Cloud DashScope (LLM + Embeddings)
    # Set DASHSCOPE_API_KEY in your .env file
    # ------------------------------------------------------------------
    dashscope_api_key: str = Field(default="", alias="DASHSCOPE_API_KEY")

    # Chat model used by the general RAG agent and AIOps nodes
    dashscope_model: str = Field(default="qwen-plus", alias="DASHSCOPE_MODEL")

    # Chat model used specifically for the Plan-Execute-Replan agent
    # (can be the same as dashscope_model or a more powerful model)
    rag_model: str = Field(default="qwen-plus", alias="RAG_MODEL")

    # Embedding model used to convert text chunks into vectors
    dashscope_embedding_model: str = Field(
        default="text-embedding-v4",
        alias="DASHSCOPE_EMBEDDING_MODEL",
    )

    # ------------------------------------------------------------------
    # Milvus (vector database)
    # ------------------------------------------------------------------
    milvus_host: str = Field(default="localhost", alias="MILVUS_HOST")
    milvus_port: int = Field(default=19530, alias="MILVUS_PORT")
    milvus_timeout: int = Field(default=10000, alias="MILVUS_TIMEOUT")  # milliseconds

    # ------------------------------------------------------------------
    # Document chunking (RAG ingestion pipeline)
    # ------------------------------------------------------------------
    # Maximum characters per chunk (used by DocumentSplitterService)
    chunk_max_size: int = Field(default=1000, alias="CHUNK_MAX_SIZE")

    # Overlap between consecutive chunks (preserves context at boundaries)
    chunk_overlap: int = Field(default=200, alias="CHUNK_OVERLAP")

    # ------------------------------------------------------------------
    # MCP (Model Context Protocol) server registry
    # ------------------------------------------------------------------
    # Format: { "server_name": { "transport": "sse", "url": "http://..." } }
    # Leave empty ({}) if no MCP servers are configured.
    # Override by setting MCP_SERVERS as a JSON string in your .env file.
    mcp_servers: Dict[str, Any] = Field(default_factory=dict, alias="MCP_SERVERS")


# ---------------------------------------------------------------------------
# Global singleton — import and use as:  from app.config import config
# ---------------------------------------------------------------------------
config = Config()
