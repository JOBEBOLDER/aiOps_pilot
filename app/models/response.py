"""
Outbound response models.

Using typed response models lets FastAPI:
- Auto-generate accurate OpenAPI / Swagger docs
- Serialize Python objects to JSON automatically
- Strip fields the client should not see (via response_model_exclude)
"""

from typing import Any, List, Optional
from pydantic import BaseModel


class ApiResponse(BaseModel):
    """Generic API response wrapper used by simple operation endpoints."""

    status: str          # "success" | "error"
    message: str
    data: Optional[Any] = None


class SessionInfoResponse(BaseModel):
    """Response for GET /chat/session/{session_id}."""

    session_id: str
    message_count: int

    # List of historical messages, each shaped as:
    # { "role": "user" | "assistant", "content": "...", "timestamp": "..." }
    history: List[Any] = []
