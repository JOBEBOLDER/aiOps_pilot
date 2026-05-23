"""
Inbound request models.

FastAPI automatically validates JSON request bodies against these Pydantic schemas
and returns a 422 Unprocessable Entity if a required field is missing or has the
wrong type — before your handler function is even called.
"""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request body for POST /chat and POST /chat_stream."""

    # Unique identifier for the conversation session.
    # All messages with the same id share the same LangGraph thread,
    # so the agent can remember previous turns.
    id: str = Field(
        default="default",
        description="Session / conversation ID (used as LangGraph thread_id)",
    )

    # The user's message text.
    question: str = Field(
        ...,   # required — no default
        description="User question or instruction",
    )


class ClearRequest(BaseModel):
    """Request body for POST /chat/clear."""

    session_id: str = Field(
        ...,
        description="The session whose history should be erased",
    )
