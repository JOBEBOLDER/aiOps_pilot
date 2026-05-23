"""Request model for the AIOps diagnostic endpoint."""

from typing import Optional
from pydantic import BaseModel, Field


class AIOpsRequest(BaseModel):
    """Request body for POST /aiops."""

    # Optional session ID.  When provided, the diagnostic run is tied to
    # that LangGraph thread so intermediate state can be inspected later.
    session_id: Optional[str] = Field(
        default="default",
        description="Session ID for the diagnostic run",
    )
