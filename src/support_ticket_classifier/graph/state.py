from pydantic import BaseModel
from typing import Literal

from support_ticket_classifier.schema import TicketClassification

class TicketState(BaseModel):
    """LangGraph state passed between nodes."""
    raw_ticket: str
    channel: str = "web_form"

    redacted_ticket: str | None = None
    classification: TicketClassification | None = None

    validation_status: Literal["passed", "failed", "blocked"] | None = None
    cost_info: dict | None = None
    error: str | None = None

    pii_detected: bool = False
    # prompt_version: str | None = None
    injection_blocked: bool = False

    attempts: int = 0
    fallback_used: bool = False
