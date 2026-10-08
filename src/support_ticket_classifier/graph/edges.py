from typing import Literal

from support_ticket_classifier.graph.state import TicketState
from support_ticket_classifier.settings import settings


def route_after_validate(state: TicketState) -> Literal["classify", "fallback", "cost_log"]:
    """
    Route the ticket based on validation status and remaining attempts.

    Passed or blocked tickets proceed to cost logging. Failed tickets retry
    classification while attempts remain; otherwise, they fall back.
    """
    if state.validation_status in ("passed", "blocked"):
        return "cost_log"

    if state.attempts < settings.MAX_CLASSIFICATION_ATTEMPTS:
        return "classify"

    return "fallback"
