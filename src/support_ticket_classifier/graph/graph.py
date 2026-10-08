from groq import APITimeoutError, RateLimitError
from langgraph.graph import StateGraph, END
from loguru import logger

from support_ticket_classifier.graph.state import TicketState
from support_ticket_classifier.graph.nodes import (
    pii_redact_node,
    check_prompt_injection_node,
    classify_ticket_node,
    response_validation_node,
    fallback_node,
    cost_log_node,
    CLASSIFY_RETRY_POLICY,
)
from support_ticket_classifier.graph.edges import route_after_validate
from support_ticket_classifier.schema import SAFE_CLASSIFICATION


def build_graph():
    builder = StateGraph(TicketState)

    builder.add_node("pii_redact", pii_redact_node)
    builder.add_node("check_prompt_injection", check_prompt_injection_node)
    builder.add_node("classify", classify_ticket_node, retry_policy=CLASSIFY_RETRY_POLICY)
    builder.add_node("validate", response_validation_node)
    builder.add_node("fallback", fallback_node)
    builder.add_node("cost_log", cost_log_node)

    builder.set_entry_point("pii_redact")
    builder.add_edge("pii_redact", "check_prompt_injection")
    builder.add_edge("check_prompt_injection", "classify")
    builder.add_edge("classify", "validate")
    builder.add_conditional_edges(
        "validate",
        route_after_validate,
        {"classify": "classify", "fallback": "fallback", "cost_log": "cost_log"},
    )
    builder.add_edge("fallback", "cost_log")
    builder.add_edge("cost_log", END)

    return builder.compile()


graph = build_graph()


def classify_ticket(raw_ticket: str, channel: str = "web_form") -> TicketState:
    """
    Run the graph end-to-end and return a fully populated TicketState.

    LangGraph returns only state fields written during the run, so the raw result
    may be sparse depending on the execution path. Re-validating it through
    TicketState fills missing fields with their defaults and gives callers a
    consistent state shape.

    If classification retries are exhausted due to a transient error, the
    exception propagates from graph.invoke() before the fallback node can run.
    In that case, return the same safe defaults used by the fallback node.
    Attempts and cost information cannot be recovered because no state is
    persisted after the failed run.
    """

    try:
        result = graph.invoke({"raw_ticket": raw_ticket, "channel": channel})
    except (RateLimitError, APITimeoutError) as e:
        logger.error(
            f"Classification crashed after exhausting transient-error retries: {e}. "
            "Falling back to a safe default; attempts/cost for this run are not recoverable."
        )
        return TicketState(
            raw_ticket=raw_ticket,
            channel=channel,
            classification=SAFE_CLASSIFICATION,
            validation_status="failed",
            fallback_used=True,
            error=str(e),
        )

    return TicketState(**result)
