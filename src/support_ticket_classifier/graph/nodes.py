
from support_ticket_classifier.graph.state import TicketState
from support_ticket_classifier.schema import SAFE_CLASSIFICATION
from support_ticket_classifier.utils import ( 
    redact_pii, check_prompt_injection, 
    classify_ticket_with_function_calling,
    validate_classification
)
from support_ticket_classifier.utils.structured_outputs import (
    DEFAULT_SYSTEM_PROMPT, CONSERVATIVE_SYSTEM_PROMPT
)

from groq import APITimeoutError, RateLimitError
from langgraph.types import RetryPolicy
from pydantic import ValidationError
from loguru import logger

from support_ticket_classifier.settings import settings
from support_ticket_classifier.utils.cost_calculator import calculate_cost, count_tokens


# Transient errors (rate limits, timeouts) are retried in-place by LangGraph 
CLASSIFY_RETRY_POLICY = RetryPolicy(max_attempts=2, retry_on=(RateLimitError, APITimeoutError))

# initial cost info for a ticket with no attempts yet
ZERO_TOKENS = {"input_tokens": 0, "output_tokens": 0}

def pii_redact_node(state: TicketState) -> dict:
    """Redact PII from the ticket text."""
    result = redact_pii(state.raw_ticket)

    if result.pii_detected:
        logger.info("PII detected and redacted.")

    return {
        "redacted_ticket": result.redacted_text,
        "pii_detected": result.pii_detected
    }


def check_prompt_injection_node(state: TicketState) -> dict:
    """Check for prompt injection in the ticket text."""
    result = check_prompt_injection(state.raw_ticket)

    if result.is_injection:
        logger.warning(f"Prompt injection detected: {result.detected_pattern}")
        return {
            "classification": SAFE_CLASSIFICATION,
            "injection_blocked": True,
            "error": f"Injection detected: {result.detected_pattern}",
        }

    return {"injection_blocked": False}


def classify_ticket_node(state: TicketState) -> dict:
    """
    Classify the ticket using the normal prompt first, then a simpler prompt on retries.
    Tracks current and cumulative token usage in cost_info across attempts.
    """
    if state.injection_blocked:
        return {}

    ticket_text = state.redacted_ticket or state.raw_ticket
    attempt = state.attempts
    system_prompt = DEFAULT_SYSTEM_PROMPT if attempt == 0 else CONSERVATIVE_SYSTEM_PROMPT

    input_tokens = count_tokens(system_prompt) + count_tokens(ticket_text)
    classification = None
    output_tokens = 0
    error = None

    try:
        classification = classify_ticket_with_function_calling(
            ticket_text=ticket_text, system_prompt=system_prompt
        )
        output_tokens = count_tokens(classification.model_dump_json())
    except (ValidationError, ValueError) as e:
        logger.error(f"Classification failed on attempt {attempt + 1}: {e}")
        error = str(e)

    prior_cumulative = (state.cost_info or {}).get("cumulative", ZERO_TOKENS)
    current_cost = {"input_tokens": input_tokens, "output_tokens": output_tokens}
    cumulative_cost = {
        "input_tokens": prior_cumulative["input_tokens"] + input_tokens,
        "output_tokens": prior_cumulative["output_tokens"] + output_tokens,
    }

    return {
        "classification": classification,
        "error": error,
        "attempts": attempt + 1,
        "cost_info": {"current": current_cost, "cumulative": cumulative_cost},
    }


def response_validation_node(state: TicketState) -> dict:
    """Validate the classification result."""
    if state.injection_blocked:
        return {"validation_status": "blocked"}

    classification = state.classification
    if classification is None:
        return {"validation_status": "failed", "error": "No classification to validate."}

    validation_result = validate_classification(classification)
    if not validation_result.is_valid:
        logger.warning(f"Validation failed: {validation_result.error_details}")
        return {
            "validation_status": "failed",
            "error": f"Validation failed: {validation_result.error_details}",
        }

    return {
        "validation_status": "passed",
        "classification": validation_result.validated_classification
    }


def fallback_node(state: TicketState) -> dict:
    """Retry attempts exhausted — skip the LLM, use a safe default, and flag for human review."""
    logger.error(
        f"Classification exhausted after {state.attempts} attempts; "
        f"routing to fallback. Last error: {state.error}"
    )

    return {
        "classification": SAFE_CLASSIFICATION,
        "validation_status": "failed",
        "fallback_used": True,
    }


def cost_log_node(state: TicketState) -> TicketState:
    """
    Log this ticket's current and cumulative cost.
    Only the cumulative total is added to SessionCostTracker to avoid double-counting.
    """
    tokens = state.cost_info or {}
    current_tokens = tokens.get("current", ZERO_TOKENS)
    cumulative_tokens = tokens.get("cumulative", ZERO_TOKENS)

    current_cost = calculate_cost(
        settings.GROQ_GENERATION_MODEL_NAME,
        current_tokens["input_tokens"],
        current_tokens["output_tokens"],
        record_to_session=False,
    )
    cumulative_cost = calculate_cost(
        settings.GROQ_GENERATION_MODEL_NAME,
        cumulative_tokens["input_tokens"],
        cumulative_tokens["output_tokens"],
        record_to_session=True,    # add to the session tracker the aggregated cost across attempts for this ticket 
    )

    if settings.LOG_COSTS:
        logger.info(
            f"Cost — model: {cumulative_cost.model} | "
            f"last attempt in:{current_cost.input_tokens} out:{current_cost.output_tokens} "
            f"cost:${current_cost.total_cost_usd:.6f} | "
            f"ticket total in:{cumulative_cost.input_tokens} out:{cumulative_cost.output_tokens} "
            f"cost:${cumulative_cost.total_cost_usd:.6f} | "
            f"attempts: {state.attempts} | fallback_used: {state.fallback_used}"
        )

    return {
        "cost_info": {
            "model": cumulative_cost.model,
            "current": {
                "input_tokens": current_cost.input_tokens,
                "output_tokens": current_cost.output_tokens,
                "cost_usd": current_cost.total_cost_usd,
            },
            "cumulative": {
                "input_tokens": cumulative_cost.input_tokens,
                "output_tokens": cumulative_cost.output_tokens,
                "cost_usd": cumulative_cost.total_cost_usd,
            },
        },
    }