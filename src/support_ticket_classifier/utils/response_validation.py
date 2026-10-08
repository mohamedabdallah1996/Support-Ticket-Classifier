"""
LLM response validation using Pydantic.

# PRODUCTION NOTE: In a real system, add custom business-rule validators
# (e.g., CRITICAL priority must always require_human_review), log validation
# failures to a monitoring system (Datadog, Sentry), and alert on high
# failure rates which could indicate a model degradation event.
"""

from dataclasses import dataclass, field
from pydantic import ValidationError
from typing import Any

from support_ticket_classifier.schema import TicketClassification


@dataclass
class ValidationResult:
    is_valid: bool
    validated_classification: TicketClassification | None = None
    error_details: list[str] = field(default_factory=list)


def validate_classification(raw: Any) -> ValidationResult:
    """Validate LLM output against the TicketClassification schema."""
    if isinstance(raw, TicketClassification):
        data = raw.model_dump()
    elif isinstance(raw, dict):
        data = raw
    else:
        return ValidationResult(
            is_valid=False,
            error_details=[f"Unexpected type: {type(raw).__name__}"],
        )

    try:
        classification = TicketClassification.model_validate(data)
    except ValidationError as e:
        errors = [
            f"{'.'.join(str(x) for x in err['loc'])}: {err['msg']}"
            for err in e.errors()
        ]
        return ValidationResult(
            is_valid=False,
            error_details=errors,
        )

    if classification.confidence_score < 0.5 and not classification.require_human_review:
        return ValidationResult(
            is_valid=False,
            error_details=["Low confidence score should set require_human_review=True"],
        )

    return ValidationResult(
        is_valid=True,
        validated_classification=classification,
    )
