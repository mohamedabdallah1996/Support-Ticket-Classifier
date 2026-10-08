"""
PII detection and redaction using regex.

# PRODUCTION NOTE: In a real system, extend this with an ML-based recognizer
# (e.g., presidio with a spaCy model) to catch name and address PII that regex
# cannot reliably detect. Log every redaction event to an audit trail, and
# store the original text encrypted at rest separately from the redacted copy.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from loguru import logger

import phonenumbers
from email_validator import EmailNotValidError, validate_email
from luhn_validator import validate

from support_ticket_classifier.settings import settings

# ---------------------------------------------------------------------------
# Candidate extraction
# ---------------------------------------------------------------------------

# extract candidates that look like email addresses, to be validated by email-validator
_EMAIL_CANDIDATE_RE = re.compile(
    r"(?<![\w.!#$%&'*+/=?^`{|}~-])"
    r"[A-Za-z0-9.!#$%&'*+/=?^`{|}~-]+"
    r"@"
    r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
    r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)+"
    r"(?![\w.-])",
)


# extract candidates to be sent to the credit-card validator. #
# Matches 13–19 digits with optional spaces or hyphens:
#   4111111111111111
#   4111 1111 1111 1111
#   4111-1111-1111-1111
_CREDIT_CARD_CANDIDATE_RE = re.compile(
    r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)",
)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _is_valid_email(value: str) -> bool:
    """Return True when `value` is a valid email address."""
    try:
        validate_email(
            value,
            check_deliverability=False,
        )
        return True
    except EmailNotValidError:
        return False


def _find_emails(text: str) -> list[re.Match[str]]:
    """Return email candidates that pass email-validator validation."""
    return [
        match
        for match in _EMAIL_CANDIDATE_RE.finditer(text)
        if _is_valid_email(match.group())
    ]


def _find_phone_numbers(
    text: str,
    default_region: str | None = settings.DEFAULT_REGION_PII,
) -> list[phonenumbers.PhoneNumberMatch]:
    """
    Find valid phone numbers in arbitrary text.

    E.164 numbers can be detected without a default region. Local/national
    numbers require `default_region`, such as "US", "GB", or "EG".
    """
    matches: list[phonenumbers.PhoneNumberMatch] = []

    for match in phonenumbers.PhoneNumberMatcher(
        text,
        default_region,
        leniency=phonenumbers.Leniency.STRICT_GROUPING,
    ):
        if phonenumbers.is_valid_number(match.number):
            matches.append(match)

    return matches


def _find_credit_cards(text: str) -> list[re.Match[str]]:
    """Return credit-card candidates that pass Luhn validation."""
    matches: list[re.Match[str]] = []

    for match in _CREDIT_CARD_CANDIDATE_RE.finditer(text):
        candidate = match.group()

        # luhn-validator accepts numbers containing spaces/hyphens.
        if validate(candidate):
            matches.append(match)

    return matches


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

@dataclass
class RedactionResult:
    redacted_text: str
    detected_entity_types: list[str] = field(default_factory=list)
    pii_detected: bool = False


def redact_pii(
    text: str,
    *,
    default_phone_region: str | None = None,
) -> RedactionResult:
    """Detect and redact supported PII from text."""
    spans = _collect_pii_spans(
        text,
        default_phone_region=default_phone_region,
    )

    if not spans:
        logger.info("PII scan complete — no PII detected.")
        return RedactionResult(redacted_text=text)

    spans = _remove_overlapping_spans(spans)
    redacted_text, detected_types = _apply_redactions(text, spans)

    logger.info(
        f"PII redaction complete — {len(spans)} item(s) redacted, types: {', '.join(detected_types)}"
    )

    return RedactionResult(
        redacted_text=redacted_text,
        detected_entity_types=detected_types,
        pii_detected=True,
    )


def _collect_pii_spans(
    text: str,
    *,
    default_phone_region: str | None = None,
) -> list[tuple[int, int, str]]:
    """Collect candidate PII spans from all supported detectors."""
    spans: list[tuple[int, int, str]] = []

    for match in _find_emails(text):
        spans.append((match.start(), match.end(), "[EMAIL REDACTED]"))

    for match in _find_credit_cards(text):
        spans.append((match.start(), match.end(), "[CREDIT CARD REDACTED]"))

    for match in _find_phone_numbers(text, default_phone_region):
        spans.append((match.start, match.end, "[PHONE REDACTED]"))

    return spans


def _remove_overlapping_spans(
    spans: list[tuple[int, int, str]],
) -> list[tuple[int, int, str]]:
    """Remove overlapping spans, keeping the longest match."""
    spans = sorted(
        spans,
        key=lambda span: (span[0], -(span[1] - span[0])),
    )

    result: list[tuple[int, int, str]] = []

    for span in spans:
        start, end, _ = span

        if any(
            start < existing_end and end > existing_start
            for existing_start, existing_end, _ in result
        ):
            continue

        result.append(span)

    return result


def _apply_redactions(
    text: str,
    spans: list[tuple[int, int, str]],
) -> tuple[str, list[str]]:
    """Replace detected PII with redaction labels without logging its values."""
    detected_types: list[str] = []
    result = text

    for start, end, label in sorted(
        spans,
        key=lambda span: span[0],
        reverse=True,
    ):
        entity_type = label.removeprefix("[").removesuffix(" REDACTED]")

        if entity_type not in detected_types:
            detected_types.append(entity_type)

        logger.warning(
            f"PII detected — type: {entity_type} | span: [{start}:{end}] | replaced with: {label}"
        )

        result = result[:start] + label + result[end:]

    return result, detected_types


if __name__ == "__main__":

    samples = [
        "Hi, my email is john.smith@example.com and phone is 555-867-5309.",
        "I was charged twice. Card: 4111 1111 1111 1111. Call me at (800) 555-0199.",
        "My number is +1 415 555 2671 and my card 4539 1488 0343 6467 was billed.",
        "No personal data here, just a regular support question about my order.",
    ]

    for sample in samples:
        r = redact_pii(sample)
        print(f"Original : {sample}")
        print(f"Redacted : {r.redacted_text}")
        print(f"Detected : {r.detected_entity_types}")
        print()
