"""
Token cost calculation and session-level cost tracking.

# PRODUCTION NOTE: In a real system, persist cost data to a database per
# user/org/request, set up budget alerts, expose a cost dashboard, and
# integrate with billing systems. Use OpenAI's usage API for reconciliation.
"""

import tiktoken
from dataclasses import dataclass

from support_ticket_classifier.settings import settings


PRICING: dict[str, dict[str, float]] = {
    "openai/gpt-oss-20b": {
        "input": 0.000075,
        "output": 0.00030,
    },
}

@dataclass
class CostInfo:
    model: str
    input_tokens: int
    output_tokens: int
    input_cost_usd: float
    output_cost_usd: float
    total_cost_usd: float


class SessionCostTracker:
    """Accumulates cost across multiple LLM calls in a session."""
    def __init__(self):
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0
        self._total_cost_usd: float = 0.0
        self._call_count: int = 0

    def record(self, cost_info: CostInfo) -> None:
        self._total_input_tokens += cost_info.input_tokens
        self._total_output_tokens += cost_info.output_tokens
        self._total_cost_usd += cost_info.total_cost_usd
        self._call_count += 1

    @property
    def summary(self) -> dict:
        return {
            "calls": self._call_count,
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "total_cost_usd": round(self._total_cost_usd, 6),
        }


# Module-level session tracker (reset per process)
session_tracker = SessionCostTracker()

def count_tokens(text: str) -> int:
    """Approximate token count using cl100k_base."""
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def calculate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    record_to_session: bool = True,
) -> CostInfo:
    """Calculate cost for an LLM call and optionally record to session tracker."""
    pricing = PRICING.get(model, PRICING[settings.GROQ_GENERATION_MODEL_NAME])  
    input_cost = (input_tokens / 1000) * pricing["input"]
    output_cost = (output_tokens / 1000) * pricing["output"]
    total = input_cost + output_cost

    info = CostInfo(
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        input_cost_usd=round(input_cost, 6),
        output_cost_usd=round(output_cost, 6),
        total_cost_usd=round(total, 6),
    )

    if record_to_session:
        session_tracker.record(info)

    return info

