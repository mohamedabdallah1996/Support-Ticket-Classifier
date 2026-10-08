"""
Structured output from LLM — the single place in the codebase that makes
classification LLM calls. Both approaches accept a system_prompt argument
so the caller (graph node or retry wrapper) controls prompt content without
touching LLM wiring.

# PRODUCTION NOTE: In a real system, prefer function-calling (approach 1) as
# it is more reliable and model-native. Fall back to JSON mode only for models
# that don't support tool/function calling. Always validate the output with
# Pydantic regardless of which approach you use.
"""

import os
import json

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from support_ticket_classifier.settings import settings
from support_ticket_classifier.schema import TicketClassification


os.environ["GROQ_API_KEY"] = settings.GROQ_API_KEY

DEFAULT_SYSTEM_PROMPT = """
You are an expert customer support ticket classifier.

Classify the customer's message using the provided `TicketClassification` schema.

Rules:

1. **Issue category:** Identify the customer's primary issue.

   * `order_issue`: placing, modifying, or canceling an order
   * `payment_issue`: failed payments, declined transactions, or charges
   * `delivery_issue`: shipping, tracking, delays, or missing deliveries
   * `product_issue`: defective, damaged, incorrect, or poor-quality products
   * `account_issue`: login, password, profile, or account access
   * `refund_request`: explicit refund or reimbursement requests
   * `technical_issue`: website, app, or system problems
   * `other`: does not fit the above

2. **Department:** Assign the team best equipped to resolve the primary issue:
   `sales`, `customer_support`, `billing`, `shipping`, or `technical`.

3. **Priority:**

   * `low`: minor issue or general question
   * `medium`: normal customer-impacting issue
   * `high`: significant impact, repeated failure, or financial concern
   * `urgent`: critical financial, security, safety, or severe service issue
     Do not increase priority solely because the customer is angry.

4. **Sentiment:** Classify the customer's emotional tone as `positive`, `neutral`, or `negative`.

5. **Confidence:** Return a score from `0.0` to `1.0` reflecting confidence in the classification.

6. **Human review:** Set to `true` if the request is ambiguous, confidence is low, involves serious financial/security/safety concerns, or explicitly requests human escalation.

7. **Reasoning:** Provide one concise sentence explaining the classification.

Focus on the primary issue, do not invent missing information, Keep confidence low and set requires_human_review=True if unsure.
"""

CONSERVATIVE_SYSTEM_PROMPT = """
You are a cautious customer support ticket classifier retrying after a prior
classification attempt failed to produce a valid result.

Classify the customer's message using the provided `TicketClassification` schema,
favoring the simplest, most defensible interpretation:

* If the primary issue is not immediately obvious, use `issue_category=other`.
* Prefer `priority=low` or `medium` unless the ticket clearly states a severe
  financial, security, or safety impact.
* Keep `confidence_score` low (below 0.5) whenever there is any ambiguity.
* Set `require_human_review=True` whenever confidence is low or the ticket is
  unclear in any way.

Return a complete, schema-valid object even if the classification is uncertain —
do not omit fields or leave the response unstructured.
"""

def classify_ticket_with_function_calling(
    ticket_text: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    model_name: str = settings.GROQ_GENERATION_MODEL_NAME,
) -> TicketClassification:
    """
    Classify a support ticket using function calling.
    
    Approach 1: Function-calling (recommended)
      - Use when: the model supports tool/function calling.
      - Pros: Model is explicitly told the schema; more reliable JSON adherence.
      - Cons: Slightly higher token overhead for the schema definition.
    """
    llm = ChatGroq(model_name=model_name, temperature=0, max_retries=0)
    structured_llm = llm.with_structured_output(TicketClassification)

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "Classify this support ticket:\n\n{ticket_text}")
    ])

    chain = prompt | structured_llm
    return chain.invoke({"ticket_text": ticket_text})


def classify_ticket_with_json(
    ticket_text: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    model_name: str = settings.GROQ_GENERATION_MODEL_NAME,
) -> TicketClassification:
    """
    Classify a support ticket using JSON output.

    Approach 2: JSON output (fallback)
      - Use when: the model does not support tool/function calling, 
        or you need the model to freely structure its output as JSON 
        and don't pin to a specific schema via tool calling.
      - Pros: Works with any model that can output JSON.
      - Cons: Model may not adhere to schema; requires post-processing and validation.
    """
    schema_json = json.dumps(TicketClassification.model_json_schema(), indent=2)
    # prevent langchain from interpreting schema braces as template variables
    schema_escaped = schema_json.replace("{", "{{").replace("}", "}}")   
    full_system_prompt = f"{system_prompt}\n\nReturn ONLY valid JSON matching this schema:\n{schema_escaped}"

    llm = ChatGroq(
        model_name=model_name, 
        temperature=0, 
        max_retries=0,
        model_kwargs={"response_format": {"type": "json_object"}}
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", full_system_prompt),
        ("user", "Classify this support ticket:\n\n{ticket_text}")
    ])

    chain = prompt | llm
    response = chain.invoke({"ticket_text": ticket_text})

    raw_content = json.loads(response.content)
    return TicketClassification.model_validate(raw_content)


if __name__ == "__main__":

    ticket_text = "I was charged twice for order #9981. Please refund immediately!"

    print("=== Approach 1: Function-calling ===")
    result1 = classify_ticket_with_function_calling(ticket_text)
    print(result1.model_dump_json(indent=2))

    print("\n=== Approach 2: JSON mode ===")
    result2 = classify_ticket_with_json(ticket_text)
    print(result2.model_dump_json(indent=2))