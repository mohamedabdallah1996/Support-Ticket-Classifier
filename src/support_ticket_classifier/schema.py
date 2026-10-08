from pydantic import BaseModel, Field
from enum import Enum


class IssueCategory(str, Enum):
    ORDER = "order_issue"
    PAYMENT = "payment_issue"
    DELIVERY = "delivery_issue"
    PRODUCT = "product_issue"
    ACCOUNT = "account_issue"
    REFUND = "refund_request"
    TECHNICAL = "technical_issue"
    OTHER = "other"

class Department(str, Enum):
    SALES = "sales"
    CUSTOMER_SUPPORT = "customer_support"
    BILLING = "billing"
    SHIPPING = "shipping"
    TECHNICAL = "technical"

class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"

class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"

class TicketClassification(BaseModel):
    issue_category: IssueCategory = Field(..., description="The category of the support ticket issue.")
    assigned_department: Department = Field(..., description="The department to which the ticket is assigned.")
    priority: Priority = Field(..., description="The priority level of the support ticket.")
    user_sentiment: Sentiment = Field(..., description="The sentiment of the user's message, indicating their emotional state.")
    confidence_score: float = Field(ge=0.0, le=1.0, description="The confidence score of the classification model's prediction, ranging from 0 to 1.")
    reasoning: str = Field(description="The reasoning behind the classification, explaining in one line why the ticket was categorized in a certain way.")
    require_human_review: bool = Field(..., description="Indicates whether the ticket requires human review based on the classification and confidence score.")


SAFE_CLASSIFICATION = TicketClassification(
    issue_category=IssueCategory.OTHER,
    assigned_department=Department.CUSTOMER_SUPPORT,
    priority=Priority.MEDIUM,
    user_sentiment=Sentiment.NEUTRAL,
    confidence_score=0.0,
    reasoning="Automatic fallback: classification failed after all retries",
    require_human_review=True
)
