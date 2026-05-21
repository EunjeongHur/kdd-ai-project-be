"""Pydantic models for /decisions. Matches docs/api.yaml."""
from datetime import date, datetime
from enum import Enum
from uuid import UUID
from typing import Optional

from pydantic import BaseModel, Field, model_validator, ConfigDict

from schemas.calculate import CalculateResponse, DecisionPriceSource, Direction, Outcome, ScenarioType


class EmotionType(str, Enum):
    """User's self-reported emotion at the time the decision was made.

    Captured via a 7-option picker on the new-reflection form. Kept structured
    (rather than free-text) so pattern analysis can aggregate consistently.
    """
    CONFIDENT = "confident"
    OPTIMISTIC = "optimistic"
    NEUTRAL = "neutral"
    CAUTIOUS = "cautious"
    ANXIOUS = "anxious"
    FEARFUL = "fearful"
    GREEDY = "greedy"


class DecisionInput(BaseModel):
    """Payload for POST /decisions. Requires quantity or amount."""
    ticker: str = Field(..., examples=["NVDA"])
    scenario_type: ScenarioType
    decision_date: date = Field(..., description="Date the user considered the decision.")
    end_date: Optional[date] = Field(None, description="Optional end date for the window.")
    quantity: Optional[float] = Field(None, gt=0, description="Shares considered.")
    amount: Optional[float] = Field(None, gt=0, description="USD amount considered.")
    notes: Optional[str] = Field(None, max_length=500, description="User notes.")
    title: Optional[str] = Field(
        None,
        max_length=120,
        description=(
            "Short event-style headline (3-7 words). Frontend either pulls from "
            "/parse-decision or accepts a manual override. Null when blank."
        ),
    )
    emotion: Optional[EmotionType] = Field(
        None,
        description="User's emotion at decision time. Null when not selected.",
    )

    @model_validator(mode="after")
    def check_quantity_or_amount(self) -> "DecisionInput":
        if self.quantity is None and self.amount is None:
            raise ValueError("Must provide at least one of quantity or amount.")
        if self.end_date and self.end_date < self.decision_date:
            raise ValueError("end_date cannot be before decision_date.")
        return self


class DecisionReflectionPatch(BaseModel):
    """Body for PATCH /decisions/{id}. Currently only reflection text is patchable.

    Used by the frontend after POST /decisions + POST /reflect: once we have the
    LLM-generated narrative we attach it to the saved decision so list/detail
    views can display it. Pass null to clear an existing reflection.
    """
    reflection: Optional[str] = Field(
        None,
        max_length=2000,
        description="Narrative reflection text (1-2 sentences) from /reflect.",
    )


class Decision(BaseModel):
    """Saved decision model mirroring Supabase decisions table."""
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    user_id: UUID
    ticker: str
    scenario_type: ScenarioType
    decision_date: date
    actual_date_used: Optional[date] = None
    end_date: Optional[date] = None
    quantity: Optional[float] = None
    amount: Optional[float] = None
    decision_price_snapshot: float
    notes: Optional[str] = None
    title: Optional[str] = None
    emotion: Optional[EmotionType] = None
    created_at: datetime
    updated_at: datetime

    # Optional snapshot computation fields stored in DB
    current_price: Optional[float] = None
    diff_amount: Optional[float] = None
    diff_percent: Optional[float] = None
    direction: Optional[Direction] = None
    outcome: Optional[Outcome] = None
    was_decision_correct: Optional[bool] = None
    current_date_snapshot: Optional[date] = None
    decision_price_source: Optional[DecisionPriceSource] = None
    reflection: Optional[str] = None


class DecisionWithCurrent(Decision):
    """Decision entry accompanied by a fresh calculation result."""
    current: CalculateResponse


class DecisionListResponse(BaseModel):
    """Response shape for GET /decisions."""
    items: list[DecisionWithCurrent]
    total: int
