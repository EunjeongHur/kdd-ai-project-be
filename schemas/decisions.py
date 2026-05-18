"""Pydantic models for /decisions. Matches docs/api.yaml."""
from datetime import date, datetime
from uuid import UUID
from typing import Optional

from pydantic import BaseModel, Field, model_validator, ConfigDict

from schemas.calculate import CalculateResponse, DecisionPriceSource, Direction, Outcome, ScenarioType


class DecisionInput(BaseModel):
    """Payload for POST /decisions. Requires quantity or amount."""
    ticker: str = Field(..., examples=["NVDA"])
    scenario_type: ScenarioType
    decision_date: date = Field(..., description="Date the user considered the decision.")
    end_date: Optional[date] = Field(None, description="Optional end date for the window.")
    quantity: Optional[float] = Field(None, gt=0, description="Shares considered.")
    amount: Optional[float] = Field(None, gt=0, description="USD amount considered.")
    notes: Optional[str] = Field(None, max_length=500, description="User notes.")

    @model_validator(mode="after")
    def check_quantity_or_amount(self) -> "DecisionInput":
        if self.quantity is None and self.amount is None:
            raise ValueError("Must provide at least one of quantity or amount.")
        if self.end_date and self.end_date < self.decision_date:
            raise ValueError("end_date cannot be before decision_date.")
        return self


class Decision(BaseModel):
    """Saved decision model mirroring Supabase decisions table."""
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    user_id: UUID
    ticker: str
    scenario_type: ScenarioType
    decision_date: date
    actual_date_used: date
    end_date: Optional[date] = None
    quantity: Optional[float] = None
    amount: Optional[float] = None
    decision_price_snapshot: float
    notes: Optional[str] = None
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
