"""Pydantic models for /calculate (F-01). Matches docs/api.yaml v0.3."""
from datetime import date
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator


class ScenarioType(str, Enum):
    """What the user did — or didn't do. See docs/api.yaml ScenarioType."""
    NO_BUY = "no_buy"                  # Considered buying but didn't.
    NO_SELL = "no_sell"                # Held a position; considered selling but didn't.
    SOLD_TOO_EARLY = "sold_too_early"  # Sold and price kept rising.


class DecisionPriceSource(str, Enum):
    """Where `decision_price` came from in the response."""
    USER = "user"          # Derived from user-supplied amount/quantity.
    YFINANCE = "yfinance"  # Adjusted close on actual_date_used.


class Direction(str, Enum):
    """7-case combination of scenario + price direction. See docs/conventions.md §13."""
    MISSED_GAIN = "missed_gain"            # no_buy + price up
    AVOIDED_LOSS = "avoided_loss"          # no_buy + price down
    KEPT_GAIN = "kept_gain"                # no_sell + price up
    ENDURED_LOSS = "endured_loss"          # no_sell + price down
    CUT_SHORT_GAIN = "cut_short_gain"      # sold_too_early + price up
    WELL_TIMED_EXIT = "well_timed_exit"    # sold_too_early + price down
    NEUTRAL = "neutral"                    # |diff_percent| < 0.5%


class Outcome(str, Enum):
    """Frontend-friendly bucket derived from Direction."""
    FAVORABLE = "favorable"      # avoided_loss | kept_gain | well_timed_exit
    UNFAVORABLE = "unfavorable"  # missed_gain | endured_loss | cut_short_gain
    NEUTRAL = "neutral"


class CalculateRequest(BaseModel):
    """Request body for POST /calculate. At least one of `quantity` / `amount`
    is required; both may be supplied (e.g. for an actual transaction with a
    known price). When both are present, `decision_price` in the response is
    derived as `amount / quantity` (user's truth) instead of the yfinance close.
    """
    ticker: str = Field(..., examples=["AAPL"], description="Uppercase US ticker.")
    scenario_type: ScenarioType
    decision_date: date = Field(..., description="Date the user was making the decision.")
    quantity: Optional[float] = Field(
        None, gt=0, examples=[10],
        description="Shares (fractional allowed).",
    )
    amount: Optional[float] = Field(
        None, gt=0, examples=[1500.00],
        description=(
            "USD amount. If given alongside `quantity`, treated as the actual "
            "transaction value — `decision_price = amount / quantity`."
        ),
    )

    @model_validator(mode="after")
    def _check_at_least_one(self) -> "CalculateRequest":
        if self.quantity is None and self.amount is None:
            raise ValueError(
                "Provide `quantity`, `amount`, or both."
            )
        return self


class CalculateResponse(BaseModel):
    """Response from POST /calculate. Matches docs/api.yaml CalculateResponse."""
    ticker: str
    scenario_type: ScenarioType
    decision_date: date = Field(..., description="The date the user supplied.")
    actual_date_used: date = Field(
        ...,
        description="Trading day actually used; differs from decision_date when "
                    "that date was a market holiday or weekend.",
    )
    decision_price: float = Field(
        ...,
        description=(
            "Reference price at decision_date. When the request supplied both "
            "`quantity` and `amount`, this is `amount / quantity` (user's "
            "actual fill). Otherwise it's the yfinance adjusted close on "
            "`actual_date_used`. See `decision_price_source`."
        ),
    )
    decision_price_source: DecisionPriceSource = Field(
        ...,
        description="Whether decision_price came from the user or yfinance.",
    )
    current_price: float = Field(..., description="Adjusted close on current_date.")
    current_date: date = Field(..., description="Trading day used for current_price.")
    diff_amount: float = Field(
        ...,
        description="Profit/loss the position would represent today. "
                    "Sign reflects price move (positive when price went up).",
    )
    diff_percent: float = Field(
        ...,
        description="((current_price - decision_price) / decision_price) * 100.",
    )
    direction: Direction
    outcome: Outcome
    was_decision_correct: Optional[bool] = Field(
        None,
        description="True if outcome favorable, False if unfavorable, null if neutral.",
    )
    split_adjusted: bool = Field(True, description="Always true (yfinance Adjusted Close).")
