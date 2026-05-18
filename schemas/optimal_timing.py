"""Pydantic models for /optimal-timing (F-02). Hindsight-only data."""
from datetime import date

from pydantic import BaseModel, Field, model_validator


class OptimalTimingRequest(BaseModel):
    """Request body for POST /optimal-timing.

    Scans the full [start_date, end_date] window for the single highest and
    lowest adjusted-close prices, enabling the frontend to present factual
    hindsight data for pattern-recognition purposes.
    """

    ticker: str = Field(..., examples=["AAPL"], description="Uppercase US ticker.")
    start_date: date = Field(..., description="First day of the holding period (inclusive).")
    end_date: date = Field(..., description="Last day of the holding period (inclusive).")

    @model_validator(mode="after")
    def _check_date_order(self) -> "OptimalTimingRequest":
        if self.end_date < self.start_date:
            raise ValueError("`end_date` must be on or after `start_date`.")
        return self


class PricePoint(BaseModel):
    """A single date/price observation."""

    date: date
    price: float = Field(..., description="Split/dividend-adjusted close price (USD).")


class OptimalTimingResponse(BaseModel):
    """Response from POST /optimal-timing.

    All data is hindsight-based.  The `summary_message` field is a factual
    statement about the period's price range; it is NOT investment advice and
    MUST NOT be presented as a prescriptive action or recommendation.
    """

    ticker: str
    start_date: date
    end_date: date
    best_buy: PricePoint = Field(
        ...,
        description=(
            "Trading day with the lowest adjusted-close price in the period. "
            "Buying here would have minimised cost — purely factual."
        ),
    )
    best_sell: PricePoint = Field(
        ...,
        description=(
            "Trading day with the highest adjusted-close price in the period. "
            "Selling here would have maximised proceeds — purely factual."
        ),
    )
    max_return_percent: float = Field(
        ...,
        description=(
            "((best_sell.price - best_buy.price) / best_buy.price) * 100. "
            "Represents the theoretical maximum return within the period."
        ),
    )
    summary_message: str = Field(
        ...,
        description=(
            "Factual, non-prescriptive summary of the period's price extremes. "
            "Never contains actionable advice."
        ),
    )
    data_points: int = Field(
        ...,
        description="Number of trading days found in the requested window.",
    )
    split_adjusted: bool = Field(True, description="Always true (yfinance Adjusted Close).")
