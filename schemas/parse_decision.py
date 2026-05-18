"""Pydantic models for /parse-decision (F-05). See docs/api.yaml ParseDecisionRequest / ParseDecisionResponse."""
from datetime import date
from typing import Optional

from pydantic import BaseModel, Field

from schemas.calculate import ScenarioType


class ParseDecisionRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        examples=["Late March I was thinking about buying 10 shares of Nvidia but didn't"],
        description="Free-text description of a trading decision (or non-decision).",
    )


class ExtractedFields(BaseModel):
    """Fields extracted from natural language. All optional — model returns null
    when it cannot confidently determine a field. Frontend uses these to prefill
    the form; the user always reviews/edits before submission.
    """
    ticker: Optional[str] = None
    scenario_type: Optional[ScenarioType] = None
    decision_date: Optional[date] = None
    quantity: Optional[float] = None
    amount: Optional[float] = None
    title: Optional[str] = Field(
        None,
        description=(
            "Short event-style label generated from the decision context "
            "(3-7 words, Title Case, e.g., 'NVDA Earnings Dip Skipped'). "
            "Used by the reflections list page. Null when input lacks "
            "enough context to summarize."
        ),
    )


class ConfidenceScores(BaseModel):
    """Per-field confidence in [0, 1]. Self-rated by the model. Frontend should
    treat scores below ~0.7 as low-confidence and clear those fields so the user
    fills them manually.
    """
    ticker: float = Field(0.0, ge=0, le=1)
    scenario_type: float = Field(0.0, ge=0, le=1)
    decision_date: float = Field(0.0, ge=0, le=1)
    quantity: float = Field(0.0, ge=0, le=1)
    amount: float = Field(0.0, ge=0, le=1)
    title: float = Field(0.0, ge=0, le=1)


class ParseDecisionResponse(BaseModel):
    extracted: ExtractedFields
    confidence: ConfidenceScores
    ticker_validated: bool = Field(
        ...,
        description=(
            "Whether the extracted ticker exists. Runs format check -> static "
            "NASDAQ index -> yfinance fallback via services.ticker_service. "
            "False when ticker is null, malformed, or not found."
        ),
    )
    reasoning: str = Field(
        ...,
        description="1-2 sentences explaining how each non-null field was derived.",
    )
