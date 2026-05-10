"""Pydantic models for /tickers/* endpoints. Matches docs/api.yaml v0.2."""
from typing import Optional

from pydantic import BaseModel, Field


class TickerValidationResponse(BaseModel):
    """Response from GET /tickers/validate.

    When `valid` is False, all metadata fields are None.
    When `valid` is True, `name` is populated; `exchange` may still be None
    if the upstream source didn't return it.
    """
    ticker: str = Field(..., description="Normalized uppercase ticker.")
    valid: bool
    name: Optional[str] = None
    exchange: Optional[str] = None


class TickerSearchItem(BaseModel):
    """One autocomplete suggestion. Search only returns existing tickers, so
    `name` is always populated. EQUITY and ETF quote types only.
    """
    ticker: str
    name: str
    exchange: Optional[str] = None


class TickerSearchResponse(BaseModel):
    items: list[TickerSearchItem]
