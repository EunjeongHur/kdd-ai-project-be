"""POST /calculate — F-01 opportunity-cost computation.

See docs/api.yaml CalculateRequest / CalculateResponse and docs/conventions.md
for the contract this implements.
"""
import re
from datetime import date

from fastapi import APIRouter, HTTPException

from schemas.calculate import (
    CalculateRequest,
    CalculateResponse,
)
from services.calculate_service import (
    compute_diffs,
    derive_direction,
    derive_outcome,
    derive_was_correct,
)
from services.yfinance_service import get_market_data

# Inline ticker normalization + format check.
# When the ticker PR (services/ticker_service.py) merges, swap these for
# the shared helpers from there.
_TICKER_PATTERN = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")


def _normalize_ticker(raw: str) -> str:
    t = raw.strip().upper()
    t = re.sub(r"[\s/.]+", "-", t)
    return t.strip("-")


def _is_valid_ticker_format(ticker: str) -> bool:
    return bool(_TICKER_PATTERN.fullmatch(ticker))


router = APIRouter(prefix="/calculate", tags=["calculate"])


@router.post("", response_model=CalculateResponse)
def calculate_opportunity_cost(req: CalculateRequest) -> CalculateResponse:
    # --- Ticker normalization + format check ---
    ticker = _normalize_ticker(req.ticker)
    if not _is_valid_ticker_format(ticker):
        raise HTTPException(
            status_code=422,
            detail={
                "error": {
                    "code": "INVALID_TICKER_FORMAT",
                    "message": (
                        "Ticker must be 1-10 characters, start with a letter, "
                        "and contain only letters, digits, '.' or '-'."
                    ),
                }
            },
        )

    # --- Future-date guard ---
    if req.decision_date > date.today():
        raise HTTPException(
            status_code=422,
            detail={
                "error": {
                    "code": "DATE_IN_FUTURE",
                    "message": "decision_date cannot be in the future.",
                }
            },
        )

    # --- Market data lookup (raises 422/502 internally on failure) ---
    market = get_market_data(ticker, req.decision_date)

    # --- Pure derivations ---
    diff_amount, diff_percent, decision_price, price_source = compute_diffs(
        decision_price_yf=market.decision_price,
        current_price=market.current_price,
        quantity=req.quantity,
        amount=req.amount,
    )
    direction = derive_direction(req.scenario_type, diff_percent)
    outcome = derive_outcome(direction)
    was_correct = derive_was_correct(outcome)

    return CalculateResponse(
        ticker=ticker,
        scenario_type=req.scenario_type,
        decision_date=req.decision_date,
        actual_date_used=market.actual_date_used,
        decision_price=round(decision_price, 4),
        decision_price_source=price_source,
        current_price=round(market.current_price, 4),
        current_date=market.current_date,
        diff_amount=round(diff_amount, 2),
        diff_percent=round(diff_percent, 2),
        direction=direction,
        outcome=outcome,
        was_decision_correct=was_correct,
        split_adjusted=True,
    )
