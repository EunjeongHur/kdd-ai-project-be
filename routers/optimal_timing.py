"""POST /optimal-timing — F-02 Optimal Timing Search.

Scans the holding period [start_date, end_date] and returns the best-buy
(lowest price) and best-sell (highest price) trading days, plus a factual
summary of the price spread.

This endpoint returns hindsight data only.  It MUST NOT be used to generate
prescriptive investment advice on the frontend.
"""
import re
from datetime import date

from fastapi import APIRouter, HTTPException

from schemas.optimal_timing import OptimalTimingRequest, OptimalTimingResponse
from services.optimal_timing_service import get_optimal_timing

# ---- Ticker helpers (mirrors routers/calculate.py until a shared service lands) ----
_TICKER_PATTERN = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")


def _normalize_ticker(raw: str) -> str:
    t = raw.strip().upper()
    t = re.sub(r"[\s/.]+", "-", t)
    return t.strip("-")


def _is_valid_ticker_format(ticker: str) -> bool:
    return bool(_TICKER_PATTERN.fullmatch(ticker))


# ---------------------------------------------------------------------------------

router = APIRouter(prefix="/optimal-timing", tags=["optimal-timing"])


@router.post("", response_model=OptimalTimingResponse)
def optimal_timing_search(req: OptimalTimingRequest) -> OptimalTimingResponse:
    """Identify the best-buy and best-sell points within a holding period.

    Scans every trading day in [start_date, end_date] and surfaces the
    minimum (best_buy) and maximum (best_sell) adjusted-close prices as
    factual reference data.

    - `best_buy`  — trading day with the lowest price in the window.
    - `best_sell` — trading day with the highest price in the window.
    - `max_return_percent` — ((best_sell − best_buy) / best_buy) × 100.
    - `summary_message` — a neutral, factual statement; never prescriptive.
    """
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
    today = date.today()
    if req.start_date > today:
        raise HTTPException(
            status_code=422,
            detail={
                "error": {
                    "code": "DATE_IN_FUTURE",
                    "message": "`start_date` cannot be in the future.",
                }
            },
        )
    # Clamp end_date to today so callers can pass today's date safely.
    effective_end = min(req.end_date, today)

    # --- Service call ---
    data = get_optimal_timing(ticker, req.start_date, effective_end)

    return OptimalTimingResponse(
        ticker=ticker,
        start_date=req.start_date,
        end_date=effective_end,
        best_buy=data.best_buy,
        best_sell=data.best_sell,
        max_return_percent=data.max_return_percent,
        summary_message=data.summary_message,
        data_points=data.data_points,
        split_adjusted=True,
    )
