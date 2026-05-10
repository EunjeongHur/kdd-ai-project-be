"""yfinance wrapper for /calculate.

Returns adjusted-close prices and the actual trading dates used (so the API
can echo `actual_date_used` and `current_date` back to the client per
docs/conventions.md §2).
"""
from dataclasses import dataclass
from datetime import date, timedelta

import yfinance as yf
from fastapi import HTTPException


@dataclass(frozen=True)
class MarketData:
    decision_price: float
    actual_date_used: date
    current_price: float
    current_date: date


def get_market_data(ticker: str, decision_date: date) -> MarketData:
    """Fetch adjusted close at `decision_date` (or the next trading day if that
    date was a market holiday/weekend) and the most recent close.

    Raises HTTPException(422) if no data is available.
    Raises HTTPException(502) on upstream yfinance failures.
    """
    tk = yf.Ticker(ticker)

    # --- Decision-date price ---
    # Pull a 7-day window starting at decision_date so we catch the next
    # trading day if the user picked a non-trading day.
    end_date = decision_date + timedelta(days=7)
    try:
        hist_decision = tk.history(
            start=decision_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            auto_adjust=True,  # explicit: returns split/dividend-adjusted Close
        )
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail={
                "error": {
                    "code": "YAHOO_FINANCE_ERROR",
                    "message": f"Upstream price lookup failed: {exc}",
                }
            },
        )

    if hist_decision.empty:
        raise HTTPException(
            status_code=422,
            detail={
                "error": {
                    "code": "DATE_BEFORE_LISTING",
                    "message": (
                        "No trading data available on or shortly after the "
                        "decision_date you supplied. The ticker may not have "
                        "been listed yet."
                    ),
                }
            },
        )

    decision_row = hist_decision.iloc[0]
    decision_price = float(decision_row["Close"])
    # `name` on a Series row from yfinance is the Timestamp (date) of that row
    actual_date_used = decision_row.name.date()

    # --- Current price (always last close, never intraday) ---
    try:
        hist_current = tk.history(period="5d", auto_adjust=True)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail={
                "error": {
                    "code": "YAHOO_FINANCE_ERROR",
                    "message": f"Upstream price lookup failed: {exc}",
                }
            },
        )

    if hist_current.empty:
        raise HTTPException(
            status_code=502,
            detail={
                "error": {
                    "code": "YAHOO_FINANCE_ERROR",
                    "message": "Upstream returned no recent data for ticker.",
                }
            },
        )

    current_row = hist_current.iloc[-1]
    current_price = float(current_row["Close"])
    current_date = current_row.name.date()

    return MarketData(
        decision_price=decision_price,
        actual_date_used=actual_date_used,
        current_price=current_price,
        current_date=current_date,
    )
