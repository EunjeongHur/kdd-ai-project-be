"""Business logic for /optimal-timing (F-02).

Scans a user-specified date range and identifies the trading days with the
lowest (best_buy) and highest (best_sell) adjusted-close prices.

All outputs are hindsight facts — no prescriptive signals are generated here.
"""
from dataclasses import dataclass
from datetime import date

import yfinance as yf
from fastapi import HTTPException

from schemas.optimal_timing import PricePoint


@dataclass(frozen=True)
class OptimalTimingData:
    best_buy: PricePoint
    best_sell: PricePoint
    max_return_percent: float
    data_points: int
    summary_message: str


def get_optimal_timing(ticker: str, start_date: date, end_date: date) -> OptimalTimingData:
    """Fetch the full price history for [start_date, end_date] and derive the
    best-buy / best-sell extremes.

    Args:
        ticker: Validated, normalised ticker symbol.
        start_date: First day of the holding period (inclusive).
        end_date: Last day of the holding period (inclusive).

    Returns:
        OptimalTimingData with all derived fields.

    Raises:
        HTTPException(422): No trading data found in the window.
        HTTPException(502): Upstream yfinance failure.
    """
    tk = yf.Ticker(ticker)

    # yfinance `end` is exclusive — add one day to include end_date.
    from datetime import timedelta
    yf_end = end_date + timedelta(days=1)

    try:
        hist = tk.history(
            start=start_date.strftime("%Y-%m-%d"),
            end=yf_end.strftime("%Y-%m-%d"),
            auto_adjust=True,  # split/dividend-adjusted Close
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

    if hist.empty:
        raise HTTPException(
            status_code=422,
            detail={
                "error": {
                    "code": "NO_DATA_IN_RANGE",
                    "message": (
                        "No trading data found for the requested period. "
                        "Check that the ticker was listed during this window "
                        "and that the date range contains at least one trading day."
                    ),
                }
            },
        )

    # Build a plain dict {date: close} to avoid pandas dependency in callers.
    prices: dict[date, float] = {
        row.name.date(): float(row["Close"])
        for _, row in hist.iterrows()
    }

    if not prices:
        raise HTTPException(
            status_code=422,
            detail={
                "error": {
                    "code": "NO_DATA_IN_RANGE",
                    "message": "No valid price rows returned for the requested window.",
                }
            },
        )

    # Find extremes.
    min_date = min(prices, key=prices.__getitem__)
    max_date = max(prices, key=prices.__getitem__)

    best_buy = PricePoint(date=min_date, price=round(prices[min_date], 4))
    best_sell = PricePoint(date=max_date, price=round(prices[max_date], 4))

    max_return_percent = round(
        ((best_sell.price - best_buy.price) / best_buy.price) * 100, 2
    )

    # Factual, non-prescriptive summary — no "you should have" language.
    sign = "+" if max_return_percent >= 0 else ""
    summary_message = (
        f"Within this period, the price range spanned from "
        f"${best_buy.price:,.2f} ({best_buy.date}) to "
        f"${best_sell.price:,.2f} ({best_sell.date}), "
        f"representing a maximum spread of {sign}{max_return_percent:.2f}%."
    )

    return OptimalTimingData(
        best_buy=best_buy,
        best_sell=best_sell,
        max_return_percent=max_return_percent,
        data_points=len(prices),
        summary_message=summary_message,
    )
