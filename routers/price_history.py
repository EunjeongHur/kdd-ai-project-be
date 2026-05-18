from datetime import date, timedelta

from fastapi import APIRouter, HTTPException, Query

from services.yfinance_service import get_market_data
import yfinance as yf

router = APIRouter(prefix="/price-history", tags=["price-history"])


"""GET /price-history/{ticker} — Returns price history around the decision date.

Fetches stock price data from 60 days before to 90 days after the decision_date using yfinance.
"""
@router.get("/{ticker}")
def get_price_history(
    ticker: str,
    decision_date: date = Query(..., description="결정일 (YYYY-MM-DD)"),
):
    """
    Returns:
        {
            ticker, decision_date, decision_price, current_price,
            history: [{ date, price }]
        }
    """
    ticker = ticker.strip().upper()

    start = decision_date - timedelta(days=60)
    end = min(date.today(), decision_date + timedelta(days=90))

    tk = yf.Ticker(ticker)

    try:
        hist = tk.history(
            start=start.strftime("%Y-%m-%d"),
            end=(end + timedelta(days=1)).strftime("%Y-%m-%d"),
            auto_adjust=True,
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
                    "code": "NO_DATA",
                    "message": "No trading data available for the selected period.",
                }
            },
        )

    try:
        market = get_market_data(ticker, decision_date)
    except HTTPException:
        raise

    history = [
        {
            "date": row.name.date().isoformat(),
            "price": round(float(row["Close"]), 4),
        }
        for _, row in hist.iterrows()
    ]

    return {
        "ticker": ticker,
        "decision_date": decision_date.isoformat(),
        "decision_price": round(market.decision_price, 4),
        "current_price": round(market.current_price, 4),
        "history": history,
    }