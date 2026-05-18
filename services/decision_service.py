"""Business logic service for decisions.

Handles Supabase DB operations, 50-cap limit check, and real-time calculation.
"""
from datetime import date
from typing import Optional
from uuid import UUID

from fastapi import HTTPException

from schemas.calculate import (
    CalculateResponse,
    DecisionPriceSource,
    Direction,
    Outcome,
)
from schemas.decisions import Decision, DecisionInput, DecisionListResponse, DecisionWithCurrent
from services.calculate_service import compute_diffs, derive_direction, derive_outcome, derive_was_correct
from services.supabase_service import get_supabase
from services.yfinance_service import get_market_data


def get_user_decision_count(user_id: str) -> int:
    """Get the exact count of decisions for a user."""
    client = get_supabase()
    res = client.table("decisions").select("id", count="exact").eq("user_id", user_id).execute()
    return res.count if res.count is not None else len(res.data)


def save_decision(user_id: str, input_data: DecisionInput) -> Decision:
    """Save a decision for a user after verifying cap limit and computing snapshot."""
    count = get_user_decision_count(user_id)
    if count >= 50:
        raise HTTPException(
            status_code=409,
            detail={
                "error": {
                    "code": "DECISION_LIMIT_REACHED",
                    "message": "Decision history limit (50) reached. Please delete old decisions.",
                }
            },
        )

    # Fetch market data
    ticker = input_data.ticker.strip().upper()
    try:
        market = get_market_data(ticker, input_data.decision_date)
    except Exception as e:
        # Re-raise standard HTTP exceptions or wrap unknown upstream errors
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=502,
            detail={"error": {"code": "YAHOO_FINANCE_ERROR", "message": f"Market data fetch failed: {str(e)}"}},
        )

    diff_amount, diff_percent, decision_price, price_source = compute_diffs(
        decision_price_yf=market.decision_price,
        current_price=market.current_price,
        quantity=input_data.quantity,
        amount=input_data.amount,
    )
    direction = derive_direction(input_data.scenario_type, diff_percent)
    outcome = derive_outcome(direction)
    was_correct = derive_was_correct(outcome)

    db_row = {
        "user_id": user_id,
        "ticker": ticker,
        "scenario_type": input_data.scenario_type.value,
        "decision_date": input_data.decision_date.isoformat(),
        "actual_date_used": market.actual_date_used.isoformat(),
        "end_date": input_data.end_date.isoformat() if input_data.end_date else None,
        "quantity": float(input_data.quantity) if input_data.quantity is not None else None,
        "amount": float(input_data.amount) if input_data.amount is not None else None,
        "decision_price_snapshot": round(decision_price, 4),
        "notes": input_data.notes,
        "title": input_data.title,
        "emotion": input_data.emotion.value if input_data.emotion else None,
        "current_price": round(market.current_price, 4),
        "diff_amount": round(diff_amount, 2),
        "diff_percent": round(diff_percent, 2),
        "direction": direction.value,
        "outcome": outcome.value,
        "was_decision_correct": was_correct,
        "current_date_snapshot": market.current_date.isoformat(),
        "decision_price_source": price_source.value,
        "reflection": None,
    }

    client = get_supabase()
    res = client.table("decisions").insert(db_row).execute()
    if not res.data:
        raise HTTPException(
            status_code=500,
            detail={"error": {"code": "DATABASE_ERROR", "message": "Failed to insert decision."}},
        )

    return Decision.model_validate(res.data[0])


def get_user_decisions(
    user_id: str,
    ticker: Optional[str] = None,
    scenario_type: Optional[str] = None,
    from_date: Optional[date] = None,
    to_date: Optional[date] = None,
    sort: str = "-created_at",
) -> DecisionListResponse:
    """Retrieve user decisions with filtering, sorting, and real-time opportunity cost calculation."""
    client = get_supabase()
    query = client.table("decisions").select("*", count="exact").eq("user_id", user_id)

    if ticker:
        query = query.eq("ticker", ticker.strip().upper())
    if scenario_type:
        query = query.eq("scenario_type", scenario_type)
    if from_date:
        query = query.gte("decision_date", from_date.isoformat())
    if to_date:
        query = query.lte("decision_date", to_date.isoformat())

    # Parse sorting
    desc = sort.startswith("-")
    order_col = sort.lstrip("-")
    # Allowed sort columns
    if order_col not in ["decision_date", "created_at", "updated_at", "ticker"]:
        order_col = "created_at"

    query = query.order(order_col, desc=desc)
    res = query.execute()

    items = []
    for row in res.data:
        decision = Decision.model_validate(row)

        # List view uses saved snapshots only — no per-row yfinance recompute.
        # With N=50 decisions even cached calls run into multi-second territory,
        # and the list doesn't surface live opportunity cost (just the saved
        # outcome badge). Detail view (GET /decisions/{id}) does the fresh fetch.
        current_calc = CalculateResponse(
            ticker=decision.ticker,
            scenario_type=decision.scenario_type,
            decision_date=decision.decision_date,
            actual_date_used=decision.actual_date_used,
            decision_price=decision.decision_price_snapshot,
            decision_price_source=decision.decision_price_source or DecisionPriceSource.YFINANCE,
            current_price=decision.current_price or decision.decision_price_snapshot,
            current_date=decision.current_date_snapshot or decision.actual_date_used,
            diff_amount=decision.diff_amount or 0.0,
            diff_percent=decision.diff_percent or 0.0,
            direction=decision.direction or Direction.NEUTRAL,
            outcome=decision.outcome or Outcome.NEUTRAL,
            was_decision_correct=decision.was_decision_correct,
            split_adjusted=True,
        )

        item_with_curr = DecisionWithCurrent(
            **decision.model_dump(),
            current=current_calc,
        )
        items.append(item_with_curr)

    total_count = res.count if res.count is not None else len(items)
    return DecisionListResponse(items=items, total=total_count)


def update_decision_reflection(
    user_id: str,
    decision_id: UUID,
    reflection: Optional[str],
) -> Decision:
    """Attach (or clear) a reflection text on a saved decision.

    Used by PATCH /decisions/{id} so the frontend can write the LLM-generated
    narrative back onto the row that POST /decisions just created.
    """
    client = get_supabase()
    res = (
        client.table("decisions")
        .update({"reflection": reflection})
        .eq("id", str(decision_id))
        .eq("user_id", user_id)
        .execute()
    )
    if not res.data:
        raise HTTPException(
            status_code=404,
            detail={
                "error": {
                    "code": "DECISION_NOT_FOUND",
                    "message": "Decision not found or unauthorized.",
                }
            },
        )
    return Decision.model_validate(res.data[0])


def get_decision_by_id(user_id: str, decision_id: UUID) -> DecisionWithCurrent:
    """Fetch one decision with freshly-recomputed current values.

    Used by GET /decisions/{id}. Mirrors get_user_decisions for a single row.
    """
    client = get_supabase()
    res = (
        client.table("decisions")
        .select("*")
        .eq("id", str(decision_id))
        .eq("user_id", user_id)
        .limit(1)
        .execute()
    )
    if not res.data:
        raise HTTPException(
            status_code=404,
            detail={
                "error": {
                    "code": "DECISION_NOT_FOUND",
                    "message": "Decision not found or unauthorized.",
                }
            },
        )

    decision = Decision.model_validate(res.data[0])

    try:
        market = get_market_data(decision.ticker, decision.decision_date)
        diff_amt, diff_pct, d_price, source = compute_diffs(
            decision_price_yf=market.decision_price,
            current_price=market.current_price,
            quantity=decision.quantity,
            amount=decision.amount,
        )
        dir_val = derive_direction(decision.scenario_type, diff_pct)
        out_val = derive_outcome(dir_val)
        corr_val = derive_was_correct(out_val)
        current_calc = CalculateResponse(
            ticker=decision.ticker,
            scenario_type=decision.scenario_type,
            decision_date=decision.decision_date,
            actual_date_used=market.actual_date_used,
            decision_price=round(d_price, 4),
            decision_price_source=source,
            current_price=round(market.current_price, 4),
            current_date=market.current_date,
            diff_amount=round(diff_amt, 2),
            diff_percent=round(diff_pct, 2),
            direction=dir_val,
            outcome=out_val,
            was_decision_correct=corr_val,
            split_adjusted=True,
        )
    except Exception:
        current_calc = CalculateResponse(
            ticker=decision.ticker,
            scenario_type=decision.scenario_type,
            decision_date=decision.decision_date,
            actual_date_used=decision.actual_date_used,
            decision_price=decision.decision_price_snapshot,
            decision_price_source=decision.decision_price_source,
            current_price=decision.current_price or decision.decision_price_snapshot,
            current_date=decision.current_date_snapshot or decision.actual_date_used,
            diff_amount=decision.diff_amount or 0.0,
            diff_percent=decision.diff_percent or 0.0,
            direction=decision.direction,
            outcome=decision.outcome,
            was_decision_correct=decision.was_decision_correct,
            split_adjusted=True,
        )

    return DecisionWithCurrent(**decision.model_dump(), current=current_calc)


def delete_decision(user_id: str, decision_id: UUID) -> None:
    """Delete a user decision by ID."""
    client = get_supabase()
    res = client.table("decisions").delete().eq("id", str(decision_id)).eq("user_id", user_id).execute()
    if not res.data:
        raise HTTPException(
            status_code=404,
            detail={"error": {"code": "DECISION_NOT_FOUND", "message": "Decision not found or unauthorized."}},
        )
