"""API router for /decisions. Matches docs/api.yaml."""
from datetime import date
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, Query, status

from schemas.decisions import Decision, DecisionInput, DecisionListResponse
from services.auth_service import get_current_user
from services.decision_service import delete_decision, get_user_decisions, save_decision

router = APIRouter(prefix="/decisions", tags=["decisions"])


@router.post("", response_model=Decision, status_code=status.HTTP_201_CREATED)
def create_decision(
    req: DecisionInput,
    user_id: str = Depends(get_current_user),
) -> Decision:
    """Save a user decision history after verifying 50-cap limit."""
    return save_decision(user_id, req)


@router.get("", response_model=DecisionListResponse)
def list_decisions(
    ticker: Optional[str] = Query(None, description="Filter by ticker"),
    scenario_type: Optional[str] = Query(None, description="Filter by scenario_type"),
    from_date: Optional[date] = Query(None, alias="from", description="Filter decisions on or after date"),
    to_date: Optional[date] = Query(None, alias="to", description="Filter decisions on or before date"),
    sort: str = Query("-created_at", description="Sorting field with optional '-' desc prefix"),
    user_id: str = Depends(get_current_user),
) -> DecisionListResponse:
    """List current user decisions with real-time recalculated opportunity costs."""
    return get_user_decisions(user_id, ticker, scenario_type, from_date, to_date, sort)


@router.delete("/{decision_id}", status_code=status.HTTP_204_NO_CONTENT)
def remove_decision(
    decision_id: UUID,
    user_id: str = Depends(get_current_user),
) -> None:
    """Delete a user decision."""
    delete_decision(user_id, decision_id)
