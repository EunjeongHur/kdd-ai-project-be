"""API router for /patterns. Matches docs/api.yaml."""
from fastapi import APIRouter, Depends

from schemas.patterns import InsightsResponse, PatternsResponse
from services.auth_service import get_current_user
from services.pattern_service import analyze_patterns, get_patterns_insights

router = APIRouter(prefix="/patterns", tags=["patterns"])


@router.get("", response_model=PatternsResponse)
def get_user_patterns(user_id: str = Depends(get_current_user)) -> PatternsResponse:
    """Retrieve locked status or unlocked investment pattern metrics (F-04)."""
    return analyze_patterns(user_id)


@router.get("/insights", response_model=InsightsResponse)
def get_user_insights(user_id: str = Depends(get_current_user)) -> InsightsResponse:
    """Retrieve locked status or unlocked F-06 AI insights."""
    return get_patterns_insights(user_id)
