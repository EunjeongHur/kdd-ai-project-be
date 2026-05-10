"""POST /parse-decision — F-05 natural-language to structured extraction.

See docs/api.yaml ParseDecisionRequest / ParseDecisionResponse.
"""
from fastapi import APIRouter

from schemas.parse_decision import ParseDecisionRequest, ParseDecisionResponse
from services.llm_service import parse_decision_text

router = APIRouter(prefix="/parse-decision", tags=["ai"])


@router.post("", response_model=ParseDecisionResponse)
def parse_decision(req: ParseDecisionRequest) -> ParseDecisionResponse:
    """Extract structured trading-decision fields from free-text user input.

    Always returns a 200-shaped result. Fields the model couldn't determine
    are null; frontend should clear corresponding form fields when
    `confidence.<field> < 0.7` and let the user fill manually.

    Note: per spec this endpoint is bearer-authed and rate-limited per user
    (30/day). Auth + rate-limit middleware is TODO — for MVP demo this is
    intentionally open. Add before production traffic.
    """
    return parse_decision_text(req.text)
