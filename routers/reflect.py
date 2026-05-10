"""POST /reflect — single-decision narrative reflection.

Distinct from PRD F-06 (`/patterns/insights`), which analyzes 10+ accumulated
decisions. This is a per-submission LLM commentary that runs every time a
user logs a decision, surfacing a 1-2 sentence neutral observation alongside
the numeric result from /calculate.
"""
from fastapi import APIRouter

from schemas.reflect import ReflectRequest, ReflectResponse
from services.llm_service import generate_reflection

router = APIRouter(prefix="/reflect", tags=["ai"])


@router.post("", response_model=ReflectResponse)
def reflect(req: ReflectRequest) -> ReflectResponse:
    """Generate a per-decision reflection. Always returns a 200-shaped
    response. When guardrails (PRD 5.2) fail three regenerations, returns
    `reflection=""` with `degraded=true` — the frontend should fall back to
    showing only the numeric result.

    Note: per spec this endpoint should be bearer-authed and rate-limited
    (similar to /parse-decision). Auth + rate limit middleware is TODO; for
    MVP demo this is intentionally open. Add before production traffic.
    """
    return generate_reflection(req)
