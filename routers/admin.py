"""API router for /admin. Matches docs/api.yaml."""
from fastapi import APIRouter, Depends, status

from schemas.admin import SeedDemoRequest, SeedDemoResponse
from services.admin_service import seed_demo_decisions
from services.auth_service import get_admin_token

router = APIRouter(prefix="/admin", tags=["admin"])


@router.post(
    "/seed-demo",
    response_model=SeedDemoResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(get_admin_token)],
)
def seed_demo_account(req: SeedDemoRequest) -> SeedDemoResponse:
    """Seed demo account with realistic decision history."""
    return seed_demo_decisions(req)
