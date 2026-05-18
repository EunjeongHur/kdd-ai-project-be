"""Pydantic models for admin ops. Matches docs/api.yaml."""
from uuid import UUID
from pydantic import BaseModel, Field


class SeedDemoRequest(BaseModel):
    user_id: UUID = Field(..., description="Target user UUID.")
    count: int = Field(50, ge=1, le=50, description="Number of decisions to seed.")
    clear_existing: bool = Field(False, description="If true, delete existing decisions for the user first.")


class SeedDemoResponse(BaseModel):
    decisions_created: int
    user_id: UUID
