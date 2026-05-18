"""Pydantic models for /patterns (F-04, F-06). Matches docs/api.yaml."""
from datetime import datetime
from typing import Literal, Union
from pydantic import BaseModel, Field

from schemas.decisions import DecisionWithCurrent


class RegrettedDecision(DecisionWithCurrent):
    """Decision with current calculation and regret score."""
    regret_score: float = Field(..., description="|diff_percent| of current. Higher = more regretted.")


class PatternMetrics(BaseModel):
    """Calculated metrics for F-04."""
    avg_distance_from_peak_percent: float = Field(
        ...,
        description="Mean distance from peak across all decisions. Always <= 0.",
    )
    consistency_score: float = Field(
        ...,
        ge=0, le=1,
        description="Variability normalized to [0, 1] (higher = more consistent).",
    )
    scenario_distribution: dict[str, int] = Field(
        ...,
        description="Count of decisions per scenario_type.",
    )
    most_regretted_top3: list[RegrettedDecision] = Field(
        ...,
        max_length=3,
        description="Top 3 decisions by regret_score (= |diff_percent|).",
    )


class PatternsLocked(BaseModel):
    unlocked: Literal[False] = False
    current_count: int = Field(..., ge=0)
    required_count: int = Field(10, description="10 decisions required to unlock F-04.")


class PatternsUnlocked(BaseModel):
    unlocked: Literal[True] = True
    decision_count: int
    metrics: PatternMetrics


PatternsResponse = Union[PatternsLocked, PatternsUnlocked]


class InsightsLocked(BaseModel):
    unlocked: Literal[False] = False


class InsightsUnlocked(BaseModel):
    unlocked: Literal[True] = True
    insights: list[str] = Field(..., description="Each item is a single sentence describing a pattern.")
    generated_at: datetime
    cached: bool
    degraded: bool


InsightsResponse = Union[InsightsLocked, InsightsUnlocked]
