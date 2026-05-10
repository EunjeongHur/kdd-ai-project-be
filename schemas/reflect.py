"""Pydantic models for POST /reflect (per-decision narrative reflection).

This is *not* PRD F-06 (pattern analysis from 10+ decisions); that endpoint is
`GET /patterns/insights` and only meaningful with accumulated history. This is
a single-decision LLM commentary that runs on every submission to surface a
short, neutral observation alongside the numeric result.

When the user has logged decisions before, the frontend should include them in
`previous_decisions` so the reflection can reference actual history (e.g.
"your third missed_gain in a row") instead of speculating about patterns from
a single data point.

PRD AI guardrails (5.2) apply: no stock recommendations, no price predictions,
no prescriptive advice, no self-blame reinforcement.
"""
from datetime import date

from pydantic import BaseModel, Field

from schemas.calculate import Direction, Outcome, ScenarioType


class DecisionContext(BaseModel):
    """Minimal decision shape used both for the current decision and for
    each entry in `previous_decisions`."""
    ticker: str = Field(..., examples=["NVDA"])
    scenario_type: ScenarioType
    decision_date: date
    diff_percent: float = Field(
        ...,
        description="Signed percent change from decision_price to current_price.",
        examples=[193.32],
    )
    direction: Direction
    outcome: Outcome


class ReflectRequest(DecisionContext):
    """Current decision plus optional history.

    `previous_decisions` should be ordered most-recent-first and capped at 10
    entries by the frontend. With history, the model anchors observations in
    the user's actual record; without it, the reflection describes only this
    decision.
    """
    previous_decisions: list[DecisionContext] = Field(
        default_factory=list,
        max_length=10,
        description=(
            "Up to 10 previous decisions for context, most recent first. "
            "Empty list means this is the user's first reflection."
        ),
    )


class ReflectResponse(BaseModel):
    reflection: str = Field(
        ...,
        description=(
            "1-2 sentence neutral observation. Empty string when guardrails "
            "forced a fallback (see `degraded`)."
        ),
    )
    degraded: bool = Field(
        False,
        description=(
            "True when the model's response failed guardrail post-processing "
            "three times. In that case `reflection` is empty and the frontend "
            "should show only the numeric result."
        ),
    )
    attempts: int = Field(
        ...,
        ge=1,
        le=3,
        description="Number of generation attempts (1-3). Telemetry only.",
    )
