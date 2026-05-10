"""Pure derivation functions for the calculate endpoint.

Kept separate from routers/calculate.py so the logic is unit-testable without
spinning up FastAPI or yfinance.
"""
from typing import Optional

from schemas.calculate import Direction, Outcome, ScenarioType

# Below this absolute percent move, we call the outcome neutral.
NEUTRAL_THRESHOLD_PERCENT = 0.5


def derive_direction(scenario: ScenarioType, diff_percent: float) -> Direction:
    """Map (scenario, price-direction) to one of the 7 Direction values.

    See docs/conventions.md §13 for the matrix.
    """
    if abs(diff_percent) < NEUTRAL_THRESHOLD_PERCENT:
        return Direction.NEUTRAL

    price_went_up = diff_percent > 0

    if scenario is ScenarioType.NO_BUY:
        return Direction.MISSED_GAIN if price_went_up else Direction.AVOIDED_LOSS
    if scenario is ScenarioType.NO_SELL:
        return Direction.KEPT_GAIN if price_went_up else Direction.ENDURED_LOSS
    if scenario is ScenarioType.SOLD_TOO_EARLY:
        return Direction.CUT_SHORT_GAIN if price_went_up else Direction.WELL_TIMED_EXIT

    raise ValueError(f"Unknown scenario_type: {scenario}")


_FAVORABLE_DIRECTIONS = frozenset({
    Direction.AVOIDED_LOSS,
    Direction.KEPT_GAIN,
    Direction.WELL_TIMED_EXIT,
})
_UNFAVORABLE_DIRECTIONS = frozenset({
    Direction.MISSED_GAIN,
    Direction.ENDURED_LOSS,
    Direction.CUT_SHORT_GAIN,
})


def derive_outcome(direction: Direction) -> Outcome:
    """Bucket Direction into favorable / unfavorable / neutral."""
    if direction in _FAVORABLE_DIRECTIONS:
        return Outcome.FAVORABLE
    if direction in _UNFAVORABLE_DIRECTIONS:
        return Outcome.UNFAVORABLE
    return Outcome.NEUTRAL


def derive_was_correct(outcome: Outcome) -> Optional[bool]:
    """True / False / None (neutral)."""
    if outcome is Outcome.FAVORABLE:
        return True
    if outcome is Outcome.UNFAVORABLE:
        return False
    return None


def compute_diffs(
    decision_price: float,
    current_price: float,
    quantity: Optional[float],
    amount: Optional[float],
) -> tuple[float, float]:
    """Return (diff_amount, diff_percent). Sign reflects price move.

    XOR enforcement is the request schema's job; this helper just trusts the
    inputs and falls through to a sensible default if neither is set (which
    shouldn't happen in production).
    """
    diff_percent = ((current_price - decision_price) / decision_price) * 100

    if quantity is not None:
        diff_amount = (current_price - decision_price) * quantity
    elif amount is not None:
        # Ratio-based: preserves fractional shares. amount * (current/decision - 1)
        diff_amount = amount * (current_price / decision_price - 1)
    else:
        # Schema validator should have blocked this; fall through harmlessly.
        diff_amount = 0.0

    return diff_amount, diff_percent
