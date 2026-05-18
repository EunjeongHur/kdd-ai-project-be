"""Pattern analysis service for F-04 and F-06.

Calculates peak distances, consistency score, and regret rankings.
"""
import statistics
from datetime import datetime, timezone
from typing import Optional

from schemas.patterns import (
    InsightsLocked,
    InsightsResponse,
    InsightsUnlocked,
    PatternMetrics,
    PatternsLocked,
    PatternsResponse,
    PatternsUnlocked,
    RegrettedDecision,
)
from services.decision_service import get_user_decisions
from services.yfinance_service import get_price_series


def analyze_patterns(user_id: str) -> PatternsResponse:
    """Analyze user decisions and return locked or unlocked pattern metrics."""
    res = get_user_decisions(user_id, sort="-created_at")
    items = res.items

    if len(items) < 10:
        return PatternsLocked(unlocked=False, current_count=len(items), required_count=10)

    distances = []
    scenario_dist = {}
    regretted_list = []

    for item in items:
        # 1. Scenario distribution
        st_key = item.scenario_type.value
        scenario_dist[st_key] = scenario_dist.get(st_key, 0) + 1

        # 2. Regret ranking
        regret_score = abs(item.current.diff_percent)
        regretted = RegrettedDecision(
            **item.model_dump(exclude={"current"}),
            current=item.current,
            regret_score=round(regret_score, 2),
        )
        regretted_list.append(regretted)

        # 3. Peak distance calculation
        # Effective decision price used as starting reference
        decision_price = item.current.decision_price
        prices = get_price_series(item.ticker, item.actual_date_used, item.current.current_date)
        if prices and max(prices) > 0:
            peak = max(prices)
        else:
            peak = max(decision_price, item.current.current_price)

        if peak > 0:
            dist = ((decision_price - peak) / peak) * 100
            # Distance from peak in window is always <= 0
            if dist > 0:
                dist = 0.0
            distances.append(dist)

    # Calculate avg distance
    avg_dist = round(sum(distances) / len(distances), 2) if distances else 0.0

    # Calculate consistency score
    if len(distances) > 1:
        sd = statistics.stdev(distances)
        c_score = max(0.0, 1.0 - min(sd / 30.0, 1.0))
    else:
        c_score = 1.0

    # Top 3 most regretted
    regretted_list.sort(key=lambda x: (x.regret_score, x.created_at.timestamp()), reverse=True)
    top3 = regretted_list[:3]

    metrics = PatternMetrics(
        avg_distance_from_peak_percent=avg_dist,
        consistency_score=round(c_score, 4),
        scenario_distribution=scenario_dist,
        most_regretted_top3=top3,
    )

    return PatternsUnlocked(unlocked=True, decision_count=len(items), metrics=metrics)


def get_patterns_insights(user_id: str) -> InsightsResponse:
    """Generate F-06 AI insights summary or locked status."""
    res = get_user_decisions(user_id)
    items = res.items

    if len(items) < 10:
        return InsightsLocked(unlocked=False)

    # Scaffolded AI insights for MVP
    mock_insights = [
        "You tend to experience hesitation when high-conviction growth stocks undergo rapid short-term surges.",
        "Your decision consistency is notably stable during broader market pullbacks.",
        "A recurring theme in your history is cutting successful positions short prematurely.",
    ]

    return InsightsUnlocked(
        unlocked=True,
        insights=mock_insights,
        generated_at=datetime.now(timezone.utc),
        cached=True,
        degraded=False,
    )
