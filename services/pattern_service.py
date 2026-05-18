"""Pattern analysis service for F-04 and F-06.

Calculates peak distances, consistency score, and regret rankings.
"""
import logging
import statistics
from datetime import datetime, timezone
from typing import Optional

from cachetools import TTLCache

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
from services.llm_service import generate_insights
from services.yfinance_service import get_price_series

logger = logging.getLogger(__name__)

# Per-user insights cache. Sonnet 4.6 is the most expensive call in the app,
# so we cache aggressively. TTL chosen so a user adding a decision sees fresh
# insights within an hour without forcing a regen on every dashboard render.
_INSIGHTS_TTL_SECONDS = 60 * 60  # 1 hour
_insights_cache: TTLCache = TTLCache(maxsize=1024, ttl=_INSIGHTS_TTL_SECONDS)


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
    """Generate F-06 AI insights summary or locked status.

    Flow:
      1. < 10 decisions -> locked
      2. Cache hit on user_id -> return cached insights (cached=True)
      3. Cache miss -> LLM call via llm_service.generate_insights, store on
         success, return (cached=False). On guardrail failure return
         (degraded=True, insights=[]) without caching the empty result.
    """
    res = get_user_decisions(user_id)
    items = res.items

    if len(items) < 10:
        return InsightsLocked(unlocked=False)

    cached_entry = _insights_cache.get(user_id)
    if cached_entry is not None:
        cached_insights, cached_at = cached_entry
        return InsightsUnlocked(
            unlocked=True,
            insights=cached_insights,
            generated_at=cached_at,
            cached=True,
            degraded=False,
        )

    result = generate_insights(items)
    generated_at = datetime.now(timezone.utc)

    if not result.degraded:
        _insights_cache[user_id] = (result.insights, generated_at)
    else:
        logger.warning("Insights generation degraded for user %s", user_id)

    return InsightsUnlocked(
        unlocked=True,
        insights=result.insights,
        generated_at=generated_at,
        cached=False,
        degraded=result.degraded,
    )


def invalidate_insights_cache(user_id: str) -> None:
    """Drop any cached insights for the user. Call after a new decision so
    the next dashboard load regenerates against the updated history."""
    _insights_cache.pop(user_id, None)
