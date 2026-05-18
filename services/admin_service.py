"""Admin service for demo data seeding.

Seeds realistic decision history to unlock F-04 and F-06 for presentation.
"""
from datetime import date, timedelta
import random
from uuid import UUID

from fastapi import HTTPException

from schemas.admin import SeedDemoRequest, SeedDemoResponse
from schemas.calculate import ScenarioType
from schemas.decisions import DecisionInput
from services.decision_service import save_decision
from services.supabase_service import get_supabase

# Curated list of realistic historical scenarios for demo
DEMO_SCENARIOS = [
    {"ticker": "NVDA", "scenario_type": ScenarioType.NO_BUY, "days_ago": 400, "quantity": 10},
    {"ticker": "AAPL", "scenario_type": ScenarioType.NO_SELL, "days_ago": 300, "quantity": 25},
    {"ticker": "TSLA", "scenario_type": ScenarioType.SOLD_TOO_EARLY, "days_ago": 250, "quantity": 15},
    {"ticker": "MSFT", "scenario_type": ScenarioType.NO_BUY, "days_ago": 350, "quantity": 20},
    {"ticker": "AMZN", "scenario_type": ScenarioType.NO_BUY, "days_ago": 450, "quantity": 30},
    {"ticker": "GOOG", "scenario_type": ScenarioType.NO_SELL, "days_ago": 180, "quantity": 40},
    {"ticker": "AMD", "scenario_type": ScenarioType.SOLD_TOO_EARLY, "days_ago": 120, "quantity": 50},
    {"ticker": "META", "scenario_type": ScenarioType.NO_BUY, "days_ago": 500, "quantity": 15},
    {"ticker": "NFLX", "scenario_type": ScenarioType.NO_SELL, "days_ago": 220, "quantity": 10},
    {"ticker": "INTC", "scenario_type": ScenarioType.NO_BUY, "days_ago": 90, "quantity": 100},
    {"ticker": "QCOM", "scenario_type": ScenarioType.NO_BUY, "days_ago": 150, "quantity": 35},
    {"ticker": "SMCI", "scenario_type": ScenarioType.NO_BUY, "days_ago": 380, "quantity": 20},
    {"ticker": "PLTR", "scenario_type": ScenarioType.SOLD_TOO_EARLY, "days_ago": 210, "quantity": 80},
    {"ticker": "COIN", "scenario_type": ScenarioType.NO_BUY, "days_ago": 270, "quantity": 25},
    {"ticker": "LLY", "scenario_type": ScenarioType.NO_BUY, "days_ago": 330, "quantity": 12},
]


def seed_demo_decisions(req: SeedDemoRequest) -> SeedDemoResponse:
    """Seed target user account with demo decisions."""
    user_id_str = str(req.user_id)
    client = get_supabase()

    if req.clear_existing:
        client.table("decisions").delete().eq("user_id", user_id_str).execute()

    today = date.today()
    created = 0

    # Ensure we have enough scenarios to cycle through
    pool = DEMO_SCENARIOS * (req.count // len(DEMO_SCENARIOS) + 1)
    selected_scenarios = pool[: req.count]

    for item in selected_scenarios:
        dec_date = today - timedelta(days=item["days_ago"])

        input_data = DecisionInput(
            ticker=item["ticker"],
            scenario_type=item["scenario_type"],
            decision_date=dec_date,
            quantity=item["quantity"],
            notes=f"Demo seeded scenario for {item['ticker']} ({item['scenario_type'].value}).",
        )

        try:
            save_decision(user_id_str, input_data)
            created += 1
        except Exception:
            # Skip any scenario that fails due to market holiday or delisting
            continue

    return SeedDemoResponse(decisions_created=created, user_id=req.user_id)
