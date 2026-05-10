from fastapi import APIRouter
from schemas.calculate import CalcRequest, CalcResponse, ScenarioType
from services.yfinance_service import get_market_data

router = APIRouter(prefix="/calculate", tags=["Calculation"])

@router.post("", response_model=CalcResponse)
@router.post("", response_model=CalcResponse)
async def calculate_opportunity_cost(req: CalcRequest):
    past_price, current_price = get_market_data(req.ticker, req.target_date)

    qty = req.quantity if req.quantity else int(req.amount // past_price)
    if qty <= 0: qty = 1

    # Enum 비교 방식으로 변경
    if req.scenario_type in [ScenarioType.NO_BUY, ScenarioType.SELL_THEN_RISE]:
        diff_val = current_price - past_price
    elif req.scenario_type == ScenarioType.NO_SELL:
        diff_val = past_price - current_price
    else:
        diff_val = 0

    diff_amount = diff_val * qty
    diff_percent = (diff_val / past_price) * 100

    # Principle 5.2 준수: 중립적인 영어 메시지
    # "This decision resulted in a +15.50% difference in outcome."
    message = f"This decision resulted in a {abs(diff_percent):.2f}% difference in outcome."

    if diff_amount > 0:
        status_color, icon = "red", "📉"
    elif diff_amount < 0:
        status_color, icon = "green", "📈"
    else:
        status_color, icon = "gray", "➖"

    return CalcResponse(
        past_price=round(past_price, 2),
        current_price=round(current_price, 2),
        diff_amount=round(abs(diff_amount), 2),
        diff_percent=round(abs(diff_percent), 2),
        message=message,
        status_color=status_color,
        icon=icon
    )