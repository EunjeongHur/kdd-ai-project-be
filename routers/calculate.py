from fastapi import APIRouter
from schemas.calculate import CalcRequest, CalcResponse
from services.yfinance_service import get_market_data

router = APIRouter(prefix="/calculate", tags=["Calculation"])

@router.post("", response_model=CalcResponse)
async def calculate_opportunity_cost(req: CalcRequest):
    past_price, current_price = get_market_data(req.ticker, req.target_date)

    # 수량/금액 처리
    qty = req.quantity if req.quantity else int(req.amount // past_price)
    if qty <= 0: qty = 1

    # 시나리오별 결과 계산
    # 기본 개념: (비교 대상 - 현재 상태)
    if req.scenario_type in ["NO_BUY", "SELL_THEN_RISE"]:
        # 안 샀거나 이미 팔았을 때: (현재가 - 과거가)만큼 수익 기회 상실
        diff_val = current_price - past_price
    elif req.scenario_type == "NO_SELL":
        # 안 팔고 버텼을 때: (과거가 - 현재가)만큼의 하락분(손실) 발생
        diff_val = past_price - current_price
    else:
        diff_val = 0

    diff_amount = diff_val * qty
    # 과거가 대비 변동률 (절댓값으로 표현하여 중립성 유지)
    diff_percent = (diff_val / past_price) * 100

    # Principle 5.2 준수: "놓쳤습니다" 대신 "차이를 낳았습니다"
    # 절댓값을 사용하여 감정적 단어 배제
    message = f"이 결정은 {abs(diff_percent):.2f}%의 결과 차이를 낳았습니다."

    # 상태값에 따른 시각적 지표 (UX)
    if diff_amount > 0:
        status_color, icon = "red", "📉"  # 아쉬운 결과 (기회비용 발생)
    elif diff_amount < 0:
        status_color, icon = "green", "📈" # 다행인 결과 (손실 회피 등)
    else:
        status_color, icon = "gray", "➖"

    return CalcResponse(
        past_price=round(past_price, 2),
        current_price=round(current_price, 2),
        difference_amount=round(abs(diff_amount), 2),
        difference_percent=round(abs(diff_percent), 2),
        message=message,
        status_color=status_color,
        icon=icon
    )