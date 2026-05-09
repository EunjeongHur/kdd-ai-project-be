from pydantic import BaseModel
from datetime import date
from typing import Optional

class CalcRequest(BaseModel):
    ticker: str
    scenario_type: str  # "NO_BUY", "NO_SELL", "SELL_THEN_RISE"
    target_date: date
    quantity: Optional[int] = None
    amount: Optional[float] = None

class CalcResponse(BaseModel):
    past_price: float
    current_price: float
    difference_amount: float
    difference_percent: float
    # 원칙 준수: 중립적인 메시지 구성
    message: str
    status_color: str # "green", "red", "gray"
    icon: str