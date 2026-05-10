from pydantic import BaseModel, Field
from datetime import date
from typing import Optional
from enum import Enum

class ScenarioType(str, Enum):
    NO_BUY = "NO_BUY"
    NO_SELL = "NO_SELL"
    SELL_THEN_RISE = "SELL_THEN_RISE"

class CalcRequest(BaseModel):
    # examples=["AAPL"] 로 수정하여 노란 줄 해결
    ticker: str = Field(..., examples=["AAPL"])
    scenario_type: ScenarioType
    target_date: date
    quantity: Optional[int] = Field(None, examples=[10])
    amount: Optional[float] = Field(None, examples=[1500.0])

class CalcResponse(BaseModel):
    past_price: float
    current_price: float
    diff_amount: float
    diff_percent: float
    message: str
    status_color: str
    icon: str