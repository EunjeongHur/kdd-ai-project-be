import yfinance as yf
from datetime import date, timedelta
from fastapi import HTTPException

def get_market_data(ticker: str, target_date: date):
    tk = yf.Ticker(ticker)

    # 1. 과거 가격 조회: 해당 날짜가 휴장일일 경우를 대비해 7일치 데이터를 가져와 첫 거래일을 선택
    end_date = target_date + timedelta(days=7)
    hist = tk.history(start=target_date.strftime('%Y-%m-%d'),
                      end=end_date.strftime('%Y-%m-%d'))

    if hist.empty:
        raise HTTPException(status_code=404, detail="해당 시점의 데이터를 찾을 수 없습니다.")

    past_price = hist['Close'].iloc[0]

    # 2. 현재 가격 조회 (실시간 가격 또는 최근 종가)
    try:
        current_price = tk.fast_info['last_price']
    except:
        current_price = tk.history(period="1d")['Close'].iloc[-1]

    return past_price, current_price