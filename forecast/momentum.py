import yfinance as yf
import pandas as pd
from datetime import date

def get_momentum_returns(ticker="AAPL", horizons=[1,3,6,9], years_back=10, start="2000-01-01"):
    raw = yf.download(ticker, start=start, interval="1d", progress=False)
    data = raw["Close"].squeeze()
    monthly = data.resample("ME").last()

    df = pd.DataFrame(index=monthly.index, data={"Price": monthly})
    for h in horizons:
        df[f"{h}m_return"] = df["Price"].pct_change(periods=h) * 100

    # Pick January closes
    yearly = df[df.index.month == 1].copy()
    yearly["Year"] = yearly.index.year

    cutoff = date.today().year - years_back
    yearly = yearly[yearly["Year"] >= cutoff]

    momentum_dict = {}
    for _, row in yearly.iterrows():
        fy = f"FY{int(row['Year'])}"
        momentum_dict[fy] = {
            "Price": row["Price"],  # Jan close stock price
            **{f"{h}m_return": row[f"{h}m_return"] for h in horizons}
        }

    return momentum_dict

if __name__ == "__main__":
    print(get_momentum_returns("AAPL", years_back=5))
