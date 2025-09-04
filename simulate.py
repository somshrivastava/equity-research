import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from config import load_config, get_output_path

## Load configuration
config = load_config()
session_id = config.get("session_id", "default")
INPUT_FILE = get_output_path("portfolio_top5.csv", session_id)
OUTPUT_FILE = get_output_path("portfolio_values.csv", session_id)
START_CAPITAL = config["start_capital"]

## Load portfolio data
df = pd.read_csv(INPUT_FILE, parse_dates=["date"])
df = df.sort_values("date")

## Extract all tickers
all_tickers = set()
for _, row in df.iterrows():
    if pd.notna(row.get("holdings", "")) and row["holdings"] and row["holdings"] != "nan":
        all_tickers.update(row["holdings"].split(","))
all_tickers = list(all_tickers)

## Download price data for all tickers
price_data = {}
for tic in all_tickers:
    try:
        data = yf.download(tic, start=str(df["date"].min().date()), end=str(df["date"].max().date()), 
                          interval="1mo", auto_adjust=True)["Close"]
        data.name = tic
        price_data[tic] = data
    except:
        pass

## Download price data for all tickers
price_data = {}
for tic in all_tickers:
    try:
        data = yf.download(tic, start=str(df["date"].min().date()), end=str(df["date"].max().date()), 
                          interval="1mo", auto_adjust=True)["Close"]
        data.name = tic
        price_data[tic] = data
    except:
        pass

prices = pd.concat(price_data.values(), axis=1)

## Simulate portfolio rebalancing
holdings = {}  # {ticker: shares}
history = []
initialized = False

for _, row in df.iterrows():
    date = row["date"]
    
    # Get holdings for this date
    if pd.notna(row.get("holdings")) and row["holdings"] and row["holdings"] != "nan":
        top_stocks = [t.strip() for t in row["holdings"].split(",") if t.strip()]
    else:
        top_stocks = []

    ## Initial portfolio setup
    if not initialized and top_stocks:
        per_stock = START_CAPITAL / len(top_stocks)
        for tic in top_stocks:
            if tic in prices.columns:
                price_series = prices[tic].dropna()
                if not price_series.empty:
                    first_price = price_series.iloc[0]
                    shares = per_stock / first_price if first_price > 0 else 0
                    holdings[tic] = shares
        initialized = True

    ## Handle portfolio rebalancing
    if initialized and top_stocks:
        current_tics = set(holdings.keys())
        new_tics = set(top_stocks)
        dropped = current_tics - new_tics
        added = new_tics - current_tics

        # Sell dropped stocks
        cash = 0.0
        for tic in dropped:
            if tic in holdings and tic in prices.columns:
                valid_prices = prices.loc[:date, tic].dropna()
                if not valid_prices.empty:
                    cash += holdings[tic] * valid_prices.iloc[-1]
                del holdings[tic]

        # Buy new stocks with cash
        if added and cash > 0:
            per_stock = cash / len(added)
            for tic in added:
                if tic in prices.columns:
                    valid_prices = prices.loc[:date, tic].dropna()
                    if not valid_prices.empty:
                        price = valid_prices.iloc[-1]
                        shares = per_stock / price if price > 0 else 0
                        holdings[tic] = shares

    ## Calculate portfolio value
    total_value = 0.0
    holding_values = {}
    for tic in holdings:
        if tic in prices.columns:
            valid_prices = prices.loc[:date, tic].dropna()
            if not valid_prices.empty:
                price = valid_prices.iloc[-1]
                val = holdings[tic] * price
                total_value += val
                holding_values[f"{tic}_shares"] = holdings[tic]
                holding_values[f"{tic}_value"] = val

    history.append({"date": date, "portfolio_value": total_value, **holding_values})

## Save results
pd.DataFrame(history).to_csv(OUTPUT_FILE, index=False)
