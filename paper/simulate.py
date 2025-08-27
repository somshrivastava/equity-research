import pandas as pd
import yfinance as yf
import json
import sys

# === Load Config ===
def load_config():
    """Load configuration from command line argument or use defaults"""
    if len(sys.argv) > 1 and sys.argv[1] == "--config" and len(sys.argv) > 2:
        with open(sys.argv[2], 'r') as f:
            config = json.load(f)
        return config
    else:
        return {
            "start_capital": 100.0,
            "session_id": "default"
        }

config = load_config()

session_id = config.get("session_id", "default")
PORTFOLIO_FILE = f"portfolio_top5_{session_id}.csv" if session_id != "default" else "portfolio_top5.csv"
OUTPUT_FILE = f"portfolio_values_{session_id}.csv" if session_id != "default" else "portfolio_values.csv"
START_CAPITAL = config["start_capital"]

print(f"Session ID: {session_id}")

# Load portfolio history
df = pd.read_csv(PORTFOLIO_FILE, parse_dates=["date"])
df = df.sort_values("date")

# Gather all tickers
all_tickers = set()
for h in df["holdings"]:
    all_tickers.update(h.split(","))
all_tickers = list(all_tickers)

print(f"Fetching price data for {len(all_tickers)} tickers...")

# Download monthly adjusted prices one by one
price_data = {}
for tic in all_tickers:
    try:
        data = yf.download(
            tic,
            start=str(df["date"].min().date()),
            end=str(df["date"].max().date()),
            interval="1mo",
            auto_adjust=True
        )["Close"]
        data.name = tic
        price_data[tic] = data
    except Exception as e:
        print(f"Failed to fetch {tic}: {e}")

prices = pd.concat(price_data.values(), axis=1)

holdings = {}   # {tic: shares}
history = []
initialized = False

for _, row in df.iterrows():
    date = row["date"]
    top5 = row["holdings"].split(",")

    if not initialized:
        # --- Buy initial 5 stocks ---
        per_stock = START_CAPITAL / len(top5)
        for tic in top5:
            if tic in prices.columns:
                price_series = prices[tic].dropna()
                if not price_series.empty:
                    first_price = price_series.iloc[0]  # earliest available
                    shares = per_stock / first_price if first_price > 0 else 0
                    holdings[tic] = shares
        initialized = True

    # --- Handle drops and additions ---
    current_tics = set(holdings.keys())
    new_tics = set(top5)
    dropped = current_tics - new_tics
    added = new_tics - current_tics

    # Sell dropped
    cash = 0.0
    for tic in dropped:
        if tic in holdings and tic in prices.columns:
            valid_prices = prices.loc[:date, tic].dropna()
            if not valid_prices.empty:
                cash += holdings[tic] * valid_prices.iloc[-1]
            del holdings[tic]

    # Buy new
    if added and cash > 0:
        per_stock = cash / len(added)
        for tic in added:
            if tic in prices.columns:
                valid_prices = prices.loc[:date, tic].dropna()
                if not valid_prices.empty:
                    price = valid_prices.iloc[-1]
                    shares = per_stock / price if price > 0 else 0
                    holdings[tic] = shares

    # --- Compute portfolio value at this date ---
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

    history.append({
        "date": date,
        "portfolio_value": total_value,
        **holding_values
    })

# === Results ===
final_value = history[-1]["portfolio_value"]
total_return = (final_value / START_CAPITAL - 1) * 100

print(f"Final Portfolio Value = ${final_value:.2f} | Total Return = {total_return:.2f}%")

pd.DataFrame(history).to_csv(OUTPUT_FILE, index=False)
print(f"Saved portfolio history with holdings breakdown to {OUTPUT_FILE}")
