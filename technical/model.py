# rf_walkforward_fundamentals.py
# Random Forest with walk-forward validation, technicals + fundamentals

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# --- Config ---
TICKER = "NVDA"
PERIOD = "15y"
INTERVAL = "1d"
SMAS = [5, 10, 30, 90, 200]
HORIZON = 10
SEED = 42
MIN_TRAIN_SIZE = 1000
TEST_WINDOW = 250

# --- Download stock & market data ---
stock = yf.download(TICKER, period=PERIOD, interval=INTERVAL, auto_adjust=True)
close = stock["Close"].copy()
spx = yf.download("^GSPC", period=PERIOD, interval=INTERVAL, auto_adjust=True)["Close"].squeeze()
vix = yf.download("^VIX", period=PERIOD, interval=INTERVAL, auto_adjust=True)["Close"].squeeze()

df = pd.DataFrame(index=close.index)
df["close"] = close
df["ret_1d"] = close.pct_change()

# --- SMA ratios ---
for period in SMAS:
    sma = close.rolling(period).mean()
    df[f"price_vs_sma{period}"] = (close - sma) / sma

# --- Momentum ---
df["ret_5d"] = close.pct_change(5)
df["ret_20d"] = close.pct_change(20)
df["spx_ret"] = spx.pct_change()
df["rel_strength"] = df["ret_5d"] - spx.pct_change(5)
df["corr20_spx"] = df["ret_1d"].rolling(20).corr(spx.pct_change())

# --- Volatility ---
df["vol20"] = df["ret_1d"].rolling(20).std()
df["maxdd20"] = df["close"].rolling(20).apply(
    lambda x: (x / np.maximum.accumulate(x) - 1).min(), raw=True
)

# --- Technical Indicators ---
delta = df["close"].diff()
gain = np.where(delta > 0, delta, 0)
loss = np.where(delta < 0, -delta, 0)
roll_up = pd.Series(gain, index=df.index).rolling(14).mean()
roll_down = pd.Series(loss, index=df.index).rolling(14).mean()
rs = roll_up / (roll_down + 1e-9)
df["rsi14"] = 100 - (100 / (1 + rs))

ema12 = close.ewm(span=12, adjust=False).mean()
ema26 = close.ewm(span=26, adjust=False).mean()
df["macd"] = ema12 - ema26

sma20 = close.rolling(20).mean()
std20 = close.rolling(20).std()
df["bb_pos"] = (close - sma20) / (2 * std20)

df["vix_level"] = vix
df["vix_chg5"] = vix.pct_change(5)

# --- Fundamentals (quarterly, forward-filled safely) ---
ticker = yf.Ticker(TICKER)
df["eps"] = 0.0
df["revenue"] = 0.0
df["eps_surprise"] = 0.0

try:
    fin = ticker.quarterly_financials.T  # rows = quarters
    fin.index = pd.to_datetime(fin.index)

    if "Total Revenue" in fin.columns:
        revenue = fin["Total Revenue"]
        revenue = revenue.reindex(pd.date_range(df.index.min(), df.index.max(), freq="D"))
        df["revenue"] = revenue.ffill().reindex(df.index).fillna(0)

    if "Net Income" in fin.columns:
        net_income = fin["Net Income"]
        net_income = net_income.reindex(pd.date_range(df.index.min(), df.index.max(), freq="D"))
        df["eps"] = net_income.ffill().reindex(df.index).fillna(0)

except Exception as e:
    print("Fundamentals fetch failed:", e)

# --- Label: average forward return ---
fwd_rets = [close.shift(-i) / close - 1 for i in range(1, HORIZON + 1)]
avg_fwd = pd.concat(fwd_rets, axis=1).mean(axis=1)
df["y"] = avg_fwd

# --- NaN-safe cleanup ---
print("Rows before cleanup:", len(df))
print("NaN counts before cleanup:\n", df.isna().sum())

df = df.fillna(0).iloc[:-HORIZON]

print("Rows after cleanup:", len(df))

# --- Features ---
feature_cols = [
    f"price_vs_sma{p}" for p in SMAS
] + [
    "ret_1d", "ret_5d", "ret_20d", "rel_strength", "corr20_spx",
    "vol20", "maxdd20", "rsi14", "macd", "bb_pos",
    "spx_ret", "vix_level", "vix_chg5",
    "eps", "revenue", "eps_surprise"
]

X = df[feature_cols].values
y = df["y"].values
dates = df.index

# --- Walk-forward loop ---
results = []
start = MIN_TRAIN_SIZE
while start + TEST_WINDOW <= len(df):
    train_idx = np.arange(0, start)
    test_idx = np.arange(start, start + TEST_WINDOW)

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    test_dates = dates[test_idx]

    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=6,
        random_state=SEED
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    corr = np.corrcoef(y_test, preds)[0, 1]

    results.append({
        "train_end": dates[start],
        "test_start": test_dates[0],
        "test_end": test_dates[-1],
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "mse": mse, "mae": mae, "r2": r2, "corr": corr
    })

    start += TEST_WINDOW

# --- Results ---
res_df = pd.DataFrame(results)

if not res_df.empty:
    print("\n--- Walk-Forward Regression Results (With Fundamentals) ---")
    print(res_df[["test_start", "test_end", "n_train", "n_test", "mse", "mae", "r2", "corr"]])

    print("\nAverage metrics across walk-forward tests:")
    print(res_df[["mse", "mae", "r2", "corr"]].mean())
else:
    print("No results generated — likely due to insufficient data after cleanup.")

# --- Feature Importances (last model) ---
if not res_df.empty:
    importances = model.feature_importances_
    print("\n--- Feature Importances (last model) ---")
    for i, col in enumerate(feature_cols):
        print(f"{col}: {importances[i]:.4f}")
