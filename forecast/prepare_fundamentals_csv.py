# prepare_fundamentals_csv.py
#
# Clean timestep CSV (rows = months, cols = features).
# Includes helper to generate training windows for MLP or RNN.

import pandas as pd
import numpy as np
from pathlib import Path

# === Config ===
INPUT_CSV  = "AAPL_fundamentals_last10y.csv"
OUTPUT_CSV = "fundamentals_AAPL_clean.csv"
TICKER     = "AAPL"

FUNDAMENTAL_COLS = [
    "Revenue","COGS","EBIT","SGA","NetIncome",
    "Cash","Receivables","Inventories","OtherCurrentAssets",
    "PPE","OtherAssets","DebtCurrent","AccountsPayable",
    "OtherCurrentLiabilities","LiabilitiesCurrent","TotalLiabilities"
]
MOMENTUM_COLS = ["1m_return","3m_return","6m_return","9m_return"]
ALL_FEATURES = FUNDAMENTAL_COLS + MOMENTUM_COLS

def parse_period_series(s: pd.Series) -> pd.Series:
    def _parse_one(val: str):
        if not isinstance(val, str):
            return pd.NaT
        v = val.strip().upper().replace("-", " ")
        if v.startswith("FY"):
            v = v[2:].strip()
        parts = v.split()
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].startswith("Q"):
            year = int(parts[0]); q = int(parts[1][1])
            month = {1:3, 2:6, 3:9, 4:12}[q]
            return pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
        try:
            return pd.to_datetime(val)
        except:
            return pd.NaT
    return s.apply(_parse_one)

# --- Step 1: save clean timestep CSV ---
df = pd.read_csv(INPUT_CSV)

if "Period" in df.columns:
    df["Period"] = parse_period_series(df["Period"])
elif "Date" in df.columns:
    df["Period"] = pd.to_datetime(df["Date"])
else:
    raise ValueError("CSV must have Period or Date column")

df = df.sort_values("Period")
df["Ticker"] = TICKER

df = df.set_index("Period").asfreq("M").ffill()
df["Ticker"] = TICKER
df = df.reset_index()

df = df[["Ticker","Period"] + ALL_FEATURES]
df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved clean CSV: {OUTPUT_CSV} | shape={df.shape}")

# =====================================================
# Step 2: Helper to build windows
# =====================================================

def make_training_windows(csv_path, window_years=5, horizon_months=12, model_type="rnn"):
    """
    Build training windows for MLP or RNN.

    Returns:
        X: np.ndarray (N, steps, features) or (N, steps*features)
        y: np.ndarray (N, targets)
        t_index: np.ndarray (N,)
    """
    df = pd.read_csv(csv_path, parse_dates=["Period"])
    steps = [12*i for i in range(window_years)][::-1]  # [48,36,24,12,0]

    X_list, y_list, t_list = [], [], []
    periods = df["Period"].tolist()
    idx_by_date = {pd.Timestamp(p): i for i,p in enumerate(periods)}

    for i, t in enumerate(periods):
        ok = True; past_rows = []
        for k in steps:
            dt = t - pd.DateOffset(months=k)
            if dt in idx_by_date:
                past_rows.append(df.loc[idx_by_date[dt], ALL_FEATURES].astype(float).values)
            else:
                ok = False; break
        tgt_dt = t + pd.DateOffset(months=horizon_months)
        if ok and (tgt_dt in idx_by_date):
            target = df.loc[idx_by_date[tgt_dt], FUNDAMENTAL_COLS].astype(float).values
            if np.any(np.isnan(target)) or np.any(np.isnan(past_rows)):
                continue
            if model_type == "rnn":
                X_list.append(np.stack(past_rows).astype(np.float32))       # (steps, features)
            else:
                X_list.append(np.concatenate(past_rows).astype(np.float32)) # flattened
            y_list.append(target.astype(np.float32))
            t_list.append(t)

    return (
        np.array(X_list, dtype=np.float32),
        np.array(y_list, dtype=np.float32),
        np.array(t_list)
    )
