# backtest.py
#
# Denormalize EBIT predictions and compute proper EBIT/EV ratios.

import pandas as pd
import numpy as np
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
            "input_csv": "wrds.csv",
            "model_type": "mlp",
            "session_id": "default"
        }

config = load_config()

# === Inputs ===
session_id = config.get("session_id", "default")
PRED_FILE = f"predictions_{config['model_type']}_{session_id}.csv" if session_id != "default" else f"predictions_{config['model_type']}.csv"
WRDS_FILE = config["input_csv"]      # Use input file from config
OUTPUT_FILE = f"ebit_ev_{session_id}.csv" if session_id != "default" else "ebit_ev.csv"

print(f"Session ID: {session_id}")

# === Load data ===
preds = pd.read_csv(PRED_FILE, parse_dates=["anchor_date"])
wrds  = pd.read_csv(WRDS_FILE)
wrds.columns = [c.lower() for c in wrds.columns]
wrds["datadate"] = pd.to_datetime(wrds["datadate"])

# === Recompute EBIT normalization stats (μ, σ) ===
# If oiadpq (Compustat EBIT) not present, approximate as revtq - cogsq - xsgaq
if "oiadpq" in wrds.columns:
    wrds["ebit_raw"] = wrds["oiadpq"]
else:
    wrds["ebit_raw"] = (
        wrds["revtq"].fillna(0)
        - wrds["cogsq"].fillna(0)
        - wrds["xsgaq"].fillna(0)
    )

# Scale EBIT by MarketCap (as training did)
wrds["ebit_scaled"] = wrds["ebit_raw"] / wrds["mkvaltq"]

mu_ebit = wrds["ebit_scaled"].mean()
sd_ebit = wrds["ebit_scaled"].std(ddof=0)

print(f"EBIT z-score params: mu={mu_ebit:.6f}, sigma={sd_ebit:.6f}")

# === Compute EV with new formula ===
wrds["EV"] = (
    wrds["mkvaltq"].fillna(0)
    + wrds["dlttq"].fillna(0)
    + wrds["lctq"].fillna(0)
    - wrds["dd1q"].fillna(0)
    - wrds["chq"].fillna(0)
)

# === Merge WRDS with predictions ===
df = preds.merge(
    wrds[["tic","datadate","mkvaltq","EV"]],
    left_on=["tic","anchor_date"],
    right_on=["tic","datadate"],
    how="left"
).drop(columns=["datadate"])

# === Denormalize EBIT ===
# Pred_EBIT and True_EBIT in preds are z-scored EBIT/Cap values.
# So invert: EBIT_dollar = ((z * σ) + μ) * MarketCap
df["Pred_EBIT_dollar"] = ((df["Pred_EBIT"] * sd_ebit) + mu_ebit) * df["mkvaltq"]
df["True_EBIT_dollar"] = ((df["True_EBIT"] * sd_ebit) + mu_ebit) * df["mkvaltq"]

# === Compute EBIT/EV ratios ===
df["Pred_EBIT_EV"] = df["Pred_EBIT_dollar"] / df["EV"]
df["True_EBIT_EV"] = df["True_EBIT_dollar"] / df["EV"]

# === Save output ===
out = df[[
    "tic","anchor_date","mkvaltq","EV",
    "Pred_EBIT_dollar","True_EBIT_dollar",
    "Pred_EBIT_EV","True_EBIT_EV"
]].sort_values(by=["anchor_date", "Pred_EBIT_EV"], ascending=False)
out.to_csv(OUTPUT_FILE, index=False)
print(f"Saved {OUTPUT_FILE} with {len(out)} rows")
