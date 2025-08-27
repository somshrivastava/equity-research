# backtest_top5_latest.py
#
# Rolling Top-N Strategy (latest EBIT/EV values):
# - Maintain a dictionary of most recent EBIT/EV for each ticker.
# - For each unique reporting date, update all tickers that reported.
# - After all updates for that date, save one snapshot of the top N.

import pandas as pd
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
            "portfolio_size": 5,
            "model_type": "mlp",
            "start_date": "2005-01-01",
            "use_predicted_ebit": True,
            "session_id": "default"
        }

config = load_config()

# === Inputs ===
session_id = config.get("session_id", "default")
INPUT_FILE = f"ebit_ev_{session_id}.csv" if session_id != "default" else "ebit_ev.csv"
OUTPUT_FILE = f"portfolio_top5_{session_id}.csv" if session_id != "default" else "portfolio_top5.csv"
RATIO_COL = "Pred_EBIT_EV" if config["use_predicted_ebit"] else "True_EBIT_EV"
PORTFOLIO_SIZE = config["portfolio_size"]

print(f"Using {'Predicted' if config['use_predicted_ebit'] else 'True'} EBIT/EV ratios for portfolio selection")
print(f"Session ID: {session_id}")

# === Load data ===
df = pd.read_csv(INPUT_FILE, parse_dates=["anchor_date"])
df = df.sort_values("anchor_date")

# Restrict to start date from config
df = df[df["anchor_date"] >= config["start_date"]]

# === Rolling top-5 logic ===
latest_scores = {}    # {tic: latest EBIT/EV}
portfolio_history = []

# Loop over unique dates instead of every row
for date, group in df.groupby("anchor_date"):
    # Update all tickers reporting on this date
    for _, row in group.iterrows():
        latest_scores[row["tic"]] = row[RATIO_COL]

    # Rank all tickers by most recent score
    ranked = sorted(latest_scores.items(), key=lambda x: x[1], reverse=True)

    # Select top N based on portfolio size
    topN = ranked[:PORTFOLIO_SIZE]

    # Save one snapshot for this date
    portfolio_history.append({
        "date": date,
        "holdings": ",".join([t for t, _ in topN]),
        "scores": ",".join([f"{t}:{s:.4f}" for t, s in topN])
    })

# === Save output ===
out = pd.DataFrame(portfolio_history)
out.to_csv(OUTPUT_FILE, index=False)
print(f"Saved rolling top-{PORTFOLIO_SIZE} portfolio to {OUTPUT_FILE} | {len(out)} unique dates")
