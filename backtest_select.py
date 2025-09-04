import pandas as pd
import numpy as np
from config import load_config, get_output_path

## Load configuration
config = load_config()
session_id = config.get("session_id", "default")
INPUT_FILE = get_output_path("ebit_ev.csv", session_id)
OUTPUT_FILE = get_output_path("portfolio_top5.csv", session_id)
RATIO_COL = "Pred_EBIT_EV" if config["use_predicted_ebit"] else "True_EBIT_EV"
PORTFOLIO_SIZE = config["portfolio_size"]

## Load and filter data
df = pd.read_csv(INPUT_FILE, parse_dates=["anchor_date"])
df = df.sort_values("anchor_date")
df = df[df["anchor_date"] >= config["start_date"]]

## Rolling top-N portfolio selection
latest_scores = {}  # Track most recent EBIT/EV for each ticker
portfolio_history = []

for date, group in df.groupby("anchor_date"):
    # Update scores for all tickers reporting on this date
    for _, row in group.iterrows():
        latest_scores[row["tic"]] = row[RATIO_COL]
    
    # Rank all tickers and select top N
    ranked = sorted(latest_scores.items(), key=lambda x: x[1], reverse=True)
    topN = ranked[:PORTFOLIO_SIZE]
    
    # Save portfolio snapshot for this date
    portfolio_history.append({
        "date": date,
        "holdings": ",".join([t for t, _ in topN]),
        "scores": ",".join([f"{t}:{s:.4f}" for t, s in topN])
    })

## Save results
out = pd.DataFrame(portfolio_history)
out.to_csv(OUTPUT_FILE, index=False)
