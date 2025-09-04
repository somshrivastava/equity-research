import pandas as pd
import numpy as np
import yfinance as yf
from config import load_config, get_output_path

## Load configuration
config = load_config()
session_id = config.get("session_id", "default")
PRED_FILE = get_output_path(f"predictions_{config['model_type']}.csv", session_id)
WRDS_FILE = config["input_csv"]
OUTPUT_FILE = get_output_path("ebit_ev.csv", session_id)

## Load and prepare data
preds = pd.read_csv(PRED_FILE, parse_dates=["anchor_date"])
wrds = pd.read_csv(WRDS_FILE)
wrds.columns = [c.lower() for c in wrds.columns]
wrds["datadate"] = pd.to_datetime(wrds["datadate"])

## Calculate EBIT (use oiadpq if available, otherwise approximate)
if "oiadpq" in wrds.columns:
    wrds["ebit_raw"] = wrds["oiadpq"]
else:
    wrds["ebit_raw"] = wrds["revtq"].fillna(0) - wrds["cogsq"].fillna(0) - wrds["xsgaq"].fillna(0)

## Compute normalization stats for denormalization
wrds["ebit_scaled"] = wrds["ebit_raw"] / wrds["mkvaltq"]
mu_ebit = wrds["ebit_scaled"].mean()
sd_ebit = wrds["ebit_scaled"].std(ddof=0)

## Calculate Enterprise Value
wrds["EV"] = (wrds["mkvaltq"].fillna(0) + wrds["dlttq"].fillna(0) + wrds["lctq"].fillna(0) 
              - wrds["dd1q"].fillna(0) - wrds["chq"].fillna(0))

## Merge predictions with WRDS data
df = preds.merge(wrds[["tic","datadate","mkvaltq","EV"]], 
                 left_on=["tic","anchor_date"], right_on=["tic","datadate"], how="left").drop(columns=["datadate"])

## Denormalize EBIT predictions back to dollar amounts
df["Pred_EBIT_dollar"] = ((df["Pred_EBIT"] * sd_ebit) + mu_ebit) * df["mkvaltq"]
df["True_EBIT_dollar"] = ((df["True_EBIT"] * sd_ebit) + mu_ebit) * df["mkvaltq"]

## Calculate EBIT/EV ratios
df["Pred_EBIT_EV"] = df["Pred_EBIT_dollar"] / df["EV"]
df["True_EBIT_EV"] = df["True_EBIT_dollar"] / df["EV"]

## Save results
out = df[["tic","anchor_date","mkvaltq","EV","Pred_EBIT_dollar","True_EBIT_dollar","Pred_EBIT_EV","True_EBIT_EV"]].sort_values(by=["anchor_date", "Pred_EBIT_EV"], ascending=False)
out.to_csv(OUTPUT_FILE, index=False)
