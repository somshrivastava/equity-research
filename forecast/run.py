# run.py
#
# End-to-end training + prediction for Lookahead Factor Models (MLP or RNN).
# After training, saves predictions vs actuals to predictions.csv.

import torch
import numpy as np
import pandas as pd
import json
import os

from prepare_fundamentals_csv import make_training_windows, FUNDAMENTAL_COLS
from dataset import make_loaders
from models import LFM_MLP, LFM_LSTM
from train import train_model
from train_utils import mse_by_feature

# =======================
# Config Loading
# =======================
def load_config():
    config_path = "config.json"
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return json.load(f)
    else:
        # Default values for when config.json doesn't exist (backwards compatibility)
        return {
            "clean_csv": "fundamentals_AAPL_clean.csv",
            "model_type": "mlp",
            "batch_size": 128,
            "lr": 1e-3,
            "epochs": 200,
            "patience": 25,
            "session_id": "default"
        }

config = load_config()

session_id = config.get("session_id", "default")
print(f"Session ID: {session_id}")

CLEAN_CSV   = config["clean_csv"]
MODEL_TYPE  = config["model_type"]
BATCH_SIZE  = config["batch_size"]
LR          = config["lr"]
EPOCHS      = config["epochs"]
PATIENCE    = config["patience"]
ALPHA1      = 0.75 if MODEL_TYPE=="mlp" else 0.5

# =======================
# Step 1: Make training windows
# =======================
X, y, t_idx = make_training_windows(
    CLEAN_CSV,
    window_years=5,
    horizon_months=12,
    model_type=MODEL_TYPE
)
X, y = X.astype("float32"), y.astype("float32")

# Normalize inputs and targets (z-score)
X_mean, X_std = X.mean(), X.std()
y_mean, y_std = y.mean(), y.std()
X = (X - X_mean) / (X_std + 1e-8)
y = (y - y_mean) / (y_std + 1e-8)

# =======================
# Step 2: Build loaders
# =======================
train_loader, val_loader = make_loaders(X, y, batch=BATCH_SIZE, val_frac=0.3)

# =======================
# Step 3: Init model
# =======================
if MODEL_TYPE == "mlp":
    model = LFM_MLP(in_dim=X.shape[1], out_dim=y.shape[1])
else:
    model = LFM_LSTM(in_feat=X.shape[2], out_dim=y.shape[1])

# =======================
# Step 4: Train
# =======================
model, best_val = train_model(
    model,
    train_loader,
    val_loader,
    max_epochs=EPOCHS,
    patience=PATIENCE,
    lr=LR,
    alpha1=ALPHA1
)

print(f"Finished training. Best validation MSE = {best_val:.4f}")

# =======================
# Step 5: Evaluate
# =======================
mse_feat, mse_avg = mse_by_feature(model, X, y, FUNDAMENTAL_COLS)
print("Per-feature MSE:", mse_feat)
print("Average MSE:", mse_avg)

# Save metrics to file
metrics_file = f"metrics_{MODEL_TYPE}_{session_id}.txt" if session_id != "default" else f"metrics_{MODEL_TYPE}.txt"
with open(metrics_file, "w") as f:
    f.write(f"Model Type: {MODEL_TYPE}\n")
    f.write(f"Best Validation MSE: {best_val:.4f}\n")
    f.write(f"Average MSE: {mse_avg:.4f}\n")
    f.write("Per-feature MSE:\n")
    for col, mse in zip(FUNDAMENTAL_COLS, mse_feat):
        f.write(f"  {col}: {mse:.4f}\n")
print(f"Saved metrics to {metrics_file}")

# =======================
# Step 6: Generate Predictions
# =======================
model.eval()
with torch.no_grad():
    X_tensor = torch.tensor(X, dtype=torch.float32)
    preds = model(X_tensor).numpy()

# Inverse transform targets
preds_denorm = preds * (y_std + 1e-8) + y_mean
y_true_denorm = y * (y_std + 1e-8) + y_mean

# Save to CSV
pred_df = pd.DataFrame(preds_denorm, columns=[f"Pred_{c}" for c in FUNDAMENTAL_COLS])
true_df = pd.DataFrame(y_true_denorm, columns=[f"True_{c}" for c in FUNDAMENTAL_COLS])
out_df = pd.concat([pd.Series(t_idx, name="Period"), pred_df, true_df], axis=1)

predictions_file = f"predictions_{MODEL_TYPE}_{session_id}.csv" if session_id != "default" else f"predictions_{MODEL_TYPE}.csv"
out_df.to_csv(predictions_file, index=False)

print(f"Saved predictions to {predictions_file}")
