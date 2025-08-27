# run_wrds_sp50.py
#
# End-to-end replication of the paper's setup using WRDS fundamentals + Yahoo Finance prices.
# Input:  WRDS SP50.csv  (quarterly fundamentals, Compustat style, with tic, datadate, mkvaltq, revtq, cogsq, etc.)
# Output:
#   - metrics_mlp.txt / metrics_lstm.txt (MSE per feature)
#   - predictions_mlp.csv / predictions_lstm.csv (Pred_* vs True_*)

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path
import yfinance as yf
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import sys

# -----------------------------
# Config
# -----------------------------
def load_config():
    """Load configuration from command line argument or use defaults"""
    if len(sys.argv) > 1 and sys.argv[1] == "--config" and len(sys.argv) > 2:
        with open(sys.argv[2], 'r') as f:
            config = json.load(f)
        return config
    else:
        # Default configuration
        return {
            "input_csv": "wrds.csv",
            "batch_size": 256,
            "learning_rate": 1e-3,
            "epochs": 200,
            "patience": 25,
            "alpha1_mlp": 0.75,
            "alpha1_lstm": 0.5,
            "fundamental_features": ["revtq", "cogsq", "xsgaq", "niq", "chq", "rectq", "invtq", "acoq", "ppentq", "aoq", "dlcq", "apq", "txpq", "lcoq", "ltq"],
            "momentum_features": ["1 month", "3 months", "6 months", "9 months"],
            "session_id": "default"
        }

config = load_config()
session_id = config.get("session_id", "default")
print(f"Session ID: {session_id}")

INPUT_CSV = config["input_csv"]
BATCH = config["batch_size"]
LR = config["learning_rate"]
EPOCHS = config["epochs"]
PATIENCE = config["patience"]
ALPHA1_MLP = config["alpha1_mlp"]
ALPHA1_LSTM = config["alpha1_lstm"]

STEPS = [16, 12, 8, 4, 0]   # quarterly offsets (≈ yearly snapshots)
HORIZON_Q = 4               # predict 4 quarters (~12 months) ahead

# Map user selections to processing columns
momentum_mapping = {
    "1 month": "mom_1m",
    "3 months": "mom_3m", 
    "6 months": "mom_6m",
    "9 months": "mom_9m"
}

# Create momentum columns based on user selection
MOMENTUM_COLS = [momentum_mapping[period] for period in config["momentum_features"] if period in momentum_mapping]

# Build the fundamental columns list based on what features were selected and successfully created
# This will be updated after we process the data
FUNDAMENTAL_COLS = [
    "Revenue","COGS","EBIT","SGA","NetIncome",
    "Cash","Receivables","Inventories","OtherCurrentAssets",
    "PPE","OtherAssets","DebtCurrent","AccountsPayable",
    "TaxesPayable","LiabilitiesCurrent","TotalLiabilities"
]
ALL_FEATURES = FUNDAMENTAL_COLS + MOMENTUM_COLS
TARGET_COLS = FUNDAMENTAL_COLS
EBIT_IDX = None  # Will be set later if EBIT exists

# -----------------------------
# Utilities
# -----------------------------
def safe_div(x, y):
    return x.div(y.replace(0, np.nan)).fillna(0.0)

def ttm_sum(s):
    return s.rolling(4, min_periods=4).sum()

def zscore_cols(df, cols):
    for c in cols:
        mu = df[c].mean()
        sd = df[c].std()
        if not np.isfinite(sd) or sd == 0:
            sd = 1.0
        df[c] = (df[c] - mu) / sd

def compute_momentum_features(tickers, start, end):
    """Fetch monthly prices from yfinance and compute selected momentum return percentiles."""
    data = yf.download(tickers, start=start, end=end, interval="1mo", group_by="ticker", auto_adjust=True)

    # Determine which momentum periods to compute based on config
    periods_to_compute = {}
    for period in config["momentum_features"]:
        if period == "1 month":
            periods_to_compute["ret_1m"] = 1
        elif period == "3 months":
            periods_to_compute["ret_3m"] = 3
        elif period == "6 months":
            periods_to_compute["ret_6m"] = 6
        elif period == "9 months":
            periods_to_compute["ret_9m"] = 9

    frames = []
    for tic in tickers:
        try:
            prices = data[tic]["Close"].dropna()
            rets_dict = {}
            for ret_name, period in periods_to_compute.items():
                rets_dict[ret_name] = prices.pct_change(period)
            
            rets = pd.DataFrame(rets_dict)
            rets["tic"] = tic
            frames.append(rets)
        except Exception as e:
            print(f"Skipping {tic}: {e}")
    
    if not frames:
        # Return empty dataframe with expected structure if no data
        return pd.DataFrame(columns=["Date", "tic"] + list(momentum_mapping.values()))
    
    mom = pd.concat(frames).reset_index().rename(columns={"index":"Date"})

    # Cross-sectional percentiles per month for computed periods
    for ret_col in periods_to_compute.keys():
        mom[ret_col+"_pct"] = mom.groupby("Date")[ret_col].rank(pct=True)

    # Rename to final momentum column names
    rename_dict = {}
    for period in config["momentum_features"]:
        if period == "1 month":
            rename_dict["ret_1m_pct"] = "mom_1m"
        elif period == "3 months":
            rename_dict["ret_3m_pct"] = "mom_3m"
        elif period == "6 months":
            rename_dict["ret_6m_pct"] = "mom_6m"
        elif period == "9 months":
            rename_dict["ret_9m_pct"] = "mom_9m"
    
    mom = mom.rename(columns=rename_dict)
    
    # Return only the columns we need
    final_cols = ["Date", "tic"] + [momentum_mapping[period] for period in config["momentum_features"] if period in momentum_mapping]
    return mom[final_cols]

# -----------------------------
# Load WRDS fundamentals
# -----------------------------
raw = pd.read_csv(INPUT_CSV)
raw.columns = [c.lower() for c in raw.columns]
df = raw.copy()
df["datadate"] = pd.to_datetime(df["datadate"])
df = df.sort_values(["tic","datadate"]).reset_index(drop=True)

# Build TTM & MRQ features
def build_features(g):
    g = g.sort_values("datadate").copy()
    
    # Only build features for columns that exist in the data and were selected by user
    selected_fundamental_cols = config["fundamental_features"]
    
    # Map fundamental features to their calculations
    if "revtq" in selected_fundamental_cols and "revtq" in g.columns:
        g["Revenue"] = ttm_sum(g["revtq"])
    if "cogsq" in selected_fundamental_cols and "cogsq" in g.columns:
        g["COGS"] = ttm_sum(g["cogsq"])
    if "xsgaq" in selected_fundamental_cols and "xsgaq" in g.columns:
        g["SGA"] = ttm_sum(g["xsgaq"])
    if "niq" in selected_fundamental_cols and "niq" in g.columns:
        g["NetIncome"] = ttm_sum(g["niq"])
    
    # Calculate EBIT if we have the components
    if all(col in g.columns for col in ["Revenue", "COGS", "SGA"]):
        g["EBIT"] = g["Revenue"] - g["COGS"] - g["SGA"]
    
    # MRQ features
    if "chq" in selected_fundamental_cols and "chq" in g.columns:
        g["Cash"] = g["chq"]
    if "rectq" in selected_fundamental_cols and "rectq" in g.columns:
        g["Receivables"] = g["rectq"]
    if "invtq" in selected_fundamental_cols and "invtq" in g.columns:
        g["Inventories"] = g["invtq"]
    if "acoq" in selected_fundamental_cols and "acoq" in g.columns:
        g["OtherCurrentAssets"] = g["acoq"]
    if "ppentq" in selected_fundamental_cols and "ppentq" in g.columns:
        g["PPE"] = g["ppentq"]
    if "aoq" in selected_fundamental_cols and "aoq" in g.columns:
        g["OtherAssets"] = g["aoq"]
    if "dlcq" in selected_fundamental_cols and "dlcq" in g.columns:
        g["DebtCurrent"] = g["dlcq"]
    if "apq" in selected_fundamental_cols and "apq" in g.columns:
        g["AccountsPayable"] = g["apq"]
    if "txpq" in selected_fundamental_cols and "txpq" in g.columns:
        g["TaxesPayable"] = g["txpq"]
    if "lcoq" in selected_fundamental_cols and "lcoq" in g.columns:
        g["LiabilitiesCurrent"] = g["lcoq"]
    if "ltq" in selected_fundamental_cols and "ltq" in g.columns:
        g["TotalLiabilities"] = g["ltq"]
    
    return g

df = df.groupby("tic", group_keys=False).apply(build_features)

# Filter to only keep rows where we have the key financial data
required_cols = [col for col in ["Revenue","COGS","SGA","NetIncome","EBIT"] if col in df.columns]
for col in required_cols:
    df = df[~df[col].isna()]

# -----------------------------
# Fetch momentum features
# -----------------------------
tickers = df["tic"].unique().tolist()
start, end = df["datadate"].min(), df["datadate"].max()
mom_feats = compute_momentum_features(tickers, start, end)

# Merge monthly momentum into fundamentals (align to quarter-end)
mom_feats["Date"] = pd.to_datetime(mom_feats["Date"]) + pd.offsets.MonthEnd(0)
df = df.merge(mom_feats, left_on=["datadate","tic"], right_on=["Date","tic"], how="left")
df = df.drop(columns=["Date"])

# Update feature lists based on what's actually available in the processed data
available_fundamental_cols = [col for col in FUNDAMENTAL_COLS if col in df.columns]
available_momentum_cols = [col for col in MOMENTUM_COLS if col in df.columns]

FUNDAMENTAL_COLS = available_fundamental_cols
MOMENTUM_COLS = available_momentum_cols
ALL_FEATURES = FUNDAMENTAL_COLS + MOMENTUM_COLS
TARGET_COLS = FUNDAMENTAL_COLS

# Set EBIT index if EBIT is available
if "EBIT" in FUNDAMENTAL_COLS:
    EBIT_IDX = FUNDAMENTAL_COLS.index("EBIT")
else:
    EBIT_IDX = 0  # Default to first feature if EBIT not available

# -----------------------------
# Scale fundamentals by MarketCap, z-score everything
# -----------------------------
for c in FUNDAMENTAL_COLS:
    df[c] = safe_div(df[c], df["mkvaltq"])
feat_df = df[["tic","datadate"] + ALL_FEATURES].copy()
zscore_cols(feat_df, ALL_FEATURES)

# -----------------------------
# Build windows (5 steps, horizon=+4q)
# -----------------------------
def make_windows(panel, steps=STEPS, horizon=HORIZON_Q):
    X_list,y_list,t_list,tic_list=[],[],[],[]
    for tic,g in panel.groupby("tic"):
        g=g.sort_values("datadate").reset_index(drop=True)
        idx={pd.Timestamp(d):i for i,d in enumerate(g["datadate"])}
        dates=g["datadate"].tolist()
        for t in dates:
            past=[]
            ok=True
            for k in steps:
                dt=t-pd.DateOffset(months=3*k)
                if dt in idx:
                    past.append(g.loc[idx[dt],ALL_FEATURES].values.astype(np.float32))
                else: ok=False; break
            tgt_dt=t+pd.DateOffset(months=3*horizon)
            if ok and (tgt_dt in idx):
                X_seq=np.stack(past)
                y_vec=g.loc[idx[tgt_dt],FUNDAMENTAL_COLS].values.astype(np.float32)
                if np.any(np.isnan(X_seq)) or np.any(np.isnan(y_vec)): continue
                X_list.append(X_seq); y_list.append(y_vec); t_list.append(t); tic_list.append(tic)
    return np.array(X_list), np.array(y_list), np.array(t_list), np.array(tic_list)

X_seq,y,t_idx,tics=make_windows(feat_df)

# -----------------------------
# Torch Dataset/Models/Training
# -----------------------------
class WindowDS(Dataset):
    def __init__(self,X,y,kind="rnn"):
        self.kind=kind
        if kind=="mlp":
            self.X=torch.tensor(X.reshape(len(X),-1),dtype=torch.float32)
        else:
            self.X=torch.tensor(X,dtype=torch.float32)
        self.y=torch.tensor(y,dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self,i): return self.X[i],self.y[i]

def make_loaders(X,y,kind):
    n=len(X); idx=np.arange(n); np.random.shuffle(idx)
    n_val=int(n*0.3); val_idx, tr_idx=idx[:n_val], idx[n_val:]
    return (DataLoader(WindowDS(X[tr_idx],y[tr_idx],kind),batch_size=BATCH,shuffle=True),
            DataLoader(WindowDS(X[val_idx],y[val_idx],kind),batch_size=BATCH,shuffle=False))

class LFM_MLP(nn.Module):
    def __init__(self,in_dim=100,out_dim=16,hidden=1024,dropout=0.5):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(in_dim,hidden),nn.ReLU(),nn.BatchNorm1d(hidden),nn.Dropout(dropout),
            nn.Linear(hidden,hidden),nn.ReLU(),nn.BatchNorm1d(hidden),nn.Dropout(dropout),
            nn.Linear(hidden,out_dim))
    def forward(self,x): return self.net(x)

class LFM_LSTM(nn.Module):
    def __init__(self,in_feat=20,hidden=64,layers=2,out_dim=16):
        super().__init__()
        self.lstm=nn.LSTM(in_feat,hidden,num_layers=layers,batch_first=True)
        self.head=nn.Linear(hidden,out_dim)
    def forward(self,x):
        out,_=self.lstm(x); return self.head(out[:,-1,:])

def weighted_mse_loss(pred,target,alpha1=0.75):
    w=torch.ones(target.shape[-1],device=pred.device)
    w[EBIT_IDX]=1.0+alpha1
    return ((pred-target)**2*w).mean()

def train_model(model,tr_loader,va_loader,alpha1,max_epochs=EPOCHS,patience=PATIENCE,lr=LR):
    device="cuda" if torch.cuda.is_available() else "cpu"
    model=model.to(device); opt=torch.optim.Adam(model.parameters(),lr=lr)
    best=float("inf"); bad=0; best_state=None
    for ep in range(1,max_epochs+1):
        model.train(); tr_loss=0.0
        for xb,yb in tr_loader:
            xb,yb=xb.to(device),yb.to(device); opt.zero_grad()
            pred=model(xb); loss=weighted_mse_loss(pred,yb,alpha1); loss.backward(); opt.step()
            tr_loss+=loss.item()*len(xb)
        tr_loss/=len(tr_loader.dataset)
        model.eval(); va_loss=0.0
        with torch.no_grad():
            for xb,yb in va_loader:
                xb,yb=xb.to(device),yb.to(device)
                pred=model(xb); loss=weighted_mse_loss(pred,yb,alpha1)
                va_loss+=loss.item()*len(xb)
        va_loss/=len(va_loader.dataset)
        print(f"epoch {ep:03d} | train {tr_loss:.4f} | val {va_loss:.4f}")
        if va_loss<best-1e-6: best=va_loss; best_state={k:v.cpu() for k,v in model.state_dict().items()}; bad=0
        else: bad+=1; 
        if bad>=patience: print("early stopping."); break
    if best_state: model.load_state_dict(best_state)
    return model,best

def mse_by_feature(model,X,y,kind):
    device="cuda" if torch.cuda.is_available() else "cpu"; model.eval()
    with torch.no_grad():
        Xb=torch.tensor(X.reshape(len(X),-1),dtype=torch.float32) if kind=="mlp" else torch.tensor(X,dtype=torch.float32)
        pred=model(Xb.to(device)).cpu().numpy()
    mse_vec=((pred-y)**2).mean(axis=0)
    return dict(zip(FUNDAMENTAL_COLS,mse_vec)), float(mse_vec.mean()), pred

def save_predictions_csv(fn,preds,y_true,t_idx,tics):
    out=pd.DataFrame({"tic":tics,"anchor_date":t_idx})
    pred_df=pd.DataFrame(preds,columns=[f"Pred_{c}" for c in FUNDAMENTAL_COLS])
    true_df=pd.DataFrame(y_true,columns=[f"True_{c}" for c in FUNDAMENTAL_COLS])
    out=pd.concat([out,pred_df,true_df],axis=1); out.to_csv(fn,index=False)

# -----------------------------
# Train & Evaluate
# -----------------------------
print("\n=== Training MLP ===")
tr_mlp,va_mlp=make_loaders(X_seq,y,"mlp")
mlp=LFM_MLP(in_dim=X_seq.shape[1]*X_seq.shape[2],out_dim=y.shape[1])
mlp,best_val=train_model(mlp,tr_mlp,va_mlp,alpha1=ALPHA1_MLP)
per_feat,avg,preds=mse_by_feature(mlp,X_seq,y,"mlp")

# Use session-specific filename
mlp_predictions_file = f"predictions_mlp_{session_id}.csv" if session_id != "default" else "predictions_mlp.csv"
save_predictions_csv(mlp_predictions_file,preds,y,t_idx,tics)

per_feat, avg, preds = mse_by_feature(mlp, X_seq, y, "mlp")
print("MLP avg MSE:", avg)
for k,v in per_feat.items():
    print(f"{k}: {v:.4f}")

# Save MLP metrics to file
mlp_metrics_file = f"metrics_mlp_{session_id}.txt" if session_id != "default" else "metrics_mlp.txt"
with open(mlp_metrics_file, "w") as f:
    f.write(f"Model Type: MLP\n")
    f.write(f"Average MSE: {avg:.4f}\n")
    f.write("Per-feature MSE:\n")
    for k,v in per_feat.items():
        f.write(f"  {k}: {v:.4f}\n")
print(f"Saved MLP metrics to {mlp_metrics_file}")

print("\n=== Training LSTM ===")
tr_lstm,va_lstm=make_loaders(X_seq,y,"rnn")
lstm=LFM_LSTM(in_feat=X_seq.shape[2],out_dim=y.shape[1])
lstm,best_val=train_model(lstm,tr_lstm,va_lstm,alpha1=ALPHA1_LSTM)
per_feat,avg,preds=mse_by_feature(lstm,X_seq,y,"rnn")

# Use session-specific filename
lstm_predictions_file = f"predictions_lstm_{session_id}.csv" if session_id != "default" else "predictions_lstm.csv"
save_predictions_csv(lstm_predictions_file,preds,y,t_idx,tics)

per_feat, avg, preds = mse_by_feature(lstm, X_seq, y, "lstm")
print("LSTM avg MSE:", avg)
for k,v in per_feat.items():
    print(f"{k}: {v:.4f}")

# Save LSTM metrics to file  
lstm_metrics_file = f"metrics_lstm_{session_id}.txt" if session_id != "default" else "metrics_lstm.txt"
with open(lstm_metrics_file, "w") as f:
    f.write(f"Model Type: LSTM\n")
    f.write(f"Average MSE: {avg:.4f}\n")
    f.write("Per-feature MSE:\n")
    for k,v in per_feat.items():
        f.write(f"  {k}: {v:.4f}\n")
print(f"Saved LSTM metrics to {lstm_metrics_file}")

print("\n=== DONE ===")
