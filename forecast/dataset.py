# dataset.py
import numpy as np, torch
from torch.utils.data import Dataset, DataLoader

EBIT_IDX = 2  # y[..., 2] is EBIT (per your column order)

class LFMWindowDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i], self.y[i]

def make_loaders(X, y, batch=128, val_frac=0.3, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(X)); rng.shuffle(idx)
    n_val = int(len(idx)*val_frac)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    ds_tr, ds_val = LFMWindowDataset(X[tr_idx], y[tr_idx]), LFMWindowDataset(X[val_idx], y[val_idx])
    return (DataLoader(ds_tr, batch_size=batch, shuffle=True),
            DataLoader(ds_val, batch_size=batch, shuffle=False))
