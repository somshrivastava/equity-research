# models.py
import torch, torch.nn as nn

class LFM_MLP(nn.Module):
    def __init__(self, in_dim=100, out_dim=16, hidden=1024, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.BatchNorm1d(hidden), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.BatchNorm1d(hidden), nn.Dropout(dropout),
            nn.Linear(hidden, out_dim)
        )
    def forward(self, x):  # x: (B, 100)
        return self.net(x)

class LFM_LSTM(nn.Module):
    def __init__(self, in_feat=20, hidden=64, layers=2, out_dim=16):
        super().__init__()
        self.lstm = nn.LSTM(in_feat, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Linear(hidden, out_dim)
    def forward(self, x):  # x: (B, 5, 20)
        out, _ = self.lstm(x)     # (B, 5, H)
        last = out[:, -1, :]      # (B, H) — predict for t+12 from final step
        return self.head(last)    # (B, 16)
