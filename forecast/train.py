# train.py
import torch, torch.nn as nn, torch.optim as optim
from dataset import make_loaders, EBIT_IDX
from models import LFM_MLP, LFM_LSTM
from train_utils import weighted_mse_loss, make_target_weights

def train_model(model, train_loader, val_loader, max_epochs=200, patience=25, lr=1e-3, alpha1=0.75, device="cuda"):
    device = device if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    w = make_target_weights(alpha1=alpha1, ebit_idx=EBIT_IDX).to(device)
    best_val, best_state, epochs_no_improve = float("inf"), None, 0

    for epoch in range(1, max_epochs+1):
        model.train(); tr_loss=0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(Xb)
            loss = weighted_mse_loss(pred, yb, w)
            loss.backward(); opt.step()
            tr_loss += loss.item()*len(Xb)
        tr_loss /= len(train_loader.dataset)

        model.eval(); val_loss=0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                pred = model(Xb)
                loss = weighted_mse_loss(pred, yb, w)
                val_loss += loss.item()*len(Xb)
        val_loss /= len(val_loader.dataset)

        print(f"epoch {epoch:03d} | train {tr_loss:.4f} | val {val_loss:.4f}")

        if val_loss < best_val - 1e-6:
            best_val, best_state, epochs_no_improve = val_loss, {k:v.cpu() for k,v in model.state_dict().items()}, 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("early stopping.")
                break

    if best_state: model.load_state_dict(best_state)
    return model, best_val
