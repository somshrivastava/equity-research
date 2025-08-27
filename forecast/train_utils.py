# train_utils.py
import torch, torch.nn as nn

def weighted_mse_loss(pred, target, weights):
    # pred/target: (B, 16), weights: (16,)
    return ((pred - target)**2 * weights).mean()

def make_target_weights(alpha1=0.75, out_dim=16, ebit_idx=2):
    w = torch.ones(out_dim)
    w[ebit_idx] = alpha1 + 1.0  # e.g., amplify EBIT contribution
    return w

def mse_by_feature(model, X, y, feature_names):
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(X, dtype=torch.float32))
    mse = ((pred.numpy() - y)**2).mean(axis=0)
    return dict(zip(feature_names, mse)), mse.mean()
