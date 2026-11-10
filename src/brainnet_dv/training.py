import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from .model import SpectralGCN
from .metrics import compute_metrics

def _predict(model, loader, device):
    model.eval()
    preds, probs, trues = [], [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            preds.extend(out.argmax(1).cpu().numpy())
            probs.extend(out.exp()[:, 1].cpu().numpy())
            trues.extend(data.y.view(-1).cpu().numpy())
    return np.asarray(preds), np.asarray(probs), np.asarray(trues)

def train_gcn(train_data, test_data, in_channels, cfg, return_model=False):
    device = cfg.resolved_device()
    if not train_data or not test_data:
        raise ValueError("Training and test graph lists must both be non-empty.")
    train_loader = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=min(len(test_data), cfg.batch_size), shuffle=False)
    model = SpectralGCN(in_channels, cfg.hidden_dim, cfg.cheb_k, cfg.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate,
                           weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    for _ in range(cfg.epochs):
        model.train()
        for data in train_loader:
            data = data.to(device)
            opt.zero_grad(set_to_none=True)
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            loss = F.nll_loss(out, data.y.view(-1))
            loss.backward()
            opt.step()
        scheduler.step()
    y_pred, y_prob, y_true = _predict(model, test_loader, device)
    metrics = compute_metrics(y_true, y_pred, y_prob)
    return (metrics, model) if return_model else metrics

def predict_with_model(model, data_list, cfg):
    loader = DataLoader(data_list, batch_size=min(len(data_list), cfg.batch_size), shuffle=False)
    return _predict(model, loader, cfg.resolved_device())
