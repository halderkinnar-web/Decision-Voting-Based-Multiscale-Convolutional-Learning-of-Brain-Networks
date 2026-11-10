"""Gradient-saliency explanations for node/ROI features.

The score is a model sensitivity measure, not a causal effect. Node indices
are atlas parcel indices; meaningful anatomical names require the matching
atlas label file, which is not included in the supplied dataset/code.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch

def explain_graph(model, graph, device):
    model.eval()
    one = graph.clone().to(device)
    one.x = one.x.detach().clone().requires_grad_(True)
    batch = torch.zeros(one.num_nodes, dtype=torch.long, device=device)
    model.zero_grad(set_to_none=True)
    log_probs = model(one.x, one.edge_index, one.edge_attr, batch)
    target = int(log_probs.argmax(dim=1).item())
    log_probs[0, target].backward()
    saliency = one.x.grad.detach().abs().mean(dim=1).cpu().numpy()
    return target, saliency

def save_fold_explanations(model, graphs, subject_indices, scale, fold, out_dir, cfg):
    rows = []
    for graph, subject_idx in zip(graphs, subject_indices):
        predicted, scores = explain_graph(model, graph, cfg.resolved_device())
        for node_idx, score in enumerate(scores):
            rows.append({
                "fold": fold, "scale_roi": scale, "subject_index": int(subject_idx),
                "predicted_class": predicted, "node_index": node_idx,
                "gradient_saliency": float(score),
            })
    out = Path(out_dir) / "explainability"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / f"fold_{fold}_scale_{scale}_saliency.csv", index=False)
