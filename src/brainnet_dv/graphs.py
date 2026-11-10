import numpy as np
import torch
from torch_geometric.data import Data
from .preprocessing import preprocess_subject, augment_fc

def make_graph(fc, sc, label):
    x = torch.as_tensor(np.asarray(fc), dtype=torch.float32)
    sc_sym = (np.asarray(sc) + np.asarray(sc).T) / 2.0
    np.fill_diagonal(sc_sym, 0.0)
    rows, cols = np.where(sc_sym != 0.0)
    if len(rows):
        edge_index = torch.as_tensor(np.vstack([rows, cols]), dtype=torch.long)
        edge_weight = torch.as_tensor(sc_sym[rows, cols], dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_weight = torch.empty((0,), dtype=torch.float32)
    data = Data(
        x=x, edge_index=edge_index, edge_attr=edge_weight,
        y=torch.tensor([int(label)], dtype=torch.long), num_nodes=x.shape[0]
    )
    return data

def build_split_datasets(fc_scales, sc_scales, labels, train_idx, test_idx, cfg,
                         augment_train=False, seed=42):
    train_ds, test_ds = {i: [] for i in range(3)}, {i: [] for i in range(3)}
    rng = np.random.default_rng(seed)
    for si in range(3):
        for idx in train_idx:
            fc, sc = preprocess_subject(fc_scales[si][idx], sc_scales[si][idx], cfg)
            views = augment_fc(fc, cfg.noise_sigma, cfg.n_augments, rng) if augment_train else [fc]
            train_ds[si].extend(make_graph(view, sc, labels[idx]) for view in views)
        for idx in test_idx:
            fc, sc = preprocess_subject(fc_scales[si][idx], sc_scales[si][idx], cfg)
            test_ds[si].append(make_graph(fc, sc, labels[idx]))
    return train_ds, test_ds
