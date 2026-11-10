from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from .data import load_data
from .graphs import build_split_datasets
from .training import train_gcn, predict_with_model
from .logistic import run_logistic
from .metrics import aggregate
from .explainability import save_fold_explanations

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _format_table(results, method):
    rows = []
    for name in ("83", "129", "234", "multi"):
        agg = aggregate(results[name])
        rows.append({"model": method, "scale": name,
                     **{k: f"{v[0]:.4f} ± {v[1]:.4f}" for k, v in agg.items()}})
    return pd.DataFrame(rows)

def run_experiment(cfg):
    if not cfg.data_path:
        raise ValueError("Set data_path in config.json or pass --data.")
    set_seed(cfg.seed)
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Device: {cfg.resolved_device()}")
    fc, sc, labels = load_data(cfg.data_path, cfg.n_subjects, cfg.scales_roi)
    in_channels = list(cfg.scales_roi)
    skf = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
    gcn_results = {k: [] for k in ("83", "129", "234", "multi")}
    lr_results = {k: [] for k in ("83", "129", "234", "multi")}

    for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(labels)), labels), 1):
        print(f"\\nFOLD {fold}/{cfg.n_folds}: train={len(train_idx)}, test={len(test_idx)}")
        single_train, single_test = build_split_datasets(
            fc, sc, labels, train_idx, test_idx, cfg, augment_train=False, seed=cfg.seed + fold)
        multi_train, multi_test = build_split_datasets(
            fc, sc, labels, train_idx, test_idx, cfg, augment_train=True, seed=cfg.seed + fold)

        for si, key in enumerate(("83", "129", "234")):
            print(f"  GCN single scale {cfg.scales_roi[si]}")
            gcn_results[key].append(train_gcn(single_train[si], single_test[si],
                                               in_channels[si], cfg))

        # Train one independent model per atlas resolution, then average probabilities.
        probs_by_scale, true_labels = [], None
        for si, key in enumerate(("83", "129", "234")):
            print(f"  DV-GCN independent scale {cfg.scales_roi[si]}")
            metrics, model = train_gcn(multi_train[si], multi_test[si],
                                       in_channels[si], cfg, return_model=True)
            pred, probs, trues = predict_with_model(model, multi_test[si], cfg)
            probs_by_scale.append(probs)
            true_labels = trues
            print(f"    scale accuracy={metrics['accuracy']:.4f}")
            if cfg.explain:
                save_fold_explanations(model, multi_test[si], test_idx,
                                       cfg.scales_roi[si], fold, out, cfg)
        mean_probs = np.mean(probs_by_scale, axis=0)
        from .metrics import compute_metrics
        gcn_results["multi"].append(compute_metrics(
            true_labels, (mean_probs >= 0.5).astype(int), mean_probs))

        for si, key in enumerate(("83", "129", "234")):
            lr_results[key].append(run_logistic(fc, sc, labels, [si],
                                                train_idx, test_idx, cfg))
        lr_results["multi"].append(run_logistic(fc, sc, labels, [0, 1, 2],
                                                train_idx, test_idx, cfg))

    gcn_df = _format_table(gcn_results, "GCN")
    lr_df = _format_table(lr_results, "LogisticRegression")
    gcn_df.to_csv(out / "table1_gcn.csv", index=False)
    lr_df.to_csv(out / "table2_logistic.csv", index=False)
    pd.concat([gcn_df, lr_df], ignore_index=True).to_csv(out / "all_results.csv", index=False)
    cfg.save(out / "run_config.json")
    print("\\nGCN results\\n", gcn_df.to_string(index=False))
    print("\\nLogistic regression results\\n", lr_df.to_string(index=False))
    print(f"\\nSaved outputs in: {out.resolve()}")
