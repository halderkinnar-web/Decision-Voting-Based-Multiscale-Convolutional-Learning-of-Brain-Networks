import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, f1_score, roc_auc_score,
    matthews_corrcoef, recall_score, confusion_matrix,
    fowlkes_mallows_score
)

def compute_metrics(y_true, y_pred, y_prob):
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else float("nan")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": auc,
        "mcc": matthews_corrcoef(y_true, y_pred),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "fm": fowlkes_mallows_score(y_true, y_pred),
        "npv": tn / (tn + fn) if tn + fn else 0.0,
    }

def aggregate(fold_results):
    return {k: (float(np.mean([m[k] for m in fold_results])),
                float(np.std([m[k] for m in fold_results])))
            for k in fold_results[0]}
