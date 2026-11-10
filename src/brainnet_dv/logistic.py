import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from .preprocessing import preprocess_subject
from .metrics import compute_metrics

def fc_upper_tri_features(fc_scales, sc_scales, scale_idxs, subj_idxs, cfg):
    blocks = []
    for si in scale_idxs:
        rows = []
        for subject in subj_idxs:
            fc, _ = preprocess_subject(fc_scales[si][subject], sc_scales[si][subject], cfg)
            rows.append(fc[np.triu_indices(fc.shape[0], k=1)])
        blocks.append(np.asarray(rows))
    return np.concatenate(blocks, axis=1)

def run_logistic(fc_scales, sc_scales, labels, scale_idxs, train_idx, test_idx, cfg):
    x_train = fc_upper_tri_features(fc_scales, sc_scales, scale_idxs, train_idx, cfg)
    x_test = fc_upper_tri_features(fc_scales, sc_scales, scale_idxs, test_idx, cfg)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    clf = LogisticRegression(max_iter=3000, C=1.0, solver="lbfgs", random_state=cfg.seed)
    clf.fit(x_train, labels[train_idx])
    return compute_metrics(labels[test_idx], clf.predict(x_test), clf.predict_proba(x_test)[:, 1])
