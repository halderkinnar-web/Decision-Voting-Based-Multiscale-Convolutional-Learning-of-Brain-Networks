import numpy as np
from scipy.linalg import eigh

def graph_diffusion(fc, t=1.0, beta=0.5):
    """Heat-kernel diffusion followed by convex mixing with the original FC."""
    a = (np.asarray(fc, dtype=float) + np.asarray(fc, dtype=float).T) / 2.0
    np.fill_diagonal(a, 0.0)
    degree = np.diag(np.abs(a).sum(axis=1))
    laplacian = degree - a
    try:
        eigval, eigvec = eigh(laplacian, check_finite=True)
        eigval = np.maximum(eigval, 0.0)
        diffused = (eigvec * np.exp(-t * eigval)) @ eigvec.T
    except (np.linalg.LinAlgError, ValueError):
        diffused = a.copy()
    out = beta * a + (1.0 - beta) * diffused
    np.fill_diagonal(out, 0.0)
    return out

def sparsify_fc(fc, m=10):
    """Keep up to M largest positive and M smallest negative entries per row."""
    n = fc.shape[0]
    sparse = np.zeros_like(fc)
    for i in range(n):
        row = fc[i].copy()
        row[i] = 0.0
        pos = np.flatnonzero(row > 0)
        neg = np.flatnonzero(row < 0)
        if len(pos):
            keep = pos[np.argsort(row[pos])[-m:]]
            sparse[i, keep] = row[keep]
        if len(neg):
            keep = neg[np.argsort(row[neg])[:m]]
            sparse[i, keep] = row[keep]
    out = (sparse + sparse.T) / 2.0
    np.fill_diagonal(out, 0.0)
    return out

def sparsify_sc(sc, percentile=50.0):
    """Remove SC values below the requested percentile among positive edges."""
    out = np.asarray(sc, dtype=float).copy()
    vals = out[out > 0]
    if vals.size:
        out[out < np.percentile(vals, percentile)] = 0.0
    np.fill_diagonal(out, 0.0)
    return (out + out.T) / 2.0

def preprocess_subject(fc_raw, sc_raw, cfg):
    fc = graph_diffusion(fc_raw, cfg.diffusion_t, cfg.beta)
    fc = sparsify_fc(fc, cfg.fc_top_bottom_m)
    sc = sparsify_sc(sc_raw, cfg.sc_percentile)
    return fc, sc

def augment_fc(fc, sigma=2.0, n=9, rng=None):
    """Return original FC plus n symmetric Gaussian perturbations."""
    rng = rng or np.random.default_rng()
    copies = [fc]
    for _ in range(n):
        p = rng.standard_normal(fc.shape)
        copies.append(fc + sigma * (p + p.T) / 2.0)
    return copies
