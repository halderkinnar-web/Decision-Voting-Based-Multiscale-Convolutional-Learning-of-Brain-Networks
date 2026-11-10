import h5py
import numpy as np

def _to_subject_first(arr, n_per_class=27):
    """Convert supported MATLAB/HDF5 layouts to (subjects, ROIs, ROIs)."""
    if arr.ndim != 3:
        raise ValueError(f"Expected a 3-D connectome array; got shape {arr.shape}")
    if arr.shape[0] == n_per_class:
        out = arr
    elif arr.shape[2] == n_per_class:
        out = arr.transpose(2, 0, 1)
    else:
        raise ValueError(
            f"Cannot identify subject axis in {arr.shape}; expected an axis of {n_per_class}. "
            "Inspect the MAT file and adapt _to_subject_first."
        )
    if out.shape[1] != out.shape[2]:
        raise ValueError(f"Connectomes must be square; got {out.shape}")
    return np.asarray(out, dtype=np.float64)

def load_data(path, n_subjects=54, scales=(83, 129, 234)):
    """Load FC_correlation and SC_number_of_fibers from the supplied HDF5 MAT file."""
    n_per_class = n_subjects // 2
    fc, sc = {}, {}
    with h5py.File(path, "r") as f:
        required = [
            "SC_FC_Connectomes/FC_correlation/ctrl",
            "SC_FC_Connectomes/FC_correlation/schz",
            "SC_FC_Connectomes/SC_number_of_fibers/ctrl",
            "SC_FC_Connectomes/SC_number_of_fibers/schz",
        ]
        missing = [k for k in required if k not in f]
        if missing:
            raise KeyError(f"Missing expected HDF5 groups/datasets: {missing}")
        refs = {k: f[k] for k in required}
        for si, expected_roi in enumerate(scales):
            arrays = []
            for key in required:
                ds = refs[key]
                # The source file stores references in the first column.
                ref = ds[si, 0] if ds.ndim >= 2 else ds[si]
                arrays.append(_to_subject_first(f[ref][()], n_per_class))
            fc[si] = np.concatenate(arrays[:2], axis=0)
            sc[si] = np.concatenate(arrays[2:], axis=0)
            if fc[si].shape[1:] != (expected_roi, expected_roi):
                raise ValueError(
                    f"Scale index {si} expected {expected_roi}x{expected_roi}, "
                    f"but FC shape is {fc[si].shape}"
                )
            if sc[si].shape != fc[si].shape:
                raise ValueError(f"FC/SC shape mismatch at scale {expected_roi}")
    labels = np.array([0] * n_per_class + [1] * n_per_class, dtype=np.int64)
    return fc, sc, labels
