# Decision Voting Based Multiscale Convolutional Learning of Brain Networks With Explainability

A modular research-code organization of the supplied Python implementation for schizophrenia/control classification from structural and functional connectomes at 83-, 129-, and 234-ROI resolutions.

> **Important research note:** The supplied script describes reproduction of GCN and logistic-regression tables. It did not contain an explainability method despite the paper title. This repository adds **gradient-based node-feature saliency** for the independently trained multi-scale GCN models. These are model-sensitivity explanations, not causal explanations. Anatomical ROI names are not generated because atlas label files were not supplied.
>
> The code is an implementation derived from the supplied script, not a verified reproduction of the paper. The numerical targets mentioned in the original script are expectations, not guaranteed results. Validate the dataset schema, experimental protocol, and all results before publication.

## 1. Project structure

```text
decision-voting-brain-networks/
├── configs/
│   └── config.json
├── src/
│   └── brainnet_dv/
│       ├── __init__.py
│       ├── config.py
│       ├── data.py
│       ├── preprocessing.py
│       ├── graphs.py
│       ├── model.py
│       ├── training.py
│       ├── logistic.py
│       ├── metrics.py
│       ├── explainability.py
│       └── experiment.py
├── tests/
├── results/
├── original_submitted_code.txt
├── requirements.txt
├── run.py
└── README.md
```

## 2. Method pipeline, step by step

1. **Load connectomes** — read `FC_correlation` and `SC_number_of_fibers` for controls and schizophrenia subjects from the HDF5-based MATLAB file. Labels are encoded as control `0` and schizophrenia `1`.
2. **Functional-connectivity diffusion** — symmetrise FC, construct the absolute-degree Laplacian, apply the heat kernel, and mix the diffused matrix with the original FC using `beta`.
3. **FC sparsification** — retain up to `M` strongest positive and `M` strongest negative connections per node, then symmetrise.
4. **SC sparsification** — remove positive structural edges below the selected percentile.
5. **Graph construction** — each ROI is a node; its FC connectivity profile is the node feature; non-zero SC entries form weighted edges.
6. **Training augmentation** — for the proposed multi-scale GCN only, create the original FC plus nine symmetric Gaussian-noise variants. Single-scale GCN baselines and logistic regression use unaugmented data, matching the supplied script's stated protocol.
7. **Independent scale-specific GCNs** — train a separate two-layer Chebyshev GCN for each atlas resolution.
8. **Decision voting** — average the three models' schizophrenia-class probabilities with equal weights, then threshold at 0.5.
9. **Logistic-regression baselines** — flatten the upper triangular FC entries, concatenate scales for the multi-scale baseline, standardise using training-fold statistics, and fit logistic regression.
10. **Evaluation** — run stratified five-fold cross-validation and report mean ± standard deviation for accuracy, precision, F1, ROC-AUC, MCC, recall, specificity, Fowlkes–Mallows score, and NPV.
11. **Explainability** — calculate absolute input gradients for each test subject and aggregate feature sensitivity by node/ROI. CSV files are saved per fold and scale.

## 3. Environment setup

Python 3.10 or 3.11 is recommended. Install a PyTorch build compatible with your CUDA/CPU environment first. Then install PyTorch Geometric following its official installation instructions:

- PyTorch: https://pytorch.org/get-started/locally/
- PyTorch Geometric: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

Then install the remaining packages:

```bash
pip install -r requirements.txt
```

If the PyTorch Geometric installation fails, do not assume a plain `pip install` has configured the correct CUDA extensions; follow the official compatibility instructions for your installed PyTorch version.

## 4. Dataset setup

The dataset is **not included** in this repository. It may be subject to its own access, licensing, and redistribution terms.

Place the dataset somewhere on your machine or compute environment, then edit `configs/config.json`:

```json
"data_path": "/absolute/path/to/27_SCHZ_CTRL_dataset(1).mat"
```

The loader expects the following HDF5 hierarchy:

```text
SC_FC_Connectomes/
├── FC_correlation/
│   ├── ctrl
│   └── schz
└── SC_number_of_fibers/
    ├── ctrl
    └── schz
```

The code expects three atlas resolutions in the order 83, 129, 234 and 27 subjects per class (54 total). If your MAT file uses a different layout, inspect it with `h5py` and adapt `src/brainnet_dv/data.py`. Do not infer that the file is compatible merely because its extension is `.mat`.

## 5. Run the experiment

From the repository root:

```bash
python run.py --config configs/config.json
```

Or pass paths explicitly:

```bash
python run.py --config configs/config.json \
  --data "/absolute/path/to/27_SCHZ_CTRL_dataset(1).mat" \
  --output results
```

For a quick pipeline smoke test, temporarily reduce `epochs` in `configs/config.json` (for example, to 2) and verify that loading, graph construction, training, and output writing work. Such a run is only a software test, not a scientific result.

## 6. Outputs

After a successful run:

```text
results/
├── table1_gcn.csv
├── table2_logistic.csv
├── all_results.csv
├── run_config.json
└── explainability/
    ├── fold_1_scale_83_saliency.csv
    ├── fold_1_scale_129_saliency.csv
    ├── fold_1_scale_234_saliency.csv
    └── ...
```

The saliency CSV contains fold, scale, subject index, predicted class, node index, and gradient saliency. Node indices are not anatomical labels. To map them to anatomical regions, use the exact atlas/parcellation label file corresponding to each resolution.

## 7. Important validation checklist before reporting results

- Confirm that the four HDF5 groups and reference-array shapes match the loader assumptions.
- Confirm subject order and labels. The current loader assumes controls are stored first and schizophrenia subjects second, with 27 in each group.
- Verify whether the source paper's fold construction, preprocessing, augmentation, and metrics are exactly represented by this implementation.
- The original code uses ordinary `StratifiedKFold`; this is appropriate only if each row is one independent subject and no repeated scans or family/site grouping is present.
- Run multiple random seeds and report variability if required by the target venue.
- Check class balance and confusion matrices in every fold.
- Treat the expected accuracy values in the original script as targets only, never as guaranteed outputs.
- The gradient saliency is a post-hoc sensitivity map. It does not establish that a brain region causes schizophrenia or that the model's explanation is clinically valid.
- For stronger explainability, consider validating saliency with perturbation/deletion tests, stability across seeds/folds, and atlas-level aggregation.

## 8. Reproducibility and limitations

The repository preserves the core method and hyperparameters in the supplied script. It separates configuration, data loading, preprocessing, graph construction, model, training, baselines, metrics, and explanation into modules.

The implementation does **not** include raw MRI processing, connectome generation, an atlas label mapping, external validation, nested hyperparameter tuning, or a formal clinical interpretability study. The supplied script itself did not establish these components. The new saliency output should therefore be described accurately as an added model-explanation analysis.
