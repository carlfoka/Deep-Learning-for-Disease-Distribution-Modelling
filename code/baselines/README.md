# Baseline models

This folder contains three runnable baselines for ecological niche modelling:

- `random_forest.py`: Random Forest probability baseline.
- `maxent.py`: MaxEnt baseline implemented with `elapid`.
- `mlp_dre.py`: pixel-level MLP density-ratio estimator.
- `baseline_metrics.py`: shared ROC AUC and continuous Boyce metrics.


## Installation

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r baselines/requirements.txt
```

Change `baselines/` in the command if this folder is placed elsewhere.

## Random Forest

The training Parquet file must contain a fold column named `fold`, a binary
label column named `mod_label`, coordinates named `x` and `y`, and numeric
feature columns. The test Parquet file needs the same feature and label columns.

```bash
python baselines/random_forest.py \
  --train-path data/processed/cv_point_features.parquet \
  --test-path data/processed/test_point_features.parquet
```

To also create a suitability map, provide both mapping arguments:

```bash
python baselines/random_forest.py \
  --train-path data/processed/cv_point_features.parquet \
  --test-path data/processed/test_point_features.parquet \
  --raster-dir data/processed/covariate_rasters \
  --landmask-tif data/processed/landmask.tif
```

## MaxEnt

MaxEnt uses the same point-feature inputs as Random Forest:

```bash
python baselines/maxent.py \
  --train-path data/processed/cv_point_features.parquet \
  --test-path data/processed/test_point_features.parquet
```

The optional `--raster-dir` and `--landmask-tif` arguments enable suitability
map generation.

## MLP-DRE

The cross-validation directory must contain `X.npy`, `M.npy`, `y.npy`, and
`fold.npy`. The test directory must contain `X.npy`, `M.npy`, and `y.npy`.
`X.npy` and `M.npy` may have shape `(N, C)` or `(N, C, 1, 1)`.

```bash
python baselines/mlp_dre.py \
  --cv-dir data/processed/mlp/cv \
  --test-dir data/processed/mlp/test
```

## Outputs

By default, results are written under:

```text
outputs/baselines/
├── random_forest/
├── maxent/
└── mlp_dre/
```


