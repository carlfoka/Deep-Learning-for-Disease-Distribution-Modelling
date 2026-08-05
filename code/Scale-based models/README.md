# Scale-based CNN-DRE models

This folder contains the multiscale CNN-DRE experiment and its three
single-scale controls. The single-scale encoders match the corresponding
branches of the multiscale model so their results can be compared directly.

## Files

| File | Model | Required patch size |
|---|---|---:|
| `01_multiscale_cnn_dre.py` | Learned fusion of scales 3, 13, and 33 | 33 x 33 |
| `02_scale_3.py` | Single-scale CNN-DRE | 3 x 3 |
| `03_scale_13.py` | Single-scale CNN-DRE | 13 x 13 |
| `04_scale_33.py` | Single-scale CNN-DRE | 33 x 33 |

The multiscale script expects stored 33 x 33 patches and extracts centered
13 x 13 and 3 x 3 crops inside the model. Smaller stored patches cannot be
used to reconstruct the 33 x 33 context.

## Installation

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r notebooks/scale_based_models/requirements.txt
```

## Data layout

Each CV directory must contain:

```text
X.npy
M.npy
y.npy
fold.npy
```

Each test directory must contain:

```text
X.npy
M.npy
y.npy
```

`X.npy` contains patch values, `M.npy` contains matching masks, `y.npy`
contains binary labels, and `fold.npy` assigns every CV sample to a fold.
The scripts validate file presence, array lengths, masks, folds, and spatial
patch dimensions before training.

## Running the models

Pass the appropriate scale-specific data directories. For example:

```bash
python notebooks/scale_based_models/01_multiscale_cnn_dre.py \
  --cv-dir data/processed/patches/cv_scale_33 \
  --test-dir data/processed/patches/test_scale_33

python notebooks/scale_based_models/02_scale_3.py \
  --cv-dir data/processed/patches/cv_scale_3 \
  --test-dir data/processed/patches/test_scale_3

python notebooks/scale_based_models/03_scale_13.py \
  --cv-dir data/processed/patches/cv_scale_13 \
  --test-dir data/processed/patches/test_scale_13

python notebooks/scale_based_models/04_scale_33.py \
  --cv-dir data/processed/patches/cv_scale_33 \
  --test-dir data/processed/patches/test_scale_33
```

Use `--output-dir PATH` to override the default location under
`outputs/scale_based_models/`. Run `python FILE.py --help` for the complete
command-line interface.

## Outputs

Each run writes fold checkpoints, fold AUCs and ensemble weights, out-of-fold
scores, and external-test predictions to its output directory. Generated
models and arrays should usually be excluded from Git and reproduced from the
code and data.

## Reproducibility note

The random seed and training hyperparameters remain defined near the top of
each script. Record any changes when reporting results. Exact results can still
vary between CPU, CUDA, and Apple MPS backends.
