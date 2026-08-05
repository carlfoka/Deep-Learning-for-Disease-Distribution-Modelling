# CNN density-ratio estimation approaches

This folder contains four cleaned research notebooks:

- `01_kliep.ipynb`: KLIEP objective.
- `02_classifier_dre.ipynb`: classifier-based density-ratio estimation.
- `03_kulsif.ipynb`: KuLSIF objective.
- `04_ulsif.ipynb`: uLSIF objective.

The original experiment variants for different patch sizes and convolution
types are preserved. Stored cell outputs and empty cells were removed, and all
machine-specific paths were converted to portable project paths.

## Important execution rule

Each large experiment cell is a complete training workflow. Run the
configuration cell first, then run only the experiment section for the patch
size and convolution variant you want. Running every experiment cell in order
would train many models and overwrite global class and function definitions.

## Expected project layout

Launch Jupyter from the repository root. The default paths expect:

```text
data/
└── processed/
    ├── cv_patches_3/
    ├── test_patches_3/
    ├── cv_patches_5/
    ├── test_patches_5/
    ├── ...
    ├── covariate_rasters/
    └── landmask.tif
```

Each cross-validation patch directory must contain:

```text
X.npy
M.npy
y.npy
fold.npy
```

Each test patch directory must contain `X.npy`, `M.npy`, and `y.npy`.

## Configuration

The first code cell defines the following paths:

- `PROJECT_ROOT`: repository root.
- `DATA_ROOT`: processed data directory.
- `OUTPUT_ROOT`: model and prediction output directory.
- `RASTER_DIR`: standardized covariate TIFF directory.
- `LANDMASK_PATH`: land-mask TIFF.

The defaults can be overridden without modifying the notebooks:

```bash
export DRE_PROJECT_ROOT=/path/to/repository
export DRE_DATA_ROOT=/path/to/processed/data
export DRE_OUTPUT_ROOT=/path/to/outputs/dre_approaches
export DRE_RASTER_DIR=/path/to/covariate/rasters
export DRE_LANDMASK_PATH=/path/to/landmask.tif
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r notebooks/dre_approaches/requirements.txt
jupyter notebook
```

PyTorch installation can depend on the desired CUDA version. If GPU support is
required, use the appropriate installation command from the PyTorch project
instead of relying on the generic requirement.

## Outputs

Generated checkpoints, predictions, metrics, and TIFF maps are written beneath
`outputs/dre_approaches/` by default. These generated artifacts should normally
be excluded from Git because they can be large and should be reproducible from
the code, configuration, and documented data.
