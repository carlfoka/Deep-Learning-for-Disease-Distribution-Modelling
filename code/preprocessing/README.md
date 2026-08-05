# Preprocessing

Portable preprocessing scripts for spatial splitting, raster normalization,
point-feature extraction, and multiscale patch extraction.

## Contents

| Path | Purpose |
|---|---|
| `point_extraction/preprocess_rasters.py` | Median/IQR-normalize aligned raster covariates |
| `point_extraction/extract_points.py` | Sample one value per raster at each point |
| `extract_patches.py` | Create mask-aware patches at scales 1, 3, 13, and 33 |
| `spatial_cross_validation.py` | Build spherical-cluster CV and external-test splits |

## Installation

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r preprocessing/requirements.txt
```

## Recommended workflow

### 1. Normalize aligned raster covariates

```bash
python preprocessing/point_extraction/preprocess_rasters.py \
  --input-dir data/processed/rasters/aligned \
  --output-dir data/processed/rasters/normalized
```

Missing cells remain missing and are marked as NaN nodata. Normalization
statistics are saved to `normalization_stats.json`. Add `--log-transform` only
for positive variables that have not already been log-transformed.

All rasters used together must have the same CRS, transform, width, and height.


### 2. Extract point features

Run once for CV and once for test:

```bash
python preprocessing/point_extraction/extract_points.py \
  --points-csv data/processed/splits/cv_split.csv \
  --raster-dir data/processed/rasters/normalized \
  --output data/processed/point_features/cv.parquet

python preprocessing/point_extraction/extract_points.py \
  --points-csv data/processed/splits/test_split.csv \
  --raster-dir data/processed/rasters/normalized \
  --output data/processed/point_features/test.parquet
```

The extractor preserves the input metadata columns and adds one feature column
per raster. Coordinates are assumed to be EPSG:4326 and are reprojected to the
raster CRS when required.

### 3. Extract CNN patches

Run separately for CV and test so their arrays remain isolated:

```bash
python preprocessing/extract_patches.py \
  --raster-dir data/processed/rasters/normalized \
  --points-csv data/processed/splits/cv_split.csv \
  --output-root data/processed/patches/cv \
  --patch-sizes 1 3 13 33 \
  --num-workers 4

python preprocessing/extract_patches.py \
  --raster-dir data/processed/rasters/normalized \
  --points-csv data/processed/splits/test_split.csv \
  --output-root data/processed/patches/test \
  --patch-sizes 1 3 13 33 \
  --num-workers 4
```

Each output scale directory contains:

| File | Contents |
|---|---|
| `X.npy` | Values with shape `[N, C, H, W]`; missing cells median-imputed |
| `M.npy` | Observation mask with the same shape; 1 observed, 0 imputed |
| `y.npy` | Binary labels |
| `fold.npy` | CV fold IDs, or NaN when absent |
| `cluster.npy` | Spatial cluster IDs, or -1 when absent |
| `patch_names.npy` | Raster/channel names |
| `meta.parquet` | Original rows for traceability |

Patch labels may be numeric `0`/`1` or common text labels such as `presence`
and `background`. Coordinates are deduplicated during extraction and mapped
back to the original row order.

For the scale-based models, use `cv/scale_33` and `test/scale_33` for the
multiscale model, and the matching `scale_3`, `scale_13`, or `scale_33`
directories for each single-scale model.


