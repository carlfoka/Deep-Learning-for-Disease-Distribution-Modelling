"""Extract aligned, mask-aware raster patches for CNN-DRE models."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.windows import Window
from rasterio.warp import transform as transform_coordinates
from tqdm import tqdm


class RasterPatchExtractor:
    """Extract centered patches from a folder of aligned, preprocessed rasters."""

    def __init__(
        self,
        raster_dir: Path,
        patch_size: int,
        pattern: str = "*.tif",
    ) -> None:
        if patch_size <= 0 or patch_size % 2 == 0:
            raise ValueError(f"patch_size must be a positive odd integer, got {patch_size}")

        self.patch_size = patch_size
        self.half_size = patch_size // 2
        self.raster_paths = sorted(Path(raster_dir).expanduser().glob(pattern))
        if not self.raster_paths:
            raise FileNotFoundError(
                f"No rasters matched '{pattern}' in {Path(raster_dir).expanduser()}"
            )
        self.raster_names = [path.stem for path in self.raster_paths]
        self._cached_medians: Optional[List[float]] = None

        with rasterio.open(self.raster_paths[0]) as reference:
            self.crs = reference.crs
            self._grid = (
                reference.crs,
                tuple(reference.transform)[:6],
                reference.width,
                reference.height,
            )
        self._validate_aligned_grids()

    def _validate_aligned_grids(self, atol: float = 1e-9) -> None:
        ref_crs, ref_transform, ref_width, ref_height = self._grid
        for path in self.raster_paths[1:]:
            with rasterio.open(path) as src:
                same = (
                    src.crs == ref_crs
                    and src.width == ref_width
                    and src.height == ref_height
                    and np.allclose(
                        tuple(src.transform)[:6], ref_transform, atol=atol, rtol=0
                    )
                )
            if not same:
                raise RuntimeError(
                    f"Grid or CRS mismatch for {path.name}; align rasters before extraction."
                )

    def project_coordinates(
        self, coordinates: np.ndarray, coordinates_crs: Optional[str]
    ) -> np.ndarray:
        """Transform an N x 2 coordinate array into the raster CRS."""
        projected = np.asarray(coordinates, dtype="float64")
        if projected.ndim != 2 or projected.shape[1] != 2:
            raise ValueError("coordinates must have shape [N, 2]")
        if coordinates_crs is None:
            return projected
        if self.crs is None:
            raise ValueError("Raster CRS is undefined, so coordinates cannot be reprojected.")

        source_crs = CRS.from_user_input(coordinates_crs)
        if source_crs == self.crs:
            return projected
        xs, ys = transform_coordinates(
            source_crs, self.crs, projected[:, 0], projected[:, 1]
        )
        return np.column_stack([xs, ys]).astype("float64")

    def _extract_patch(
        self, raster: rasterio.io.DatasetReader, x: float, y: float
    ) -> np.ndarray:
        row_center, col_center = raster.index(x, y)
        window = Window(
            col_center - self.half_size,
            row_center - self.half_size,
            self.patch_size,
            self.patch_size,
        )
        patch = raster.read(
            1,
            window=window,
            boundless=True,
            masked=True,
            fill_value=np.nan,
            out_dtype="float32",
        )
        return np.asarray(patch.filled(np.nan), dtype="float32")

    def _compute_raster_medians(self, downscale: int = 8) -> List[float]:
        medians: List[float] = []
        for path in self.raster_paths:
            with rasterio.open(path) as src:
                out_height = max(1, src.height // downscale)
                out_width = max(1, src.width // downscale)
                thumbnail = src.read(
                    1,
                    out_shape=(out_height, out_width),
                    resampling=Resampling.nearest,
                    masked=True,
                    out_dtype="float32",
                )
                values = np.asarray(thumbnail.filled(np.nan), dtype="float32")
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                raise ValueError(f"Raster {path.name} has no finite values for imputation.")
            medians.append(float(np.median(finite)))
        return medians

    def _process_indices(
        self,
        raster_path: Path,
        coordinates: np.ndarray,
        indices: Sequence[int],
        median: float,
    ) -> List[Tuple[int, np.ndarray, np.ndarray]]:
        results: List[Tuple[int, np.ndarray, np.ndarray]] = []
        with rasterio.open(raster_path) as raster:
            for index in indices:
                x, y = coordinates[index]
                patch = self._extract_patch(raster, float(x), float(y))
                observed = np.isfinite(patch)
                filled = np.where(observed, patch, median).astype("float32")
                results.append((int(index), filled, observed.astype("float32")))
        return results

    def extract(
        self, coordinates: np.ndarray, num_workers: int = 1
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Return values and masks with shape [N, C, H, W]."""
        coordinates = np.asarray(coordinates, dtype="float64")
        if coordinates.ndim != 2 or coordinates.shape[1] != 2:
            raise ValueError("coordinates must have shape [N, 2]")
        if coordinates.shape[0] == 0:
            raise ValueError("No coordinates were provided.")
        if not np.isfinite(coordinates).all():
            raise ValueError("Coordinates contain NaN or infinite values.")
        if num_workers < 1:
            raise ValueError("num_workers must be at least 1")

        if self._cached_medians is None:
            self._cached_medians = self._compute_raster_medians()

        n_points = len(coordinates)
        n_rasters = len(self.raster_paths)
        data = np.empty(
            (n_points, n_rasters, self.patch_size, self.patch_size), dtype="float32"
        )
        mask = np.empty_like(data, dtype="float32")
        worker_count = min(num_workers, n_points)
        chunks = [
            chunk.tolist()
            for chunk in np.array_split(np.arange(n_points), worker_count)
            if len(chunk)
        ]

        for channel, raster_path in enumerate(
            tqdm(self.raster_paths, desc=f"Extracting {self.patch_size}x{self.patch_size}")
        ):
            median = self._cached_medians[channel]
            if worker_count == 1:
                batches = [
                    self._process_indices(raster_path, coordinates, chunks[0], median)
                ]
            else:
                with ThreadPoolExecutor(max_workers=worker_count) as executor:
                    batches = list(
                        executor.map(
                            lambda idx: self._process_indices(
                                raster_path, coordinates, idx, median
                            ),
                            chunks,
                        )
                    )

            for batch in batches:
                for index, patch, observed in batch:
                    data[index, channel] = patch
                    mask[index, channel] = observed

        return data, mask, self.raster_names


def coerce_binary_labels(series: pd.Series) -> np.ndarray:
    """Convert numeric 0/1 or common presence/background labels to float32."""
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all() and set(numeric.unique()).issubset({0, 1}):
        return numeric.to_numpy(dtype="float32")

    normalized = series.astype(str).str.strip().str.lower()
    mapping = {
        "presence": 1.0,
        "present": 1.0,
        "pres": 1.0,
        "true": 1.0,
        "yes": 1.0,
        "background": 0.0,
        "absence": 0.0,
        "absent": 0.0,
        "false": 0.0,
        "no": 0.0,
    }
    converted = normalized.map(mapping)
    converted.loc[converted.isna() & normalized.str.startswith("pres")] = 1.0
    converted.loc[
        converted.isna()
        & (normalized.str.startswith("back") | normalized.str.startswith("abs"))
    ] = 0.0
    if converted.isna().any():
        bad = sorted(normalized[converted.isna()].unique().tolist())
        raise ValueError(f"Unrecognized labels: {bad}")
    return converted.to_numpy(dtype="float32")


def _existing_outputs(output_dir: Path) -> Dict[str, Path]:
    return {
        "X": output_dir / "X.npy",
        "M": output_dir / "M.npy",
        "y": output_dir / "y.npy",
        "fold": output_dir / "fold.npy",
        "cluster": output_dir / "cluster.npy",
        "patch_names": output_dir / "patch_names.npy",
        "meta": output_dir / "meta.parquet",
    }


def extract_patches_with_metadata(
    extractor: RasterPatchExtractor,
    points_csv: Path,
    output_dir: Path,
    longitude_col: str = "Longitude",
    latitude_col: str = "Latitude",
    label_col: str = "label",
    fold_col: str = "fold",
    cluster_col: str = "cluster_id",
    coordinates_crs: Optional[str] = "EPSG:4326",
    num_workers: int = 1,
    overwrite: bool = False,
) -> Dict[str, Path]:
    """Extract patches aligned to all input CSV rows and save model-ready arrays."""
    output_dir = Path(output_dir).expanduser()
    outputs = _existing_outputs(output_dir)
    if not overwrite and all(path.is_file() for path in outputs.values()):
        print(f"[skip] Complete patch dataset already exists in {output_dir}")
        return outputs
    if not overwrite:
        existing = [path.name for path in outputs.values() if path.exists()]
        if existing:
            raise FileExistsError(
                f"Partial outputs exist in {output_dir}: {', '.join(existing)}. "
                "Pass --overwrite to replace them."
            )

    frame = pd.read_csv(points_csv)
    required = [longitude_col, latitude_col, label_col]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"CSV columns missing: {', '.join(missing)}")

    coordinates_frame = frame[[longitude_col, latitude_col]].apply(
        pd.to_numeric, errors="coerce"
    )
    valid = np.isfinite(coordinates_frame.to_numpy(dtype=float)).all(axis=1)
    if not valid.all():
        bad_rows = np.flatnonzero(~valid)[:10].tolist()
        raise ValueError(f"Invalid coordinates at row indices {bad_rows}")

    labels = coerce_binary_labels(frame[label_col])
    folds = pd.to_numeric(
        frame.get(fold_col, pd.Series(np.nan, index=frame.index)), errors="coerce"
    ).to_numpy()
    clusters = pd.to_numeric(
        frame.get(cluster_col, pd.Series(-1, index=frame.index)), errors="coerce"
    ).fillna(-1).to_numpy(dtype="int32")

    unique_coordinates = coordinates_frame.drop_duplicates().reset_index(drop=True)
    unique_index = pd.MultiIndex.from_frame(unique_coordinates)
    row_index = pd.MultiIndex.from_frame(coordinates_frame)
    inverse = unique_index.get_indexer(row_index)
    if (inverse < 0).any():
        raise RuntimeError("Failed to map extracted unique coordinates back to CSV rows.")

    projected = extractor.project_coordinates(
        unique_coordinates.to_numpy(dtype="float64"), coordinates_crs
    )
    unique_values, unique_masks, patch_names = extractor.extract(
        projected, num_workers=num_workers
    )
    values = unique_values[inverse]
    masks = unique_masks[inverse]

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(outputs["X"], values.astype("float32"))
    np.save(outputs["M"], masks.astype("float32"))
    np.save(outputs["y"], labels)
    np.save(outputs["fold"], folds)
    np.save(outputs["cluster"], clusters)
    np.save(outputs["patch_names"], np.asarray(patch_names, dtype=str))
    frame.to_parquet(outputs["meta"], index=False)

    print(
        f"[save] {output_dir}: X={values.shape}, M={masks.shape}, "
        f"y={labels.shape}, channels={len(patch_names)}"
    )
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract mask-aware raster patches for one or more odd scales."
    )
    parser.add_argument("--raster-dir", type=Path, required=True)
    parser.add_argument("--points-csv", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--patch-sizes", type=int, nargs="+", default=[1, 3, 13, 33])
    parser.add_argument("--raster-pattern", default="*.tif")
    parser.add_argument("--longitude-col", default="Longitude")
    parser.add_argument("--latitude-col", default="Latitude")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--fold-col", default="fold")
    parser.add_argument("--cluster-col", default="cluster_id")
    parser.add_argument("--coordinates-crs", default="EPSG:4326")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    coordinates_crs = None if args.coordinates_crs.lower() == "none" else args.coordinates_crs
    for patch_size in args.patch_sizes:
        print(f"\n=== Extracting patch size {patch_size} ===")
        extractor = RasterPatchExtractor(
            raster_dir=args.raster_dir,
            patch_size=patch_size,
            pattern=args.raster_pattern,
        )
        extract_patches_with_metadata(
            extractor=extractor,
            points_csv=args.points_csv,
            output_dir=args.output_root / f"scale_{patch_size}",
            longitude_col=args.longitude_col,
            latitude_col=args.latitude_col,
            label_col=args.label_col,
            fold_col=args.fold_col,
            cluster_col=args.cluster_col,
            coordinates_crs=coordinates_crs,
            num_workers=args.num_workers,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
