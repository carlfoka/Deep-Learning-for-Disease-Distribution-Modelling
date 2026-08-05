"""Normalize raster covariates before extracting values at point locations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import rasterio


def _read_normalize_band(
    src: rasterio.io.DatasetReader,
    log_transform: bool = False,
    offset: float = 1e-6,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Read band 1 and median/IQR-normalize finite values without filling gaps."""
    masked = src.read(1, masked=True).astype("float32")
    arr = np.asarray(masked.filled(np.nan), dtype="float32")
    arr[(arr < -1e16) | (arr > 1e16)] = np.nan

    if log_transform:
        positive = arr > 0
        if np.any(positive):
            arr[positive] = np.log(arr[positive] + offset)

    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return arr, {
            "median_raw": np.nan,
            "iqr_raw": np.nan,
            "q1_raw": np.nan,
            "q3_raw": np.nan,
        }

    median = float(np.median(finite))
    q1 = float(np.percentile(finite, 25))
    q3 = float(np.percentile(finite, 75))
    iqr = max(q3 - q1, 1e-8)
    normalized = (arr - median) / iqr

    return normalized.astype("float32"), {
        "median_raw": median,
        "iqr_raw": iqr,
        "q1_raw": q1,
        "q3_raw": q3,
    }


def preprocess_raster_folder(
    input_dir: Path,
    output_dir: Path,
    pattern: str = "*.tif",
    log_transform: bool = False,
    overwrite: bool = False,
) -> Dict[str, Dict[str, float]]:
    """Normalize band 1 of every matching raster and preserve missing cells as NaN."""
    input_dir = Path(input_dir).expanduser()
    output_dir = Path(output_dir).expanduser()
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input raster directory does not exist: {input_dir}")

    rasters = sorted(input_dir.glob(pattern))
    if not rasters:
        raise FileNotFoundError(f"No rasters matched '{pattern}' in {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if not overwrite:
        existing = [output_dir / path.name for path in rasters if (output_dir / path.name).exists()]
        if existing:
            names = ", ".join(path.name for path in existing[:10])
            raise FileExistsError(
                f"Output rasters already exist ({names}); pass --overwrite to replace them."
            )

    stats_all: Dict[str, Dict[str, float]] = {}
    for raster_path in rasters:
        output_path = output_dir / raster_path.name
        print(f"[preprocess] Processing {raster_path.name}")
        with rasterio.open(raster_path) as src:
            normalized, stats = _read_normalize_band(
                src, log_transform=log_transform
            )
            profile = src.profile.copy()
            profile.update(dtype="float32", nodata=np.nan, count=1)

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(normalized, 1)

        stats_all[raster_path.name] = stats
        print(
            f"[preprocess] Saved {output_path}; "
            f"median={stats['median_raw']:.4f}, IQR={stats['iqr_raw']:.4f}"
        )

    return stats_all


def save_stats(stats: Dict[str, Dict[str, float]], output_path: Path) -> None:
    """Write normalization statistics as standards-compliant JSON."""
    safe_stats = {
        raster: {
            key: (float(value) if np.isfinite(value) else None)
            for key, value in values.items()
        }
        for raster, values in stats.items()
    }
    output_path = Path(output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(safe_stats, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Median/IQR-normalize raster covariates before point extraction."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pattern", default="*.tif")
    parser.add_argument("--log-transform", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--stats-json",
        type=Path,
        help="Statistics file (default: OUTPUT_DIR/normalization_stats.json).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = preprocess_raster_folder(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        pattern=args.pattern,
        log_transform=args.log_transform,
        overwrite=args.overwrite,
    )
    stats_path = args.stats_json or args.output_dir / "normalization_stats.json"
    save_stats(stats, stats_path)
    print(f"[preprocess] Saved normalization statistics to {stats_path}")


if __name__ == "__main__":
    main()
