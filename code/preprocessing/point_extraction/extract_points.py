"""Extract raster covariates at tabular point coordinates."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
from rasterio.transform import rowcol
from rasterio.warp import transform as transform_coordinates


GridSignature = Tuple[Optional[CRS], Tuple[float, ...], int, int]


def _grid_signature(src: rasterio.io.DatasetReader) -> GridSignature:
    return src.crs, tuple(src.transform)[:6], src.width, src.height


def _same_grid(a: GridSignature, b: GridSignature, atol: float = 1e-9) -> bool:
    crs_a, transform_a, width_a, height_a = a
    crs_b, transform_b, width_b, height_b = b
    return (
        crs_a == crs_b
        and width_a == width_b
        and height_a == height_b
        and np.allclose(transform_a, transform_b, atol=atol, rtol=0)
    )


def _valid_coordinates(
    frame: pd.DataFrame, longitude_col: str, latitude_col: str
) -> pd.DataFrame:
    missing = [c for c in (longitude_col, latitude_col) if c not in frame.columns]
    if missing:
        raise ValueError(f"Coordinate columns missing from CSV: {', '.join(missing)}")

    result = frame.copy()
    result[longitude_col] = pd.to_numeric(result[longitude_col], errors="coerce")
    result[latitude_col] = pd.to_numeric(result[latitude_col], errors="coerce")
    valid = np.isfinite(result[longitude_col]) & np.isfinite(result[latitude_col])
    print(f"[coordinates] Keeping {int(valid.sum())} / {len(result)} valid rows")
    return result.loc[valid].copy()


def extract_points_from_rasters(
    raster_dir: Path,
    coordinates: pd.DataFrame,
    longitude_col: str,
    latitude_col: str,
    pattern: str = "*.tif",
    check_same_grid: bool = True,
    coordinates_crs: Optional[str] = "EPSG:4326",
) -> pd.DataFrame:
    """Add one sampled-value column per aligned raster to a point table."""
    raster_dir = Path(raster_dir).expanduser()
    rasters = sorted(raster_dir.glob(pattern))
    if not rasters:
        raise FileNotFoundError(f"No rasters matched '{pattern}' in {raster_dir}")

    with rasterio.open(rasters[0]) as reference:
        reference_signature = _grid_signature(reference)
        reference_transform = reference.transform
        reference_crs = reference.crs
        height, width = reference.height, reference.width

    xs = coordinates[longitude_col].to_numpy(dtype=float)
    ys = coordinates[latitude_col].to_numpy(dtype=float)

    if coordinates_crs is not None:
        if reference_crs is None:
            raise ValueError("Raster CRS is undefined, so coordinates cannot be reprojected.")
        source_crs = CRS.from_user_input(coordinates_crs)
        if source_crs != reference_crs:
            xs_out, ys_out = transform_coordinates(source_crs, reference_crs, xs, ys)
            xs = np.asarray(xs_out, dtype=float)
            ys = np.asarray(ys_out, dtype=float)

    rows, cols = rowcol(reference_transform, xs, ys, op=np.floor)
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    in_bounds = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    if not np.all(in_bounds):
        print(
            f"[extract] Warning: {int((~in_bounds).sum())} coordinates are outside "
            "the raster extent; their values will be NaN."
        )

    output = coordinates.copy()
    collisions = sorted({path.stem for path in rasters} & set(output.columns))
    if collisions:
        raise ValueError(
            "Raster names would overwrite existing table columns: "
            + ", ".join(collisions)
        )
    for raster_path in rasters:
        with rasterio.open(raster_path) as src:
            if check_same_grid and not _same_grid(
                _grid_signature(src), reference_signature
            ):
                raise RuntimeError(
                    f"Grid or CRS mismatch for {raster_path.name}; align rasters first."
                )
            band = np.asarray(src.read(1, masked=True).filled(np.nan), dtype="float32")

        values = np.full(rows.shape, np.nan, dtype="float32")
        values[in_bounds] = band[rows[in_bounds], cols[in_bounds]]
        output[raster_path.stem] = values

    return output


def save_table(frame: pd.DataFrame, output_path: Path) -> Path:
    output_path = Path(output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix == ".parquet":
        frame.to_parquet(output_path, index=False)
    elif suffix == ".csv":
        frame.to_csv(output_path, index=False)
    else:
        raise ValueError("Output must use a .parquet or .csv extension.")
    return output_path


def run_extraction_and_save(
    points_csv: Path,
    raster_dir: Path,
    output_path: Path,
    longitude_col: str = "Longitude",
    latitude_col: str = "Latitude",
    raster_pattern: str = "*.tif",
    coordinates_crs: Optional[str] = "EPSG:4326",
) -> Path:
    frame = pd.read_csv(points_csv)
    frame = _valid_coordinates(frame, longitude_col, latitude_col)
    features = extract_points_from_rasters(
        raster_dir=raster_dir,
        coordinates=frame,
        longitude_col=longitude_col,
        latitude_col=latitude_col,
        pattern=raster_pattern,
        check_same_grid=True,
        coordinates_crs=coordinates_crs,
    )
    saved_path = save_table(features, output_path)
    print(f"[save] Saved {features.shape[0]} rows x {features.shape[1]} columns to {saved_path}")
    return saved_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract aligned raster values at point coordinates."
    )
    parser.add_argument("--points-csv", type=Path, required=True)
    parser.add_argument("--raster-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--longitude-col", default="Longitude")
    parser.add_argument("--latitude-col", default="Latitude")
    parser.add_argument("--raster-pattern", default="*.tif")
    parser.add_argument(
        "--coordinates-crs",
        default="EPSG:4326",
        help="CRS of the CSV coordinates; use 'none' only if already in raster CRS.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    coordinates_crs = None if args.coordinates_crs.lower() == "none" else args.coordinates_crs
    run_extraction_and_save(
        points_csv=args.points_csv,
        raster_dir=args.raster_dir,
        output_path=args.output,
        longitude_col=args.longitude_col,
        latitude_col=args.latitude_col,
        raster_pattern=args.raster_pattern,
        coordinates_crs=coordinates_crs,
    )


if __name__ == "__main__":
    main()
