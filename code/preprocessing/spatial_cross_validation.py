"""Create spatial CV folds and an external test split using spherical k-means."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Dict, Iterable, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


WGS84 = "EPSG:4326"
EARTH_RADIUS_KM = 6371.0088


def load_points_from_csv(
    csv_path: Path,
    longitude_col: str = "Longitude",
    latitude_col: str = "Latitude",
) -> gpd.GeoDataFrame:
    """Load finite longitude/latitude rows as a WGS84 GeoDataFrame."""
    frame = pd.read_csv(csv_path)
    missing = [c for c in (longitude_col, latitude_col) if c not in frame.columns]
    if missing:
        raise ValueError(f"Presence CSV columns missing: {', '.join(missing)}")

    longitude = pd.to_numeric(frame[longitude_col], errors="coerce")
    latitude = pd.to_numeric(frame[latitude_col], errors="coerce")
    valid = np.isfinite(longitude) & np.isfinite(latitude)
    if not valid.all():
        warnings.warn(f"Dropping {int((~valid).sum())} rows with invalid coordinates.")
    frame = frame.loc[valid].copy()
    longitude = longitude.loc[valid]
    latitude = latitude.loc[valid]
    return gpd.GeoDataFrame(
        frame,
        geometry=gpd.points_from_xy(longitude, latitude),
        crs=WGS84,
    )


def _assert_wgs84(frame: gpd.GeoDataFrame, name: str) -> None:
    if frame.crs is None or frame.crs.to_epsg() != 4326:
        raise ValueError(f"{name} must use WGS84 (EPSG:4326), got {frame.crs}")
    if frame.geometry.isna().any() or frame.geometry.is_empty.any():
        raise ValueError(f"{name} contains missing or empty geometries.")
    if not frame.geometry.geom_type.eq("Point").all():
        raise ValueError(f"{name} must contain point geometries only.")
    coordinates = np.column_stack([frame.geometry.x, frame.geometry.y])
    if not np.isfinite(coordinates).all():
        raise ValueError(f"{name} contains non-finite point coordinates.")


def spherical_unit_vectors(frame: gpd.GeoDataFrame) -> np.ndarray:
    longitude = np.deg2rad(frame.geometry.x.to_numpy())
    latitude = np.deg2rad(frame.geometry.y.to_numpy())
    return np.column_stack(
        [
            np.cos(latitude) * np.cos(longitude),
            np.cos(latitude) * np.sin(longitude),
            np.sin(latitude),
        ]
    )


def fit_presence_kmeans_then_label(
    presences: gpd.GeoDataFrame,
    background: gpd.GeoDataFrame,
    n_clusters: int,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray, KMeans]:
    if n_clusters < 2:
        raise ValueError("n_clusters must be at least 2")
    if n_clusters > len(presences):
        raise ValueError(
            f"n_clusters={n_clusters} exceeds the {len(presences)} presence points."
        )
    model = KMeans(n_clusters=n_clusters, n_init=20, random_state=random_state)
    presence_labels = model.fit_predict(spherical_unit_vectors(presences))
    background_labels = model.predict(spherical_unit_vectors(background))
    return presence_labels, background_labels, model


def summarize_clusters(
    presence_labels: np.ndarray, background_labels: np.ndarray
) -> pd.DataFrame:
    presence_counts = pd.Series(presence_labels).value_counts().rename("n_pres")
    background_counts = pd.Series(background_labels).value_counts().rename("n_bg")
    total_counts = presence_counts.add(background_counts, fill_value=0).rename("n_all")
    return (
        pd.concat([presence_counts, background_counts, total_counts], axis=1)
        .fillna(0)
        .astype(int)
        .rename_axis("cluster")
        .reset_index()
    )


def choose_test_clusters(
    cluster_stats: pd.DataFrame,
    n_test: int,
    rng: np.random.Generator,
    strategy: str = "staggered",
) -> np.ndarray:
    """Select external-test clusters that contain at least one presence."""
    if n_test <= 0:
        raise ValueError("n_test_clusters must be at least 1")
    candidates = (
        cluster_stats.loc[cluster_stats["n_pres"] > 0]
        .sort_values(["n_pres", "n_all"], ascending=[False, False])
        .reset_index(drop=True)
    )
    if candidates.empty:
        raise RuntimeError("No presence-bearing clusters are available for testing.")
    if n_test >= len(cluster_stats):
        raise ValueError("The external test cannot contain every cluster.")
    if n_test > len(candidates):
        warnings.warn(
            f"Only {len(candidates)} presence-bearing clusters are available; using all."
        )
        n_test = len(candidates)

    if strategy == "top":
        chosen_indices = list(range(n_test))
    elif strategy == "random_presence":
        chosen_indices = rng.choice(len(candidates), size=n_test, replace=False).tolist()
    elif strategy == "mixed_margins":
        top_count = n_test // 2
        top_indices = list(range(top_count))
        marginal_order = candidates.sort_values(
            ["n_pres", "n_all"], ascending=[True, True]
        ).index.tolist()
        available = [index for index in marginal_order if index not in top_indices]
        margin_count = n_test - len(top_indices)
        if len(available) > margin_count:
            available = rng.choice(available, size=margin_count, replace=False).tolist()
        chosen_indices = top_indices + available[:margin_count]
    elif strategy == "staggered":
        positions = np.linspace(0, len(candidates) - 1, n_test, dtype=int)
        max_shift = max(1, len(candidates) // max(n_test, 1))
        shift = int(rng.integers(0, max_shift))
        chosen_indices = ((positions + shift) % len(candidates)).tolist()
        chosen_indices = list(dict.fromkeys(chosen_indices))
        if len(chosen_indices) < n_test:
            remaining = [i for i in range(len(candidates)) if i not in chosen_indices]
            rng.shuffle(remaining)
            chosen_indices.extend(remaining[: n_test - len(chosen_indices)])
    else:
        raise ValueError(f"Unknown test selection strategy: {strategy}")

    return candidates.iloc[chosen_indices[:n_test]]["cluster"].to_numpy(dtype=int)


def balanced_fold_assignment(
    cluster_stats: pd.DataFrame,
    n_folds: int,
    min_background_per_fold: int,
) -> Dict[int, int]:
    """Greedily assign whole clusters while balancing presence/background counts."""
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    if min_background_per_fold < 0:
        raise ValueError("min_background_per_fold cannot be negative")
    if len(cluster_stats) < n_folds:
        raise ValueError(
            f"Only {len(cluster_stats)} clusters remain for {n_folds} folds."
        )
    presence_clusters = cluster_stats.loc[cluster_stats["n_pres"] > 0]
    if len(presence_clusters) < n_folds:
        raise ValueError(
            f"Only {len(presence_clusters)} remaining clusters contain presences; "
            f"cannot give each of {n_folds} folds a presence cluster."
        )

    folds = [
        {"n_pres": 0, "n_bg": 0, "n_all": 0, "clusters": []}
        for _ in range(n_folds)
    ]
    ordered = cluster_stats.sort_values(
        ["n_pres", "n_bg", "n_all"], ascending=[False, False, False]
    )
    with_presence = ordered.loc[ordered["n_pres"] > 0].to_dict("records")
    background_only = ordered.loc[ordered["n_pres"] == 0].to_dict("records")

    def add_cluster(fold_index: int, cluster: Dict[str, int]) -> None:
        folds[fold_index]["n_pres"] += cluster["n_pres"]
        folds[fold_index]["n_bg"] += cluster["n_bg"]
        folds[fold_index]["n_all"] += cluster["n_all"]
        folds[fold_index]["clusters"].append(int(cluster["cluster"]))

    for cluster in with_presence:
        empty_presence = [i for i, fold in enumerate(folds) if fold["n_pres"] == 0]
        if empty_presence:
            target = min(empty_presence, key=lambda i: folds[i]["n_all"])
        else:
            target = min(
                range(n_folds),
                key=lambda i: (
                    folds[i]["n_pres"], folds[i]["n_bg"], folds[i]["n_all"]
                ),
            )
        add_cluster(target, cluster)

    for cluster in background_only:
        below_minimum = [
            i for i, fold in enumerate(folds)
            if fold["n_bg"] < min_background_per_fold
        ]
        candidates: Iterable[int] = below_minimum or range(n_folds)
        target = min(
            candidates,
            key=lambda i: (folds[i]["n_bg"], folds[i]["n_all"]),
        )
        add_cluster(target, cluster)

    mapping: Dict[int, int] = {}
    for fold_id, fold in enumerate(folds, start=1):
        for cluster in fold["clusters"]:
            mapping[cluster] = fold_id
    if any(fold["n_bg"] == 0 for fold in folds):
        warnings.warn("At least one CV fold has no background points.")
    return mapping


def min_gc_distance_to_set_km(
    points: np.ndarray, reference: np.ndarray, chunk_size: int = 10_000
) -> np.ndarray:
    """Compute nearest great-circle distance without allocating one huge matrix."""
    if len(reference) == 0:
        raise ValueError("Reference point set is empty.")
    distances = np.empty(len(points), dtype=float)
    for start in range(0, len(points), chunk_size):
        stop = min(start + chunk_size, len(points))
        dots = np.clip(points[start:stop] @ reference.T, -1.0, 1.0)
        distances[start:stop] = EARTH_RADIUS_KM * np.min(np.arccos(dots), axis=1)
    return distances


def test_buffer_mask(
    frame: gpd.GeoDataFrame, buffer_km: float
) -> np.ndarray:
    if buffer_km <= 0:
        return np.zeros(len(frame), dtype=bool)
    test = frame["split"].eq("test")
    if not test.any():
        return np.zeros(len(frame), dtype=bool)
    distances = min_gc_distance_to_set_km(
        spherical_unit_vectors(frame), spherical_unit_vectors(frame.loc[test])
    )
    return distances <= buffer_km


def leakage_diagnostics(presences: gpd.GeoDataFrame) -> None:
    test = presences.loc[presences["split"].eq("test")]
    cv = presences.loc[presences["split"].eq("cv")]
    if test.empty or cv.empty:
        print("[diagnostics] Not enough presence points for leakage diagnostics.")
        return
    distances = min_gc_distance_to_set_km(
        spherical_unit_vectors(cv), spherical_unit_vectors(test)
    )
    print("\n[CV presences -> nearest test presence]")
    print(f"  minimum:       {np.min(distances):.1f} km")
    print(f"  5th percentile:{np.percentile(distances, 5):.1f} km")
    print(f"  median:        {np.median(distances):.1f} km")
    print(f"  fraction <100 km: {(distances < 100).mean():.3f}")
    print(f"  fraction <300 km: {(distances < 300).mean():.3f}")


def fold_separation_diagnostics(presences: gpd.GeoDataFrame) -> None:
    effective_cv = presences["split"].eq("cv") & ~presences["in_test_buffer"]
    unit_vectors = spherical_unit_vectors(presences)
    for fold_id in sorted(presences.loc[effective_cv, "fold"].dropna().unique()):
        current = effective_cv & presences["fold"].eq(fold_id)
        other = effective_cv & ~presences["fold"].eq(fold_id)
        if not current.any() or not other.any():
            continue
        distances = min_gc_distance_to_set_km(
            unit_vectors[current.to_numpy()], unit_vectors[other.to_numpy()]
        )
        print(f"\n[Fold {int(fold_id)} separation]")
        print(f"  minimum: {np.min(distances):.1f} km")
        print(f"  median:  {np.median(distances):.1f} km")
        print(f"  fraction <100 km: {(distances < 100).mean():.3f}")


def save_diagnostic_plots(
    presences: gpd.GeoDataFrame,
    background: gpd.GeoDataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    presences.loc[presences["split"].eq("cv") & ~presences["in_test_buffer"]].plot(
        ax=ax, markersize=5, alpha=0.4, color="blue", label="CV outside buffer"
    )
    presences.loc[presences["split"].eq("cv") & presences["in_test_buffer"]].plot(
        ax=ax, markersize=5, alpha=0.6, color="orange", label="excluded buffer"
    )
    presences.loc[presences["split"].eq("test")].plot(
        ax=ax, markersize=8, alpha=0.8, color="red", label="test"
    )
    ax.set(xlabel="Longitude", ylabel="Latitude", title="Presence spatial splits")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "spatial_splits.png", dpi=200)
    plt.close(fig)

    cv_presence = presences.loc[
        presences["split"].eq("cv") & ~presences["in_test_buffer"]
    ].groupby("fold").size()
    cv_background = background.loc[
        background["split"].eq("cv") & ~background["in_test_buffer"]
    ].groupby("fold").size()
    folds = sorted(set(cv_presence.index) | set(cv_background.index))
    x = np.arange(len(folds))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - 0.18, cv_presence.reindex(folds, fill_value=0), 0.36, label="presence")
    ax.bar(x + 0.18, cv_background.reindex(folds, fill_value=0), 0.36, label="background")
    ax.set_xticks(x, [int(value) for value in folds])
    ax.set(xlabel="Fold", ylabel="Count", title="Effective CV fold sizes")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "fold_counts.png", dpi=200)
    plt.close(fig)


def export_test_cv_csv(
    presences: gpd.GeoDataFrame,
    background: gpd.GeoDataFrame,
    test_csv: Path,
    cv_csv: Path,
    exclude_buffer_from_cv: bool = True,
    overwrite: bool = False,
) -> None:
    for path in (test_csv, cv_csv):
        if path.exists() and not overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite to replace it.")
        path.parent.mkdir(parents=True, exist_ok=True)

    presence = presences.copy()
    background_copy = background.copy()
    presence["label"] = "presence"
    background_copy["label"] = "background"
    combined = pd.concat([presence, background_copy], ignore_index=True)
    combined["Longitude"] = combined.geometry.x
    combined["Latitude"] = combined.geometry.y
    columns = ["Longitude", "Latitude", "label", "cluster_id", "fold"]

    test_frame = combined.loc[combined["split"].eq("test"), columns]
    cv_mask = combined["split"].eq("cv")
    if exclude_buffer_from_cv:
        cv_mask &= ~combined["in_test_buffer"]
    cv_frame = combined.loc[cv_mask, columns]
    test_frame.to_csv(test_csv, index=False)
    cv_frame.to_csv(cv_csv, index=False)
    print(f"[save] Test CSV: {test_csv} ({len(test_frame)} rows)")
    print(f"[save] CV CSV:   {cv_csv} ({len(cv_frame)} rows)")


def partition_with_spherical_kmeans(
    presences: gpd.GeoDataFrame,
    background: gpd.GeoDataFrame,
    n_clusters: int,
    n_test_clusters: int,
    n_folds: int,
    seed: int,
    test_buffer_km: float,
    strategy: str,
    min_background_per_fold: int,
    run_diagnostics: bool = True,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    _assert_wgs84(presences, "presences")
    _assert_wgs84(background, "background")
    if presences.empty or background.empty:
        raise ValueError("Presence and background datasets must both be non-empty.")
    if test_buffer_km < 0:
        raise ValueError("test_buffer_km cannot be negative")

    rng = np.random.default_rng(seed)
    presence_labels, background_labels, _ = fit_presence_kmeans_then_label(
        presences, background, n_clusters, seed
    )
    cluster_stats = summarize_clusters(presence_labels, background_labels)
    test_clusters = set(
        choose_test_clusters(
            cluster_stats, n_test_clusters, rng, strategy=strategy
        ).tolist()
    )
    cv_stats = cluster_stats.loc[~cluster_stats["cluster"].isin(test_clusters)].copy()
    fold_map = balanced_fold_assignment(
        cv_stats, n_folds, min_background_per_fold
    )

    def attach_labels(
        frame: gpd.GeoDataFrame, labels: np.ndarray
    ) -> gpd.GeoDataFrame:
        result = frame.copy()
        result["cluster_id"] = labels
        result["split"] = np.where(result["cluster_id"].isin(test_clusters), "test", "cv")
        result["fold"] = result["cluster_id"].map(fold_map)
        result.loc[result["split"].eq("test"), "fold"] = np.nan
        result["in_test_buffer"] = False
        return result

    presence_split = attach_labels(presences, presence_labels)
    background_split = attach_labels(background, background_labels)
    presence_split["in_test_buffer"] = test_buffer_mask(
        presence_split, test_buffer_km
    )
    background_split["in_test_buffer"] = test_buffer_mask(
        background_split, test_buffer_km
    )

    effective_presence = presence_split.loc[
        presence_split["split"].eq("cv") & ~presence_split["in_test_buffer"]
    ]
    effective_background = background_split.loc[
        background_split["split"].eq("cv") & ~background_split["in_test_buffer"]
    ]
    sanity = (
        effective_presence.groupby("fold").size().rename("n_pres_cv").to_frame()
        .join(
            effective_background.groupby("fold").size().rename("n_bg_cv"),
            how="outer",
        )
        .fillna(0)
        .astype(int)
        .sort_index()
    )
    print("\n[Effective CV fold sizes after test-buffer exclusion]")
    print(sanity)
    expected_folds = set(range(1, n_folds + 1))
    folds_with_presence = set(effective_presence["fold"].dropna().astype(int))
    if folds_with_presence != expected_folds:
        missing = sorted(expected_folds - folds_with_presence)
        raise RuntimeError(
            f"Test-buffer exclusion leaves no presences in CV fold(s): {missing}. "
            "Reduce --test-buffer-km or change the split settings."
        )

    print(f"\nExternal test clusters: {sorted(test_clusters)}")
    print(
        f"Presence points: test={presence_split['split'].eq('test').sum()}, "
        f"effective CV={len(effective_presence)}"
    )
    print(
        f"Background points: test={background_split['split'].eq('test').sum()}, "
        f"effective CV={len(effective_background)}"
    )
    if run_diagnostics:
        leakage_diagnostics(presence_split)
        fold_separation_diagnostics(presence_split)
    return presence_split, background_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create spherical-cluster external-test and spatial CV splits."
    )
    parser.add_argument("--presence-csv", type=Path, required=True)
    parser.add_argument("--background-gpkg", type=Path, required=True)
    parser.add_argument("--background-layer", required=True)
    parser.add_argument("--output-gpkg", type=Path, required=True)
    parser.add_argument("--presence-layer", default="presence_spatial_splits")
    parser.add_argument("--background-output-layer", default="background_spatial_splits")
    parser.add_argument("--test-csv", type=Path)
    parser.add_argument("--cv-csv", type=Path)
    parser.add_argument("--longitude-col", default="Longitude")
    parser.add_argument("--latitude-col", default="Latitude")
    parser.add_argument("--n-clusters", type=int, default=50)
    parser.add_argument("--n-test-clusters", type=int, default=5)
    parser.add_argument("--n-folds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-buffer-km", type=float, default=300.0)
    parser.add_argument(
        "--test-selection-strategy",
        choices=["staggered", "top", "random_presence", "mixed_margins"],
        default="staggered",
    )
    parser.add_argument("--min-background-per-fold", type=int, default=300)
    parser.add_argument("--no-diagnostics", action="store_true")
    parser.add_argument("--plots-dir", type=Path)
    parser.add_argument("--keep-buffer-in-cv", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.background_gpkg.is_file():
        raise FileNotFoundError(f"Background GeoPackage not found: {args.background_gpkg}")
    if args.output_gpkg.exists() and not args.overwrite:
        raise FileExistsError(
            f"{args.output_gpkg} exists; pass --overwrite to replace its output layers."
        )
    test_csv = args.test_csv or args.output_gpkg.with_name("test_split.csv")
    cv_csv = args.cv_csv or args.output_gpkg.with_name("cv_split.csv")
    if not args.overwrite:
        existing_csv = [path for path in (test_csv, cv_csv) if path.exists()]
        if existing_csv:
            raise FileExistsError(
                f"Split CSV already exists: {', '.join(map(str, existing_csv))}. "
                "Pass --overwrite to replace it."
            )

    presences = load_points_from_csv(
        args.presence_csv, args.longitude_col, args.latitude_col
    )
    background = gpd.read_file(args.background_gpkg, layer=args.background_layer)
    _assert_wgs84(background, "background")
    presence_split, background_split = partition_with_spherical_kmeans(
        presences=presences,
        background=background,
        n_clusters=args.n_clusters,
        n_test_clusters=args.n_test_clusters,
        n_folds=args.n_folds,
        seed=args.seed,
        test_buffer_km=args.test_buffer_km,
        strategy=args.test_selection_strategy,
        min_background_per_fold=args.min_background_per_fold,
        run_diagnostics=not args.no_diagnostics,
    )

    args.output_gpkg.parent.mkdir(parents=True, exist_ok=True)
    presence_split.to_file(
        args.output_gpkg, layer=args.presence_layer, driver="GPKG"
    )
    background_split.to_file(
        args.output_gpkg, layer=args.background_output_layer, driver="GPKG"
    )
    print(
        f"[save] Spatial split layers written to {args.output_gpkg}: "
        f"{args.presence_layer}, {args.background_output_layer}"
    )

    export_test_cv_csv(
        presence_split,
        background_split,
        test_csv=test_csv,
        cv_csv=cv_csv,
        exclude_buffer_from_cv=not args.keep_buffer_in_cv,
        overwrite=args.overwrite,
    )
    if args.plots_dir:
        save_diagnostic_plots(presence_split, background_split, args.plots_dir)
        print(f"[save] Diagnostic plots written to {args.plots_dir}")


if __name__ == "__main__":
    main()
