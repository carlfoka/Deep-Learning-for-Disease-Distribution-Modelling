# ===== MaxEnt baseline for Ecological Niche Modelling (elapid MaxentModel) =====
# Goal: match RF/CNN baseline training/evaluation style while keeping MaxEnt as the model.
#
# What this does:
# 1) K-fold CV on point features parquet
# 2) Stage search over beta_multiplier:
#       BETA_STAGES = (0.5, 1.0, 2.0, 4.0)
# 3) Per-fold model selection:
#       optional AUC floor -> logloss window -> best Boyce
#       tie-break lower logloss
# 4) OOF predictions and OOF metrics using MaxEnt suitability predictions
# 5) Ensemble over folds on held-out test using CNN-style softmax(AUC) weights
# 6) Map prediction using already standardized rasters, tiled inference
#
# Dependencies:
#   pip install pandas pyarrow scikit-learn numpy rasterio tqdm elapid
#
# NOTE:
# - This keeps MaxEnt as the model.
# - "Stage" for MaxEnt = beta_multiplier regularization value.
# - Uses model.predict(X) as the suitability/probability-like score.
# - For logloss-based selection, scores are clipped to (EPS_PROBA, 1-EPS_PROBA).
# - No final full-data refit is done.
# - No parallelisation is used here.
# - Ensemble is softmax(AUC)-weighted mean of MaxEnt predictions.
# - This matches the CNN fold-weighting logic, but not CNN log-ratio ensembling.

import argparse
import os
import warnings
from pathlib import Path
from typing import Sequence, Dict, Tuple, Iterable, Optional, Union

import numpy as np
import pandas as pd

import rasterio
from rasterio.windows import Window
from tqdm import tqdm

import elapid as ela
from sklearn.metrics import log_loss

try:
    from .baseline_metrics import compute_metrics
except ImportError:  # Allow direct execution: python baselines/maxent.py
    from baseline_metrics import compute_metrics

FOLD_COL   = "fold"
LABEL_COL  = "mod_label"
LON_COL    = "x"
LAT_COL    = "y"
DROP_COLS  = ("Longitude", "Latitude", "label", "cluster_id")

NBINS_BOYCE = 20


# -------------------- MaxEnt beta grid --------------------
BETA_STAGES = (0.5, 1.0, 2.0)
MAXENT_TRANSFORM = "cloglog"   # or "logistic"


# -------------------- Selection knobs --------------------
USE_AUC_FLOOR_MAXENT = True
AUC_FLOOR_MAXENT = 0.80
LOSS_WINDOW_DELTA = 0.03
EPS_PROBA = 1e-7


# -------------------- CNN-style ensemble weighting --------------------
AUC_SOFTMAX_TAU = 50.0


TILE = 1024
DST_NODATA = -9999.0
LANDMASK_VALUE = 1

SEED = 42
np.random.seed(SEED)


# ------------------- label + feature utilities ----------------------------
def ensure_mod_label_binary_inplace(df: pd.DataFrame, col=LABEL_COL) -> None:
    if df[col].dtype == object:
        mapper = {
            "presence": 1,
            "pres": 1,
            "pos": 1,
            "positive": 1,
            "1": 1,
            "true": 1,
            "t": 1,
            "yes": 1,
            "background": 0,
            "absence": 0,
            "abs": 0,
            "bg": 0,
            "neg": 0,
            "negative": 0,
            "0": 0,
            "false": 0,
            "f": 0,
            "no": 0,
        }
        df[col] = df[col].astype(str).str.lower().str.strip().map(mapper)

    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int).clip(0, 1)


def select_numeric_features(
    df: pd.DataFrame,
    label_col: str,
    fold_col: str,
    lon_col: str,
    lat_col: str,
    drop_cols=DROP_COLS,
) -> list[str]:
    exclude = {label_col, fold_col, lon_col, lat_col, *drop_cols}

    base = [
        c for c in df.columns
        if c not in exclude and np.issubdtype(df[c].dtype, np.number)
    ]

    keep = []
    for c in base:
        v = df[c].to_numpy()
        finite = np.isfinite(v)

        if not finite.any():
            continue

        if np.nanstd(v, ddof=0) == 0:
            continue

        keep.append(c)

    return keep


def _nan_to_zero_df(X: pd.DataFrame) -> pd.DataFrame:
    return X.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _nan_to_zero(X: np.ndarray) -> np.ndarray:
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


# ------------------- Raster grid helpers ----------------------------
def tiles(height: int, width: int, tile: int) -> Iterable[Tuple[int, int, int, int]]:
    for r0 in range(0, height, tile):
        for c0 in range(0, width, tile):
            r1 = min(r0 + tile, height)
            c1 = min(c0 + tile, width)
            yield r0, r1, c0, c1


def _grid_signature(src: rasterio.io.DatasetReader) -> Tuple[str, Tuple[float, ...], int, int, Optional[float]]:
    return (
        str(src.crs),
        tuple(src.transform)[:6],
        src.width,
        src.height,
        src.nodata,
    )


def _same_grid(sig_a, sig_b, atol: float = 1e-9) -> bool:
    crs_a, t_a, w_a, h_a, _ = sig_a
    crs_b, t_b, w_b, h_b, _ = sig_b

    if crs_a != crs_b:
        return False

    if (w_a != w_b) or (h_a != h_b):
        return False

    return np.allclose(np.array(t_a), np.array(t_b), atol=atol, rtol=0)


# ------------------- MaxEnt helpers ----------------------------
def fit_maxent(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    beta_multiplier: float,
    transform: str = MAXENT_TRANSFORM,
) -> ela.MaxentModel:
    model = ela.MaxentModel(
        transform=transform,
        beta_multiplier=float(beta_multiplier),
    )
    model.fit(X_tr, y_tr)
    return model


def predict_maxent_safe(
    model: ela.MaxentModel,
    X: Union[pd.DataFrame, np.ndarray],
) -> np.ndarray:
    try:
        p = np.asarray(model.predict(X), dtype=np.float64)
    except Exception:
        n = len(X) if hasattr(X, "__len__") else 0
        p = np.full(n, np.nan, dtype=np.float64)

    return p


# ------------------- Per-fold training with stage selection ---------------------
def train_maxent_fold_stage_search(
    df: pd.DataFrame,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    feature_cols: Sequence[str],
    beta_stages: Sequence[float],
    transform: str,
    use_auc_floor: bool,
    auc_floor: float,
    loss_window_delta: float,
) -> Tuple[ela.MaxentModel, dict, dict]:
    """
    Train MaxEnt on train split. Evaluate multiple beta_multiplier values.

    Selection rule:
      1) optionally keep stages with AUC >= auc_floor
      2) among kept, find min validation logloss
      3) keep stages with logloss <= min_logloss + loss_window_delta
      4) choose max Boyce inside that window
      5) tie-break by lower logloss

    Fallback:
      - if no stage passes the filter, choose best AUC
      - if AUC invalid, choose first beta stage

    Returns:
      best_model, hist, best_row
    """
    X_tr = _nan_to_zero_df(df.loc[train_mask, feature_cols].astype(float))
    y_tr = df.loc[train_mask, LABEL_COL].to_numpy(dtype=int)

    X_va = _nan_to_zero_df(df.loc[val_mask, feature_cols].astype(float))
    y_va = df.loc[val_mask, LABEL_COL].to_numpy(dtype=int)

    hist = {
        "beta_multiplier": [],
        "val_auc": [],
        "val_boyce": [],
        "val_logloss": [],
    }

    for beta in beta_stages:
        mdl = fit_maxent(
            X_tr,
            y_tr,
            beta_multiplier=float(beta),
            transform=transform,
        )

        p_va = predict_maxent_safe(mdl, X_va)

        if np.unique(y_va).size > 1 and np.isfinite(p_va).all():
            mets = compute_metrics(y_va, p_va, nbins_boyce=NBINS_BOYCE)
            auc = float(mets["ROC_AUC"])
            boy = float(mets["Boyce"])
            ll = float(
                log_loss(
                    y_va,
                    np.clip(p_va, EPS_PROBA, 1.0 - EPS_PROBA),
                    labels=[0, 1],
                )
            )
        else:
            auc = np.nan
            boy = np.nan
            ll = np.nan

        hist["beta_multiplier"].append(float(beta))
        hist["val_auc"].append(auc if np.isfinite(auc) else np.nan)
        hist["val_boyce"].append(boy if np.isfinite(boy) else np.nan)
        hist["val_logloss"].append(ll if np.isfinite(ll) else np.nan)

    val_auc = np.asarray(hist["val_auc"], dtype=float)
    val_boy = np.asarray(hist["val_boyce"], dtype=float)
    val_ll = np.asarray(hist["val_logloss"], dtype=float)

    finite_auc = np.isfinite(val_auc)
    finite_ll = np.isfinite(val_ll)

    ok = np.zeros_like(val_auc, dtype=bool)

    if use_auc_floor and finite_auc.any():
        ok = finite_auc & (val_auc >= auc_floor)
    else:
        ok = finite_auc

    ok = ok & finite_ll

    if ok.any():
        ll_min = float(np.min(val_ll[ok]))
        in_window = ok & (val_ll <= ll_min + loss_window_delta)

        idxs = np.where(in_window)[0]
        boy_eff = np.where(np.isfinite(val_boy), val_boy, -np.inf)

        best_idx = None
        best_tuple = (-np.inf, np.inf)  # (Boyce, logloss)

        for i in idxs:
            tup = (float(boy_eff[i]), float(val_ll[i]))
            if (tup[0] > best_tuple[0]) or (
                tup[0] == best_tuple[0] and tup[1] < best_tuple[1]
            ):
                best_tuple = tup
                best_idx = int(i)

        assert best_idx is not None

    else:
        if finite_auc.any():
            best_idx = int(np.nanargmax(val_auc))
        else:
            best_idx = 0

    best_beta = float(hist["beta_multiplier"][best_idx])

    # Refit selected beta on this fold's training split.
    # This is necessary because stage-search models are temporary.
    mdl_best = fit_maxent(
        X_tr,
        y_tr,
        beta_multiplier=best_beta,
        transform=transform,
    )

    best_row = {
        "best_beta_multiplier": best_beta,
        "best_val_auc": float(val_auc[best_idx]) if np.isfinite(val_auc[best_idx]) else np.nan,
        "best_val_boyce": float(val_boy[best_idx]) if np.isfinite(val_boy[best_idx]) else np.nan,
        "best_val_logloss": float(val_ll[best_idx]) if np.isfinite(val_ll[best_idx]) else np.nan,
    }

    return mdl_best, hist, best_row


# ------------------- Cross-validation driver ---------------------------
def run_maxent_cv_baseline(
    parquet_path: str,
    fold_col=FOLD_COL,
    label_col=LABEL_COL,
    lon_col=LON_COL,
    lat_col=LAT_COL,
    drop_cols=DROP_COLS,
    beta_stages=BETA_STAGES,
    transform: str = MAXENT_TRANSFORM,
    use_auc_floor: bool = USE_AUC_FLOOR_MAXENT,
    auc_floor: float = AUC_FLOOR_MAXENT,
    loss_window_delta: float = LOSS_WINDOW_DELTA,
):
    df = pd.read_parquet(parquet_path).copy()
    df = df.dropna(subset=[fold_col, label_col]).reset_index(drop=True)

    ensure_mod_label_binary_inplace(df, col=label_col)

    feature_cols = select_numeric_features(
        df,
        label_col,
        fold_col,
        lon_col,
        lat_col,
        drop_cols,
    )

    if not feature_cols:
        raise RuntimeError("No numeric feature columns selected. Check parquet contents.")

    folds = pd.Series(df[fold_col]).dropna().unique()

    try:
        folds = np.sort(folds.astype(int))
    except Exception:
        folds = np.sort(folds)

    models_per_fold: Dict[int, ela.MaxentModel] = {}
    oof_pred = np.full(len(df), np.nan, dtype=np.float64)

    per_fold_rows = []
    stage_rows = []

    y_all = df[label_col].to_numpy(dtype=int)

    for k in folds:
        k_int = int(k)

        tr_mask = (df[fold_col] != k).to_numpy()
        va_mask = (df[fold_col] == k).to_numpy()

        mdl_k, hist, best_row = train_maxent_fold_stage_search(
            df=df,
            train_mask=tr_mask,
            val_mask=va_mask,
            feature_cols=feature_cols,
            beta_stages=beta_stages,
            transform=transform,
            use_auc_floor=use_auc_floor,
            auc_floor=auc_floor,
            loss_window_delta=loss_window_delta,
        )

        models_per_fold[k_int] = mdl_k

        # OOF predictions for this fold's validation rows
        X_va = _nan_to_zero_df(df.loc[va_mask, feature_cols].astype(float))
        p_va = predict_maxent_safe(mdl_k, X_va)
        oof_pred[np.where(va_mask)[0]] = p_va

        per_fold_rows.append(
            {
                "fold": k_int,
                **best_row,
            }
        )

        for b, auc, boy, ll in zip(
            hist["beta_multiplier"],
            hist["val_auc"],
            hist["val_boyce"],
            hist["val_logloss"],
        ):
            stage_rows.append(
                {
                    "fold": k_int,
                    "beta_multiplier": float(b),
                    "val_auc": float(auc) if np.isfinite(auc) else np.nan,
                    "val_boyce": float(boy) if np.isfinite(boy) else np.nan,
                    "val_logloss": float(ll) if np.isfinite(ll) else np.nan,
                }
            )

        print(
            f"[fold {k_int}] selected beta={best_row['best_beta_multiplier']} | "
            f"AUC={best_row['best_val_auc']:.4f} "
            f"Boyce={best_row['best_val_boyce']:.4f} "
            f"logloss={best_row['best_val_logloss']:.4f}"
        )

    if np.isfinite(oof_pred).all():
        m_oof = compute_metrics(y_all, oof_pred, nbins_boyce=NBINS_BOYCE)
        oof_metrics = {
            "OOF_Boyce": float(m_oof["Boyce"]),
            "OOF_AUC": float(m_oof["ROC_AUC"]),
        }
    else:
        oof_metrics = {
            "OOF_Boyce": np.nan,
            "OOF_AUC": np.nan,
        }
        warnings.warn("OOF predictions contain NaN values; OOF metrics set to NaN.")

    cv_df = pd.DataFrame(per_fold_rows).sort_values("fold").reset_index(drop=True)
    stage_df = pd.DataFrame(stage_rows).sort_values(["fold", "beta_multiplier"]).reset_index(drop=True)

    print("\n===== MAXENT BASELINE CV OOF METRICS =====")
    print(oof_metrics)

    print("\nPer-fold summary:")
    print(cv_df)

    cv_df.to_csv("maxent_baseline_cv_summary.csv", index=False)
    stage_df.to_csv("maxent_baseline_cv_stage_history.csv", index=False)
    pd.DataFrame([oof_metrics]).to_csv("maxent_baseline_cv_oof_metrics.csv", index=False)
    np.save("maxent_baseline_oof_pred.npy", oof_pred)

    return dict(
        df_cv=df,
        cv_df=cv_df,
        stage_df=stage_df,
        feature_cols=feature_cols,
        models_per_fold=models_per_fold,
        oof_pred=oof_pred,
        oof_metrics=oof_metrics,
        transform=transform,
    )


# ------------------- CNN-style softmax(AUC) fold weights ---------------------------
def softmax_auc_weights_from_cv_auc(
    cv_df: pd.DataFrame,
    tau: float = AUC_SOFTMAX_TAU,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    CNN-style fold weights:

        w_k ∝ exp(tau * (AUC_k - mean_AUC))

    Then normalize so sum_k w_k = 1.

    NaN AUCs receive zero effective weight.
    If all AUCs are invalid, falls back to equal weights.
    """
    cv_df = cv_df.sort_values("fold").reset_index(drop=True)

    folds = cv_df["fold"].to_numpy(dtype=int)
    aucs = cv_df["best_val_auc"].to_numpy(dtype=np.float64)

    if aucs.size == 0:
        return folds, np.array([], dtype=np.float64)

    finite = np.isfinite(aucs)

    if finite.sum() == 0:
        weights = np.ones_like(aucs, dtype=np.float64) / len(aucs)
        return folds, weights

    mean_auc = float(np.nanmean(aucs))

    # Same logic as CNN:
    # center by mean AUC, multiply by temperature, then softmax.
    x = np.where(finite, aucs - mean_auc, -1e9)
    x = tau * x

    # Numerical stability.
    x_max = np.max(x[finite])
    ex = np.exp(np.clip(x - x_max, -200, 200))
    ex[~finite] = 0.0

    s = ex.sum()

    if s <= 0 or not np.isfinite(s):
        weights = np.ones_like(aucs, dtype=np.float64) / len(aucs)
    else:
        weights = ex / s

    return folds, weights.astype(np.float64)


# ------------------- Test evaluation: ensemble of fold models ---------------------------
def evaluate_maxent_on_test_ensemble(
    models_per_fold: Dict[int, ela.MaxentModel],
    feature_cols: Sequence[str],
    test_path: str,
    label_col: str = LABEL_COL,
    fold_ids: Optional[np.ndarray] = None,
    fold_weights: Optional[np.ndarray] = None,
):
    df_te = pd.read_parquet(test_path).copy()

    if label_col not in df_te.columns:
        raise ValueError(f"Test parquet lacks '{label_col}'.")

    ensure_mod_label_binary_inplace(df_te, col=label_col)

    avail = [c for c in feature_cols if c in df_te.columns]
    missing = [c for c in feature_cols if c not in df_te.columns]

    if missing:
        warnings.warn(f"Test set missing {len(missing)} feature(s) from training: {missing}")

    if not avail:
        raise RuntimeError("No training features found in test parquet.")

    # Preserve the exact training feature order; absent test features become zero.
    X_te = _nan_to_zero_df(
        df_te.reindex(columns=list(feature_cols), fill_value=0.0).astype(float)
    )
    y_te = df_te[label_col].to_numpy(dtype=int)

    if fold_ids is None:
        fold_ids = np.array(sorted(models_per_fold.keys()), dtype=int)

    models = [models_per_fold[int(f)] for f in fold_ids]
    K = len(models)

    if K < 1:
        raise ValueError("No fold models available for ensemble.")

    if fold_weights is None:
        w = np.ones(K, dtype=np.float64) / K
    else:
        w = np.asarray(fold_weights, dtype=np.float64)

        if w.shape[0] != K:
            raise ValueError(f"fold_weights length {w.shape[0]} != #models {K}")

        if (not np.isfinite(w).all()) or (w.sum() <= 0):
            w = np.ones(K, dtype=np.float64) / K
        else:
            w = w / w.sum()

    P = np.empty((K, X_te.shape[0]), dtype=np.float64)

    for i, mdl in enumerate(models):
        P[i, :] = predict_maxent_safe(mdl, X_te)

    # MaxEnt ensemble remains in prediction/suitability space.
    p_ens = (w.reshape(K, 1) * P).sum(axis=0)

    m = compute_metrics(y_te, p_ens, nbins_boyce=NBINS_BOYCE)

    out = dict(
        test_Boyce=float(m["Boyce"]),
        test_AUC=float(m["ROC_AUC"]),
        n_test=int(len(y_te)),
        n_pos=int(y_te.sum()),
        n_bg=int((1 - y_te).sum()),
    )

    print("\n===== HELD-OUT TEST METRICS (MAXENT SOFTMAX-AUC ENSEMBLE) =====")
    for k, v in out.items():
        print(f"{k}: {v}")

    return out, p_ens.astype(np.float64), y_te, avail


# ------------------- Mapping: MaxEnt ensemble suitability ---------------------------
def build_maxent_suitability_map_ensemble_prestandardized(
    raster_dir: str,
    landmask_tif: str,
    out_dir: str,
    models_per_fold: Dict[int, ela.MaxentModel],
    fold_weights: np.ndarray,
    feature_cols: Sequence[str],
    tile: int = TILE,
    dst_nodata: float = DST_NODATA,
    landmask_value: int | float | None = LANDMASK_VALUE,
    output_name_mean: str = "maxent_baseline_suitability_mean_softmax_auc.tif",
):
    """
    Assumes rasters in raster_dir are already standardized exactly as used
    during point-feature extraction.

    Suitability score = softmax(AUC)-weighted mean of MaxEnt predictions
    across fold models.
    """
    os.makedirs(out_dir, exist_ok=True)
    raster_dir = str(raster_dir)

    fold_ids = np.array(sorted(models_per_fold.keys()), dtype=int)
    models = [models_per_fold[int(f)] for f in fold_ids]
    K = len(models)

    if K < 1:
        raise ValueError("No models provided for ensemble.")

    w = np.asarray(fold_weights, dtype=np.float64)

    if w.shape[0] != K or (not np.isfinite(w).all()) or (w.sum() <= 0):
        w = np.ones(K, dtype=np.float64) / K
    else:
        w = w / w.sum()

    ras_paths = sorted(Path(raster_dir).glob("*.tif"))

    if not ras_paths:
        raise FileNotFoundError(f"No rasters found in {raster_dir}")

    name_to_path = {p.stem: p for p in ras_paths}

    present_feats = [f for f in feature_cols if f in name_to_path]
    missing_feats = [f for f in feature_cols if f not in name_to_path]

    if missing_feats:
        warnings.warn(f"Missing {len(missing_feats)} raster(s); filling with zeros: {missing_feats}")

    if not present_feats:
        raise RuntimeError("None of the feature_cols exist as rasters in raster_dir.")

    # Reference grid
    ref_path = name_to_path[present_feats[0]]

    with rasterio.open(str(ref_path)) as ref:
        ref_sig = _grid_signature(ref)
        profile = ref.profile.copy()
        H, W = ref.height, ref.width

    # Grid checks
    for f in present_feats:
        with rasterio.open(str(name_to_path[f])) as src:
            if not _same_grid(_grid_signature(src), ref_sig):
                raise RuntimeError(f"Grid/CRS mismatch: {name_to_path[f].name}. Align rasters first.")

    # Landmask
    with rasterio.open(landmask_tif) as lm:
        if not _same_grid(_grid_signature(lm), ref_sig):
            raise RuntimeError("Grid/CRS mismatch for landmask.")
        landmask = lm.read(1)

    out_map = np.full((H, W), np.nan, dtype=np.float32)

    n_tiles = ((H + tile - 1) // tile) * ((W + tile - 1) // tile)

    for r0, r1, c0, c1 in tqdm(
        tiles(H, W, tile),
        total=n_tiles,
        desc="MaxEnt baseline map",
    ):
        lm_tile = landmask[r0:r1, c0:c1]

        if landmask_value is not None and not np.any(lm_tile == landmask_value):
            continue

        if landmask_value is not None:
            rc = np.argwhere(lm_tile == landmask_value)
        else:
            rc = np.argwhere(np.isfinite(lm_tile))

        if rc.size == 0:
            continue

        rr, cc = rc[:, 0], rc[:, 1]
        n = rr.size

        X = np.empty((n, len(feature_cols)), dtype=np.float32)
        win = Window(c0, r0, c1 - c0, r1 - r0)

        for j, fname in enumerate(feature_cols):
            if fname not in name_to_path:
                X[:, j] = 0.0
                continue

            path = name_to_path[fname]

            with rasterio.open(str(path)) as src:
                a = src.read(1, window=win, masked=True).astype("float32")

            a = np.array(a.filled(np.nan), dtype=np.float32)

            v = a[rr, cc]
            v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)

            X[:, j] = v.astype(np.float32)

        X_df = pd.DataFrame(X, columns=list(feature_cols))

        p_ens = np.zeros(n, dtype=np.float64)

        for k, mdl in enumerate(models):
            p = predict_maxent_safe(mdl, X_df)
            p_ens += w[k] * p

        out_map[r0 + rr, c0 + cc] = p_ens.astype(np.float32)

    prof = profile.copy()
    prof.update(
        count=1,
        dtype="float32",
        nodata=dst_nodata,
    )

    out_path = os.path.join(out_dir, output_name_mean)

    with rasterio.open(out_path, "w", **prof) as dst:
        dst.write(
            np.where(np.isfinite(out_map), out_map, dst_nodata).astype("float32"),
            1,
        )

    print(f"Written: {out_path}")
    return out_path


# ------------------- Command-line interface ----------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the MaxEnt ENM baseline."
    )
    parser.add_argument(
        "--train-path",
        type=Path,
        required=True,
        help="Cross-validation point-feature Parquet file.",
    )
    parser.add_argument(
        "--test-path",
        type=Path,
        required=True,
        help="Held-out point-feature Parquet file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/baselines/maxent"),
        help="Directory for metrics, predictions, and suitability maps.",
    )
    parser.add_argument(
        "--raster-dir",
        type=Path,
        help="Optional directory of pre-standardized covariate TIFF files.",
    )
    parser.add_argument(
        "--landmask-tif",
        type=Path,
        help="Optional land-mask TIFF. Required when --raster-dir is used.",
    )
    args = parser.parse_args()
    if (args.raster_dir is None) != (args.landmask_tif is None):
        parser.error("--raster-dir and --landmask-tif must be supplied together")
    return args


# ------------------- MAIN ----------------------------
def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running with elapid version: {ela.__version__}")

    # 1) CV: OOF metrics + per-fold beta selection
    cv_out = run_maxent_cv_baseline(
        str(args.train_path),
        fold_col=FOLD_COL,
        label_col=LABEL_COL,
        lon_col=LON_COL,
        lat_col=LAT_COL,
        drop_cols=DROP_COLS,
        beta_stages=BETA_STAGES,
        transform=MAXENT_TRANSFORM,
        use_auc_floor=USE_AUC_FLOOR_MAXENT,
        auc_floor=AUC_FLOOR_MAXENT,
        loss_window_delta=LOSS_WINDOW_DELTA,
    )

    # 2) CNN-style softmax(AUC) ensemble weights
    fold_ids, fold_w = softmax_auc_weights_from_cv_auc(
        cv_out["cv_df"],
        tau=AUC_SOFTMAX_TAU,
    )

    print(f"\n[Ensemble] CNN-style softmax(AUC) weights, tau={AUC_SOFTMAX_TAU}")
    for f, w in zip(fold_ids, fold_w):
        print(f"  fold {int(f)}: weight={float(w):.6f}")

    np.save(args.output_dir / "fold_ids.npy", fold_ids)
    np.save(args.output_dir / "fold_weights_softmax_auc.npy", fold_w)
    np.save(
        args.output_dir / "fold_aucs.npy",
        cv_out["cv_df"].sort_values("fold")["best_val_auc"].to_numpy(dtype=float),
    )

    pd.DataFrame(
        {
            "fold": fold_ids,
            "softmax_auc_weight": fold_w,
            "best_val_auc": cv_out["cv_df"]
                .sort_values("fold")["best_val_auc"]
                .to_numpy(dtype=float),
        }
    ).to_csv(args.output_dir / "fold_weights_softmax_auc.csv", index=False)

    # Enforce fold order
    models_per_fold = cv_out["models_per_fold"]
    models_per_fold = {int(f): models_per_fold[int(f)] for f in fold_ids}

    # 3) Held-out test ensemble metrics
    test_metrics, test_scores, test_labels, used_features = evaluate_maxent_on_test_ensemble(
        models_per_fold=models_per_fold,
        feature_cols=cv_out["feature_cols"],
        test_path=str(args.test_path),
        label_col=LABEL_COL,
        fold_ids=fold_ids,
        fold_weights=fold_w,
    )

    pd.DataFrame([test_metrics]).to_csv(
        args.output_dir / "test_metrics.csv",
        index=False,
    )

    np.save(args.output_dir / "test_scores.npy", test_scores)

    # 4) Optional mapping with the fold ensemble
    if args.raster_dir is not None:
        map_dir = args.output_dir / "suitability_maps"
        build_maxent_suitability_map_ensemble_prestandardized(
            raster_dir=args.raster_dir,
            landmask_tif=args.landmask_tif,
            out_dir=map_dir,
            models_per_fold=models_per_fold,
            fold_weights=fold_w,
            feature_cols=cv_out["feature_cols"],
            tile=TILE,
            dst_nodata=DST_NODATA,
            landmask_value=LANDMASK_VALUE,
            output_name_mean="maxent_suitability_mean_softmax_auc.tif",
        )


if __name__ == "__main__":
    main()
