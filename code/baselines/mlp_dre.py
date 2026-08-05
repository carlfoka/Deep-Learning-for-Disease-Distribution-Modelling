"""
MLP-only DRE training (patch_size=1, no spatial context)
========================================================
Identical pipeline to the CNN version (ranking loss, softmax-AUC
ensemble weights, Boyce+AUC selection) but with the CNN encoder
removed.  Each sample is a single pixel's C covariate values fed
directly into the MLP head.

Data layout expected:
  X.npy  — (N, C, 1, 1)  or  (N, C)
  M.npy  — (N, C, 1, 1)  or  (N, C)
  y.npy  — (N,)
  fold.npy — (N,)
"""

import argparse
import csv
import os
import random
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from sklearn.metrics import roc_auc_score


# =========================================================
# CONFIG
# =========================================================

BATCH_SIZE  = 256
NUM_WORKERS = 0

MAX_EPOCHS = 100
PATIENCE   = 10

LR           = 1e-3
WEIGHT_DECAY = 1e-3

CLIP_LOGITS = 10.0
SEED        = 42

HIDDEN_DIMS = [32]   # MLP hidden layers (tune as needed)
DROPOUT     = 0.35

PATCH_SIZE = 1  # single pixel — no spatial context

# Data augmentation (only noise makes sense for 1×1)
NOISE_STD = 0.04

# Selection rule
USE_AUC_FLOOR    = False
AUC_FLOOR        = 0.80
LOSS_WINDOW_DELTA = 0.03

# Numeric stability
MAX_ABS_LOG_RATIO = 50.0

# Ensemble
ENSEMBLE_IN_LOGSPACE = True

# Ranking loss
LAMBDA_RANK      = 0.05
MAX_RANK_PAIRS   = 4096
RANK_TEMPERATURE = 1.0

# Softmax(AUC) weighting
AUC_SOFTMAX_TAU = 50.0


# =========================================================
# Device
# =========================================================
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# =========================================================
# Seed
# =========================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================================================
# Metrics (unchanged)
# =========================================================
def _average_ranks(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(x) + 1, dtype=float)
    sx = x[order]
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and sx[j] == sx[i]:
            j += 1
        if j - i > 1:
            avg = (i + 1 + j) / 2.0
            ranks[order[i:j]] = avg
        i = j
    return ranks


def _spearman_tieaware(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.size < 3 or y.size < 3:
        return np.nan
    rx, ry = _average_ranks(x), _average_ranks(y)
    if np.all(rx == rx[0]) or np.all(ry == ry[0]):
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def continuous_boyce(y_true, scores, nbins_max=20, min_per_group=10):
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    s_bg, s_pr = s[y == 0], s[y == 1]
    if s_bg.size < min_per_group or s_pr.size < min_per_group:
        return np.nan
    uq = np.unique(s_bg[np.isfinite(s_bg)])
    if uq.size < 3:
        return np.nan
    nb = min(nbins_max, max(3, uq.size - 1))
    qs = np.quantile(s_bg, np.linspace(0.0, 1.0, nb + 1))
    qs[0] -= 1e-12
    qs[-1] += 1e-12
    pratio, centers = [], []
    Lb, Lp = float(len(s_bg)), float(len(s_pr))
    for a, b in zip(qs[:-1], qs[1:]):
        in_bg = (s_bg >= a) & (s_bg < b)
        nbk = int(in_bg.sum())
        if nbk == 0:
            continue
        npk = int(((s_pr >= a) & (s_pr < b)).sum())
        pratio.append((npk / Lp) / (nbk / Lb))
        centers.append(0.5 * (a + b))
    if len(pratio) < 3:
        return np.nan
    return _spearman_tieaware(np.asarray(centers), np.asarray(pratio))


def compute_auc(y_true, scores):
    y = np.asarray(y_true, int)
    s = np.asarray(scores, float)
    m = np.isfinite(s)
    y, s = y[m], s[m]
    if y.size < 2 or np.unique(y).size < 2:
        return np.nan
    return float(roc_auc_score(y, s))


def compute_boyce_and_auc(y_true, scores, nbins_boyce=20):
    y = np.asarray(y_true, int)
    s = np.asarray(scores, float)
    m = np.isfinite(s)
    y, s = y[m], s[m]
    if y.size == 0:
        return dict(Boyce=np.nan, ROC_AUC=np.nan)
    return dict(
        Boyce=continuous_boyce(y, s, nbins_max=nbins_boyce),
        ROC_AUC=compute_auc(y, s),
    )


# =========================================================
# Helpers
# =========================================================
def sigmoid_np(x):
    x = np.clip(np.asarray(x, np.float64), -MAX_ABS_LOG_RATIO, MAX_ABS_LOG_RATIO)
    return 1.0 / (1.0 + np.exp(-x))


def logits_to_logratio(logits_np, pi0, pi1):
    eps = 1e-12
    pi0 = float(np.clip(pi0, eps, 1.0 - eps))
    pi1 = float(np.clip(pi1, eps, 1.0 - eps))
    return logits_np + np.log(pi0 / pi1)


def logits_to_bounded_score(logits_np, pi0, pi1):
    return sigmoid_np(logits_to_logratio(logits_np, pi0, pi1))


def compute_epoch_metrics_from_logits(logits_np, y_np, loss_val, pi0, pi1):
    score01 = logits_to_bounded_score(logits_np, pi0, pi1)
    mets = compute_boyce_and_auc(y_np, score01)
    mets["loss"] = float(loss_val)
    return mets


def make_cv_indices(fold_cv):
    return sorted(int(v) for v in np.unique(fold_cv))


# =========================================================
# Dataset — flat pixel vectors
# =========================================================
class PixelDREDataset(Dataset):
    """
    Each sample is (x_values, x_mask, y) where x_values and x_mask
    are 1-D vectors of length C (one value per covariate).
    """
    def __init__(self, X_values, X_masks, y, indices, train: bool):
        super().__init__()
        self.Xv = X_values.astype(np.float32)   # (N, C)
        self.Xm = X_masks.astype(np.float32)     # (N, C)
        self.y  = y.astype(np.float32)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.train = train

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        xv = torch.from_numpy(self.Xv[idx])     # (C,)
        xm = torch.from_numpy(self.Xm[idx])     # (C,)
        yy = torch.tensor(self.y[idx], dtype=torch.float32)

        # Augmentation: additive noise on valid channels
        if self.train and NOISE_STD > 0.0:
            noise = torch.randn_like(xv) * NOISE_STD
            valid = (xm > 0).to(dtype=xv.dtype)
            xv = xv + noise * valid

        return xv, xm, yy


# =========================================================
# MLP DRE model (no CNN encoder)
# =========================================================
class MLPDRE(nn.Module):
    def __init__(self, in_dim: int, hidden_dims=(64, 32), dropout=0.35, clip_logits=10.0):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
        self.clip_logits = clip_logits

    def forward(self, x):
        logits = self.net(x).squeeze(-1)
        if self.clip_logits is not None:
            logits = torch.clamp(logits, -self.clip_logits, self.clip_logits)
        return logits


class PixelDRE(nn.Module):
    """
    Pixel-level DRE: raw covariates (C,) → MLP → logit.
    Masks are applied as element-wise zeroing before the MLP.
    """
    def __init__(self, in_channels: int, hidden_dims=(64, 32), dropout=0.35, clip_logits=10.0):
        super().__init__()
        self.mlp = MLPDRE(
            in_dim=in_channels,
            hidden_dims=hidden_dims,
            dropout=dropout,
            clip_logits=clip_logits,
        )

    def forward(self, x_val, x_mask):
        # x_val: (B, C),  x_mask: (B, C)
        # zero out invalid channels
        m = (x_mask > 0).to(dtype=x_val.dtype)
        x = x_val * m
        return self.mlp(x)


# =========================================================
# Ranking loss (unchanged)
# =========================================================
def pairwise_ranking_loss(logits, y, max_pairs=4096, temperature=1.0):
    y = (y > 0.5)
    pos, neg = logits[y], logits[~y]
    if pos.numel() == 0 or neg.numel() == 0:
        return logits.new_tensor(0.0)
    diff = (pos[:, None] - neg[None, :]) / float(max(1e-8, temperature))
    n_pairs = diff.numel()
    if n_pairs > max_pairs:
        idx = torch.randint(0, n_pairs, (max_pairs,), device=logits.device)
        diff = diff.reshape(-1)[idx]
    return F.softplus(-diff).mean()


# =========================================================
# Softmax(AUC) weights (unchanged)
# =========================================================
def softmax_auc_weights(fold_aucs, tau=50.0):
    a = np.asarray(fold_aucs, np.float64)
    m = np.isfinite(a)
    if m.sum() == 0:
        return np.ones_like(a) / len(a)
    mean_auc = float(np.nanmean(a))
    x = np.where(m, a - mean_auc, -1e9) * tau
    x_max = np.max(x[m])
    ex = np.exp(np.clip(x - x_max, -200, 200))
    ex[~m] = 0.0
    s = ex.sum()
    return ex / s if (s > 0 and np.isfinite(s)) else np.ones_like(a) / len(a)


# =========================================================
# Training per fold
# =========================================================
def train_and_save_fold_model(
    fold_id: int,
    X_values: np.ndarray,
    X_masks: np.ndarray,
    y_cv: np.ndarray,
    fold_cv: np.ndarray,
    model_dir: str,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:

    os.makedirs(model_dir, exist_ok=True)

    f = fold_cv.astype(int)
    train_idx = np.where(f != fold_id)[0]
    val_idx   = np.where(f == fold_id)[0]

    pi1_fold = float((y_cv[train_idx] == 1).mean())
    pi0_fold = 1.0 - pi1_fold
    print(f"\n[fold {fold_id}] train N={len(train_idx)}, val N={len(val_idx)} "
          f"| pi1={pi1_fold:.6f}, pi0={pi0_fold:.6f}")

    in_channels = X_values.shape[1]

    ds_tr  = PixelDREDataset(X_values, X_masks, y_cv, train_idx, train=True)
    ds_val = PixelDREDataset(X_values, X_masks, y_cv, val_idx,   train=False)

    dl_tr  = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=NUM_WORKERS, drop_last=False)
    dl_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, drop_last=False)

    model = PixelDRE(
        in_channels=in_channels,
        hidden_dims=HIDDEN_DIMS,
        dropout=DROPOUT,
        clip_logits=CLIP_LOGITS,
    ).to(device)

    bce = nn.BCEWithLogitsLoss()
    opt = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    def run_epoch(loader, opt_obj=None):
        train = opt_obj is not None
        model.train() if train else model.eval()

        total_loss = 0.0
        all_logits, all_y = [], []

        for xv, xm, yb in loader:
            xv, xm, yb = xv.to(device), xm.to(device), yb.to(device)
            if train:
                opt_obj.zero_grad(set_to_none=True)

            with torch.set_grad_enabled(train):
                logits = model(xv, xm)
                loss_bce = bce(logits, yb)
                loss_rank = pairwise_ranking_loss(
                    logits, yb,
                    max_pairs=MAX_RANK_PAIRS,
                    temperature=RANK_TEMPERATURE,
                )
                loss = loss_bce + LAMBDA_RANK * loss_rank

                if train:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt_obj.step()

            total_loss += loss.item() * yb.size(0)
            all_logits.append(logits.detach().cpu().numpy())
            all_y.append(yb.detach().cpu().numpy())

        total_loss /= len(loader.dataset)
        all_logits = np.concatenate(all_logits)
        all_y = np.concatenate(all_y)
        mets = compute_epoch_metrics_from_logits(all_logits, all_y, total_loss, pi0_fold, pi1_fold)
        return mets, all_logits, all_y

    candidates = []
    best_loss_seen = np.inf
    no_improve = 0

    for ep in range(1, MAX_EPOCHS + 1):
        tr, _, _ = run_epoch(dl_tr, opt)
        va, val_logits, val_y = run_epoch(dl_val, None)

        print(
            f"[fold {fold_id}] epoch {ep:03d} | "
            f"train loss {tr['loss']:.4f}, AUC {tr['ROC_AUC']:.4f}, Boyce {tr['Boyce']:.4f} | "
            f"val loss {va['loss']:.4f}, AUC {va['ROC_AUC']:.4f}, Boyce {va['Boyce']:.4f}"
        )

        auc_ok = True
        if USE_AUC_FLOOR:
            auc_ok = np.isfinite(va["ROC_AUC"]) and (va["ROC_AUC"] >= AUC_FLOOR)

        if auc_ok:
            state = {
                "model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "in_channels": in_channels,
                "patch_size": PATCH_SIZE,
                "hidden_dims": list(HIDDEN_DIMS),
                "pi0": float(pi0_fold),
                "pi1": float(pi1_fold),
            }
            candidates.append({
                "loss": float(va["loss"]),
                "boyce": float(va["Boyce"]) if np.isfinite(va["Boyce"]) else np.nan,
                "auc": float(va["ROC_AUC"]) if np.isfinite(va["ROC_AUC"]) else np.nan,
                "state": state,
                "val_logits": val_logits,
                "val_y": val_y,
                "epoch": ep,
            })

            if va["loss"] < best_loss_seen - 1e-9:
                best_loss_seen = float(va["loss"])
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1

        if no_improve >= PATIENCE:
            print(f"[fold {fold_id}] early stopping at epoch {ep}")
            break

    if len(candidates) == 0:
        raise RuntimeError(
            f"[fold {fold_id}] No epoch met AUC floor ({AUC_FLOOR}). "
            f"Lower AUC_FLOOR or set USE_AUC_FLOOR=False."
        )

    loss_min = min(c["loss"] for c in candidates)
    thresh = loss_min + LOSS_WINDOW_DELTA
    window = [c for c in candidates if c["loss"] <= thresh]

    def key(c):
        b = c["boyce"]
        b_val = -np.inf if (b is None or not np.isfinite(b)) else float(b)
        return (b_val, -c["loss"])

    best = max(window, key=key)

    ckpt_path = os.path.join(model_dir, f"mlp_dre_fold_{fold_id}.pt")
    torch.save(best["state"], ckpt_path)
    print(
        f"[fold {fold_id}] selected epoch={best['epoch']} | "
        f"loss={best['loss']:.4f}, Boyce={best['boyce']:.4f}, AUC={best['auc']:.4f} | "
        f"saved -> {ckpt_path}"
    )

    val_logits = best["val_logits"]
    val_score01 = logits_to_bounded_score(val_logits, best["state"]["pi0"], best["state"]["pi1"])
    best_val = {"Boyce": best["boyce"], "ROC_AUC": best["auc"], "loss": best["loss"]}
    return best_val, val_score01, val_idx


# =========================================================
# Ensemble prediction
# =========================================================
def ensemble_predict_on_indices(
    X_values, X_masks, y, indices, model_dir, fold_ids, weights,
):
    in_channels = X_values.shape[1]

    models = []
    per_model_logpi = []
    for fid in fold_ids:
        ckpt_path = os.path.join(model_dir, f"mlp_dre_fold_{fid}.pt")
        state = torch.load(ckpt_path, map_location="cpu")

        model = PixelDRE(
            in_channels=in_channels,
            hidden_dims=tuple(state.get("hidden_dims", HIDDEN_DIMS)),
            dropout=DROPOUT,
            clip_logits=CLIP_LOGITS,
        ).to(device)
        model.load_state_dict(state["model_state"])
        model.eval()
        models.append(model)

        pi0_k = float(state["pi0"])
        pi1_k = float(state["pi1"])
        per_model_logpi.append(np.log(pi0_k / pi1_k))

    w = np.asarray(weights, np.float64)
    if (not np.isfinite(w).all()) or w.sum() <= 0:
        w = np.ones(len(fold_ids), np.float64) / len(fold_ids)
    else:
        w = w / w.sum()

    weights_t = torch.tensor(w, dtype=torch.float32, device=device)
    logpi_t = torch.tensor(np.asarray(per_model_logpi, np.float32), dtype=torch.float32, device=device)

    idx = np.asarray(indices, dtype=np.int64)
    all_score01, all_logratio, all_y = [], [], []

    with torch.no_grad():
        for start in range(0, len(idx), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(idx))
            idx_batch = idx[start:end]

            xv = torch.from_numpy(X_values[idx_batch].astype(np.float32)).to(device)
            xm = torch.from_numpy(X_masks[idx_batch].astype(np.float32)).to(device)

            logratios = []
            for k, model in enumerate(models):
                logits_k = model(xv, xm)
                log_ratio_k = logits_k + logpi_t[k]
                log_ratio_k = torch.clamp(log_ratio_k, -MAX_ABS_LOG_RATIO, MAX_ABS_LOG_RATIO)
                logratios.append(log_ratio_k)

            if ENSEMBLE_IN_LOGSPACE:
                log_ratio_ens = torch.zeros(xv.size(0), device=device)
                for k in range(len(models)):
                    log_ratio_ens += weights_t[k] * logratios[k]
            else:
                ratio_sum = torch.zeros(xv.size(0), device=device)
                for k in range(len(models)):
                    ratio_sum += weights_t[k] * torch.exp(logratios[k])
                log_ratio_ens = torch.log(torch.clamp(ratio_sum, min=1e-30))

            log_ratio_ens = torch.clamp(log_ratio_ens, -MAX_ABS_LOG_RATIO, MAX_ABS_LOG_RATIO)
            score01 = torch.sigmoid(log_ratio_ens)

            all_score01.append(score01.cpu().numpy())
            all_logratio.append(log_ratio_ens.cpu().numpy())
            all_y.append(y[idx_batch])

    return (
        np.concatenate(all_score01),
        np.concatenate(all_logratio),
        np.concatenate(all_y),
    )


# =========================================================
# Command-line interface
# =========================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the pixel-level MLP-DRE baseline."
    )
    parser.add_argument(
        "--cv-dir",
        type=Path,
        required=True,
        help="Directory containing CV X.npy, M.npy, y.npy, and fold.npy.",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        required=True,
        help="Directory containing test X.npy, M.npy, and y.npy.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/baselines/mlp_dre"),
        help="Directory for checkpoints, metrics, and predictions.",
    )
    return parser.parse_args()


# =========================================================
# MAIN
# =========================================================
def main() -> None:
    args = parse_args()
    model_dir = str(args.output_dir)

    set_seed(SEED)
    os.makedirs(model_dir, exist_ok=True)

    cv_x_path = args.cv_dir / "X.npy"
    cv_m_path = args.cv_dir / "M.npy"
    cv_y_path = args.cv_dir / "y.npy"
    cv_fold_path = args.cv_dir / "fold.npy"

    test_x_path = args.test_dir / "X.npy"
    test_m_path = args.test_dir / "M.npy"
    test_y_path = args.test_dir / "y.npy"

    # ----------- load CV data -----------
    X_cv = np.load(cv_x_path)
    M_cv = np.load(cv_m_path)
    y_cv = np.load(cv_y_path).astype(np.float32)
    fold_cv_raw = np.load(cv_fold_path)

    # Flatten from (N, C, 1, 1) to (N, C) if needed
    if X_cv.ndim == 4 and X_cv.shape[2] == 1 and X_cv.shape[3] == 1:
        if M_cv.shape != X_cv.shape:
            raise ValueError(f"CV mask shape {M_cv.shape} does not match X shape {X_cv.shape}")
        X_cv = X_cv[:, :, 0, 0]
        M_cv = M_cv[:, :, 0, 0]
        print("[CV] Squeezed (N,C,1,1) -> (N,C)")

    if X_cv.ndim != 2:
        raise ValueError(f"CV X must have shape (N, C) or (N, C, 1, 1); got {X_cv.shape}")
    if M_cv.shape != X_cv.shape:
        raise ValueError(f"CV mask shape {M_cv.shape} does not match X shape {X_cv.shape}")

    print("[CV] X shape:", X_cv.shape)
    print("[CV] M shape:", M_cv.shape)
    print("[CV] y shape:", y_cv.shape)
    print("[CV] fold shape:", fold_cv_raw.shape)

    if X_cv.shape[0] != y_cv.shape[0]:
        raise ValueError("Mismatch between CV X and y lengths")
    if fold_cv_raw.shape[0] != y_cv.shape[0]:
        raise ValueError("Mismatch between CV fold and y lengths")

    fold_cv = fold_cv_raw.astype(int)

    bad = ~np.isfinite(X_cv)
    if bad.any():
        print(f"[CV] Warning: {int(bad.sum())} non-finite X values set to 0.")
        X_cv[bad] = 0.0
    bad_m = ~np.isfinite(M_cv)
    if bad_m.any():
        M_cv[bad_m] = 0.0

    # ----------- load TEST data -----------
    X_test = np.load(test_x_path)
    M_test = np.load(test_m_path)
    y_test = np.load(test_y_path).astype(np.float32)

    if X_test.ndim == 4 and X_test.shape[2] == 1 and X_test.shape[3] == 1:
        if M_test.shape != X_test.shape:
            raise ValueError(
                f"Test mask shape {M_test.shape} does not match X shape {X_test.shape}"
            )
        X_test = X_test[:, :, 0, 0]
        M_test = M_test[:, :, 0, 0]
        print("[TEST] Squeezed (N,C,1,1) -> (N,C)")

    if X_test.ndim != 2:
        raise ValueError(
            f"Test X must have shape (N, C) or (N, C, 1, 1); got {X_test.shape}"
        )
    if M_test.shape != X_test.shape:
        raise ValueError(
            f"Test mask shape {M_test.shape} does not match X shape {X_test.shape}"
        )
    if X_test.shape[0] != y_test.shape[0]:
        raise ValueError("Mismatch between test X and y lengths")
    if X_test.shape[1] != X_cv.shape[1]:
        raise ValueError("CV and test data have different numbers of covariates")

    print("[TEST] X shape:", X_test.shape)
    print("[TEST] M shape:", M_test.shape)
    print("[TEST] y shape:", y_test.shape)

    bad = ~np.isfinite(X_test)
    if bad.any():
        X_test[bad] = 0.0
    bad_m = ~np.isfinite(M_test)
    if bad_m.any():
        M_test[bad_m] = 0.0

    # ----------- CV training -----------
    fold_ids = make_cv_indices(fold_cv)
    print("Folds:", fold_ids)

    oof_score01 = np.full_like(y_cv, np.nan, dtype=float)
    fold_aucs, fold_losses, fold_boyces = [], [], []

    for fid in fold_ids:
        best_val, val_score01, val_idx = train_and_save_fold_model(
            fid, X_cv, M_cv, y_cv, fold_cv, model_dir,
        )
        fold_aucs.append(best_val["ROC_AUC"])
        fold_losses.append(best_val["loss"])
        fold_boyces.append(best_val["Boyce"])
        oof_score01[val_idx] = val_score01

    # ----------- CV OOF metrics -----------
    cv_metrics = compute_boyce_and_auc(y_cv, oof_score01)
    print("\n[CV] OOF metrics:", cv_metrics)
    print("[CV] per-fold AUCs:", fold_aucs)
    print("[CV] per-fold losses:", fold_losses)
    print("[CV] per-fold Boyce:", fold_boyces)

    # ----------- softmax(AUC) weights -----------
    weights = softmax_auc_weights(fold_aucs, tau=AUC_SOFTMAX_TAU)
    print(f"\n[Ensemble] softmax(AUC) weights (tau={AUC_SOFTMAX_TAU}):", weights)

    np.save(os.path.join(model_dir, "fold_ids.npy"), np.array(fold_ids, dtype=int))
    np.save(os.path.join(model_dir, "fold_aucs.npy"), np.asarray(fold_aucs, dtype=float))
    np.save(os.path.join(model_dir, "fold_weights_softmax_auc.npy"), weights)
    np.save(os.path.join(model_dir, "oof_score01.npy"), oof_score01)
    print("[Saved] fold_ids.npy, fold_aucs.npy, fold_weights_softmax_auc.npy, oof_score01.npy")

    # ----------- External test -----------
    test_idx = np.arange(len(y_test))
    print("\nExternal test N =", len(test_idx))

    test_score01, test_logratio, test_y = ensemble_predict_on_indices(
        X_values=X_test, X_masks=M_test, y=y_test,
        indices=test_idx, model_dir=model_dir,
        fold_ids=fold_ids, weights=weights,
    )

    test_metrics = compute_boyce_and_auc(test_y, test_score01)
    print("\n[Test] Ensemble metrics:", test_metrics)

    np.save(os.path.join(model_dir, "test_score01.npy"), test_score01)
    np.save(os.path.join(model_dir, "test_logratio_ens.npy"), test_logratio)
    with (args.output_dir / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["split", "Boyce", "ROC_AUC"])
        writer.writeheader()
        writer.writerow({"split": "cv_oof", **cv_metrics})
        writer.writerow({"split": "test", **test_metrics})
    print("[Test] Saved test_score01.npy, test_logratio_ens.npy")


if __name__ == "__main__":
    main()
