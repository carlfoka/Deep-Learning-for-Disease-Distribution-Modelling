"""Train and evaluate the single-scale 3x3 CNN-DRE model."""

import argparse
import os
from pathlib import Path

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

DEFAULT_OUTPUT_DIR = Path("outputs/scale_based_models/scale_3")

BATCH_SIZE = 256
NUM_WORKERS = 0

MAX_EPOCHS = 100
PATIENCE = 10

LR = 1e-3
WEIGHT_DECAY = 1e-3

CLIP_LOGITS = 10.0
SEED = 42

EMB_DIM = 32
HIDDEN_DIMS = [32]     # 3 hidden layers
DROPOUT = 0.35 

PATCH_SIZE = 3  # 3x3 patches

# Data augmentation
USE_FLIPS = True
NOISE_STD = 0.04

# Selection rule: among epochs passing AUC floor, pick max Boyce in (loss <= loss_min + delta)
USE_AUC_FLOOR = True
AUC_FLOOR = 0.80
LOSS_WINDOW_DELTA = 0.03

# Numeric stability
MAX_ABS_LOG_RATIO = 50.0

# Ensemble choice
ENSEMBLE_IN_LOGSPACE = True


# ---- NEW: ranking loss hyperparams ----
LAMBDA_RANK = 0.05          # weight of ranking term relative to BCE
MAX_RANK_PAIRS = 4096      # cap number of pos-neg pairs per batch for speed
RANK_TEMPERATURE = 1.0     # 1.0 = standard pairwise logistic; >1 softens gradients

# ---- NEW: softmax(AUC) weighting hyperparam ----
AUC_SOFTMAX_TAU = 50.0     # higher => more peaked weights (try 20..100)


# =========================================================
# Device
# =========================================================
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# Metrics: Boyce + AUC (expects any real-valued score)
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
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.size < 3 or y.size < 3:
        return np.nan
    rx, ry = _average_ranks(x), _average_ranks(y)
    if np.all(rx == rx[0]) or np.all(ry == ry[0]):
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def continuous_boyce(y_true, scores, nbins_max=20, min_per_group=10):
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)

    s_bg = s[y == 0]
    s_pr = s[y == 1]
    if (s_bg.size < min_per_group) or (s_pr.size < min_per_group):
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
        in_pr = (s_pr >= a) & (s_pr < b)
        npk = int(in_pr.sum())
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
# Helpers: stable bounded score from logits via log-ratio
# =========================================================
def sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = np.clip(x, -MAX_ABS_LOG_RATIO, MAX_ABS_LOG_RATIO)
    return 1.0 / (1.0 + np.exp(-x))


def logits_to_logratio(logits_np: np.ndarray, pi0: float, pi1: float) -> np.ndarray:
    eps = 1e-12
    pi0 = float(np.clip(pi0, eps, 1.0 - eps))
    pi1 = float(np.clip(pi1, eps, 1.0 - eps))
    return logits_np + np.log(pi0 / pi1)


def logits_to_bounded_score(logits_np: np.ndarray, pi0: float, pi1: float) -> np.ndarray:
    log_ratio = logits_to_logratio(logits_np, pi0, pi1)
    return sigmoid_np(log_ratio)


def compute_epoch_metrics_from_logits(logits_np, y_np, loss_val, pi0, pi1):
    score01 = logits_to_bounded_score(logits_np, pi0, pi1)
    mets = compute_boyce_and_auc(y_np, score01)
    mets["loss"] = float(loss_val)
    return mets


def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_cv_indices(fold_cv: np.ndarray) -> List[int]:
    vals = np.unique(fold_cv)
    return sorted(int(v) for v in vals)


# =========================================================
# Dataset + augmentation
# =========================================================
def augment_patch(values: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # values/mask: (C,H,W)
    if USE_FLIPS:
        if torch.rand(1).item() < 0.5:
            values = torch.flip(values, dims=[1])
            mask = torch.flip(mask, dims=[1])
        if torch.rand(1).item() < 0.5:
            values = torch.flip(values, dims=[2])
            mask = torch.flip(mask, dims=[2])

    # noise ONLY on valid pixels
    if NOISE_STD > 0.0:
        noise = torch.randn_like(values) * NOISE_STD
        m = (mask > 0).to(dtype=values.dtype)
        values = values + noise * m

    return values, mask


class PatchDREDataset(Dataset):
    def __init__(self, X_values, X_masks, y, indices, train: bool):
        super().__init__()
        self.Xv = X_values.astype(np.float32)
        self.Xm = X_masks.astype(np.float32)
        self.y = y.astype(np.float32)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.train = train

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = self.indices[i]
        xv = torch.from_numpy(self.Xv[idx])
        xm = torch.from_numpy(self.Xm[idx])
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        if self.train:
            xv, xm = augment_patch(xv, xm)
        return xv, xm, y


# =========================================================
# DRE head
# =========================================================
class MLPDRE(nn.Module):
    def __init__(self, in_dim: int, hidden_dims=(256, 128, 64), dropout=0.2, clip_logits=None):
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




# =========================================================
# NEW: ranking component for classifier-based DRE
# =========================================================
def pairwise_ranking_loss(logits: torch.Tensor, y: torch.Tensor,
                          max_pairs: int = 4096, temperature: float = 1.0) -> torch.Tensor:
    """
    Pairwise logistic ranking loss: encourages logits(pos) > logits(neg).
      loss = mean log(1 + exp(-(pos-neg)/T))
    Computed within a batch; subsamples pairs if too many.
    """
    y = (y > 0.5)
    pos = logits[y]
    neg = logits[~y]
    if pos.numel() == 0 or neg.numel() == 0:
        return logits.new_tensor(0.0)

    # all pairwise diffs
    # diff shape: (n_pos, n_neg)
    diff = (pos[:, None] - neg[None, :]) / float(max(1e-8, temperature))

    n_pairs = diff.numel()
    if n_pairs > max_pairs:
        # random subsample of pairs
        idx = torch.randint(0, n_pairs, (max_pairs,), device=logits.device)
        diff = diff.reshape(-1)[idx]

    # softplus(-diff) = log(1+exp(-diff))
    return F.softplus(-diff).mean()


# =========================================================
# Hybrid partial-conv encoder blocks
# =========================================================
class ChannelwisePartialConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, eps=1e-8):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.eps = eps
        kH, kW = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.kernel_area = float(kH * kW)

        w = torch.ones(in_channels, 1, kH, kW)
        self.register_buffer("mask_weight", w)

        self.stride = stride
        self.padding = padding

    def forward(self, x, m):
        if m is None:
            m = torch.ones_like(x)
        if x.dim() != 4 or m.dim() != 4:
            raise ValueError(f"Expected x,m 4D. Got x={x.shape}, m={m.shape}")
        if m.shape != x.shape:
            raise ValueError(f"ChannelwisePartialConv2d expects mask same shape as x. Got x={x.shape}, m={m.shape}")

        m = (m > 0).to(dtype=x.dtype)

        m_sum = F.conv2d(m, self.mask_weight, stride=self.stride, padding=self.padding, groups=x.size(1))  # (B,C,H,W)
        scale = self.kernel_area / (m_sum + self.eps)

        x_norm = x * m * scale
        x_norm = x_norm * (m_sum > 0).to(dtype=x.dtype)

        y = self.conv(x_norm)

        with torch.no_grad():
            m_spatial = (m_sum.sum(dim=1, keepdim=True) > 0).to(dtype=x.dtype)  # (B,1,H,W)

        return y, m_spatial


class SpatialPartialConv2d(nn.Module):
    """
    Partial conv in feature space using a spatial mask (B,1,H,W).
    Supports dilation.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 dilation=1, bias=False, eps=1e-8):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=bias)
        self.eps = eps
        kH, kW = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.kernel_area = float(kH * kW)
        self.register_buffer("mask_kernel", torch.ones(1, 1, kH, kW))
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x, m):
        if m is None:
            m = torch.ones(x.size(0), 1, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
        if m.dim() != 4 or m.size(1) != 1:
            raise ValueError(f"Spatial mask must be (B,1,H,W). Got {m.shape}")

        m = (m > 0).to(dtype=x.dtype)
        y = self.conv(x * m)

        with torch.no_grad():
            m_sum = F.conv2d(m, self.mask_kernel, stride=self.stride, padding=self.padding, dilation=self.dilation)
            m_out = (m_sum > 0).to(dtype=x.dtype)

        scale = self.kernel_area / (m_sum + self.eps)
        y = y * scale
        y = y * m_out
        return y, m_out


class CPConvBlock1(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.p = ChannelwisePartialConv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, m_cov):
        x, m = self.p(x, m_cov)  # m becomes spatial
        x = self.bn(x)
        x = self.act(x)
        return x, m


class CPConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dilation=1):
        super().__init__()
        pad = dilation  # for k=3, padding=dilation keeps spatial size
        self.p = SpatialPartialConv2d(in_ch, out_ch, 3, padding=pad, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, m):
        x, m = self.p(x, m)
        x = self.bn(x)
        x = self.act(x)
        return x, m


class MaskedMaxPool2d(nn.Module):
    def __init__(self, kernel_size=2, stride=2):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size, stride)

    def forward(self, x, m):
        x = self.pool(x)
        m = self.pool(m)
        m = (m > 0).to(dtype=x.dtype)
        return x, m


class MaskedGlobalAvgPool(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x, m):
        if m is None:
            m = torch.ones(x.size(0), 1, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
        if m.dim() != 4 or m.size(1) != 1:
            raise ValueError(f"MaskedGlobalAvgPool expects m=(B,1,H,W). Got {m.shape}")
        m = (m > 0).to(dtype=x.dtype)
        num = (x * m).sum(dim=(2, 3))                # (B,C)
        den = m.sum(dim=(2, 3)).clamp_min(self.eps)  # (B,1)
        return num / den


# =========================================================
# 3x3 ENCODER
# Exact counterpart of the 3x3 branch in the multi-scale model.
# =========================================================
class PatchEncoder3Hybrid(nn.Module):
    """
    Branch for a 3x3 patch.
    No pooling: preserve local detail.
    """
    def __init__(self, in_value_channels: int, emb_dim: int = 32):
        super().__init__()
        self.b1 = CPConvBlock1(in_value_channels, 16)
        self.b2 = CPConvBlock(16, 16, dilation=1)
        self.b3 = CPConvBlock(16, 32, dilation=1)

        self.gap = MaskedGlobalAvgPool()
        self.proj = nn.Sequential(
            nn.Linear(32, emb_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x_val, x_mask):
        m_cov = (x_mask > 0).to(dtype=x_val.dtype)
        x, m = self.b1(x_val, m_cov)
        x, m = self.b2(x, m)
        x, m = self.b3(x, m)
        h = self.gap(x, m)
        z = self.proj(h)
        return z


class CNNDRE(nn.Module):
    """
    Classifier-based DRE: patch -> encoder z -> MLP -> logit
    """
    def __init__(self, in_value_channels, emb_dim=32, hidden_dims=(32,), dropout=0.35, clip_logits=10.0):
        super().__init__()
        self.encoder = PatchEncoder3Hybrid(in_value_channels, emb_dim)
        self.dre_head = MLPDRE(in_dim=emb_dim, hidden_dims=hidden_dims, dropout=dropout, clip_logits=clip_logits)

    def forward(self, x_val, x_mask):
        z = self.encoder(x_val, x_mask)
        logits = self.dre_head(z)
        return logits


# =========================================================
# Training per fold
# =========================================================
def train_and_save_fold_model_cnn(
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
    val_idx = np.where(f == fold_id)[0]

    # fold-specific priors from TRAIN ONLY
    pi1_fold = float((y_cv[train_idx] == 1).mean())
    pi0_fold = 1.0 - pi1_fold
    print(f"\n[fold {fold_id}] train N={len(train_idx)}, val N={len(val_idx)} | priors pi1={pi1_fold:.6f}, pi0={pi0_fold:.6f}")

    ds_tr = PatchDREDataset(X_values, X_masks, y_cv, train_idx, train=True)
    ds_val = PatchDREDataset(X_values, X_masks, y_cv, val_idx, train=False)

    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True,
                       num_workers=NUM_WORKERS, drop_last=False)
    dl_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, drop_last=False)

    in_value_channels = X_values.shape[1]
    model = CNNDRE(
        in_value_channels=in_value_channels,
        emb_dim=EMB_DIM,
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

                # ---- NEW: ranking component ----
                loss_rank = pairwise_ranking_loss(
                    logits, yb,
                    max_pairs=MAX_RANK_PAIRS,
                    temperature=RANK_TEMPERATURE
                )

                loss = loss_bce + float(LAMBDA_RANK) * loss_rank

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

    # Selection: among epochs passing AUC floor, pick max Boyce in (loss <= loss_min + delta)
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
                "model_type": "CNNDRE",
                "encoder_type": "PatchEncoder3Hybrid",
                "in_value_channels": in_value_channels,
                "patch_size": PATCH_SIZE,
                "emb_dim": EMB_DIM,
                "hidden_dims": HIDDEN_DIMS,
                "pi0": float(pi0_fold),
                "pi1": float(pi1_fold),
            }
            candidates.append(
                {
                    "loss": float(va["loss"]),
                    "boyce": float(va["Boyce"]) if np.isfinite(va["Boyce"]) else np.nan,
                    "auc": float(va["ROC_AUC"]) if np.isfinite(va["ROC_AUC"]) else np.nan,
                    "state": state,
                    "val_logits": val_logits,
                    "val_y": val_y,
                    "epoch": ep,
                }
            )

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
            f"[fold {fold_id}] No epoch met AUC floor (AUC_FLOOR={AUC_FLOOR}). "
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

    ckpt_path = os.path.join(model_dir, f"cnn_dre_fold_{fold_id}.pt")
    torch.save(best["state"], ckpt_path)
    print(
        f"[fold {fold_id}] selected epoch={best['epoch']} | "
        f"loss={best['loss']:.4f}, Boyce={best['boyce']:.4f}, AUC={best['auc']:.4f} | "
        f"saved -> {ckpt_path}"
    )

    # Return OOF bounded scores for this fold's val subset
    val_logits = best["val_logits"]
    val_y = best["val_y"]
    val_score01 = logits_to_bounded_score(val_logits, best["state"]["pi0"], best["state"]["pi1"])
    best_val = {"Boyce": best["boyce"], "ROC_AUC": best["auc"], "loss": best["loss"]}
    return best_val, val_score01, val_idx


# =========================================================
# Ensemble prediction (bounded score)
# =========================================================
def ensemble_predict_on_indices_cnn(
    X_values: np.ndarray,
    X_masks: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
    model_dir: str,
    fold_ids: List[int],
    weights: np.ndarray,
):
    in_value_channels = X_values.shape[1]

    models = []
    per_model_logpi = []
    for fid in fold_ids:
        ckpt_path = os.path.join(model_dir, f"cnn_dre_fold_{fid}.pt")
        state = torch.load(ckpt_path, map_location="cpu")

        if state["in_value_channels"] != in_value_channels:
            raise RuntimeError(f"Value channel mismatch for fold {fid}")
        if state.get("patch_size", PATCH_SIZE) != PATCH_SIZE:
            raise RuntimeError(f"Patch size mismatch for fold {fid}")
        if state.get("encoder_type") != "PatchEncoder3Hybrid":
            raise RuntimeError(
                f"Encoder mismatch for fold {fid}: expected PatchEncoder3Hybrid, "
                f"got {state.get('encoder_type', 'missing metadata')}"
            )
        if ("pi0" not in state) or ("pi1" not in state):
            raise RuntimeError(f"Missing fold priors in checkpoint for fold {fid} (pi0/pi1).")

        model = CNNDRE(
            in_value_channels=in_value_channels,
            emb_dim=state.get("emb_dim", EMB_DIM),
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

    # normalize weights
    w = np.asarray(weights, dtype=np.float64)
    if (not np.isfinite(w).all()) or w.sum() <= 0:
        w = np.ones(len(fold_ids), dtype=np.float64) / len(fold_ids)
    else:
        w = w / w.sum()

    weights_t = torch.tensor(w, dtype=torch.float32, device=device)
    logpi_t = torch.tensor(np.asarray(per_model_logpi, dtype=np.float32), dtype=torch.float32, device=device)

    idx = np.asarray(indices, dtype=np.int64)

    all_score01, all_logratio, all_y = [], [], []

    with torch.no_grad():
        for start in range(0, len(idx), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(idx))
            idx_batch = idx[start:end]

            xv = torch.from_numpy(X_values[idx_batch].astype(np.float32)).to(device)
            xm = torch.from_numpy(X_masks[idx_batch].astype(np.float32)).to(device)
            yb = torch.from_numpy(y[idx_batch].astype(np.float32)).to(device)

            # per-model log-ratios
            logratios = []
            for k, model in enumerate(models):
                logits_k = model(xv, xm)
                log_ratio_k = logits_k + logpi_t[k]
                log_ratio_k = torch.clamp(log_ratio_k, -MAX_ABS_LOG_RATIO, MAX_ABS_LOG_RATIO)
                logratios.append(log_ratio_k)

            # ensemble
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
            all_y.append(yb.cpu().numpy())

    return (
        np.concatenate(all_score01),
        np.concatenate(all_logratio),
        np.concatenate(all_y),
    )


# =========================================================
# NEW: Softmax(AUC) weights
# =========================================================
def softmax_auc_weights(fold_aucs: List[float], tau: float = 50.0) -> np.ndarray:
    """
    w_k ∝ exp(tau * (auc_k - mean_auc)), normalized.
    Handles NaNs by giving them very low weight.
    """
    a = np.asarray(fold_aucs, dtype=np.float64)
    if a.size == 0:
        return np.array([], dtype=np.float64)

    m = np.isfinite(a)
    if m.sum() == 0:
        return np.ones_like(a) / len(a)

    mean_auc = float(np.nanmean(a))
    x = np.where(m, a - mean_auc, -1e9)  # NaNs -> huge negative
    x = tau * x

    # stable softmax
    x_max = np.max(x[m])  # only finite matter
    ex = np.exp(np.clip(x - x_max, -200, 200))
    ex[~m] = 0.0
    s = ex.sum()
    if s <= 0 or (not np.isfinite(s)):
        return np.ones_like(a) / len(a)
    return ex / s


# =========================================================
# MAIN
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Train the single-scale 3x3 CNN-DRE model.")
    parser.add_argument(
        "--cv-dir", type=Path, required=True,
        help="Directory containing CV X.npy, M.npy, y.npy, and fold.npy.",
    )
    parser.add_argument(
        "--test-dir", type=Path, required=True,
        help="Directory containing test X.npy, M.npy, and y.npy.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"Model and prediction directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    return parser.parse_args()


def require_data_files(directory: Path, filenames, label: str):
    directory = directory.expanduser()
    missing = [name for name in filenames if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{label} directory '{directory}' is missing: {', '.join(missing)}"
        )
    return directory


if __name__ == "__main__":
    args = parse_args()
    cv_dir = require_data_files(
        args.cv_dir, ("X.npy", "M.npy", "y.npy", "fold.npy"), "CV"
    )
    test_dir = require_data_files(
        args.test_dir, ("X.npy", "M.npy", "y.npy"), "test"
    )
    model_dir = args.output_dir.expanduser()

    set_seed(SEED)
    model_dir.mkdir(parents=True, exist_ok=True)
    print("Using device:", device)

    # ----------- load CV data -----------
    X_cv = np.load(cv_dir / "X.npy")
    M_cv = np.load(cv_dir / "M.npy")
    y_cv = np.load(cv_dir / "y.npy").astype(np.float32)
    fold_cv_raw = np.load(cv_dir / "fold.npy")

    print("[CV] X shape:", X_cv.shape)
    print("[CV] M shape:", M_cv.shape)
    print("[CV] y shape:", y_cv.shape)
    print("[CV] fold shape:", fold_cv_raw.shape)

    if X_cv.shape[0] != y_cv.shape[0] or M_cv.shape[0] != y_cv.shape[0]:
        raise ValueError("Mismatch between CV X/M and y lengths")
    if X_cv.shape[-2:] != (PATCH_SIZE, PATCH_SIZE):
        raise ValueError(
            f"Expected stored CV patches of size {PATCH_SIZE}x{PATCH_SIZE}, "
            f"got {tuple(X_cv.shape[-2:])}"
        )
    if M_cv.shape != X_cv.shape:
        raise ValueError(f"CV mask shape {M_cv.shape} must match value shape {X_cv.shape}")

    if not np.isfinite(fold_cv_raw).all():
        raise ValueError("fold.npy contains NaN/inf; CV split must be defined for all rows")
    fold_cv = fold_cv_raw.astype(int)

    bad = ~np.isfinite(X_cv)
    if bad.any():
        print(f"[CV] Warning: {int(bad.sum())} non-finite X values set to 0.")
        X_cv[bad] = 0.0
    bad_m = ~np.isfinite(M_cv)
    if bad_m.any():
        print(f"[CV] Warning: {int(bad_m.sum())} non-finite M values set to 0.")
        M_cv[bad_m] = 0.0

    # ----------- load TEST data -----------
    X_test = np.load(test_dir / "X.npy")
    M_test = np.load(test_dir / "M.npy")
    y_test = np.load(test_dir / "y.npy").astype(np.float32)

    print("[TEST] X shape:", X_test.shape)
    print("[TEST] M shape:", M_test.shape)
    print("[TEST] y shape:", y_test.shape)

    if X_test.shape[0] != y_test.shape[0] or M_test.shape[0] != y_test.shape[0]:
        raise ValueError("Mismatch between TEST X/M and y lengths")
    if X_test.shape[-2:] != (PATCH_SIZE, PATCH_SIZE):
        raise ValueError(
            f"Expected stored TEST patches of size {PATCH_SIZE}x{PATCH_SIZE}, "
            f"got {tuple(X_test.shape[-2:])}"
        )
    if M_test.shape != X_test.shape:
        raise ValueError(f"TEST mask shape {M_test.shape} must match value shape {X_test.shape}")

    bad = ~np.isfinite(X_test)
    if bad.any():
        print(f"[TEST] Warning: {int(bad.sum())} non-finite X values set to 0.")
        X_test[bad] = 0.0
    bad_m = ~np.isfinite(M_test)
    if bad_m.any():
        print(f"[TEST] Warning: {int(bad_m.sum())} non-finite M values set to 0.")
        M_test[bad_m] = 0.0

    # ----------- CV training over folds -----------
    fold_ids = make_cv_indices(fold_cv)
    print("Folds:", fold_ids)

    oof_score01 = np.full_like(y_cv, np.nan, dtype=float)

    fold_aucs = []
    fold_losses = []
    fold_boyces = []

    for fid in fold_ids:
        best_val, val_score01, val_idx = train_and_save_fold_model_cnn(
            fid, X_cv, M_cv, y_cv, fold_cv, model_dir
        )
        fold_aucs.append(best_val["ROC_AUC"])
        fold_losses.append(best_val["loss"])
        fold_boyces.append(best_val["Boyce"])
        oof_score01[val_idx] = val_score01

    # ----------- CV metrics (OOF bounded score) -----------
    cv_metrics = compute_boyce_and_auc(y_cv, oof_score01)
    print("\n[CV] OOF metrics (bounded score):", cv_metrics)
    print("[CV] per-fold chosen AUCs:", fold_aucs)
    print("[CV] per-fold chosen losses:", fold_losses)
    print("[CV] per-fold chosen Boyce:", fold_boyces)

    # ----------- NEW: softmax(AUC) weights -----------
    weights = softmax_auc_weights(fold_aucs, tau=AUC_SOFTMAX_TAU)
    print(f"\n[Ensemble] softmax(AUC) weights (tau={AUC_SOFTMAX_TAU}):", weights)

    np.save(model_dir / "fold_ids.npy", np.array(fold_ids, dtype=int))
    np.save(model_dir / "fold_aucs.npy", np.asarray(fold_aucs, dtype=float))
    np.save(model_dir / "fold_weights_softmax_auc.npy", weights)
    np.save(model_dir / "oof_score01.npy", oof_score01)
    print("[Saved] fold_ids.npy, fold_aucs.npy, fold_weights_softmax_auc.npy, oof_score01.npy")

    # ----------- External test ensemble (bounded score) -----------
    test_idx = np.arange(len(y_test))
    print("External test N =", len(test_idx))

    test_score01, test_logratio, test_y = ensemble_predict_on_indices_cnn(
        X_values=X_test,
        X_masks=M_test,
        y=y_test,
        indices=test_idx,
        model_dir=model_dir,
        fold_ids=fold_ids,
        weights=weights,
    )

    test_metrics = compute_boyce_and_auc(test_y, test_score01)
    print("\n[Test] Ensemble metrics (bounded score):", test_metrics)

    np.save(model_dir / "test_score01.npy", test_score01)
    np.save(model_dir / "test_logratio_ens.npy", test_logratio)
    print("[Test] Saved test_score01.npy, test_logratio_ens.npy")
