#!/usr/bin/env python3
"""
Analyze Stage1 latent geometry against watermark recoverability.

This tool is designed for the current ReMark research question:
  1. Do forge latents cluster by watermark accuracy?
  2. Do forge->original latent displacements share a common repair structure?
  3. Is watermark accuracy monotonic along pair-wise latent paths?
  4. Is watermark recoverability predictable from latent codes with a small probe?

Outputs:
  - summary.json
  - samples.csv
  - bucket_stats.csv
  - bucket_between.csv
  - path_sample_summary.csv
  - path_curves.csv
  - probe_predictions.csv

Usage:
  python tools/analyze_latent_geometry.py \
    --run-dir runs/stage1_20260311_145328 \
    --checkpoint best \
    --split val \
    --attack auto \
    --max-batches 8
"""

import argparse
import json
import math
import os
import random
import sys
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.dataset import ReMark_Dataset
from network.vae import build_vae
from wm_adapters.registry import build_wm_adapter
from attacks.registry import build_attack, ATTACK_REGISTRY

# Ensure registries are populated when the script is run directly.
import wm_adapters  # noqa: F401
import attacks  # noqa: F401


EPS = 1e-8


def dict_to_ns(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: dict_to_ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [dict_to_ns(v) for v in obj]
    return obj


def load_run_config(run_dir: str):
    import yaml

    cfg_path = os.path.join(run_dir, "config.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return dict_to_ns(cfg)


def resolve_ckpt_path(run_dir: str, checkpoint: str) -> str:
    if os.path.isabs(checkpoint) and os.path.isfile(checkpoint):
        return checkpoint
    if os.path.isfile(checkpoint):
        return os.path.abspath(checkpoint)
    if checkpoint in ("best", "last"):
        path = os.path.join(run_dir, "checkpoints", "vae", f"{checkpoint}.pth")
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"checkpoint not found: {checkpoint}")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_loader(cfg, split: str, batch_size: int, num_workers: int):
    csv_path = cfg.data.val_csv if split == "val" else cfg.data.train_csv
    dataset = ReMark_Dataset(
        csv_path=csv_path,
        image_size=cfg.data.image_size,
        mode="val",
        use_wm_cache=False,
        center_crop=getattr(cfg.data, "center_crop", 0),
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def apply_attack(attack_name: str, attack_obj, batch: dict,
                 images: torch.Tensor, wm_images: torch.Tensor,
                 device: torch.device) -> torch.Tensor:
    if attack_name == "offline_fake":
        if "fake_image" not in batch:
            raise RuntimeError("attack=offline_fake but batch has no fake_image.")
        return batch["fake_image"].to(device, non_blocking=True)

    if hasattr(attack_obj, "attack_with_cover"):
        try:
            return attack_obj.attack_with_cover(wm_images, images, batch=batch)
        except TypeError:
            return attack_obj.attack_with_cover(wm_images, images)
    return attack_obj(wm_images)


def load_stage1_vae(cfg, ckpt_path: str, device: torch.device):
    vae = build_vae(cfg).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(state, dict):
        sd = state.get("net", state.get("vae", state))
    else:
        sd = state
    vae.load_state_dict(sd, strict=True)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae


def decode_latent(vae, z: torch.Tensor, reference_image: Optional[torch.Tensor] = None) -> torch.Tensor:
    x_hat = vae.decode(z)
    if getattr(vae, "residual_output", False):
        if reference_image is None:
            raise RuntimeError("VAE uses residual_output=True but reference_image is missing.")
        x_hat = torch.clamp(
            reference_image + float(getattr(vae, "residual_scale", 1.0)) * x_hat,
            -1.0,
            1.0,
        )
    return x_hat


def sample_bit_accuracy(logits: torch.Tensor, messages: torch.Tensor) -> np.ndarray:
    pred = (logits > 0).float()
    eq = pred.eq(messages).float().mean(dim=1)
    return eq.detach().cpu().numpy().astype(np.float32)


def pairwise_euclidean(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    sq = np.sum(x * x, axis=1, keepdims=True)
    dist2 = sq + sq.T - 2.0 * (x @ x.T)
    np.maximum(dist2, 0.0, out=dist2)
    return np.sqrt(dist2, out=dist2)


def pairwise_cosine(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    x_norm = x / np.clip(norms, EPS, None)
    sim = x_norm @ x_norm.T
    np.clip(sim, -1.0, 1.0, out=sim)
    return (1.0 - sim).astype(np.float32)


def pearsonr_np(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2:
        return float("nan")
    if np.std(x) < EPS or np.std(y) < EPS:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearmanr_np(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(pd.Series(x).rank(method="average"), dtype=np.float64)
    y = np.asarray(pd.Series(y).rank(method="average"), dtype=np.float64)
    return pearsonr_np(x, y)


def r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.size == 0:
        return float("nan")
    denom = np.sum((y_true - y_true.mean()) ** 2)
    if denom < EPS:
        return float("nan")
    num = np.sum((y_true - y_pred) ** 2)
    return float(1.0 - num / denom)


def mae_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def bucketize(values: np.ndarray, bucket_size: float) -> np.ndarray:
    n_buckets = int(math.ceil(1.0 / bucket_size))
    labels = np.floor(np.asarray(values) / bucket_size).astype(np.int64)
    return np.clip(labels, 0, n_buckets - 1)


def bucket_label(bucket_idx: int, bucket_size: float) -> str:
    lo = bucket_idx * bucket_size
    hi = min((bucket_idx + 1) * bucket_size, 1.0)
    right = "]" if hi >= 1.0 - 1e-12 else ")"
    return f"[{lo:.1f},{hi:.1f}{right}"


def masked_upper_mean(mat: np.ndarray, indices: np.ndarray) -> float:
    if indices.size < 2:
        return float("nan")
    sub = mat[np.ix_(indices, indices)]
    mask = np.triu(np.ones_like(sub, dtype=bool), k=1)
    vals = sub[mask]
    if vals.size == 0:
        return float("nan")
    return float(vals.mean())


def silhouette_from_distance(dist: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels)
    uniq, counts = np.unique(labels, return_counts=True)
    valid_labels = uniq[counts >= 2]
    if valid_labels.size < 2:
        return float("nan")
    keep = np.isin(labels, valid_labels)
    labels = labels[keep]
    dist = dist[np.ix_(keep, keep)]
    n = labels.shape[0]
    sil = np.zeros(n, dtype=np.float64)
    for i in range(n):
        same = labels == labels[i]
        same[i] = False
        a = dist[i, same].mean() if np.any(same) else 0.0
        b = float("inf")
        for lab in valid_labels:
            if lab == labels[i]:
                continue
            other = labels == lab
            if np.any(other):
                b = min(b, float(dist[i, other].mean()))
        if not np.isfinite(b):
            sil[i] = 0.0
        else:
            sil[i] = 0.0 if max(a, b) < EPS else (b - a) / max(a, b)
    return float(np.mean(sil))


def davies_bouldin_score_np(x: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels)
    uniq, counts = np.unique(labels, return_counts=True)
    valid_labels = uniq[counts >= 2]
    if valid_labels.size < 2:
        return float("nan")
    x = np.asarray(x, dtype=np.float64)
    centroids = []
    scatters = []
    for lab in valid_labels:
        pts = x[labels == lab]
        c = pts.mean(axis=0)
        centroids.append(c)
        scatters.append(np.linalg.norm(pts - c, axis=1).mean())
    centroids = np.stack(centroids, axis=0)
    scatters = np.asarray(scatters, dtype=np.float64)
    cdist = pairwise_euclidean(centroids.astype(np.float32)).astype(np.float64)
    db_terms = []
    for i in range(valid_labels.size):
        ratios = []
        for j in range(valid_labels.size):
            if i == j:
                continue
            ratios.append((scatters[i] + scatters[j]) / max(cdist[i, j], EPS))
        db_terms.append(max(ratios))
    return float(np.mean(db_terms))


def knn_regression_from_distance(dist: np.ndarray, targets: np.ndarray, k: int) -> np.ndarray:
    n = targets.shape[0]
    k = max(1, min(k, n - 1))
    preds = np.zeros(n, dtype=np.float32)
    for i in range(n):
        order = np.argpartition(dist[i], kth=min(k, n - 1))[:k + 1]
        order = order[order != i]
        if order.size > k:
            order = order[:k]
        preds[i] = float(np.mean(targets[order])) if order.size > 0 else float(targets[i])
    return preds


def fit_delta_pca(delta_flat: np.ndarray, n_components: int) -> Dict[str, np.ndarray]:
    delta = np.asarray(delta_flat, dtype=np.float32)
    mean_delta = delta.mean(axis=0, keepdims=True)
    centered = delta - mean_delta
    n_components = max(1, min(int(n_components), centered.shape[0], centered.shape[1]))
    u, s, vh = np.linalg.svd(centered, full_matrices=False)
    vh = vh[:n_components]
    explained = (s[:n_components] ** 2) / max(centered.shape[0] - 1, 1)
    total = (s ** 2).sum() / max(centered.shape[0] - 1, 1)
    ratio = explained / max(total, EPS)
    return {
        "mean_delta": mean_delta.astype(np.float32),
        "components": vh.astype(np.float32),
        "explained_variance_ratio": ratio.astype(np.float32),
    }


class ProbeRegressor(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 0):
        super().__init__()
        if hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.net = nn.Linear(in_dim, 1)

    def forward(self, x):
        return self.net(x).squeeze(1)


def train_probe_regressor(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    seed: int,
) -> np.ndarray:
    torch.manual_seed(seed)
    model = ProbeRegressor(x_train.shape[1], hidden_dim=hidden_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    x_train_t = torch.from_numpy(x_train).to(device=device, dtype=torch.float32)
    y_train_t = torch.from_numpy(y_train).to(device=device, dtype=torch.float32)
    x_test_t = torch.from_numpy(x_test).to(device=device, dtype=torch.float32)

    n = x_train_t.shape[0]
    bs = max(1, min(batch_size, n))
    for _ in range(max(epochs, 1)):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, bs):
            idx = perm[start:start + bs]
            pred = model(x_train_t[idx])
            loss = loss_fn(pred, y_train_t[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    with torch.no_grad():
        pred = model(x_test_t).detach().cpu().numpy().astype(np.float32)
    return np.clip(pred, 0.0, 1.0)


def analyze_clustering(
    z_f_flat: np.ndarray,
    forge_acc: np.ndarray,
    bucket_size: float,
    min_bucket_samples: int,
    knn_k: int,
) -> Tuple[Dict[str, object], pd.DataFrame, pd.DataFrame]:
    labels = bucketize(forge_acc, bucket_size)
    bucket_ids, counts = np.unique(labels, return_counts=True)
    label_names = {int(b): bucket_label(int(b), bucket_size) for b in bucket_ids}

    euclidean = pairwise_euclidean(z_f_flat)
    cosine = pairwise_cosine(z_f_flat)

    bucket_rows = []
    valid_labels = []
    for bucket_idx, count in zip(bucket_ids, counts):
        idx = np.where(labels == bucket_idx)[0]
        row = {
            "bucket_id": int(bucket_idx),
            "bucket_label": label_names[int(bucket_idx)],
            "count": int(count),
            "acc_mean": float(forge_acc[idx].mean()),
            "acc_std": float(forge_acc[idx].std(ddof=0)),
            "within_euclidean_mean": masked_upper_mean(euclidean, idx),
            "within_cosine_mean": masked_upper_mean(cosine, idx),
        }
        bucket_rows.append(row)
        if count >= min_bucket_samples:
            valid_labels.append(int(bucket_idx))
    bucket_df = pd.DataFrame(bucket_rows)

    between_rows = []
    for i, bucket_i in enumerate(bucket_ids):
        idx_i = np.where(labels == bucket_i)[0]
        for bucket_j in bucket_ids[i + 1:]:
            idx_j = np.where(labels == bucket_j)[0]
            between_rows.append(
                {
                    "bucket_i": int(bucket_i),
                    "bucket_i_label": label_names[int(bucket_i)],
                    "bucket_j": int(bucket_j),
                    "bucket_j_label": label_names[int(bucket_j)],
                    "between_euclidean_mean": float(euclidean[np.ix_(idx_i, idx_j)].mean()),
                    "between_cosine_mean": float(cosine[np.ix_(idx_i, idx_j)].mean()),
                }
            )
    between_df = pd.DataFrame(between_rows)

    valid_mask = np.isin(labels, valid_labels)
    valid_count = int(valid_mask.sum())
    if valid_count >= max(4, knn_k + 1):
        sil_e = silhouette_from_distance(euclidean[valid_mask][:, valid_mask], labels[valid_mask])
        sil_c = silhouette_from_distance(cosine[valid_mask][:, valid_mask], labels[valid_mask])
        db = davies_bouldin_score_np(z_f_flat[valid_mask], labels[valid_mask])
    else:
        sil_e = float("nan")
        sil_c = float("nan")
        db = float("nan")

    knn_e = knn_regression_from_distance(euclidean, forge_acc, knn_k)
    knn_c = knn_regression_from_distance(cosine, forge_acc, knn_k)

    summary = {
        "num_samples": int(z_f_flat.shape[0]),
        "bucket_size": float(bucket_size),
        "num_buckets_present": int(bucket_ids.size),
        "num_valid_buckets": int(len(valid_labels)),
        "num_valid_samples": int(valid_count),
        "valid_bucket_labels": [label_names[b] for b in valid_labels],
        "silhouette_euclidean": sil_e,
        "silhouette_cosine": sil_c,
        "davies_bouldin_euclidean": db,
        "knn_regression_euclidean": {
            "k": int(min(knn_k, max(z_f_flat.shape[0] - 1, 1))),
            "r2": r2_score_np(forge_acc, knn_e),
            "pearson": pearsonr_np(forge_acc, knn_e),
            "spearman": spearmanr_np(forge_acc, knn_e),
            "mae": mae_np(forge_acc, knn_e),
        },
        "knn_regression_cosine": {
            "k": int(min(knn_k, max(z_f_flat.shape[0] - 1, 1))),
            "r2": r2_score_np(forge_acc, knn_c),
            "pearson": pearsonr_np(forge_acc, knn_c),
            "spearman": spearmanr_np(forge_acc, knn_c),
            "mae": mae_np(forge_acc, knn_c),
        },
    }
    return summary, bucket_df, between_df


def analyze_displacements(
    z_f_flat: np.ndarray,
    z_o_flat: np.ndarray,
    samples_df: pd.DataFrame,
    z_shape: Sequence[int],
    attacked_images: torch.Tensor,
    wm_images: torch.Tensor,
    messages: torch.Tensor,
    vae,
    wm_adapter,
    device: torch.device,
    test_fraction: float,
    split_seed: int,
    ridge_lambda: float,
    analysis_batch_size: int,
    pca_components: int,
) -> Dict[str, object]:
    deltas = z_o_flat - z_f_flat
    delta_norm = np.linalg.norm(deltas, axis=1)

    delta_unit = deltas / np.clip(delta_norm[:, None], EPS, None)
    cosine_mat = np.clip(delta_unit @ delta_unit.T, -1.0, 1.0)
    cos_vals = cosine_mat[np.triu_indices(cosine_mat.shape[0], k=1)]

    pca = fit_delta_pca(deltas, n_components=pca_components)
    explained = pca["explained_variance_ratio"]

    n = z_f_flat.shape[0]
    rng = np.random.default_rng(split_seed)
    perm = rng.permutation(n)
    test_n = max(1, int(round(n * test_fraction)))
    test_idx = np.sort(perm[:test_n])
    train_idx = np.sort(perm[test_n:])
    if train_idx.size < 2:
        train_idx = np.sort(perm[:-1])
        test_idx = np.sort(perm[-1:])

    x_train = z_f_flat[train_idx].astype(np.float32)
    y_train = z_o_flat[train_idx].astype(np.float32)
    x_test = z_f_flat[test_idx].astype(np.float32)
    y_test = z_o_flat[test_idx].astype(np.float32)

    x_mean = x_train.mean(axis=0, keepdims=True)
    y_mean = y_train.mean(axis=0, keepdims=True)
    x_train_c = x_train - x_mean
    y_train_c = y_train - y_mean
    x_test_c = x_test - x_mean

    device_map = device
    x_train_t = torch.from_numpy(x_train_c).to(device_map)
    y_train_t = torch.from_numpy(y_train_c).to(device_map)
    x_test_t = torch.from_numpy(x_test_c).to(device_map)
    eye = torch.eye(x_train_t.shape[0], device=device_map, dtype=torch.float32)
    kernel = x_train_t @ x_train_t.T
    alpha = torch.linalg.solve(kernel + float(ridge_lambda) * eye, y_train_t)
    y_pred_linear = (x_test_t @ x_train_t.T @ alpha + torch.from_numpy(y_mean).to(device_map)).detach().cpu().numpy()

    mean_delta = (y_train - x_train).mean(axis=0, keepdims=True)
    y_pred_mean_delta = x_test + mean_delta

    def decode_acc_from_flat(z_flat: np.ndarray, references: torch.Tensor) -> np.ndarray:
        refs = references[test_idx]
        msgs = messages[test_idx]
        out = []
        for start in range(0, z_flat.shape[0], analysis_batch_size):
            end = min(start + analysis_batch_size, z_flat.shape[0])
            z = torch.from_numpy(z_flat[start:end]).to(device=device, dtype=torch.float32)
            z = z.view(-1, *z_shape)
            ref = refs[start:end].to(device=device, dtype=torch.float32)
            msg = msgs[start:end].to(device=device, dtype=torch.float32)
            x_hat = decode_latent(vae, z, ref)
            logits = wm_adapter.decode(x_hat)
            out.append(sample_bit_accuracy(logits, msg))
        return np.concatenate(out, axis=0)

    forge_acc = samples_df.loc[test_idx, "forge_recon_acc"].to_numpy(dtype=np.float32)
    orig_self = samples_df.loc[test_idx, "original_selfref_acc"].to_numpy(dtype=np.float32)
    orig_forge = samples_df.loc[test_idx, "original_forgeref_acc"].to_numpy(dtype=np.float32)
    linear_acc = decode_acc_from_flat(y_pred_linear.astype(np.float32), attacked_images)
    mean_delta_acc = decode_acc_from_flat(y_pred_mean_delta.astype(np.float32), attacked_images)

    linear_delta = y_pred_linear - x_test
    true_delta = y_test - x_test
    cos_dir = np.sum(linear_delta * true_delta, axis=1) / (
        np.linalg.norm(linear_delta, axis=1) * np.linalg.norm(true_delta, axis=1) + EPS
    )

    return {
        "delta_cosine_distribution": {
            "num_pairs": int(cos_vals.size),
            "mean": float(np.mean(cos_vals)) if cos_vals.size > 0 else float("nan"),
            "std": float(np.std(cos_vals)) if cos_vals.size > 0 else float("nan"),
            "median": float(np.median(cos_vals)) if cos_vals.size > 0 else float("nan"),
            "p05": float(np.quantile(cos_vals, 0.05)) if cos_vals.size > 0 else float("nan"),
            "p95": float(np.quantile(cos_vals, 0.95)) if cos_vals.size > 0 else float("nan"),
            "positive_fraction": float(np.mean(cos_vals > 0.0)) if cos_vals.size > 0 else float("nan"),
        },
        "delta_norm": {
            "mean": float(delta_norm.mean()),
            "std": float(delta_norm.std(ddof=0)),
            "median": float(np.median(delta_norm)),
        },
        "delta_pca": {
            "n_components_reported": int(min(pca_components, explained.shape[0])),
            "explained_variance_ratio": [float(v) for v in explained.tolist()],
            "cumulative_explained_variance_ratio": [float(v) for v in np.cumsum(explained).tolist()],
        },
        "linear_ridge_map": {
            "train_samples": int(train_idx.size),
            "test_samples": int(test_idx.size),
            "ridge_lambda": float(ridge_lambda),
            "latent_mse": float(np.mean((y_pred_linear - y_test) ** 2)),
            "delta_direction_cosine_mean": float(np.mean(cos_dir)),
            "forge_acc_mean": float(np.mean(forge_acc)),
            "pred_acc_mean": float(np.mean(linear_acc)),
            "pred_acc_uplift": float(np.mean(linear_acc - forge_acc)),
            "target_acc_forgeref_mean": float(np.mean(orig_forge)),
            "target_acc_selfref_mean": float(np.mean(orig_self)),
        },
        "mean_delta_baseline": {
            "latent_mse": float(np.mean((y_pred_mean_delta - y_test) ** 2)),
            "pred_acc_mean": float(np.mean(mean_delta_acc)),
            "pred_acc_uplift": float(np.mean(mean_delta_acc - forge_acc)),
        },
    }


def slerp_tensor(z0: torch.Tensor, z1: torch.Tensor, t: float) -> torch.Tensor:
    b = z0.shape[0]
    z0_flat = z0.reshape(b, -1).float()
    z1_flat = z1.reshape(b, -1).float()
    norm0 = z0_flat.norm(dim=1, keepdim=True).clamp(min=EPS)
    norm1 = z1_flat.norm(dim=1, keepdim=True).clamp(min=EPS)
    cos_theta = (z0_flat / norm0 * z1_flat / norm1).sum(dim=1).clamp(-1.0, 1.0)
    theta = torch.acos(cos_theta)
    sin_theta = torch.sin(theta)
    lerp_mask = sin_theta.abs() < 1e-4

    w0 = torch.where(
        lerp_mask,
        torch.full_like(theta, 1.0 - t),
        torch.sin((1.0 - t) * theta) / (sin_theta + EPS),
    )
    w1 = torch.where(
        lerp_mask,
        torch.full_like(theta, t),
        torch.sin(t * theta) / (sin_theta + EPS),
    )
    view_shape = [b] + [1] * (z0.ndim - 1)
    return (w0.view(view_shape) * z0 + w1.view(view_shape) * z1).to(z0.dtype)


def build_reference(mode: str, forge_ref: torch.Tensor, original_ref: torch.Tensor, t: float) -> torch.Tensor:
    if mode == "forge":
        return forge_ref
    if mode == "original":
        return original_ref
    if mode == "blend":
        return torch.lerp(forge_ref, original_ref, float(t))
    raise ValueError(f"Unsupported decode_reference: {mode}")


def analyze_paths(
    z_f: torch.Tensor,
    z_o: torch.Tensor,
    attacked_images: torch.Tensor,
    wm_images: torch.Tensor,
    messages: torch.Tensor,
    sample_ids: Sequence[int],
    vae,
    wm_adapter,
    device: torch.device,
    path_points: int,
    path_batch_size: int,
    path_max_samples: int,
    path_seed: int,
    decode_reference: str,
    local_eps: float,
    local_repeats: int,
    pca_topk: int,
) -> Tuple[Dict[str, object], pd.DataFrame, pd.DataFrame]:
    n_total = z_f.shape[0]
    rng = np.random.default_rng(path_seed)
    if path_max_samples > 0 and n_total > path_max_samples:
        chosen = np.sort(rng.choice(n_total, size=path_max_samples, replace=False))
    else:
        chosen = np.arange(n_total)

    z_f = z_f[chosen]
    z_o = z_o[chosen]
    attacked_images = attacked_images[chosen]
    wm_images = wm_images[chosen]
    messages = messages[chosen]
    sample_ids = [int(sample_ids[i]) for i in chosen.tolist()]

    delta_flat_all = (z_o - z_f).view(z_f.shape[0], -1).numpy().astype(np.float32)
    pca = fit_delta_pca(delta_flat_all, n_components=max(1, pca_topk))
    mean_delta = torch.from_numpy(pca["mean_delta"]).to(device=device, dtype=torch.float32)
    components = torch.from_numpy(pca["components"]).to(device=device, dtype=torch.float32)

    ts = np.linspace(0.0, 1.0, path_points, dtype=np.float32)
    curve_rows = []
    sample_rows = []

    path_specs = [("slerp", 1), ("linear", 1), ("pca_topk", 1), ("local_perturbed", max(1, local_repeats))]

    for path_type, repeats in path_specs:
        for repeat_idx in range(repeats):
            for start in range(0, z_f.shape[0], path_batch_size):
                end = min(start + path_batch_size, z_f.shape[0])
                zf = z_f[start:end].to(device=device, dtype=torch.float32)
                zo = z_o[start:end].to(device=device, dtype=torch.float32)
                forge_ref = attacked_images[start:end].to(device=device, dtype=torch.float32)
                orig_ref = wm_images[start:end].to(device=device, dtype=torch.float32)
                msg = messages[start:end].to(device=device, dtype=torch.float32)
                delta = zo - zf
                delta_flat = delta.view(delta.shape[0], -1)

                if path_type == "pca_topk":
                    centered = delta_flat - mean_delta
                    scores = centered @ components.T
                    delta_proj = mean_delta + scores @ components
                    delta_proj = delta_proj.view_as(delta)
                else:
                    delta_proj = None

                if path_type == "local_perturbed":
                    noise = torch.randn_like(delta_flat)
                    proj = (
                        (noise * delta_flat).sum(dim=1, keepdim=True)
                        / (delta_flat.pow(2).sum(dim=1, keepdim=True) + EPS)
                    ) * delta_flat
                    ortho = noise - proj
                    ortho = ortho / (ortho.norm(dim=1, keepdim=True) + EPS)
                    offset = ortho * (local_eps * delta_flat.norm(dim=1, keepdim=True))
                    offset = offset.view_as(delta)
                else:
                    offset = None

                curve_batch = []
                for t in ts.tolist():
                    if path_type == "slerp":
                        z_t = slerp_tensor(zf, zo, t)
                    elif path_type == "linear":
                        z_t = torch.lerp(zf, zo, float(t))
                    elif path_type == "pca_topk":
                        z_t = zf + float(t) * delta_proj
                    elif path_type == "local_perturbed":
                        local_weight = 4.0 * float(t) * (1.0 - float(t))
                        z_t = torch.lerp(zf, zo, float(t)) + local_weight * offset
                    else:
                        raise ValueError(path_type)

                    ref_t = build_reference(decode_reference, forge_ref, orig_ref, float(t))
                    x_t = decode_latent(vae, z_t, ref_t)
                    logits_t = wm_adapter.decode(x_t)
                    acc_t = sample_bit_accuracy(logits_t, msg)
                    curve_batch.append(acc_t)

                curve_batch = np.stack(curve_batch, axis=1)
                for local_idx, sid in enumerate(sample_ids[start:end]):
                    curve = curve_batch[local_idx]
                    diffs = np.diff(curve)
                    monotonic_rate = float(np.mean(diffs >= -1e-4)) if diffs.size > 0 else 1.0
                    fully_monotonic = float(np.all(diffs >= -1e-4)) if diffs.size > 0 else 1.0
                    row = {
                        "sample_id": int(sid),
                        "path_type": path_type,
                        "repeat": int(repeat_idx),
                        "start_acc": float(curve[0]),
                        "end_acc": float(curve[-1]),
                        "uplift": float(curve[-1] - curve[0]),
                        "monotonic_rate": monotonic_rate,
                        "fully_monotonic": fully_monotonic,
                        "spearman": spearmanr_np(ts, curve),
                        "auc": float(np.trapz(curve, ts)),
                    }
                    sample_rows.append(row)
                    for t_idx, (t, acc) in enumerate(zip(ts.tolist(), curve.tolist())):
                        curve_rows.append(
                            {
                                "sample_id": int(sid),
                                "path_type": path_type,
                                "repeat": int(repeat_idx),
                                "point_idx": int(t_idx),
                                "t": float(t),
                                "acc": float(acc),
                            }
                        )

    sample_df = pd.DataFrame(sample_rows)
    curve_df = pd.DataFrame(curve_rows)

    summary = {}
    for path_type, group in sample_df.groupby("path_type"):
        summary[path_type] = {
            "num_curves": int(group.shape[0]),
            "start_acc_mean": float(group["start_acc"].mean()),
            "end_acc_mean": float(group["end_acc"].mean()),
            "uplift_mean": float(group["uplift"].mean()),
            "monotonic_rate_mean": float(group["monotonic_rate"].mean()),
            "fully_monotonic_fraction": float(group["fully_monotonic"].mean()),
            "spearman_mean": float(group["spearman"].mean()),
            "auc_mean": float(group["auc"].mean()),
        }
    summary["decode_reference"] = decode_reference
    summary["path_points"] = int(path_points)
    summary["num_selected_samples"] = int(len(sample_ids))
    summary["local_eps"] = float(local_eps)
    summary["local_repeats"] = int(local_repeats)
    summary["pca_topk"] = int(pca_topk)
    return summary, sample_df, curve_df


def analyze_probe(
    z_f_flat: np.ndarray,
    forge_acc: np.ndarray,
    bucket_size: float,
    test_fraction: float,
    split_seed: int,
    device: torch.device,
    probe_epochs: int,
    probe_batch_size: int,
    probe_lr: float,
    probe_weight_decay: float,
    probe_hidden_dim: int,
) -> Tuple[Dict[str, object], pd.DataFrame]:
    n = z_f_flat.shape[0]
    rng = np.random.default_rng(split_seed)
    perm = rng.permutation(n)
    test_n = max(1, int(round(n * test_fraction)))
    test_idx = np.sort(perm[:test_n])
    train_idx = np.sort(perm[test_n:])
    if train_idx.size < 2:
        train_idx = np.sort(perm[:-1])
        test_idx = np.sort(perm[-1:])

    x_train = z_f_flat[train_idx].astype(np.float32)
    x_test = z_f_flat[test_idx].astype(np.float32)
    y_train = forge_acc[train_idx].astype(np.float32)
    y_test = forge_acc[test_idx].astype(np.float32)

    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    x_train_std = (x_train - mean) / std
    x_test_std = (x_test - mean) / std

    pred_linear = train_probe_regressor(
        x_train_std,
        y_train,
        x_test_std,
        hidden_dim=0,
        epochs=probe_epochs,
        batch_size=probe_batch_size,
        lr=probe_lr,
        weight_decay=probe_weight_decay,
        device=device,
        seed=split_seed,
    )
    pred_mlp = train_probe_regressor(
        x_train_std,
        y_train,
        x_test_std,
        hidden_dim=probe_hidden_dim,
        epochs=probe_epochs,
        batch_size=probe_batch_size,
        lr=probe_lr,
        weight_decay=probe_weight_decay,
        device=device,
        seed=split_seed + 1,
    )

    pred_df = pd.DataFrame(
        {
            "sample_index": test_idx.astype(int),
            "target_acc": y_test.astype(np.float32),
            "pred_linear": pred_linear.astype(np.float32),
            "pred_mlp": pred_mlp.astype(np.float32),
            "target_bucket": bucketize(y_test, bucket_size).astype(int),
            "pred_linear_bucket": bucketize(pred_linear, bucket_size).astype(int),
            "pred_mlp_bucket": bucketize(pred_mlp, bucket_size).astype(int),
        }
    )

    summary = {
        "train_samples": int(train_idx.size),
        "test_samples": int(test_idx.size),
        "linear": {
            "r2": r2_score_np(y_test, pred_linear),
            "pearson": pearsonr_np(y_test, pred_linear),
            "spearman": spearmanr_np(y_test, pred_linear),
            "mae": mae_np(y_test, pred_linear),
            "bucket_acc": float(
                (pred_df["target_bucket"].to_numpy() == pred_df["pred_linear_bucket"].to_numpy()).mean()
            ),
        },
        "mlp_2layer": {
            "hidden_dim": int(probe_hidden_dim),
            "r2": r2_score_np(y_test, pred_mlp),
            "pearson": pearsonr_np(y_test, pred_mlp),
            "spearman": spearmanr_np(y_test, pred_mlp),
            "mae": mae_np(y_test, pred_mlp),
            "bucket_acc": float(
                (pred_df["target_bucket"].to_numpy() == pred_df["pred_mlp_bucket"].to_numpy()).mean()
            ),
        },
    }
    return summary, pred_df


@torch.no_grad()
def collect_pairs(
    cfg,
    run_dir: str,
    ckpt_path: str,
    split: str,
    attack_name: str,
    max_batches: int,
    batch_size: int,
    num_workers: int,
    max_samples: int,
    seed: int,
    device: torch.device,
) -> Dict[str, object]:
    set_seed(seed)
    vae = load_stage1_vae(cfg, ckpt_path, device)
    wm_adapter = build_wm_adapter(cfg.wm_model, cfg)
    attack_obj = None
    if attack_name != "offline_fake":
        if attack_name not in ATTACK_REGISTRY:
            raise KeyError(f'Attack "{attack_name}" not registered. Available: {list(ATTACK_REGISTRY.keys())}')
        attack_obj = build_attack(attack_name, cfg)

    loader = build_loader(cfg, split=split, batch_size=batch_size, num_workers=num_workers)

    z_f_list = []
    z_o_list = []
    attacked_list = []
    wm_list = []
    messages_list = []
    raw_acc_list = []
    forge_acc_list = []
    orig_self_acc_list = []
    orig_forge_acc_list = []
    sample_rows = []
    sample_id = 0

    for bidx, batch in enumerate(loader):
        if max_batches > 0 and bidx >= max_batches:
            break

        images = batch["image"].to(device, non_blocking=True)
        bs = images.shape[0]
        messages = torch.randint(
            0,
            2,
            (bs, wm_adapter.message_length),
            dtype=torch.float32,
            device=device,
        )

        wm_images = wm_adapter.encode(images, messages)
        attacked = apply_attack(attack_name, attack_obj, batch, images, wm_images, device)

        z_f, _ = vae.encode(attacked)
        z_o, _ = vae.encode(wm_images)

        raw_logits = wm_adapter.decode(attacked)
        forge_recon = decode_latent(vae, z_f, attacked)
        orig_self = decode_latent(vae, z_o, wm_images)
        orig_forge = decode_latent(vae, z_o, attacked)

        forge_logits = wm_adapter.decode(forge_recon)
        orig_self_logits = wm_adapter.decode(orig_self)
        orig_forge_logits = wm_adapter.decode(orig_forge)

        raw_acc = sample_bit_accuracy(raw_logits, messages)
        forge_acc = sample_bit_accuracy(forge_logits, messages)
        orig_self_acc = sample_bit_accuracy(orig_self_logits, messages)
        orig_forge_acc = sample_bit_accuracy(orig_forge_logits, messages)

        take = bs
        if max_samples > 0:
            remaining = max_samples - sample_id
            if remaining <= 0:
                break
            take = min(take, remaining)

        if take <= 0:
            break

        z_f = z_f[:take]
        z_o = z_o[:take]
        attacked = attacked[:take]
        wm_images = wm_images[:take]
        messages = messages[:take]
        raw_acc = raw_acc[:take]
        forge_acc = forge_acc[:take]
        orig_self_acc = orig_self_acc[:take]
        orig_forge_acc = orig_forge_acc[:take]
        img_paths = batch["img_path"][:take]

        z_f_flat = z_f.detach().cpu().view(take, -1).numpy()
        z_o_flat = z_o.detach().cpu().view(take, -1).numpy()
        delta_flat = z_o_flat - z_f_flat

        for local_idx in range(take):
            sample_rows.append(
                {
                    "sample_id": int(sample_id + local_idx),
                    "img_path": img_paths[local_idx],
                    "attack": attack_name,
                    "raw_attacked_acc": float(raw_acc[local_idx]),
                    "forge_recon_acc": float(forge_acc[local_idx]),
                    "original_selfref_acc": float(orig_self_acc[local_idx]),
                    "original_forgeref_acc": float(orig_forge_acc[local_idx]),
                    "z_f_norm": float(np.linalg.norm(z_f_flat[local_idx])),
                    "z_o_norm": float(np.linalg.norm(z_o_flat[local_idx])),
                    "delta_norm": float(np.linalg.norm(delta_flat[local_idx])),
                }
            )

        z_f_list.append(z_f.detach().cpu())
        z_o_list.append(z_o.detach().cpu())
        attacked_list.append(attacked.detach().cpu().half())
        wm_list.append(wm_images.detach().cpu().half())
        messages_list.append(messages.detach().cpu())
        raw_acc_list.append(torch.from_numpy(raw_acc))
        forge_acc_list.append(torch.from_numpy(forge_acc))
        orig_self_acc_list.append(torch.from_numpy(orig_self_acc))
        orig_forge_acc_list.append(torch.from_numpy(orig_forge_acc))

        sample_id += take
        if max_samples > 0 and sample_id >= max_samples:
            break

    if sample_id == 0:
        raise RuntimeError("No samples collected. Check split/max-batches/max-samples settings.")

    samples_df = pd.DataFrame(sample_rows)
    z_f = torch.cat(z_f_list, dim=0)
    z_o = torch.cat(z_o_list, dim=0)
    attacked_images = torch.cat(attacked_list, dim=0)
    wm_images = torch.cat(wm_list, dim=0)
    messages = torch.cat(messages_list, dim=0)

    return {
        "vae": vae,
        "wm_adapter": wm_adapter,
        "samples_df": samples_df,
        "z_f": z_f,
        "z_o": z_o,
        "z_shape": list(z_f.shape[1:]),
        "z_f_flat": z_f.view(z_f.shape[0], -1).numpy().astype(np.float32),
        "z_o_flat": z_o.view(z_o.shape[0], -1).numpy().astype(np.float32),
        "attacked_images": attacked_images,
        "wm_images": wm_images,
        "messages": messages,
        "raw_acc": torch.cat(raw_acc_list, dim=0).numpy().astype(np.float32),
        "forge_acc": torch.cat(forge_acc_list, dim=0).numpy().astype(np.float32),
        "orig_self_acc": torch.cat(orig_self_acc_list, dim=0).numpy().astype(np.float32),
        "orig_forge_acc": torch.cat(orig_forge_acc_list, dim=0).numpy().astype(np.float32),
    }


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    return obj


def determine_attacks(cfg, requested: str, split: str) -> List[str]:
    requested = str(requested).strip().lower()
    configured = [str(x) for x in getattr(cfg.attacks, "online", [])]
    csv_path = cfg.data.val_csv if split == "val" else cfg.data.train_csv
    has_offline = "fake_path" in pd.read_csv(csv_path, nrows=1).columns

    if requested == "auto":
        if configured:
            return [configured[0]]
        if has_offline:
            return ["offline_fake"]
        raise RuntimeError("No online attack configured and dataset has no fake_path column.")
    if requested == "all":
        attacks_out = list(configured)
        if has_offline:
            attacks_out.append("offline_fake")
        if not attacks_out:
            raise RuntimeError("No attack available for analysis.")
        return attacks_out
    if requested == "offline":
        if not has_offline:
            raise RuntimeError("Requested offline attack, but dataset has no fake_path column.")
        return ["offline_fake"]
    attacks_out = [x.strip() for x in requested.split(",") if x.strip()]
    if not attacks_out:
        raise RuntimeError("No valid attack names provided.")
    return attacks_out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True, help="Path to run dir containing config.yaml")
    p.add_argument("--checkpoint", default="best", help="best | last | /abs/path/to/ckpt.pth")
    p.add_argument("--split", choices=["train", "val"], default="val")
    p.add_argument("--attack", default="auto", help="auto | all | offline | comma-separated attack names")
    p.add_argument("--output-dir", default="", help="Default: <run-dir>/analysis/latent_geometry_<split>_<ckpt>")
    p.add_argument("--max-batches", type=int, default=8, help="0 means full split")
    p.add_argument("--max-samples", type=int, default=0, help="0 means no extra sample cap")
    p.add_argument("--batch-size", type=int, default=0, help="0 means use config batch size")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--bucket-size", type=float, default=0.1)
    p.add_argument("--min-bucket-samples", type=int, default=4)
    p.add_argument("--knn-k", type=int, default=10)

    p.add_argument("--test-fraction", type=float, default=0.2)
    p.add_argument("--ridge-lambda", type=float, default=1e-3)
    p.add_argument("--pca-components", type=int, default=8)

    p.add_argument("--path-points", type=int, default=11)
    p.add_argument("--path-batch-size", type=int, default=16)
    p.add_argument("--path-max-samples", type=int, default=128)
    p.add_argument("--decode-reference", choices=["forge", "original", "blend"], default="forge")
    p.add_argument("--local-eps", type=float, default=0.10)
    p.add_argument("--local-repeats", type=int, default=2)
    p.add_argument("--pca-topk", type=int, default=1)

    p.add_argument("--probe-epochs", type=int, default=200)
    p.add_argument("--probe-batch-size", type=int, default=128)
    p.add_argument("--probe-lr", type=float, default=1e-3)
    p.add_argument("--probe-weight-decay", type=float, default=1e-4)
    p.add_argument("--probe-hidden-dim", type=int, default=256)
    return p.parse_args()


def main():
    args = parse_args()
    run_dir = os.path.abspath(args.run_dir)
    cfg = load_run_config(run_dir)
    ckpt_path = resolve_ckpt_path(run_dir, args.checkpoint)
    batch_size = args.batch_size if args.batch_size > 0 else int(cfg.training.batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_tag = os.path.splitext(os.path.basename(ckpt_path))[0]
    if args.output_dir:
        base_out = os.path.abspath(args.output_dir)
    else:
        base_out = os.path.join(run_dir, "analysis", f"latent_geometry_{args.split}_{ckpt_tag}")
    os.makedirs(base_out, exist_ok=True)

    attacks_to_run = determine_attacks(cfg, args.attack, args.split)
    summaries = {}

    print(f"run_dir={run_dir}")
    print(f"checkpoint={ckpt_path}")
    print(f"device={device}")
    print(f"attacks={attacks_to_run}")

    for attack_name in attacks_to_run:
        attack_out = os.path.join(base_out, attack_name)
        os.makedirs(attack_out, exist_ok=True)
        print(f"[{attack_name}] collecting pairs...")

        collected = collect_pairs(
            cfg=cfg,
            run_dir=run_dir,
            ckpt_path=ckpt_path,
            split=args.split,
            attack_name=attack_name,
            max_batches=args.max_batches,
            batch_size=batch_size,
            num_workers=args.num_workers,
            max_samples=args.max_samples,
            seed=args.seed,
            device=device,
        )

        samples_df = collected["samples_df"]
        vae = collected["vae"]
        wm_adapter = collected["wm_adapter"]
        z_f_flat = collected["z_f_flat"]
        z_o_flat = collected["z_o_flat"]
        z_f = collected["z_f"]
        z_o = collected["z_o"]
        z_shape = collected["z_shape"]
        attacked_images = collected["attacked_images"]
        wm_images = collected["wm_images"]
        messages = collected["messages"]
        forge_acc = collected["forge_acc"]

        print(f"[{attack_name}] clustering...")
        clustering_summary, bucket_df, between_df = analyze_clustering(
            z_f_flat=z_f_flat,
            forge_acc=forge_acc,
            bucket_size=args.bucket_size,
            min_bucket_samples=args.min_bucket_samples,
            knn_k=args.knn_k,
        )

        print(f"[{attack_name}] displacement...")
        displacement_summary = analyze_displacements(
            z_f_flat=z_f_flat,
            z_o_flat=z_o_flat,
            samples_df=samples_df,
            z_shape=z_shape,
            attacked_images=attacked_images,
            wm_images=wm_images,
            messages=messages,
            vae=vae,
            wm_adapter=wm_adapter,
            device=device,
            test_fraction=args.test_fraction,
            split_seed=args.seed,
            ridge_lambda=args.ridge_lambda,
            analysis_batch_size=args.path_batch_size,
            pca_components=args.pca_components,
        )

        print(f"[{attack_name}] path robustness...")
        path_summary, path_sample_df, path_curve_df = analyze_paths(
            z_f=z_f,
            z_o=z_o,
            attacked_images=attacked_images,
            wm_images=wm_images,
            messages=messages,
            sample_ids=samples_df["sample_id"].tolist(),
            vae=vae,
            wm_adapter=wm_adapter,
            device=device,
            path_points=args.path_points,
            path_batch_size=args.path_batch_size,
            path_max_samples=args.path_max_samples,
            path_seed=args.seed,
            decode_reference=args.decode_reference,
            local_eps=args.local_eps,
            local_repeats=args.local_repeats,
            pca_topk=args.pca_topk,
        )

        print(f"[{attack_name}] probe...")
        probe_summary, probe_pred_df = analyze_probe(
            z_f_flat=z_f_flat,
            forge_acc=forge_acc,
            bucket_size=args.bucket_size,
            test_fraction=args.test_fraction,
            split_seed=args.seed,
            device=device,
            probe_epochs=args.probe_epochs,
            probe_batch_size=args.probe_batch_size,
            probe_lr=args.probe_lr,
            probe_weight_decay=args.probe_weight_decay,
            probe_hidden_dim=args.probe_hidden_dim,
        )

        summary = {
            "run_dir": run_dir,
            "checkpoint": ckpt_path,
            "split": args.split,
            "attack": attack_name,
            "num_samples": int(samples_df.shape[0]),
            "sample_acc_stats": {
                "raw_attacked_acc_mean": float(samples_df["raw_attacked_acc"].mean()),
                "forge_recon_acc_mean": float(samples_df["forge_recon_acc"].mean()),
                "original_selfref_acc_mean": float(samples_df["original_selfref_acc"].mean()),
                "original_forgeref_acc_mean": float(samples_df["original_forgeref_acc"].mean()),
                "delta_norm_mean": float(samples_df["delta_norm"].mean()),
            },
            "clustering": clustering_summary,
            "displacement": displacement_summary,
            "paths": path_summary,
            "probe": probe_summary,
        }
        summaries[attack_name] = summary

        samples_df.to_csv(os.path.join(attack_out, "samples.csv"), index=False)
        bucket_df.to_csv(os.path.join(attack_out, "bucket_stats.csv"), index=False)
        between_df.to_csv(os.path.join(attack_out, "bucket_between.csv"), index=False)
        path_sample_df.to_csv(os.path.join(attack_out, "path_sample_summary.csv"), index=False)
        path_curve_df.to_csv(os.path.join(attack_out, "path_curves.csv"), index=False)
        probe_pred_df.to_csv(os.path.join(attack_out, "probe_predictions.csv"), index=False)
        with open(os.path.join(attack_out, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(to_jsonable(summary), f, ensure_ascii=False, indent=2)

        print(
            f"[{attack_name}] done: forge_acc_mean={summary['sample_acc_stats']['forge_recon_acc_mean']:.4f}  "
            f"linear_map_uplift={summary['displacement']['linear_ridge_map']['pred_acc_uplift']:+.4f}  "
            f"slerp_monotonic={summary['paths']['slerp']['monotonic_rate_mean']:.4f}"
        )

    combined_path = os.path.join(base_out, "summary_all.json")
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(summaries), f, ensure_ascii=False, indent=2)
    print(f"saved={combined_path}")


if __name__ == "__main__":
    main()
