from __future__ import annotations

import math

import numpy as np
import torch

EPS = 1e-8


def normalize_weights(weights: np.ndarray, eps: float = EPS) -> np.ndarray:
    w = np.clip(np.asarray(weights, dtype=np.float64).reshape(-1), 0.0, None)
    if w.size == 0:
        return w
    s = float(w.sum())
    if s <= eps:
        return np.ones_like(w, dtype=np.float64) / float(w.size)
    return w / s


def weighted_lcvar(scores: np.ndarray, weights: np.ndarray, alpha: float, eps: float = EPS) -> float:
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    if s.size == 0:
        raise ValueError("weighted_lcvar requires at least one score")
    if not (0.0 < float(alpha) <= 1.0):
        raise ValueError(f"alpha must be in (0,1], got {alpha}")
    w = normalize_weights(weights, eps)
    if w.size != s.size:
        raise ValueError(f"scores and weights length mismatch: {s.size} vs {w.size}")
    order = np.argsort(s, kind="mergesort")
    remaining = float(alpha)
    total = 0.0
    for idx in order:
        take = min(float(w[idx]), remaining)
        if take > 0:
            total += take * float(s[idx])
            remaining -= take
        if remaining <= 1e-12:
            break
    if remaining > 1e-8:
        total += remaining * float(s[order[-1]])
    return float(total / float(alpha))


def weighted_mean(scores: np.ndarray, weights: np.ndarray, eps: float = EPS) -> float:
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    w = normalize_weights(weights, eps)
    if w.size != s.size:
        raise ValueError("scores and weights length mismatch")
    return float(np.sum(s * w))


def finite_sample_upper_quantile(scores: np.ndarray, delta: float, numerical_margin: float = 0.0, strict: bool = True) -> float:
    vals = np.sort(np.asarray(scores, dtype=np.float64).reshape(-1))
    n = vals.size
    if n == 0:
        raise ValueError("Cannot calibrate: no negative deployability scores in calibration split.")
    if not (0.0 < float(delta) < 1.0):
        raise ValueError(f"delta must be in (0,1), got {delta}")
    k = int(math.ceil((n + 1) * (1.0 - float(delta))))
    if strict and k > n:
        return float("inf")
    return float(vals[min(k, n) - 1] + numerical_margin)


def torch_normalize_weights(weights: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    w = torch.clamp(weights, min=0.0)
    denom = w.sum(dim=-1, keepdim=True)
    fallback = torch.ones_like(w) / max(w.shape[-1], 1)
    return torch.where(denom > eps, w / denom.clamp_min(eps), fallback)


def torch_weighted_lcvar(scores: torch.Tensor, weights: torch.Tensor, alpha: float, eps: float = EPS) -> torch.Tensor:
    if not (0.0 < float(alpha) <= 1.0):
        raise ValueError(f"alpha must be in (0,1], got {alpha}")
    w = torch_normalize_weights(weights, eps)
    sorted_scores, idx = torch.sort(scores, dim=-1, descending=False, stable=True)
    sorted_weights = torch.gather(w, -1, idx)
    cumsum_prev = torch.cumsum(sorted_weights, dim=-1) - sorted_weights
    remaining = torch.clamp(float(alpha) - cumsum_prev, min=0.0)
    take = torch.minimum(sorted_weights, remaining)
    return (take * sorted_scores).sum(dim=-1) / float(alpha)


def torch_weighted_mean(scores: torch.Tensor, weights: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    w = torch_normalize_weights(weights, eps)
    return (scores * w).sum(dim=-1)
