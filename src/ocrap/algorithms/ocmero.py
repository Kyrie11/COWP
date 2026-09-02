from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .lcv import EPS, normalize_weights, torch_normalize_weights, torch_weighted_lcvar, torch_weighted_mean, weighted_lcvar, weighted_mean


@dataclass
class OCMEROResult:
    r_dep: float
    r_orc: float
    gap: float
    odg_pos: float
    q: np.ndarray
    r_per_root: np.ndarray
    best_option: np.ndarray
    oracle_per_root: np.ndarray


def _masked_max(values: np.ndarray, option_valid: np.ndarray) -> np.ndarray:
    mask = np.asarray(option_valid, dtype=bool)
    if not mask.any():
        return np.full(values.shape[0], -1e9, dtype=np.float64)
    return np.max(np.where(mask[None, :], values, -1e9), axis=1)


def sparsify_compatibility(C: np.ndarray, top_m: int | None) -> np.ndarray:
    C = np.asarray(C, dtype=np.float64)
    if top_m is None or top_m <= 0 or top_m >= C.shape[-1]:
        return C.copy()
    out = np.zeros_like(C)
    for i in range(C.shape[0]):
        idx = np.argsort(C[i], kind="mergesort")[::-1][:top_m]
        out[i, idx] = C[i, idx]
        out[i, i] = max(out[i, i], C[i, i], 1.0)
    return out


def oc_mero(
    M: np.ndarray,
    p: np.ndarray,
    C: np.ndarray,
    alpha: float = 0.2,
    beta: float = 0.2,
    option_valid: np.ndarray | None = None,
    root_valid: np.ndarray | None = None,
    use_lcvar: bool = True,
    use_obs_kernel: bool = True,
    top_m: int | None = None,
    eps: float = EPS,
) -> OCMEROResult:
    M = np.asarray(M, dtype=np.float64)
    if M.ndim != 2:
        raise ValueError("M must be [K,L]")
    K, L = M.shape
    p_arr = np.asarray(p, dtype=np.float64).reshape(-1)[:K]
    if root_valid is not None:
        p_arr = np.where(np.asarray(root_valid, dtype=bool).reshape(-1)[:K], p_arr, 0.0)
    p_norm = normalize_weights(p_arr, eps)
    option_valid_arr = np.ones(L, dtype=bool) if option_valid is None else np.asarray(option_valid, dtype=bool).reshape(-1)[:L]
    C_arr = np.asarray(C, dtype=np.float64)
    if C_arr.shape != (K, K):
        raise ValueError(f"C must be [{K},{K}], got {C_arr.shape}")
    C_eff = np.eye(K, dtype=np.float64) if not use_obs_kernel else sparsify_compatibility(C_arr, top_m)
    agg = weighted_lcvar if use_lcvar else weighted_mean

    oracle_per_root = _masked_max(M, option_valid_arr)
    r_orc = agg(oracle_per_root, p_norm, beta) if use_lcvar else agg(oracle_per_root, p_norm)

    q = np.full((K, L), -1e9, dtype=np.float64)
    for i in range(K):
        w = normalize_weights(C_eff[i] * p_norm, eps)
        for l in range(L):
            if option_valid_arr[l]:
                q[i, l] = agg(M[:, l], w, beta) if use_lcvar else agg(M[:, l], w)
    best_option = np.argmax(q, axis=1)
    r_per_root = q[np.arange(K), best_option]
    r_dep = agg(r_per_root, p_norm, alpha) if use_lcvar else agg(r_per_root, p_norm)
    gap = float(r_orc - r_dep)
    return OCMEROResult(float(r_dep), float(r_orc), gap, max(0.0, gap), q, r_per_root, best_option, oracle_per_root)


def torch_oc_mero(
    M: torch.Tensor,
    p: torch.Tensor,
    C: torch.Tensor,
    alpha: float = 0.2,
    beta: float = 0.2,
    option_valid: torch.Tensor | None = None,
    root_valid: torch.Tensor | None = None,
    use_lcvar: bool = True,
    use_obs_kernel: bool = True,
    top_m: int | None = None,
    eps: float = EPS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, K, L = M.shape
    if option_valid is None:
        option_valid = torch.ones((B, L), dtype=torch.bool, device=M.device)
    if option_valid.dim() == 1:
        option_valid = option_valid.unsqueeze(0).expand(B, -1)
    if root_valid is not None:
        if root_valid.dim() == 1:
            root_valid = root_valid.unsqueeze(0).expand(B, -1)
        p = torch.where(root_valid.bool(), p, torch.zeros_like(p))
    p_norm = torch_normalize_weights(p, eps)
    if not use_obs_kernel:
        C_eff = torch.eye(K, dtype=M.dtype, device=M.device).unsqueeze(0).expand(B, -1, -1)
    elif top_m is not None and int(top_m) > 0 and int(top_m) < K:
        m = int(top_m)
        vals, idx = torch.topk(C, k=m, dim=-1)
        C_eff = torch.zeros_like(C).scatter(-1, idx, vals)
        eye = torch.eye(K, dtype=torch.bool, device=M.device).unsqueeze(0)
        C_eff = torch.where(eye, torch.maximum(C_eff, torch.ones_like(C_eff)), C_eff)
    else:
        C_eff = C
    M_masked = torch.where(option_valid.unsqueeze(1), M, torch.full_like(M, -1e9))
    oracle_per_root = M_masked.max(dim=-1).values
    r_orc = torch_weighted_lcvar(oracle_per_root, p_norm, beta, eps) if use_lcvar else torch_weighted_mean(oracle_per_root, p_norm, eps)
    q_list = []
    scores = M.transpose(1, 2)
    for i in range(K):
        w = torch_normalize_weights(C_eff[:, i, :] * p_norm, eps)
        w_expand = w.unsqueeze(1).expand(-1, L, -1)
        q_i = torch_weighted_lcvar(scores, w_expand, beta, eps) if use_lcvar else torch_weighted_mean(scores, w_expand, eps)
        q_i = torch.where(option_valid, q_i, torch.full_like(q_i, -1e9))
        q_list.append(q_i)
    q = torch.stack(q_list, dim=1)
    r_per_root = q.max(dim=-1).values
    r_dep = torch_weighted_lcvar(r_per_root, p_norm, alpha, eps) if use_lcvar else torch_weighted_mean(r_per_root, p_norm, eps)
    return r_dep, r_orc, r_orc - r_dep, q
