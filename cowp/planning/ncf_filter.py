from __future__ import annotations

import numpy as np


def hard_first_filter(candidate_valid: np.ndarray, conventional_safe: np.ndarray, witness_prob: np.ndarray, opr: np.ndarray, p_hard: float = 0.75, alpha: float = 0.35, use_hard_witness_rejection: bool = True, use_option_preservation: bool = True) -> np.ndarray:
    valid = np.asarray(candidate_valid, dtype=bool) & np.asarray(conventional_safe, dtype=bool)
    if witness_prob.ndim == 2:
        max_w = np.max(witness_prob, axis=-1)
    else:
        max_w = np.asarray(witness_prob)
    if opr.ndim == 2:
        min_opr = np.min(opr, axis=-1)
    else:
        min_opr = np.asarray(opr)
    if use_hard_witness_rejection:
        valid &= max_w <= p_hard
    if use_option_preservation:
        valid &= min_opr >= alpha
    return valid


def candidate_scores(ego_utility: np.ndarray, witness_prob: np.ndarray, opr: np.ndarray, c_i: np.ndarray, p_soft: float = 0.45, alpha: float = 0.35, gamma: float = 0.10, soft_burden_only: bool = False, ignore_witness_score: bool = False) -> np.ndarray:
    max_w = np.max(witness_prob, axis=-1) if witness_prob.ndim == 2 else np.asarray(witness_prob)
    min_opr = np.min(opr, axis=-1) if opr.ndim == 2 else np.asarray(opr)
    max_c = np.max(c_i, axis=-1) if c_i.ndim == 2 else np.asarray(c_i)
    if ignore_witness_score:
        return np.asarray(ego_utility, dtype=float)
    if soft_burden_only:
        return ego_utility + 2.0 * max_c + 1.0 * np.maximum(0.0, alpha - min_opr)
    return ego_utility + 5.0 * np.maximum(0.0, max_w - p_soft) + 2.0 * np.maximum(0.0, alpha - min_opr) + 2.0 * np.maximum(0.0, max_c - gamma)
