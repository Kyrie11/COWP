from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class FieldSpec:
    shape_suffix: tuple[int | str, ...]
    dtype_kind: str | tuple[str, ...]
    required: bool = True


COWP_SCHEMA: dict[str, FieldSpec] = {
    "cowp/candidates/trajectory": FieldSpec(("K", "T", 7), "f"),
    "cowp/candidates/macro_type": FieldSpec(("K",), ("i", "u")),
    "cowp/candidates/valid": FieldSpec(("K",), "b"),
    "cowp/candidates/conventional_safe": FieldSpec(("K",), "b"),
    "cowp/candidates/false_safe": FieldSpec(("K",), "b"),
    "cowp/candidates/noncoercive_feasible": FieldSpec(("K",), "b"),
    "cowp/candidates/ego_utility_prior": FieldSpec(("K",), "f"),
    "cowp/candidates/is_logged": FieldSpec(("K",), "b"),
    "cowp/candidates/is_neutral": FieldSpec(("K",), "b"),
    "cowp/candidates/topology_id": FieldSpec(("K",), ("i", "u")),
    "cowp/candidates/proposal_source": FieldSpec(("K",), ("i", "u"), required=False),
    "cowp/candidates/proposal_region_id": FieldSpec(("K",), ("i", "u"), required=False),
    "cowp/candidates/proposal_target_time_s": FieldSpec(("K",), "f", required=False),
    "cowp/candidates/proposal_timing_side": FieldSpec(("K",), ("i", "u"), required=False),
    "cowp/candidates/proposal_target_agent_index": FieldSpec(("K",), ("i", "u"), required=False),
    "cowp/candidates/proposal_gap_s": FieldSpec(("K",), "f", required=False),
    "cowp/candidates/proposal_accel_mps2": FieldSpec(("K",), "f", required=False),
    "cowp/candidates/proposal_entry_distance_m": FieldSpec(("K",), "f", required=False),
    "cowp/candidates/proposal_target_tta_error_s": FieldSpec(("K",), "f", required=False),
    "cowp/candidates/audited_pair_count": FieldSpec(("K",), ("i", "u"), required=False),
    "cowp/candidates/ncf_blocker_count": FieldSpec(("K",), ("i", "u"), required=False),
    "cowp/candidates/min_audited_opr": FieldSpec(("K",), "f", required=False),
    "cowp/candidates/max_audited_tail_burden_excess": FieldSpec(("K",), "f", required=False),
    "cowp/critical/track_index": FieldSpec(("A",), ("i", "u")),
    "cowp/critical/track_id": FieldSpec(("A",), ("i", "u"), required=False),
    "cowp/critical/input_index": FieldSpec(("A",), ("i", "u"), required=False),
    "cowp/critical/track_index_original": FieldSpec(("A",), ("i", "u"), required=False),
    "cowp/critical/input_visible": FieldSpec(("A",), "b", required=False),
    "cowp/critical/mapped_by_id": FieldSpec(("A",), "b", required=False),
    "cowp/critical/valid": FieldSpec(("A",), "b"),
    "cowp/natural/traj": FieldSpec(("A", "M", "T", 7), "f"),
    "cowp/natural/weight": FieldSpec(("A", "M"), "f"),
    "cowp/natural/source": FieldSpec(("A", "M"), ("i", "u")),
    "cowp/natural/valid": FieldSpec(("A", "M"), "b"),
    "cowp/natural/burden_neutral": FieldSpec(("A", "M"), "f"),
    "cowp/natural/priority_preserved": FieldSpec(("A", "M"), "b"),
    "cowp/natural/beta": FieldSpec(("A",), "f"),
    "cowp/natural/obs_contamination": FieldSpec(("A", "M"), "f", required=False),
    "cowp/natural/map_compliant": FieldSpec(("A", "M"), "b", required=False),
    "cowp/natural/map_distance_max": FieldSpec(("A", "M"), "f", required=False),
    "cowp/natural/map_verified": FieldSpec(("A", "M"), "b", required=False),
    "cowp/audit/pair_relevant": FieldSpec(("K", "A"), "b", required=False),
    "cowp/audit/relevance_mass": FieldSpec(("K", "A"), "f", required=False),
    "cowp/audit/root_affected": FieldSpec(("K", "A", "M"), "b", required=False),
    "cowp/audit/root_unsafe": FieldSpec(("K", "A", "M"), "b", required=False),
    "cowp/audit/root_direct_burden": FieldSpec(("K", "A", "M"), "f", required=False),
    "cowp/audit/root_budget_crossed": FieldSpec(("K", "A", "M"), "b", required=False),
    "cowp/audit/root_burden_only_affected": FieldSpec(("K", "A", "M"), "b", required=False),
    "cowp/audit/canonical_root_weight": FieldSpec(("A", "M"), "f", required=False),
    "cowp/response/traj": FieldSpec(("K", "A", "R", "T", 7), "f"),
    "cowp/response/valid": FieldSpec(("K", "A", "R"), "b"),
    "cowp/response/source": FieldSpec(("K", "A", "R"), ("i", "u")),
    "cowp/response/root_index": FieldSpec(("K", "A", "R"), ("i", "u"), required=False),
    "cowp/response/root_affinity": FieldSpec(("K", "A", "R"), "f", required=False),
    "cowp/response/is_safe": FieldSpec(("K", "A", "R"), "b"),
    "cowp/response/is_low_burden": FieldSpec(("K", "A", "R"), "b"),
    "cowp/response/burden_total": FieldSpec(("K", "A", "R"), "f"),
    "cowp/response/burden_components": FieldSpec(("K", "A", "R", 6), "f"),
    "cowp/witness/exists": FieldSpec(("K", "A"), "b"),
    "cowp/witness/token": FieldSpec(("K", "A"), ("i", "u")),
    "cowp/witness/burden_total": FieldSpec(("K", "A"), "f"),
    "cowp/witness/burden_components": FieldSpec(("K", "A", 6), "f"),
    "cowp/witness/min_safe_burden": FieldSpec(("K", "A"), "f"),
    "cowp/witness/natural_conflict_mass": FieldSpec(("K", "A"), "f"),
    "cowp/witness/causal_relevance_mass": FieldSpec(("K", "A"), "f", required=False),
    "cowp/witness/natural_conflict_mass_by_source": FieldSpec(("K", "A", 4), "f", required=False),
    "cowp/witness/natural_mass_by_source": FieldSpec(("K", "A", 4), "f", required=False),
    "cowp/witness/low_safe_mass_by_source": FieldSpec(("K", "A", 4), "f", required=False),
    "cowp/witness/opr": FieldSpec(("K", "A"), "f"),
    "cowp/witness/c_i": FieldSpec(("K", "A"), "f"),
    "cowp/witness/tail_burden_excess": FieldSpec(("K", "A"), "f", required=False),
    "cowp/witness/root_min_safe_burden": FieldSpec(("K", "A", "M"), "f", required=False),
    "cowp/witness/conflict_interval": FieldSpec(("K", "A", 2), ("i", "u")),
    "cowp/witness/conflict_region_id": FieldSpec(("K", "A"), ("i", "u")),
    "cowp/witness/critical_agent_track_index": FieldSpec(("A",), ("i", "u")),
    "cowp/witness/rho": FieldSpec(("K", "A"), ("i", "u")),
    "cowp/witness/pair_noncoercive_feasible": FieldSpec(("K", "A"), "b", required=False),
    "cowp/witness/blocker_code": FieldSpec(("K", "A"), ("i", "u"), required=False),
    "cowp/transport/mode_valid": FieldSpec(("K", "A", "M"), "b", required=False),
    "cowp/transport/mode_conflict": FieldSpec(("K", "A", "M"), "b", required=False),
    "cowp/transport/mode_affected": FieldSpec(("K", "A", "M"), "b", required=False),
    "cowp/transport/mode_retained_low_safe": FieldSpec(("K", "A", "M"), "b", required=False),
    "cowp/transport/response_root_index": FieldSpec(("K", "A", "R"), ("i", "u"), required=False),
    "cowp/transport/response_is_min_burden": FieldSpec(("K", "A", "R"), "b", required=False),
    "cowp/transport/root_recovery_mass": FieldSpec(("K", "A"), "f", required=False),
    "cowp/transport/root_low_safe_score": FieldSpec(("K", "A", "M"), "f", required=False),
    "cowp/transport/root_target_confidence": FieldSpec(("K", "A", "M"), "f", required=False),
    "cowp/transport/root_min_safe_burden": FieldSpec(("K", "A", "M"), "f", required=False),
    "cowp/transport/transported_opr": FieldSpec(("K", "A"), "f", required=False),
    "cowp/transport/canonical_root_weight": FieldSpec(("A", "M"), "f", required=False),
    "map/conflict_regions": FieldSpec(("C", 8), "f"),
    "map/conflict_region_valid": FieldSpec(("C",), "b"),
}


def _matches_dtype(arr: np.ndarray, kind: str | tuple[str, ...]) -> bool:
    kinds = (kind,) if isinstance(kind, str) else kind
    if "b" in kinds and arr.dtype == np.bool_:
        return True
    return arr.dtype.kind in kinds


def validate_schema(data: Mapping[str, np.ndarray], cfg: dict, strict: bool = True) -> list[str]:
    errors: list[str] = []
    dims = {
        "K": int(cfg.get("limits", {}).get("max_candidates", 64)),
        "A": int(cfg.get("limits", {}).get("max_critical_agents", 8)),
        "M": int(cfg.get("limits", {}).get("max_natural_alternatives", 24)),
        "R": int(cfg.get("limits", {}).get("max_safe_responses", 32)),
        "T": int(cfg.get("time", {}).get("future_steps", 80)),
        "C": int(cfg.get("limits", {}).get("max_conflict_regions", 64)),
    }
    for name, spec in COWP_SCHEMA.items():
        if name not in data:
            if strict and spec.required:
                errors.append(f"missing field {name}")
            continue
        arr = np.asarray(data[name])
        expected = tuple(dims[x] if isinstance(x, str) else x for x in spec.shape_suffix)
        if arr.shape != expected:
            errors.append(f"{name} shape {arr.shape} != {expected}")
        if not _matches_dtype(arr, spec.dtype_kind):
            errors.append(f"{name} dtype {arr.dtype} not compatible with {spec.dtype_kind}")
    return errors


def validate_numeric_invariants(data: Mapping[str, np.ndarray], cfg: dict) -> list[str]:
    errors: list[str] = []
    for key, arr in data.items():
        if isinstance(arr, np.ndarray) and arr.dtype.kind == "f" and not np.all(np.isfinite(arr) | np.isinf(arr)):
            errors.append(f"{key} contains NaN")
    if "cowp/candidates/valid" in data and int(np.sum(data["cowp/candidates/valid"])) < 8:
        errors.append("candidate_valid.sum < 8")
    if "cowp/natural/weight" in data and "cowp/natural/valid" in data:
        weights = np.asarray(data["cowp/natural/weight"])
        valid = np.asarray(data["cowp/natural/valid"], dtype=bool)
        for a in range(weights.shape[0]):
            if np.any(valid[a]) and not np.isclose(weights[a, valid[a]].sum(), 1.0, atol=1e-3):
                errors.append(f"natural weights for critical {a} do not sum to 1")
    if "cowp/witness/opr" in data:
        opr = np.asarray(data["cowp/witness/opr"])
        if np.nanmin(opr) < -1e-4 or np.nanmax(opr) > 1.0001:
            errors.append("OPR outside [0,1]")
    if "cowp/response/burden_components" in data:
        comps = np.asarray(data["cowp/response/burden_components"])
        if np.nanmin(comps) < -1e-4 or np.nanmax(comps) > 2.0001:
            errors.append("burden components outside [0,2]")
    if "cowp/audit/pair_relevant" in data:
        rel = np.asarray(data["cowp/audit/pair_relevant"], dtype=bool)
        cand_valid = np.asarray(data.get("cowp/candidates/valid", np.ones(rel.shape[0], dtype=bool)), dtype=bool)
        crit_valid = np.asarray(data.get("cowp/critical/valid", np.ones(rel.shape[1], dtype=bool)), dtype=bool)
        base_pair = cand_valid[:, None] & crit_valid[None, :]
        if "cowp/witness/pair_noncoercive_feasible" in data:
            pncf = np.asarray(data["cowp/witness/pair_noncoercive_feasible"], dtype=bool)
            if np.any(base_pair & (~rel) & (~pncf)):
                errors.append("irrelevant valid audit pair marked non-NCF")
            if "cowp/witness/exists" in data:
                exists_pair = np.asarray(data["cowp/witness/exists"], dtype=bool)
                if np.any(base_pair & rel & (~pncf) & (~exists_pair)):
                    errors.append("relevant non-NCF pair has no witness (silent blocker)")
        if "cowp/witness/blocker_code" in data:
            code = np.asarray(data["cowp/witness/blocker_code"], dtype=np.int64)
            if np.any(base_pair & (~rel) & (code != 0)):
                errors.append("irrelevant valid audit pair carries blocker code")
        if all(k in data for k in (
            "cowp/audit/root_affected", "cowp/audit/root_unsafe",
            "cowp/audit/root_budget_crossed", "cowp/audit/root_burden_only_affected",
        )):
            ra = np.asarray(data["cowp/audit/root_affected"], dtype=bool)
            ru = np.asarray(data["cowp/audit/root_unsafe"], dtype=bool)
            rb = np.asarray(data["cowp/audit/root_budget_crossed"], dtype=bool)
            rbo = np.asarray(data["cowp/audit/root_burden_only_affected"], dtype=bool)
            if not (ra.shape == ru.shape == rb.shape == rbo.shape):
                errors.append("causal-audit root tensors have inconsistent shapes")
            else:
                if np.any(np.logical_xor(ra, ru | rb)):
                    errors.append("root_affected != root_unsafe OR root_budget_crossed")
                if np.any(np.logical_xor(rbo, rb & ~ru)):
                    errors.append("root_burden_only_affected != root_budget_crossed AND NOT root_unsafe")
        if all(k in data for k in ("cowp/audit/root_affected", "cowp/transport/mode_affected")):
            ra = np.asarray(data["cowp/audit/root_affected"], dtype=bool)
            ta = np.asarray(data["cowp/transport/mode_affected"], dtype=bool)
            if ra.shape != ta.shape or np.any(np.logical_xor(ra, ta)):
                errors.append("transport/mode_affected does not exactly match audit/root_affected")
    if "cowp/witness/exists" in data:
        exists = np.asarray(data["cowp/witness/exists"], dtype=bool)
        if "cowp/audit/relevance_mass" in data:
            mass = np.asarray(data["cowp/audit/relevance_mass"], dtype=float)
            delta = float(cfg.get("ncf", {}).get("audit_min_relevance_mass", 0.10))
            mass_name = "causal relevance"
        else:
            mass = np.asarray(data.get("cowp/witness/natural_conflict_mass", np.zeros_like(exists, dtype=float)))
            delta = float(cfg.get("ncf", {}).get("positive_min_natural_conflict_mass", 0.10))
            mass_name = "conflict"
        opr = np.asarray(data.get("cowp/witness/opr", np.ones_like(exists, dtype=float)))
        tail = data.get("cowp/witness/tail_burden_excess")
        if tail is None:
            min_b = np.asarray(data.get("cowp/witness/min_safe_burden", np.zeros_like(exists, dtype=float)))
            beta = np.asarray(data.get("cowp/natural/beta", np.ones(exists.shape[1]) * 0.65))
            tail = np.maximum(min_b - beta[None, :], 0.0)
        else:
            tail = np.asarray(tail)
        alpha_opr = float(cfg.get("ncf", {}).get("alpha_opr", 0.35))
        gamma = float(cfg.get("ncf", {}).get("gamma", 0.10))
        for k, a in zip(*np.where(exists)):
            if not mass[k, a] > delta - 1e-5:
                errors.append(f"positive witness ({k},{a}) below {mass_name} mass threshold")
            if not (tail[k, a] > gamma - 1e-5 or opr[k, a] < alpha_opr):
                errors.append(f"positive witness ({k},{a}) has neither tail-burden excess nor option collapse")
    return errors
