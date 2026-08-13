#!/usr/bin/env python3
"""Diagnose whether existing COWP Waymax-attached tensor caches are sufficient.

This script is intentionally standalone: it only needs Python + NumPy and does
not import Waymax/JAX/PyTorch. It scans existing *.npz caches and answers five
separate questions:

1. Are the core natural/response/witness/planner labels structurally complete?
2. Are attached collision/offroad outcomes sufficient as *auxiliary* planner
   supervision?
3. Are attached log-divergence labels real and informative, or missing/constant?
4. Are learned-offline Waymax outcome metrics likely representative, or biased
   because only a small candidate subset was replayed?
5. Is replay/attach needed for real online Waymax closed-loop evaluation?
   (Answer: online evaluation reads tf.Example and does not require attached cache
   outcomes; this item is reported explicitly to prevent conflating the two.)

Example:
  python diagnose_waymax_cache_sufficiency.py \
    --train-cache /path/tensor_cache_train_waymax \
    --val-cache /path/tensor_cache_val_waymax \
    --workers 8 \
    --output-json outputs/cache_sufficiency.json

For a quick estimate before a full scan:
  python diagnose_waymax_cache_sufficiency.py ... --sample-scenes 2000
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def _restore_key(key: str) -> str:
    return str(key).replace("__", "/")


def _first(arrays: dict[str, np.ndarray], names: Iterable[str], default: Any = None) -> Any:
    for name in names:
        if name in arrays:
            return arrays[name]
    return default


def _scenario_id(arrays: dict[str, np.ndarray], path: str) -> str:
    value = _first(
        arrays,
        ("scenario/id", "womd/scenario/id", "scenario__id", "womd__scenario__id"),
        None,
    )
    if value is None:
        return Path(path).stem
    try:
        item = np.asarray(value).reshape(-1)[0]
        if isinstance(item, (bytes, np.bytes_)):
            return bytes(item).decode("utf-8", errors="replace")
        return str(item)
    except Exception:
        return Path(path).stem


def _bool_1d(value: Any, size: int = 0) -> np.ndarray:
    if value is None:
        return np.zeros(size, dtype=bool)
    arr = np.asarray(value, dtype=bool).reshape(-1)
    if size <= 0:
        return arr
    out = np.zeros(size, dtype=bool)
    n = min(size, arr.size)
    out[:n] = arr[:n]
    return out


def _float_1d(value: Any, size: int = 0, fill: float = np.nan) -> np.ndarray:
    if value is None:
        return np.full(size, fill, dtype=np.float32)
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if size <= 0:
        return arr
    out = np.full(size, fill, dtype=np.float32)
    n = min(size, arr.size)
    out[:n] = arr[:n]
    return out


def _int_1d(value: Any, size: int = 0, fill: int = -1) -> np.ndarray:
    if value is None:
        return np.full(size, fill, dtype=np.int64)
    arr = np.asarray(value, dtype=np.int64).reshape(-1)
    if size <= 0:
        return arr
    out = np.full(size, fill, dtype=np.int64)
    n = min(size, arr.size)
    out[:n] = arr[:n]
    return out


def _argmin_masked(values: np.ndarray, mask: np.ndarray) -> int:
    idx = np.flatnonzero(mask & np.isfinite(values))
    if idx.size == 0:
        return -1
    return int(idx[np.argmin(values[idx])])


STAGE_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "natural": (
        "cowp/natural/traj",
        "cowp/natural/weight",
        "cowp/natural/source",
        "cowp/natural/valid",
        "cowp/natural/beta",
    ),
    "response": (
        "cowp/candidates/trajectory",
        "cowp/candidates/valid",
        "cowp/response/valid",
        "cowp/response/is_safe",
        "cowp/response/is_low_burden",
        "cowp/response/burden_total",
    ),
    "witness": (
        "cowp/candidates/trajectory",
        "cowp/candidates/valid",
        "cowp/critical/valid",
        "cowp/witness/exists",
        "cowp/witness/token",
        "cowp/witness/burden_total",
        "cowp/witness/min_safe_burden",
        "cowp/witness/opr",
        "cowp/witness/conflict_interval",
    ),
    "planner_core": (
        "cowp/candidates/trajectory",
        "cowp/candidates/valid",
        "cowp/candidates/certificate_valid",
        "cowp/candidates/conventional_safe",
        "cowp/candidates/false_safe",
        "cowp/candidates/noncoercive_feasible",
        "cowp/candidates/ego_utility_prior",
        "cowp/candidates/is_logged",
        "cowp/candidates/is_neutral",
    ),
    "waymax_outcome": (
        "waymax/candidate_selected_for_rollout",
        "waymax/candidate_rollout_valid",
        "waymax/candidate_collision",
        "waymax/candidate_offroad",
        "waymax/candidate_log_divergence",
    ),
}

CLASS_KEYS = {
    "valid": "cowp/candidates/valid",
    "conventional_safe": "cowp/candidates/conventional_safe",
    "false_safe": "cowp/candidates/false_safe",
    "noncoercive_feasible": "cowp/candidates/noncoercive_feasible",
    "logged": "cowp/candidates/is_logged",
    "neutral": "cowp/candidates/is_neutral",
}


def _scan_one(path: str, logdiv_unsafe_threshold: float) -> dict[str, Any]:
    result: dict[str, Any] = {"path": path, "read_error": None}
    try:
        with np.load(path, allow_pickle=True) as data:
            arrays = {_restore_key(k): data[k] for k in data.files}
    except Exception as exc:
        result["read_error"] = f"{type(exc).__name__}: {exc}"
        return result

    keys = set(arrays)
    sid = _scenario_id(arrays, path)
    valid = _bool_1d(arrays.get("cowp/candidates/valid"))
    K = int(valid.size)
    certificate_valid = _bool_1d(arrays.get("cowp/candidates/certificate_valid", valid), K) & valid if K else valid
    result.update({
        "scenario_id": sid,
        "K": K,
        "stage_ok": {name: all(k in keys for k in req) for name, req in STAGE_REQUIREMENTS.items()},
        "missing_keys": {
            name: [k for k in req if k not in keys]
            for name, req in STAGE_REQUIREMENTS.items()
            if not all(k in keys for k in req)
        },
    })

    if K == 0:
        result["candidate_summary"] = {"valid": 0}
        return result

    class_masks: dict[str, np.ndarray] = {}
    for name, key in CLASS_KEYS.items():
        default = valid if name == "valid" else np.zeros(K, dtype=bool)
        base_mask = _bool_1d(arrays.get(key, default), K) & valid
        if name in {"false_safe", "noncoercive_feasible"}:
            base_mask &= certificate_valid
        class_masks[name] = base_mask
    class_masks["certificate_valid"] = certificate_valid

    selected = _bool_1d(arrays.get("waymax/candidate_selected_for_rollout"), K)
    rollout_valid = _bool_1d(arrays.get("waymax/candidate_rollout_valid"), K)
    collision = _bool_1d(arrays.get("waymax/candidate_collision"), K)
    offroad = _bool_1d(arrays.get("waymax/candidate_offroad"), K)
    logdiv = _float_1d(arrays.get("waymax/candidate_log_divergence"), K)
    macro = _int_1d(arrays.get("cowp/candidates/macro_type"), K)
    utility = _float_1d(arrays.get("cowp/candidates/ego_utility_prior"), K)

    rv = rollout_valid & valid
    finite_ld = rv & np.isfinite(logdiv)
    unsafe_physical = rv & (collision | offroad)
    unsafe_with_logdiv = rv & (
        collision | offroad | (np.isfinite(logdiv) & (logdiv > float(logdiv_unsafe_threshold)))
    )
    safe_physical = rv & ~collision & ~offroad

    class_stats: dict[str, dict[str, int]] = {}
    for name, mask in class_masks.items():
        replayed = mask & rv
        class_stats[name] = {
            "candidates": int(mask.sum()),
            "selected": int((mask & selected).sum()),
            "rollout_valid": int(replayed.sum()),
            "collision": int((replayed & collision).sum()),
            "offroad": int((replayed & offroad).sum()),
            "finite_logdiv": int((replayed & np.isfinite(logdiv)).sum()),
        }

    proxy: dict[str, bool | None] = {}
    proxy_indices = {
        "utility_best_valid": _argmin_masked(utility, valid),
        "utility_best_conventional": _argmin_masked(utility, class_masks["conventional_safe"]),
        "utility_best_ncf": _argmin_masked(utility, class_masks["noncoercive_feasible"]),
    }
    for name, idx in proxy_indices.items():
        proxy[name] = None if idx < 0 else bool(rv[idx])
    proxy["any_logged_replayed"] = None if not class_masks["logged"].any() else bool((class_masks["logged"] & rv).any())
    proxy["any_neutral_replayed"] = None if not class_masks["neutral"].any() else bool((class_masks["neutral"] & rv).any())

    macro_stats: dict[str, dict[str, int]] = {}
    for m in np.unique(macro[valid]):
        mm = valid & (macro == m)
        macro_stats[str(int(m))] = {
            "candidates": int(mm.sum()),
            "rollout_valid": int((mm & rv).sum()),
            "unsafe_physical": int((mm & unsafe_physical).sum()),
        }

    status = arrays.get("waymax/rollout_status")
    if status is not None:
        try:
            item = np.asarray(status).reshape(-1)[0]
            if isinstance(item, (bytes, np.bytes_)):
                item = bytes(item).decode("utf-8", errors="replace")
            status = str(item)
        except Exception:
            status = "<unreadable>"

    result.update({
        "candidate_summary": {
            "total_slots": K,
            "valid": int(valid.sum()),
            "certificate_valid": int(certificate_valid.sum()),
            "selected": int(selected.sum()),
            "rollout_valid": int(rv.sum()),
            "selected_and_rollout_valid": int((selected & rv).sum()),
            "selected_invalid_original": int((selected & ~valid).sum()),
            "rollout_valid_not_selected": int((rollout_valid & ~selected).sum()),
            "rollout_valid_invalid_original": int((rollout_valid & ~valid).sum()),
            "safe_physical": int(safe_physical.sum()),
            "unsafe_physical": int(unsafe_physical.sum()),
            "unsafe_with_logdiv": int(unsafe_with_logdiv.sum()),
            "mixed_safe_unsafe_physical": bool(safe_physical.any() and unsafe_physical.any()),
            "mixed_safe_unsafe_with_logdiv": bool((rv & ~unsafe_with_logdiv).any() and unsafe_with_logdiv.any()),
            "finite_logdiv": int(finite_ld.sum()),
            "nonzero_logdiv": int((finite_ld & (np.abs(logdiv) > 1e-6)).sum()),
            "logdiv_values": logdiv[finite_ld].astype(float).tolist(),
            "status": status,
        },
        "class_stats": class_stats,
        "macro_stats": macro_stats,
        "proxy_replay_coverage": proxy,
    })
    return result


@dataclass
class Aggregate:
    name: str
    path: str
    files_total: int = 0
    files_readable: int = 0
    read_errors: list[str] = field(default_factory=list)
    scenario_ids: set[str] = field(default_factory=set)
    stage_ok: Counter = field(default_factory=Counter)
    missing_keys: dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    sums: Counter = field(default_factory=Counter)
    scene_counts: Counter = field(default_factory=Counter)
    distributions: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    class_sums: dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    macro_sums: dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    proxy_true: Counter = field(default_factory=Counter)
    proxy_den: Counter = field(default_factory=Counter)
    status_counts: Counter = field(default_factory=Counter)

    def add(self, row: dict[str, Any]) -> None:
        self.files_total += 1
        if row.get("read_error"):
            if len(self.read_errors) < 20:
                self.read_errors.append(f"{row.get('path')}: {row['read_error']}")
            return
        self.files_readable += 1
        self.scenario_ids.add(str(row.get("scenario_id", "")))
        for stage, ok in row.get("stage_ok", {}).items():
            self.stage_ok[stage] += int(bool(ok))
        for stage, keys in row.get("missing_keys", {}).items():
            for key in keys:
                self.missing_keys[stage][key] += 1

        c = row.get("candidate_summary", {})
        for key in (
            "total_slots", "valid", "selected", "rollout_valid", "selected_and_rollout_valid",
            "selected_invalid_original", "rollout_valid_not_selected", "rollout_valid_invalid_original",
            "safe_physical", "unsafe_physical", "unsafe_with_logdiv", "finite_logdiv", "nonzero_logdiv",
        ):
            self.sums[key] += int(c.get(key, 0) or 0)
        for key in ("valid", "selected", "rollout_valid", "safe_physical", "unsafe_physical"):
            self.distributions[key].append(float(c.get(key, 0) or 0))
        self.scene_counts["with_any_selected"] += int(int(c.get("selected", 0) or 0) > 0)
        self.scene_counts["with_any_rollout_valid"] += int(int(c.get("rollout_valid", 0) or 0) > 0)
        self.scene_counts["mixed_safe_unsafe_physical"] += int(bool(c.get("mixed_safe_unsafe_physical", False)))
        self.scene_counts["mixed_safe_unsafe_with_logdiv"] += int(bool(c.get("mixed_safe_unsafe_with_logdiv", False)))
        self.distributions["logdiv"].extend(float(x) for x in c.get("logdiv_values", []) if np.isfinite(float(x)))
        if c.get("status") is not None:
            self.status_counts[str(c["status"])] += 1

        for cls, vals in row.get("class_stats", {}).items():
            self.class_sums[cls].update({k: int(v) for k, v in vals.items()})
        for macro, vals in row.get("macro_stats", {}).items():
            self.macro_sums[macro].update({k: int(v) for k, v in vals.items()})
        for name, value in row.get("proxy_replay_coverage", {}).items():
            if value is not None:
                self.proxy_den[name] += 1
                self.proxy_true[name] += int(bool(value))

    @staticmethod
    def _rate(num: float, den: float) -> float | None:
        return None if den <= 0 else float(num) / float(den)

    @staticmethod
    def _stats(values: list[float]) -> dict[str, float | int | None]:
        a = np.asarray(values, dtype=np.float64)
        a = a[np.isfinite(a)]
        if a.size == 0:
            return {"count": 0, "mean": None, "std": None, "p10": None, "p50": None, "p90": None, "min": None, "max": None}
        return {
            "count": int(a.size),
            "mean": float(a.mean()),
            "std": float(a.std()),
            "p10": float(np.percentile(a, 10)),
            "p50": float(np.percentile(a, 50)),
            "p90": float(np.percentile(a, 90)),
            "min": float(a.min()),
            "max": float(a.max()),
        }

    def finish(self) -> dict[str, Any]:
        n = self.files_readable
        selected = self.sums["selected"]
        rv = self.sums["rollout_valid"]
        valid = self.sums["valid"]
        finite_ld = self.sums["finite_logdiv"]
        logdiv_stats = self._stats(self.distributions["logdiv"])
        class_summary: dict[str, Any] = {}
        for cls, vals in sorted(self.class_sums.items()):
            denom = vals["candidates"]
            rv_cls = vals["rollout_valid"]
            class_summary[cls] = {
                **dict(vals),
                "selected_coverage": self._rate(vals["selected"], denom),
                "rollout_valid_coverage": self._rate(rv_cls, denom),
                "collision_rate": self._rate(vals["collision"], rv_cls),
                "offroad_rate": self._rate(vals["offroad"], rv_cls),
                "finite_logdiv_rate": self._rate(vals["finite_logdiv"], rv_cls),
            }
        macro_summary: dict[str, Any] = {}
        for macro, vals in sorted(self.macro_sums.items(), key=lambda x: int(x[0])):
            macro_summary[macro] = {
                **dict(vals),
                "rollout_valid_coverage": self._rate(vals["rollout_valid"], vals["candidates"]),
                "unsafe_physical_rate": self._rate(vals["unsafe_physical"], vals["rollout_valid"]),
            }
        proxy = {
            name: {
                "eligible_scenes": int(self.proxy_den[name]),
                "replayed_scenes": int(self.proxy_true[name]),
                "coverage": self._rate(self.proxy_true[name], self.proxy_den[name]),
            }
            for name in sorted(self.proxy_den)
        }
        return {
            "split": self.name,
            "cache_dir": self.path,
            "files_total": int(self.files_total),
            "files_readable": int(self.files_readable),
            "read_error_count": int(self.files_total - self.files_readable),
            "read_error_examples": self.read_errors,
            "unique_scenario_ids": int(len(self.scenario_ids)),
            "stage_key_coverage": {
                stage: {
                    "complete_scenes": int(self.stage_ok[stage]),
                    "coverage": self._rate(self.stage_ok[stage], n),
                    "top_missing_keys": self.missing_keys[stage].most_common(20),
                }
                for stage in STAGE_REQUIREMENTS
            },
            "scene_coverage": {
                "with_any_selected": self._rate(self.scene_counts["with_any_selected"], n),
                "with_any_rollout_valid": self._rate(self.scene_counts["with_any_rollout_valid"], n),
                "mixed_safe_unsafe_physical": self._rate(self.scene_counts["mixed_safe_unsafe_physical"], n),
                "mixed_safe_unsafe_with_logdiv": self._rate(self.scene_counts["mixed_safe_unsafe_with_logdiv"], n),
            },
            "candidate_totals": dict(self.sums),
            "candidate_distributions_per_scene": {
                k: self._stats(v) for k, v in self.distributions.items() if k != "logdiv"
            },
            "outcome_quality": {
                "selected_rollout_success_rate": self._rate(self.sums["selected_and_rollout_valid"], selected),
                "rollout_valid_coverage_of_valid_candidates": self._rate(rv, valid),
                "physical_unsafe_rate_among_rollout_valid": self._rate(self.sums["unsafe_physical"], rv),
                "logdiv_unsafe_rate_among_rollout_valid": self._rate(self.sums["unsafe_with_logdiv"], rv),
                "finite_logdiv_rate_among_rollout_valid": self._rate(finite_ld, rv),
                "nonzero_logdiv_rate_among_finite": self._rate(self.sums["nonzero_logdiv"], finite_ld),
                "logdiv_stats": logdiv_stats,
            },
            "consistency_errors": {
                "selected_invalid_original": int(self.sums["selected_invalid_original"]),
                "rollout_valid_not_selected": int(self.sums["rollout_valid_not_selected"]),
                "rollout_valid_invalid_original": int(self.sums["rollout_valid_invalid_original"]),
            },
            "coverage_by_candidate_class": class_summary,
            "coverage_by_macro_type_id": macro_summary,
            "proxy_selected_candidate_replay_coverage": proxy,
            "rollout_status_counts": self.status_counts.most_common(),
        }


def _scenario_paths(cache_dir: Path) -> list[Path]:
    """Return only real scenario archives, matching COWPNpzDataset."""
    return sorted(p for p in cache_dir.glob("*.npz") if not p.name.startswith("."))


def _scan_cache(
    name: str,
    cache_dir: Path,
    workers: int,
    sample_scenes: int | None,
    seed: int,
    logdiv_unsafe_threshold: float,
) -> tuple[dict[str, Any], set[str]]:
    # Hidden sampler/metadata NPZ files are not scenarios. Counting them created
    # the false 20441-vs-20440 discrepancy and one apparently incomplete scene.
    paths = _scenario_paths(cache_dir)
    if not paths:
        raise FileNotFoundError(f"No *.npz files found in {cache_dir}")
    if sample_scenes is not None and sample_scenes > 0 and sample_scenes < len(paths):
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(paths), size=int(sample_scenes), replace=False))
        paths = [paths[int(i)] for i in idx]
    agg = Aggregate(name=name, path=str(cache_dir))
    args = ((str(p), float(logdiv_unsafe_threshold)) for p in paths)
    if workers <= 1:
        for item in args:
            agg.add(_scan_one(*item))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for row in ex.map(_scan_one_star, args, chunksize=32):
                agg.add(row)
    return agg.finish(), agg.scenario_ids


def _scan_one_star(args: tuple[str, float]) -> dict[str, Any]:
    return _scan_one(*args)


def _status(level: str, reasons: list[str], action: str) -> dict[str, Any]:
    return {"status": level, "reasons": reasons, "recommended_action": action}


def _get(d: dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _min_non_none(*values: float | None) -> float | None:
    vals = [float(v) for v in values if v is not None]
    return min(vals) if vals else None


def _make_decision(train: dict[str, Any], val: dict[str, Any]) -> dict[str, Any]:
    decision: dict[str, Any] = {}

    core_coverages = []
    core_reasons = []
    for split_name, split in (("train", train), ("val", val)):
        for stage in ("natural", "response", "witness", "planner_core"):
            cov = _get(split, f"stage_key_coverage.{stage}.coverage", 0.0) or 0.0
            core_coverages.append(float(cov))
            if cov < 0.99:
                core_reasons.append(f"{split_name}.{stage} key coverage={cov:.3f} < 0.99")
    if min(core_coverages or [0.0]) >= 0.99:
        decision["core_staged_training"] = _status(
            "PASS",
            ["natural/response/witness/planner core labels are structurally complete in both splits"],
            "Reuse the existing caches for natural, response, witness, and label-based planner training; no Waymax replay is needed for these stages.",
        )
    else:
        decision["core_staged_training"] = _status(
            "FAIL", core_reasons,
            "Repair/rebuild missing core labels. Candidate replay/attach cannot fix missing natural/response/witness labels.",
        )

    safety_reasons = []
    safety_pass = True
    for split_name, split in (("train", train), ("val", val)):
        field_cov = _get(split, "stage_key_coverage.waymax_outcome.coverage", 0.0) or 0.0
        scene_cov = _get(split, "scene_coverage.with_any_rollout_valid", 0.0) or 0.0
        success = _get(split, "outcome_quality.selected_rollout_success_rate", 0.0) or 0.0
        rv_median = _get(split, "candidate_distributions_per_scene.rollout_valid.p50", 0.0) or 0.0
        mixed = _get(split, "scene_coverage.mixed_safe_unsafe_physical", 0.0) or 0.0
        if field_cov < 0.99:
            safety_pass = False
            safety_reasons.append(f"{split_name}: Waymax fields coverage={field_cov:.3f}")
        if scene_cov < 0.90:
            safety_pass = False
            safety_reasons.append(f"{split_name}: scenes with valid outcome={scene_cov:.3f} < 0.90")
        if success < 0.95:
            safety_pass = False
            safety_reasons.append(f"{split_name}: selected replay success={success:.3f} < 0.95")
        if rv_median < 8:
            safety_pass = False
            safety_reasons.append(f"{split_name}: median valid replayed candidates/scene={rv_median:.1f} < 8")
        if mixed < 0.15:
            safety_reasons.append(
                f"{split_name}: only {mixed:.3f} scenes contain both physically safe and unsafe replayed candidates; ranking supervision may be weak"
            )
    if safety_pass:
        decision["collision_offroad_auxiliary_training"] = _status(
            "PASS" if not safety_reasons else "WARN",
            safety_reasons or ["attached collision/offroad outcomes have broad scene coverage and high replay success"],
            "Reuse existing attached outcomes as an auxiliary collision/offroad planner loss. Do not make this sparse auxiliary loss the sole planner objective.",
        )
    else:
        decision["collision_offroad_auxiliary_training"] = _status(
            "FAIL", safety_reasons,
            "Do not replay all scenes immediately. First repair failed/missing rows incrementally or replay a targeted subset of scenes with low coverage.",
        )

    logdiv_reasons = []
    logdiv_pass = True
    for split_name, split in (("train", train), ("val", val)):
        finite = _get(split, "outcome_quality.finite_logdiv_rate_among_rollout_valid", 0.0) or 0.0
        nonzero = _get(split, "outcome_quality.nonzero_logdiv_rate_among_finite", 0.0) or 0.0
        std = _get(split, "outcome_quality.logdiv_stats.std", 0.0) or 0.0
        maxv = _get(split, "outcome_quality.logdiv_stats.max", 0.0) or 0.0
        if finite < 0.90:
            logdiv_pass = False
            logdiv_reasons.append(f"{split_name}: finite logdiv coverage={finite:.3f} < 0.90")
        if nonzero < 0.02 or std < 0.05 or maxv <= 0.0:
            logdiv_pass = False
            logdiv_reasons.append(
                f"{split_name}: logdiv is absent/near-constant (nonzero={nonzero:.3f}, std={std:.4f}, max={maxv:.4f})"
            )
    if logdiv_pass:
        decision["logdiv_training_and_evaluation"] = _status(
            "PASS", ["log-divergence labels are finite and non-degenerate in both splits"],
            "Existing caches can supervise and evaluate log-divergence.",
        )
    else:
        decision["logdiv_training_and_evaluation"] = _status(
            "FAIL", logdiv_reasons,
            "Do not train the logdiv head from these labels and do not report SelectedWaymaxMeanLogDivergence as evidence. Either set logdiv/outcome-risk terms to zero, or replay only the subset needed with metric-set=safety_logdiv.",
        )

    offline_reasons = []
    offline_pass = True
    for split_name, split in (("train", train), ("val", val)):
        candidate_cov = _get(split, "outcome_quality.rollout_valid_coverage_of_valid_candidates", 0.0) or 0.0
        proxies = _get(split, "proxy_selected_candidate_replay_coverage", {}) or {}
        proxy_covs = [v.get("coverage") for v in proxies.values() if isinstance(v, dict) and v.get("coverage") is not None]
        min_proxy = min(proxy_covs) if proxy_covs else 0.0
        if candidate_cov < 0.50:
            offline_pass = False
            offline_reasons.append(f"{split_name}: only {candidate_cov:.3f} of valid candidates have valid Waymax outcomes")
        if min_proxy < 0.75:
            offline_pass = False
            offline_reasons.append(f"{split_name}: minimum proxy-selected candidate replay coverage={min_proxy:.3f} < 0.75")
    if offline_pass:
        decision["learned_offline_waymax_outcome_metrics"] = _status(
            "PASS", ["candidate and proxy-selection coverage are adequate for learned-offline outcome diagnostics"],
            "Existing caches are acceptable for learned-offline outcome metrics, while real online Waymax remains the primary evaluation.",
        )
    else:
        decision["learned_offline_waymax_outcome_metrics"] = _status(
            "WARN", offline_reasons,
            "Treat learned-offline Waymax collision/offroad/logdiv numbers as partial diagnostics. Replay validation candidates selected by the trained checkpoint, or report the selected-outcome coverage denominator explicitly. Full training-set replay is not required for this.",
        )

    decision["real_online_waymax_closed_loop_evaluation"] = _status(
        "PASS_NOT_CACHE_DEPENDENT",
        ["real online evaluation loads validation tf.Example and steps the Waymax environment; attached candidate outcomes are not the evaluation environment"],
        "Run online Waymax closed-loop evaluation directly with the checkpoint. Do not spend four days replaying training labels merely to enable online evaluation.",
    )

    core_ok = decision["core_staged_training"]["status"] == "PASS"
    safety_ok = decision["collision_offroad_auxiliary_training"]["status"] in {"PASS", "WARN"}
    logdiv_ok = decision["logdiv_training_and_evaluation"]["status"] == "PASS"
    if core_ok and safety_ok and logdiv_ok:
        overall = _status(
            "REUSE_EXISTING",
            ["existing caches support core training and attached outcome supervision"],
            "Set RUN_OUTCOME_REPLAY=0 and point TRAIN_CACHE/VAL_CACHE to the existing waymax caches.",
        )
    elif core_ok and safety_ok and not logdiv_ok:
        overall = _status(
            "REUSE_WITH_LOGDIV_DISABLED",
            ["core labels and collision/offroad outcomes are usable, but logdiv is missing or degenerate"],
            "Skip full replay. Train with existing caches, disable logdiv/outcome-risk terms, and run real online Waymax. If logdiv is required for the paper, replay validation first and a targeted training subset rather than all 20k scenes.",
        )
    elif core_ok:
        overall = _status(
            "REUSE_CORE_TARGETED_REPLAY_ONLY",
            ["core labels are usable but attached outcomes are incomplete/weak"],
            "Reuse caches for natural/response/witness/planner-core training. Replay only failed/missing or model-selected hard scenes; do not automatically redo the entire training split.",
        )
    else:
        overall = _status(
            "CACHE_REBUILD_NEEDED",
            ["one or more core training stages lack required labels"],
            "Repair/rebuild the core cache. Replaying Waymax outcomes alone will not solve this.",
        )
    decision["overall"] = overall
    return decision


def _split_shift(train: dict[str, Any], val: dict[str, Any], overlap: int) -> dict[str, Any]:
    metrics = {
        "mean_valid_candidates": "candidate_distributions_per_scene.valid.mean",
        "mean_rollout_valid_candidates": "candidate_distributions_per_scene.rollout_valid.mean",
        "physical_unsafe_rate": "outcome_quality.physical_unsafe_rate_among_rollout_valid",
        "rollout_valid_candidate_coverage": "outcome_quality.rollout_valid_coverage_of_valid_candidates",
        "false_safe_replay_coverage": "coverage_by_candidate_class.false_safe.rollout_valid_coverage",
        "ncf_replay_coverage": "coverage_by_candidate_class.noncoercive_feasible.rollout_valid_coverage",
    }
    out = {"scenario_id_overlap": int(overlap), "metrics": {}}
    for name, path in metrics.items():
        tv = _get(train, path)
        vv = _get(val, path)
        out["metrics"][name] = {
            "train": tv,
            "val": vv,
            "absolute_difference": None if tv is None or vv is None else abs(float(tv) - float(vv)),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-cache", required=True, type=Path)
    ap.add_argument("--val-cache", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    ap.add_argument("--sample-scenes", type=int, default=None, help="Randomly sample this many files from each split; omit for a full scan.")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--logdiv-unsafe-threshold", type=float, default=8.0)
    ap.add_argument("--output-json", type=Path, default=None)
    args = ap.parse_args()

    train, train_ids = _scan_cache(
        "train", args.train_cache, args.workers, args.sample_scenes, args.seed, args.logdiv_unsafe_threshold
    )
    val, val_ids = _scan_cache(
        "val", args.val_cache, args.workers, args.sample_scenes, args.seed + 1, args.logdiv_unsafe_threshold
    )
    overlap = len((train_ids - {""}) & (val_ids - {""}))
    payload = {
        "scan_mode": "sample" if args.sample_scenes else "full",
        "sample_scenes_per_split": args.sample_scenes,
        "train": train,
        "val": val,
        "cross_split": _split_shift(train, val, overlap),
        "decision": _make_decision(train, val),
        "interpretation_notes": [
            "Attached candidate outcomes are optional auxiliary planner labels; natural/response/witness training does not require replay/attach.",
            "Real online Waymax closed-loop evaluation reads tf.Example and steps the simulator, so it is not enabled by replaying the training cache.",
            "A safety-only replay can still be useful for collision/offroad auxiliary training, but missing/constant logdiv must not be silently treated as a true zero target.",
            "Learned-offline selected Waymax metrics are only representative when the candidate selected by a method has a replayed valid outcome; report that coverage explicitly.",
        ],
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)
    print(text)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
