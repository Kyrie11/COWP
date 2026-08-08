from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from cowp.core.constants import ProposalSource
from cowp.data.dataset import COWPNpzDataset


WANTED = {
    "scenario/id",
    "cowp/candidates/valid",
    "cowp/candidates/conventional_safe",
    "cowp/candidates/noncoercive_feasible",
    "cowp/candidates/proposal_source",
    "cowp/candidates/proposal_target_tta_error_s",
    "cowp/critical/valid",
    "cowp/witness/exists",
    "cowp/witness/rho",
}


def _sid(row: dict[str, np.ndarray], fallback: str) -> str:
    value = np.asarray(row.get("scenario/id", fallback))
    try:
        return str(value.item()) if value.size == 1 else str(value.reshape(-1)[0])
    except Exception:
        return fallback


def _zero_summary() -> dict[str, float | bool | int]:
    return {
        "valid_count": 0,
        "any_valid": False,
        "any_conv": False,
        "any_ncf": False,
        "conv_without_ncf": False,
        "any_priority_eligible": False,
        "any_priority_ncf": False,
        "priority_eligible_without_ncf": False,
        "rmr_count": 0,
        "rmr_ncf_count": 0,
        "rmr_timing_error_count": 0,
        "rmr_timing_error_sum_s": 0.0,
        "rmr_timing_error_max_s": 0.0,
    }


def _summarize(row: dict[str, np.ndarray]) -> dict[str, float | bool | int]:
    valid = np.asarray(row.get("cowp/candidates/valid", []), dtype=bool).reshape(-1)
    if valid.size == 0:
        return _zero_summary()
    conv = np.asarray(row.get("cowp/candidates/conventional_safe", np.zeros_like(valid)), dtype=bool).reshape(-1)[: len(valid)] & valid
    ncf = np.asarray(row.get("cowp/candidates/noncoercive_feasible", np.zeros_like(valid)), dtype=bool).reshape(-1)[: len(valid)] & valid
    source = np.asarray(row.get("cowp/candidates/proposal_source", np.full_like(valid, int(ProposalSource.PAD))), dtype=np.int64).reshape(-1)[: len(valid)]
    rmr = valid & (source == int(ProposalSource.ROBUST_BCTE))
    phr = valid & (source == int(ProposalSource.PRIORITY_HOLD_RELEASE))
    timing_err = np.asarray(
        row.get("cowp/candidates/proposal_target_tta_error_s", np.full(len(valid), np.nan, dtype=np.float32)),
        dtype=np.float32,
    ).reshape(-1)[: len(valid)]
    rmr_err = np.abs(timing_err[rmr]) if len(timing_err) else np.asarray([], dtype=np.float32)
    rmr_err = rmr_err[np.isfinite(rmr_err)]

    crit = np.asarray(row.get("cowp/critical/valid", []), dtype=bool).reshape(-1)
    witness = np.asarray(row.get("cowp/witness/exists", np.zeros((len(valid), len(crit)), dtype=bool)), dtype=bool)
    rho = np.asarray(row.get("cowp/witness/rho", np.zeros_like(witness, dtype=np.int64)), dtype=np.int64)
    if witness.ndim == 2 and witness.shape[0] >= len(valid) and witness.shape[1] == len(crit) and rho.shape == witness.shape:
        protected = ((rho[: len(valid)] == 2) | (rho[: len(valid)] == 3)) & crit[None, :]
        priority_available = protected.any(axis=1)
        priority_fs = conv & (witness[: len(valid)] & protected).any(axis=1)
        priority_ncf = conv & priority_available & ~priority_fs
        priority_eligible = conv & priority_available
    else:
        priority_ncf = np.zeros_like(valid)
        priority_eligible = np.zeros_like(valid)

    return {
        "valid_count": int(valid.sum()),
        "any_valid": bool(valid.any()),
        "any_conv": bool(conv.any()),
        "any_ncf": bool(ncf.any()),
        "conv_without_ncf": bool(conv.any() and not ncf.any()),
        "any_priority_eligible": bool(priority_eligible.any()),
        "any_priority_ncf": bool(priority_ncf.any()),
        "priority_eligible_without_ncf": bool(priority_eligible.any() and not priority_ncf.any()),
        "rmr_count": int(rmr.sum()),
        "rmr_ncf_count": int((rmr & ncf).sum()),
        "phr_count": int(phr.sum()),
        "phr_ncf_count": int((phr & ncf).sum()),
        "phr_priority_ncf_count": int((phr & priority_ncf).sum()),
        "rmr_timing_error_count": int(rmr_err.size),
        "rmr_timing_error_sum_s": float(rmr_err.sum()) if rmr_err.size else 0.0,
        "rmr_timing_error_max_s": float(rmr_err.max()) if rmr_err.size else 0.0,
    }


def _load(
    cache_dir: str,
    limit: int,
    requested_ids: set[str] | None = None,
) -> dict[str, dict[str, float | bool | int]]:
    ds = COWPNpzDataset(cache_dir)
    indices = list(range(len(ds)))
    if requested_ids:
        # Tensor-cache filenames are scenario IDs by construction.  Restricting
        # the paired probe to its requested IDs avoids opening thousands of large
        # legacy NPZ files just to compare a 191/1200-scene subset.
        indices = [i for i, p in enumerate(ds.paths) if p.stem in requested_ids]
    if limit > 0:
        indices = indices[: int(limit)]
    out = {}
    for i in indices:
        row = ds.load(i, WANTED)
        out[_sid(row, ds.paths[i].stem)] = _summarize(row)
    return out


def _load_profile(path: str | None) -> tuple[dict[str, dict], Counter]:
    latest: dict[str, dict] = {}
    reasons = Counter()
    if not path:
        return latest, reasons
    p = Path(path)
    if not p.is_file():
        return latest, reasons
    for raw in p.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except Exception:
            reasons["malformed_profile_row"] += 1
            continue
        sid = str(row.get("scenario_id", "")).strip()
        if not sid:
            reasons["profile_row_without_scenario_id"] += 1
            continue
        latest[sid] = row
    for row in latest.values():
        if str(row.get("status")) == "filtered":
            reasons[str(row.get("filter_reason", "filtered"))] += 1
    return latest, reasons


def _rate(x: int, n: int) -> float:
    return float(x / max(n, 1))


def _read_ids(path: str | None) -> set[str] | None:
    if not path:
        return None
    return {line.strip().split()[0] for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()}


def _aggregate(ids: list[str], old: dict, new: dict) -> tuple[Counter, list[int], list[int]]:
    c = Counter()
    old_valid: list[int] = []
    new_valid: list[int] = []
    for sid in ids:
        a, b = old[sid], new[sid]
        old_valid.append(int(a["valid_count"]))
        new_valid.append(int(b["valid_count"]))
        c.update(
            pairs=1,
            old_any_valid=int(bool(a["any_valid"])),
            new_any_valid=int(bool(b["any_valid"])),
            old_any_ncf=int(bool(a["any_ncf"])),
            new_any_ncf=int(bool(b["any_ncf"])),
            old_floor=int(bool(a["conv_without_ncf"])),
            new_floor=int(bool(b["conv_without_ncf"])),
            old_priority_eligible=int(bool(a["any_priority_eligible"])),
            new_priority_eligible=int(bool(b["any_priority_eligible"])),
            old_priority_floor=int(bool(a["priority_eligible_without_ncf"])),
            new_priority_floor=int(bool(b["priority_eligible_without_ncf"])),
            old_hard=int(bool(a["conv_without_ncf"])),
            hard_recovered=int(bool(a["conv_without_ncf"] and b["any_ncf"])),
            ncf_lost=int(bool(a["any_ncf"] and not b["any_ncf"])),
            new_rmr_candidates=int(b["rmr_count"]),
            new_rmr_ncf_candidates=int(b["rmr_ncf_count"]),
            new_scene_with_rmr=int(int(b["rmr_count"]) > 0),
            new_scene_with_rmr_ncf=int(int(b["rmr_ncf_count"]) > 0),
            new_phr_candidates=int(b.get("phr_count", 0)),
            new_phr_ncf_candidates=int(b.get("phr_ncf_count", 0)),
            new_phr_priority_ncf_candidates=int(b.get("phr_priority_ncf_count", 0)),
            new_scene_with_phr=int(int(b.get("phr_count", 0)) > 0),
            new_scene_with_phr_ncf=int(int(b.get("phr_ncf_count", 0)) > 0),
            new_scene_with_phr_priority_ncf=int(int(b.get("phr_priority_ncf_count", 0)) > 0),
            new_rmr_timing_error_count=int(b["rmr_timing_error_count"]),
            new_rmr_timing_error_sum_s=float(b["rmr_timing_error_sum_s"]),
        )
        c["new_rmr_timing_error_max_s"] = max(
            float(c.get("new_rmr_timing_error_max_s", 0.0)), float(b["rmr_timing_error_max_s"])
        )
    return c, old_valid, new_valid


def main() -> None:
    ap = argparse.ArgumentParser(description="Paired old-vs-fresh proposal-bank comparison.")
    ap.add_argument("--old-cache", required=True)
    ap.add_argument("--new-cache", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--representative-scene-ids", default=None)
    ap.add_argument("--hard-scene-ids", default=None)
    ap.add_argument("--new-build-profile", default=None, help="Profile JSONL used to account for requested fresh scenes filtered before an NPZ can be written.")
    ap.add_argument("--min-overall-any-valid", type=float, default=0.99)
    ap.add_argument("--min-overall-any-ncf", type=float, default=0.40)
    ap.add_argument("--max-false-safe-floor", type=float, default=0.55)
    ap.add_argument("--max-pbtr-floor", type=float, default=0.45)
    ap.add_argument("--min-hard-recovery", type=float, default=0.20)
    ap.add_argument("--max-rmr-target-tta-error-s", type=float, default=0.20)
    ap.add_argument("--allow-missing-requested", action="store_true")
    args = ap.parse_args()

    representative_requested = _read_ids(args.representative_scene_ids)
    hard_requested = _read_ids(args.hard_scene_ids)
    requested_union = (representative_requested or set()).union(hard_requested or set())
    requested_for_load = requested_union if requested_union else None
    old = _load(args.old_cache, int(args.limit), requested_for_load)
    new = _load(args.new_cache, int(args.limit), requested_for_load)
    profile, filter_reasons = _load_profile(args.new_build_profile)

    missing_from_old = sorted(requested_union.difference(old))
    raw_missing_new = sorted(requested_union.difference(new))
    synthesized_filtered: list[str] = []
    unexpected_missing_new: list[str] = []
    build_errors: list[str] = []
    for sid in raw_missing_new:
        row = profile.get(sid)
        status = str((row or {}).get("status", ""))
        if status == "filtered":
            new[sid] = _zero_summary()
            synthesized_filtered.append(sid)
        elif status == "error":
            build_errors.append(sid)
            unexpected_missing_new.append(sid)
        else:
            unexpected_missing_new.append(sid)

    if (missing_from_old or unexpected_missing_new) and not args.allow_missing_requested:
        raise ValueError(
            "Paired probe is incomplete: "
            f"missing_from_old={len(missing_from_old)} examples={missing_from_old[:5]}, "
            f"unexpected_missing_from_new={len(unexpected_missing_new)} examples={unexpected_missing_new[:5]}, "
            f"filtered_zero_valid_accounted={len(synthesized_filtered)}"
        )

    common = set(old).intersection(new)
    representative_ids = sorted(common if representative_requested is None else common.intersection(representative_requested))
    if not representative_ids:
        raise ValueError("No representative scenario IDs are common/accounted in both inputs")
    hard_ids = sorted(
        [sid for sid in common if bool(old[sid]["conv_without_ncf"])] if hard_requested is None else common.intersection(hard_requested)
    )
    c, old_valid, new_valid = _aggregate(representative_ids, old, new)
    hard_c, _, _ = _aggregate(hard_ids, old, new) if hard_ids else (Counter(), [], [])
    n = int(c["pairs"])
    new_any_valid = _rate(c["new_any_valid"], n)
    new_any_ncf = _rate(c["new_any_ncf"], n)
    new_floor = _rate(c["new_floor"], n)
    new_pbtr_floor = _rate(c["new_priority_floor"], c["new_priority_eligible"])
    old_pbtr_floor = _rate(c["old_priority_floor"], c["old_priority_eligible"])
    hard_recovery = _rate(hard_c["hard_recovered"], hard_c["old_hard"])
    rmr_timing_count = int(c["new_rmr_timing_error_count"])
    rmr_timing_mean = float(c["new_rmr_timing_error_sum_s"]) / max(rmr_timing_count, 1)
    rmr_timing_max = float(c.get("new_rmr_timing_error_max_s", 0.0))
    gates = {
        "overall_any_valid": new_any_valid >= args.min_overall_any_valid,
        "overall_any_ncf": new_any_ncf >= args.min_overall_any_ncf,
        "false_safe_floor": new_floor <= args.max_false_safe_floor,
        "pbtr_floor": new_pbtr_floor <= args.max_pbtr_floor,
        "hard_scene_recovery": hard_recovery >= args.min_hard_recovery,
        "rmr_timing_consistency": rmr_timing_count > 0 and rmr_timing_max <= args.max_rmr_target_tta_error_s + 1e-6,
        "no_unexpected_build_errors": not build_errors and not unexpected_missing_new,
    }
    result = {
        "schema_version": "cowp_v16_8_6_priority_commitment_paired_proposal_probe_v3",
        "old_cache": str(Path(args.old_cache).resolve()),
        "new_cache": str(Path(args.new_cache).resolve()),
        "new_build_profile": str(Path(args.new_build_profile).resolve()) if args.new_build_profile else None,
        "num_common_or_accounted_scenes": len(common),
        "num_representative_scenes": n,
        "pairing_completeness": {
            "requested_scene_count": len(requested_union),
            "missing_from_old_count": len(missing_from_old),
            "unexpected_missing_from_new_count": len(unexpected_missing_new),
            "filtered_fresh_scene_count_accounted_as_zero_valid": len(synthesized_filtered),
            "build_error_count": len(build_errors),
            "complete": not missing_from_old and not unexpected_missing_new,
            "fresh_filter_reason_counts": dict(filter_reasons),
        },
        "old": {
            "any_valid_scene_rate": _rate(c["old_any_valid"], n),
            "any_ncf_scene_rate": _rate(c["old_any_ncf"], n),
            "best_case_selected_false_safe_lower_bound": _rate(c["old_floor"], n),
            "best_case_pbtr_lower_bound": old_pbtr_floor,
            "mean_valid_candidates": float(np.mean(old_valid)),
        },
        "new": {
            "any_valid_scene_rate": new_any_valid,
            "any_ncf_scene_rate": new_any_ncf,
            "best_case_selected_false_safe_lower_bound": new_floor,
            "best_case_pbtr_lower_bound": new_pbtr_floor,
            "mean_valid_candidates": float(np.mean(new_valid)),
            "scene_with_rmr_bcte_rate": _rate(c["new_scene_with_rmr"], n),
            "scene_with_rmr_bcte_ncf_rate": _rate(c["new_scene_with_rmr_ncf"], n),
            "rmr_bcte_candidate_count": int(c["new_rmr_candidates"]),
            "rmr_bcte_ncf_candidate_count": int(c["new_rmr_ncf_candidates"]),
            "rmr_timing_error_count": rmr_timing_count,
            "rmr_target_tta_error_mean_s": rmr_timing_mean,
            "rmr_target_tta_error_max_s": rmr_timing_max,
            "scene_with_priority_hold_release_rate": _rate(c["new_scene_with_phr"], n),
            "scene_with_priority_hold_release_ncf_rate": _rate(c["new_scene_with_phr_ncf"], n),
            "scene_with_priority_hold_release_priority_ncf_rate": _rate(c["new_scene_with_phr_priority_ncf"], n),
            "priority_hold_release_candidate_count": int(c["new_phr_candidates"]),
            "priority_hold_release_ncf_candidate_count": int(c["new_phr_ncf_candidates"]),
            "priority_hold_release_priority_ncf_candidate_count": int(c["new_phr_priority_ncf_candidates"]),
        },
        "paired": {
            "old_hard_scene_count": int(hard_c["old_hard"]),
            "hard_scene_ncf_recovery_rate": hard_recovery,
            "ncf_loss_rate": _rate(c["ncf_lost"], n),
            "mean_valid_candidate_gain": float(np.mean(new_valid) - np.mean(old_valid)),
        },
        "promotion_thresholds": {
            "min_overall_any_valid": args.min_overall_any_valid,
            "min_overall_any_ncf": args.min_overall_any_ncf,
            "max_false_safe_floor": args.max_false_safe_floor,
            "max_pbtr_floor": args.max_pbtr_floor,
            "min_hard_recovery": args.min_hard_recovery,
            "max_rmr_target_tta_error_s": args.max_rmr_target_tta_error_s,
        },
        "gate_checks": gates,
        "promote_to_full_rebuild": bool(all(gates.values())),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
