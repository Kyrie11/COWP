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
    "cowp/critical/valid",
    "cowp/witness/exists",
    "cowp/witness/rho",
}


def _rate(num: int, den: int) -> float:
    return float(num / max(den, 1))


def _scenario_summary(row: dict[str, np.ndarray], keep_sources: set[int] | None) -> tuple[dict[str, bool], int]:
    valid = np.asarray(row.get("cowp/candidates/valid", []), dtype=bool).reshape(-1)
    if valid.size == 0:
        return {
            "any_valid": False, "any_conv": False, "any_ncf": False,
            "floor": False, "priority_eligible": False, "priority_ncf": False, "priority_floor": False,
        }, 0
    source = np.asarray(row.get("cowp/candidates/proposal_source", np.zeros_like(valid, dtype=np.int64)), dtype=np.int64).reshape(-1)[: len(valid)]
    if keep_sources is not None:
        valid = valid & np.isin(source, np.asarray(sorted(keep_sources), dtype=np.int64))
    conv = np.asarray(row.get("cowp/candidates/conventional_safe", np.zeros_like(valid)), dtype=bool).reshape(-1)[: len(valid)] & valid
    ncf = np.asarray(row.get("cowp/candidates/noncoercive_feasible", np.zeros_like(valid)), dtype=bool).reshape(-1)[: len(valid)] & valid

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

    any_valid = bool(valid.any())
    any_conv = bool(conv.any())
    any_ncf = bool(ncf.any())
    any_pe = bool(priority_eligible.any())
    any_pncf = bool(priority_ncf.any())
    return {
        "any_valid": any_valid,
        "any_conv": any_conv,
        "any_ncf": any_ncf,
        "floor": bool(any_conv and not any_ncf),
        "priority_eligible": any_pe,
        "priority_ncf": any_pncf,
        "priority_floor": bool(any_pe and not any_pncf),
    }, int(valid.sum())


def main() -> None:
    ap = argparse.ArgumentParser(description="Cheap post-build ablation of COWP ego proposal sources; no retraining or Waymax required.")
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--subset-modulo", type=int, default=1)
    ap.add_argument("--subset-remainder", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    all_nonpad = {int(x) for x in ProposalSource if x != ProposalSource.PAD}
    timing = {int(ProposalSource.LEGACY_TIMING), int(ProposalSource.ROBUST_BCTE), int(ProposalSource.PRIORITY_HOLD_RELEASE)}
    schemes: dict[str, set[int] | None] = {
        "all": None,
        "without_priority_hold_release": all_nonpad - {int(ProposalSource.PRIORITY_HOLD_RELEASE)},
        "without_rmr_bcte": all_nonpad - {int(ProposalSource.ROBUST_BCTE)},
        "without_legacy_timing": all_nonpad - {int(ProposalSource.LEGACY_TIMING)},
        "without_any_interaction_timing": all_nonpad - timing,
        "base_plus_rmr_only": (all_nonpad - timing) | {int(ProposalSource.ROBUST_BCTE)},
        "base_plus_rmr_plus_priority_commitment": (all_nonpad - timing) | {int(ProposalSource.ROBUST_BCTE), int(ProposalSource.PRIORITY_HOLD_RELEASE)},
    }

    ds = COWPNpzDataset(args.cache_dir)
    modulo = max(1, int(args.subset_modulo))
    remainder = int(args.subset_remainder)
    if not 0 <= remainder < modulo:
        raise ValueError(f"subset remainder {remainder} invalid for modulo {modulo}")
    indices = [i for i in range(len(ds)) if i % modulo == remainder]
    if args.limit > 0:
        indices = indices[: int(args.limit)]

    counters = {name: Counter() for name in schemes}
    valid_counts = {name: [] for name in schemes}
    source_candidate_counts = Counter()
    missing_provenance_scenes = 0
    valid_nonpad_source_candidates = 0
    for i in indices:
        row = ds.load(i, WANTED)
        valid = np.asarray(row.get("cowp/candidates/valid", []), dtype=bool).reshape(-1)
        if "cowp/candidates/proposal_source" not in row:
            missing_provenance_scenes += 1
            # Source ablation is undefined on stale caches.  Do not silently map
            # every candidate to PAD: that makes the entire old bank look like
            # an RMR increment in the downstream difference.
            continue
        src = np.asarray(row["cowp/candidates/proposal_source"], dtype=np.int64).reshape(-1)[: len(valid)]
        valid_nonpad_source_candidates += int((valid & (src != int(ProposalSource.PAD))).sum())
        for value in src[valid]:
            try:
                source_candidate_counts[ProposalSource(int(value)).name] += 1
            except Exception:
                source_candidate_counts[f"UNKNOWN_{int(value)}"] += 1
        for name, keep in schemes.items():
            summary, nv = _scenario_summary(row, keep)
            c = counters[name]
            c["scenes"] += 1
            c["any_valid"] += int(summary["any_valid"])
            c["any_conv"] += int(summary["any_conv"])
            c["any_ncf"] += int(summary["any_ncf"])
            c["floor"] += int(summary["floor"])
            c["priority_eligible"] += int(summary["priority_eligible"])
            c["priority_ncf"] += int(summary["priority_ncf"])
            c["priority_floor"] += int(summary["priority_floor"])
            valid_counts[name].append(nv)

    if missing_provenance_scenes > 0 or valid_nonpad_source_candidates == 0:
        raise RuntimeError(
            "Proposal-source ablation requires a fresh cache with cowp/candidates/proposal_source provenance. "
            f"missing_provenance_scenes={missing_provenance_scenes}/{len(indices)}, "
            f"valid_nonpad_source_candidates={valid_nonpad_source_candidates}. "
            "The v16.8 transport overlay cannot retrofit proposal provenance into stale raw candidate tensors."
        )

    result = {
        "schema_version": "cowp_v16_8_6_proposal_source_ablation_v2",
        "cache_dir": str(Path(args.cache_dir).resolve()),
        "evaluation_subset": {"modulo": modulo, "remainder": remainder, "num_scenes": len(indices)},
        "proposal_source_candidate_counts": dict(source_candidate_counts),
        "ablations": {},
    }
    for name, c in counters.items():
        n = int(c["scenes"])
        pe = int(c["priority_eligible"])
        result["ablations"][name] = {
            "any_valid_scene_rate": _rate(c["any_valid"], n),
            "any_conventional_safe_scene_rate": _rate(c["any_conv"], n),
            "any_ncf_scene_rate": _rate(c["any_ncf"], n),
            "best_case_selected_false_safe_lower_bound": _rate(c["floor"], n),
            "any_priority_eligible_scene_rate": _rate(pe, n),
            "any_priority_ncf_scene_rate": _rate(c["priority_ncf"], n),
            "best_case_pbtr_lower_bound": _rate(c["priority_floor"], pe),
            "mean_valid_candidates": float(np.mean(valid_counts[name])) if valid_counts[name] else 0.0,
        }

    base = result["ablations"]["without_rmr_bcte"]
    full = result["ablations"]["all"]
    no_commit = result["ablations"]["without_priority_hold_release"]
    result["priority_hold_release_increment"] = {
        "delta_any_ncf_scene_rate": full["any_ncf_scene_rate"] - no_commit["any_ncf_scene_rate"],
        "delta_false_safe_floor": full["best_case_selected_false_safe_lower_bound"] - no_commit["best_case_selected_false_safe_lower_bound"],
        "delta_pbtr_floor": full["best_case_pbtr_lower_bound"] - no_commit["best_case_pbtr_lower_bound"],
        "delta_mean_valid_candidates": full["mean_valid_candidates"] - no_commit["mean_valid_candidates"],
    }
    result["rmr_increment"] = {
        "delta_any_ncf_scene_rate": full["any_ncf_scene_rate"] - base["any_ncf_scene_rate"],
        "delta_false_safe_floor": full["best_case_selected_false_safe_lower_bound"] - base["best_case_selected_false_safe_lower_bound"],
        "delta_pbtr_floor": full["best_case_pbtr_lower_bound"] - base["best_case_pbtr_lower_bound"],
        "delta_mean_valid_candidates": full["mean_valid_candidates"] - base["mean_valid_candidates"],
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
