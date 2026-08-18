from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from cowp.core.constants import MacroType, ProposalSource
from cowp.data.dataset import COWPNpzDataset


WANTED = {
    "scenario/id",
    "cowp/candidates/valid",
    "cowp/candidates/conventional_safe",
    "cowp/candidates/false_safe",
    "cowp/candidates/noncoercive_feasible",
    "cowp/candidates/certificate_valid",
    "cowp/candidates/priority_eligible",
    "cowp/candidates/priority_false_safe",
    "cowp/candidates/priority_noncoercive_feasible",
    "cowp/candidates/macro_type",
    "cowp/candidates/proposal_source",
    "cowp/critical/valid",
    "cowp/critical/mechanism_valid",
    "cowp/witness/exists",
    "cowp/witness/rho",
    "cowp/witness/pair_noncoercive_feasible",
}


def _scenario_id(row: dict[str, np.ndarray], fallback: str) -> str:
    value = row.get("scenario/id")
    if value is None:
        return fallback
    arr = np.asarray(value)
    try:
        return str(arr.item()) if arr.size == 1 else str(arr.reshape(-1)[0])
    except Exception:
        return fallback


def _name(enum_cls, value: int) -> str:
    try:
        return enum_cls(int(value)).name
    except Exception:
        return f"UNKNOWN_{int(value)}"


def _rate(num: int, den: int) -> float:
    return float(num / max(den, 1))


def _write_ids(path: str | None, ids: list[str]) -> None:
    if path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("".join(f"{sid}\n" for sid in ids), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Measure the scene-level proposal ceiling before retraining. The key "
            "lower bound is P(any conventional-safe candidate and no NCF candidate)."
        )
    )
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--hard-scene-ids", default=None)
    ap.add_argument("--hard-count", type=int, default=400, help="Number of hard scenes written for the paired probe; <=0 writes all.")
    ap.add_argument("--control-scene-ids", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--control-count", type=int, default=800)
    ap.add_argument("--random-scene-ids", default=None, help="Representative random scene IDs used for unbiased coverage estimates.")
    ap.add_argument("--random-count", type=int, default=800)
    ap.add_argument("--probe-total-count", type=int, default=0, help="If >0, keep up to --hard-count hard scenes and fill the remainder of this total with representative random scenes. This prevents a probe from failing merely because the old cache contains fewer hard scenes than the preferred target.")
    ap.add_argument("--random-exclude-hard-probe", action="store_true", help="Draw the representative-random set from scenarios not already selected into the hard probe, so a 400+800 strict probe really contains 1200 distinct scenes.")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--subset-modulo", type=int, default=1, help="Match learned-offline index-modulo partitioning.")
    ap.add_argument("--subset-remainder", type=int, default=0)
    args = ap.parse_args()

    ds = COWPNpzDataset(args.cache_dir)
    modulo = max(int(args.subset_modulo), 1)
    remainder = int(args.subset_remainder)
    if remainder < 0 or remainder >= modulo:
        raise ValueError(f"subset_remainder must be in [0, subset_modulo), got {remainder}/{modulo}")
    indices = [i for i in range(len(ds)) if i % modulo == remainder]
    if args.limit > 0:
        indices = indices[: int(args.limit)]
    n = len(indices)
    scene_counts = Counter()
    candidate_counts = Counter()
    macro_stats: dict[str, Counter] = defaultdict(Counter)
    source_stats: dict[str, Counter] = defaultdict(Counter)
    hard_ids: list[str] = []
    other_ids: list[str] = []
    all_ids: list[str] = []
    valid_candidate_counts: list[int] = []

    for idx in indices:
        row = ds.load(idx, WANTED)
        sid = _scenario_id(row, ds.paths[idx].stem)
        all_ids.append(sid)
        valid = np.asarray(row.get("cowp/candidates/valid", []), dtype=bool).reshape(-1)
        if valid.size == 0:
            other_ids.append(sid)
            continue
        conv = np.asarray(row.get("cowp/candidates/conventional_safe", np.zeros_like(valid)), dtype=bool).reshape(-1)[: len(valid)] & valid
        ncf = np.asarray(row.get("cowp/candidates/noncoercive_feasible", np.zeros_like(valid)), dtype=bool).reshape(-1)[: len(valid)] & valid
        fs = np.asarray(row.get("cowp/candidates/false_safe", np.zeros_like(valid)), dtype=bool).reshape(-1)[: len(valid)] & valid
        macro = np.asarray(row.get("cowp/candidates/macro_type", np.full_like(valid, int(MacroType.PAD))), dtype=np.int64).reshape(-1)[: len(valid)]
        source = np.asarray(row.get("cowp/candidates/proposal_source", np.full_like(valid, int(ProposalSource.PAD))), dtype=np.int64).reshape(-1)[: len(valid)]

        cert = np.asarray(row.get("cowp/candidates/certificate_valid", valid), dtype=bool).reshape(-1)[: len(valid)] & valid
        if "cowp/candidates/priority_noncoercive_feasible" in row:
            priority_eligible = np.asarray(row.get("cowp/candidates/priority_eligible", np.zeros_like(valid)), dtype=bool).reshape(-1)[: len(valid)] & valid & cert
            priority_fs = np.asarray(row.get("cowp/candidates/priority_false_safe", np.zeros_like(valid)), dtype=bool).reshape(-1)[: len(valid)] & valid & cert
            priority_ncf = np.asarray(row.get("cowp/candidates/priority_noncoercive_feasible", np.zeros_like(valid)), dtype=bool).reshape(-1)[: len(valid)] & valid & cert
        else:
            crit = np.asarray(row.get("cowp/critical/valid", []), dtype=bool).reshape(-1)
            mech = np.asarray(row.get("cowp/critical/mechanism_valid", crit), dtype=bool).reshape(-1)[: len(crit)] & crit
            witness = np.asarray(row.get("cowp/witness/exists", np.zeros((len(valid), len(crit)), dtype=bool)), dtype=bool)
            rho = np.asarray(row.get("cowp/witness/rho", np.zeros_like(witness, dtype=np.int64)), dtype=np.int64)
            pair_ncf_raw = row.get("cowp/witness/pair_noncoercive_feasible")
            if witness.ndim == 2 and witness.shape[0] >= len(valid) and witness.shape[1] == len(crit) and rho.shape == witness.shape:
                protected = ((rho[: len(valid)] == 2) | (rho[: len(valid)] == 3)) & mech[None, :]
                priority_available = protected.any(axis=1)
                priority_fs = conv & cert & (witness[: len(valid)] & protected).any(axis=1)
                priority_eligible = conv & cert & priority_available
                if pair_ncf_raw is not None:
                    pair_ncf = np.asarray(pair_ncf_raw, dtype=bool)[: len(valid), : len(crit)]
                    priority_ncf = priority_eligible & np.all((~protected) | pair_ncf, axis=1) & ~priority_fs
                else:
                    priority_ncf = priority_eligible & ~priority_fs
            else:
                priority_fs = np.zeros_like(valid)
                priority_ncf = np.zeros_like(valid)
                priority_eligible = np.zeros_like(valid)

        any_valid = bool(valid.any())
        any_conv = bool(conv.any())
        any_ncf = bool(ncf.any())
        any_priority_eligible = bool(priority_eligible.any())
        any_priority_ncf = bool(priority_ncf.any())
        scene_counts.update(
            scenes=1,
            any_valid=int(any_valid),
            any_conventional_safe=int(any_conv),
            any_ncf=int(any_ncf),
            conventional_without_ncf=int(any_conv and not any_ncf),
            any_priority_eligible=int(any_priority_eligible),
            any_priority_ncf=int(any_priority_ncf),
            priority_eligible_without_ncf=int(any_priority_eligible and not any_priority_ncf),
        )
        if any_conv and not any_ncf:
            hard_ids.append(sid)
        else:
            other_ids.append(sid)
        valid_candidate_counts.append(int(valid.sum()))
        candidate_counts.update(
            valid=int(valid.sum()),
            conventional_safe=int(conv.sum()),
            ncf=int(ncf.sum()),
            false_safe=int(fs.sum()),
        )
        for k in np.where(valid)[0]:
            flags = {
                "valid": 1,
                "conventional_safe": int(conv[k]),
                "ncf": int(ncf[k]),
                "false_safe": int(fs[k]),
                "priority_eligible": int(priority_eligible[k]),
                "priority_ncf": int(priority_ncf[k]),
            }
            macro_stats[_name(MacroType, int(macro[k]))].update(flags)
            source_stats[_name(ProposalSource, int(source[k]))].update(flags)

    rng = np.random.default_rng(int(args.seed))
    hard_total_ids = sorted(hard_ids)
    preferred_hard = len(hard_total_ids) if int(args.hard_count) <= 0 else max(int(args.hard_count), 0)
    probe_total = max(int(args.probe_total_count), 0)
    hard_count = min(preferred_hard, len(hard_total_ids))
    if probe_total > 0:
        hard_count = min(hard_count, probe_total)
    hard_probe_ids = sorted(rng.choice(np.asarray(hard_total_ids, dtype=object), size=hard_count, replace=False).tolist()) if hard_count else []
    control_count = min(max(int(args.control_count), 0), len(other_ids))
    control_ids = sorted(rng.choice(np.asarray(other_ids, dtype=object), size=control_count, replace=False).tolist()) if control_count else []
    hard_probe_set = set(hard_probe_ids)
    random_pool = [sid for sid in all_ids if (not args.random_exclude_hard_probe or sid not in hard_probe_set)]
    requested_random = max(int(args.random_count), 0)
    if probe_total > 0:
        requested_random = max(probe_total - len(hard_probe_ids), 0)
    random_count = min(requested_random, len(random_pool))
    random_ids = sorted(rng.choice(np.asarray(random_pool, dtype=object), size=random_count, replace=False).tolist()) if random_count else []
    _write_ids(args.hard_scene_ids, hard_probe_ids)
    _write_ids(args.control_scene_ids, control_ids)
    _write_ids(args.random_scene_ids, random_ids)

    scenes = int(scene_counts["scenes"])
    p_eligible = int(scene_counts["any_priority_eligible"])
    result: dict[str, Any] = {
        "schema_version": "cowp_v16_8_14_proposal_ceiling_v4",
        "cache_dir": str(Path(args.cache_dir).resolve()),
        "num_scenes": scenes,
        "evaluation_subset": {
            "subset_modulo": modulo,
            "subset_remainder": remainder,
            "dataset_size": len(ds),
            "subset_size": len(indices),
        },
        "scene_rates": {
            "any_valid": _rate(scene_counts["any_valid"], scenes),
            "any_conventional_safe": _rate(scene_counts["any_conventional_safe"], scenes),
            "any_ncf": _rate(scene_counts["any_ncf"], scenes),
            "conventional_without_ncf": _rate(scene_counts["conventional_without_ncf"], scenes),
            "best_case_selected_false_safe_lower_bound": _rate(scene_counts["conventional_without_ncf"], scenes),
            "any_priority_eligible": _rate(p_eligible, scenes),
            "any_priority_ncf": _rate(scene_counts["any_priority_ncf"], scenes),
            "priority_eligible_without_ncf": _rate(scene_counts["priority_eligible_without_ncf"], scenes),
            "best_case_pbtr_lower_bound": _rate(scene_counts["priority_eligible_without_ncf"], p_eligible),
        },
        "scene_counts": dict(scene_counts),
        "candidate_counts": dict(candidate_counts),
        "valid_candidates_per_scene": {
            "mean": float(np.mean(valid_candidate_counts)) if valid_candidate_counts else 0.0,
            "p10": float(np.percentile(valid_candidate_counts, 10)) if valid_candidate_counts else 0.0,
            "median": float(np.median(valid_candidate_counts)) if valid_candidate_counts else 0.0,
            "p90": float(np.percentile(valid_candidate_counts, 90)) if valid_candidate_counts else 0.0,
        },
        "macro_stats": {k: dict(v) for k, v in sorted(macro_stats.items())},
        "proposal_source_stats": {k: dict(v) for k, v in sorted(source_stats.items())},
        "hard_scene_total_count": len(hard_total_ids),
        "hard_scene_preferred_count": int(preferred_hard),
        "hard_scene_probe_count": len(hard_probe_ids),
        "probe_total_requested": int(probe_total),
        "probe_total_selected": int(len(hard_probe_ids) + len(random_ids)),
        "hard_scene_shortfall_from_preferred": int(max(preferred_hard - len(hard_probe_ids), 0)),
        "hard_scene_ids_path": str(Path(args.hard_scene_ids).resolve()) if args.hard_scene_ids else None,
        "control_scene_count": len(control_ids),
        "control_scene_ids_path": str(Path(args.control_scene_ids).resolve()) if args.control_scene_ids else None,
        "representative_random_scene_count": len(random_ids),
        "representative_random_scene_ids_path": str(Path(args.random_scene_ids).resolve()) if args.random_scene_ids else None,
        "decision": {
            "full_rebuild_cannot_be_justified_from_old_cache_alone": True,
            "requires_paired_fresh_proposal_probe": True,
            "suggested_promotion_thresholds": {
                "overall_any_ncf_min": 0.40,
                "best_case_selected_false_safe_lower_bound_max": 0.55,
                "hard_scene_ncf_recovery_min": 0.20,
                "best_case_pbtr_lower_bound_max": 0.45,
            },
        },
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
