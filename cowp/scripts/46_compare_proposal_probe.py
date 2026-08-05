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
}


def _sid(row: dict[str, np.ndarray], fallback: str) -> str:
    value = np.asarray(row.get("scenario/id", fallback))
    try:
        return str(value.item()) if value.size == 1 else str(value.reshape(-1)[0])
    except Exception:
        return fallback


def _summarize(row: dict[str, np.ndarray]) -> dict[str, float | bool | int]:
    valid = np.asarray(row.get("cowp/candidates/valid", []), dtype=bool).reshape(-1)
    conv = np.asarray(row.get("cowp/candidates/conventional_safe", np.zeros_like(valid)), dtype=bool).reshape(-1)[: len(valid)] & valid
    ncf = np.asarray(row.get("cowp/candidates/noncoercive_feasible", np.zeros_like(valid)), dtype=bool).reshape(-1)[: len(valid)] & valid
    source = np.asarray(row.get("cowp/candidates/proposal_source", np.full_like(valid, int(ProposalSource.PAD))), dtype=np.int64).reshape(-1)[: len(valid)]
    rmr = valid & (source == int(ProposalSource.ROBUST_BCTE))
    return {
        "valid_count": int(valid.sum()),
        "any_conv": bool(conv.any()),
        "any_ncf": bool(ncf.any()),
        "conv_without_ncf": bool(conv.any() and not ncf.any()),
        "rmr_count": int(rmr.sum()),
        "rmr_ncf_count": int((rmr & ncf).sum()),
    }


def _load(cache_dir: str, limit: int) -> dict[str, dict[str, float | bool | int]]:
    ds = COWPNpzDataset(cache_dir)
    n = len(ds) if limit <= 0 else min(len(ds), limit)
    out = {}
    for i in range(n):
        row = ds.load(i, WANTED)
        out[_sid(row, ds.paths[i].stem)] = _summarize(row)
    return out


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
            old_any_ncf=int(bool(a["any_ncf"])),
            new_any_ncf=int(bool(b["any_ncf"])),
            old_floor=int(bool(a["conv_without_ncf"])),
            new_floor=int(bool(b["conv_without_ncf"])),
            old_hard=int(bool(a["conv_without_ncf"])),
            hard_recovered=int(bool(a["conv_without_ncf"] and b["any_ncf"])),
            ncf_lost=int(bool(a["any_ncf"] and not b["any_ncf"])),
            new_rmr_candidates=int(b["rmr_count"]),
            new_rmr_ncf_candidates=int(b["rmr_ncf_count"]),
            new_scene_with_rmr=int(int(b["rmr_count"]) > 0),
            new_scene_with_rmr_ncf=int(int(b["rmr_ncf_count"]) > 0),
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
    ap.add_argument("--min-overall-any-ncf", type=float, default=0.40)
    ap.add_argument("--max-false-safe-floor", type=float, default=0.55)
    ap.add_argument("--min-hard-recovery", type=float, default=0.20)
    ap.add_argument("--allow-missing-requested", action="store_true", help="Diagnostic-only: allow requested IDs to be absent. Promotion probes should keep the default strict behavior.")
    args = ap.parse_args()

    old = _load(args.old_cache, int(args.limit))
    new = _load(args.new_cache, int(args.limit))
    common = set(old).intersection(new)
    if not common:
        raise ValueError("No common scenario IDs between old and new caches")
    representative_requested = _read_ids(args.representative_scene_ids)
    hard_requested = _read_ids(args.hard_scene_ids)
    requested_union = (representative_requested or set()).union(hard_requested or set())
    missing_from_old = sorted(requested_union.difference(old))
    missing_from_new = sorted(requested_union.difference(new))
    if (missing_from_old or missing_from_new) and not args.allow_missing_requested:
        raise ValueError(
            "Paired probe is incomplete: "
            f"missing_from_old={len(missing_from_old)} examples={missing_from_old[:5]}, "
            f"missing_from_new={len(missing_from_new)} examples={missing_from_new[:5]}"
        )
    representative_ids = sorted(common if representative_requested is None else common.intersection(representative_requested))
    if not representative_ids:
        raise ValueError("No representative scenario IDs are common to both inputs")
    hard_ids = sorted(
        [sid for sid in common if bool(old[sid]["conv_without_ncf"])]
        if hard_requested is None
        else common.intersection(hard_requested)
    )
    c, old_valid, new_valid = _aggregate(representative_ids, old, new)
    hard_c, _, _ = _aggregate(hard_ids, old, new) if hard_ids else (Counter(), [], [])
    n = int(c["pairs"])
    new_any_ncf = _rate(c["new_any_ncf"], n)
    new_floor = _rate(c["new_floor"], n)
    hard_recovery = _rate(hard_c["hard_recovered"], hard_c["old_hard"])
    gates = {
        "overall_any_ncf": new_any_ncf >= args.min_overall_any_ncf,
        "false_safe_floor": new_floor <= args.max_false_safe_floor,
        "hard_scene_recovery": hard_recovery >= args.min_hard_recovery,
    }
    result = {
        "schema_version": "cowp_v16_8_3_paired_proposal_probe_v1",
        "old_cache": str(Path(args.old_cache).resolve()),
        "new_cache": str(Path(args.new_cache).resolve()),
        "num_common_scenes": len(common),
        "num_representative_scenes": n,
        "pairing_completeness": {
            "requested_scene_count": len(requested_union),
            "missing_from_old_count": len(missing_from_old),
            "missing_from_new_count": len(missing_from_new),
            "complete": not missing_from_old and not missing_from_new,
        },
        "old": {
            "any_ncf_scene_rate": _rate(c["old_any_ncf"], n),
            "best_case_selected_false_safe_lower_bound": _rate(c["old_floor"], n),
            "mean_valid_candidates": float(np.mean(old_valid)),
        },
        "new": {
            "any_ncf_scene_rate": new_any_ncf,
            "best_case_selected_false_safe_lower_bound": new_floor,
            "mean_valid_candidates": float(np.mean(new_valid)),
            "scene_with_rmr_bcte_rate": _rate(c["new_scene_with_rmr"], n),
            "scene_with_rmr_bcte_ncf_rate": _rate(c["new_scene_with_rmr_ncf"], n),
            "rmr_bcte_candidate_count": int(c["new_rmr_candidates"]),
            "rmr_bcte_ncf_candidate_count": int(c["new_rmr_ncf_candidates"]),
        },
        "paired": {
            "old_hard_scene_count": int(hard_c["old_hard"]),
            "hard_scene_ncf_recovery_rate": hard_recovery,
            "ncf_loss_rate": _rate(c["ncf_lost"], n),
            "mean_valid_candidate_gain": float(np.mean(new_valid) - np.mean(old_valid)),
        },
        "promotion_thresholds": {
            "min_overall_any_ncf": args.min_overall_any_ncf,
            "max_false_safe_floor": args.max_false_safe_floor,
            "min_hard_recovery": args.min_hard_recovery,
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
