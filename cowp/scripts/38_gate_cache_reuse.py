from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from cowp.data.dataset import COWPNpzDataset


V9_REQUIRED = (
    "state/is_sdc",
    "cowp/critical/valid",
    "cowp/critical/input_index",
    "cowp/natural/traj",
    "cowp/natural/valid",
    "cowp/natural/weight",
    "cowp/natural/source",
    "cowp/response/valid",
    "cowp/witness/exists",
    "cowp/candidates/noncoercive_feasible",
    "cowp/transport/mode_valid",
    "cowp/transport/mode_conflict",
    "cowp/transport/response_root_index",
)

V15_LABEL_KEYS = (
    "cowp/natural/obs_contamination",
    "cowp/natural/map_compliant",
    "cowp/natural/map_distance_max",
    "cowp/natural/map_verified",
)


def _sample_indices(n: int, limit: int) -> list[int]:
    if limit <= 0 or limit >= n:
        return list(range(n))
    return sorted(set(np.linspace(0, n - 1, num=limit, dtype=np.int64).tolist()))


def _summary(cache_dir: Path) -> dict[str, Any]:
    path = cache_dir / "transport_augmentation_summary.json"
    if not path.is_file():
        return {"exists": False, "path": str(path)}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        data["exists"] = True
        data["path"] = str(path)
        return data
    except Exception as exc:
        return {"exists": True, "path": str(path), "read_error": repr(exc)}


def _scan_split(raw_dir: str, overlay_dir: str, sample_scenes: int) -> dict[str, Any]:
    raw = COWPNpzDataset(raw_dir)
    overlay = COWPNpzDataset(overlay_dir)
    overlay_by_name = {p.name: i for i, p in enumerate(overlay.paths)}
    indices = _sample_indices(len(raw), sample_scenes)

    missing_overlay = 0
    read_errors: list[dict[str, str]] = []
    missing_counts = {k: 0 for k in V9_REQUIRED}
    v15_counts = {k: 0 for k in V15_LABEL_KEYS}
    critical_valid = critical_bad = response_slots = root_bad = 0
    logdiv_finite = rollout_valid_count = 0

    wanted = {
        "state/is_sdc",
        "womd/state/is_sdc",
        "cowp/critical/",
        "cowp/natural/",
        "cowp/response/valid",
        "cowp/witness/exists",
        "cowp/candidates/noncoercive_feasible",
        "cowp/transport/",
        "waymax/candidate_rollout_valid",
        "waymax/candidate_log_divergence",
    }

    scanned = 0
    for ri in indices:
        name = raw.paths[ri].name
        oi = overlay_by_name.get(name)
        if oi is None:
            missing_overlay += 1
            continue
        try:
            d = overlay.load(oi, wanted)
        except Exception as exc:
            if len(read_errors) < 20:
                read_errors.append({"file": name, "error": repr(exc)})
            continue
        scanned += 1
        for key in V9_REQUIRED:
            if key not in d:
                missing_counts[key] += 1
        for key in V15_LABEL_KEYS:
            if key in d:
                v15_counts[key] += 1

        cv = np.asarray(d.get("cowp/critical/valid", []), dtype=bool).reshape(-1)
        ci = np.asarray(d.get("cowp/critical/input_index", []), dtype=np.int64).reshape(-1)
        n = min(len(cv), len(ci))
        critical_valid += int(cv[:n].sum())
        critical_bad += int((cv[:n] & (ci[:n] < 0)).sum())

        roots = np.asarray(d.get("cowp/transport/response_root_index", []), dtype=np.int64)
        rv = np.asarray(d.get("cowp/response/valid", []), dtype=bool)
        nat = np.asarray(d.get("cowp/natural/valid", []), dtype=bool)
        if roots.shape == rv.shape and roots.ndim == 3 and nat.ndim == 2:
            response_slots += int(rv.sum())
            m = max(int(nat.shape[-1]), 1)
            root_bad += int((rv & ((roots < 0) | (roots >= m))).sum())

        rollout = np.asarray(d.get("waymax/candidate_rollout_valid", []), dtype=bool)
        logdiv = np.asarray(d.get("waymax/candidate_log_divergence", []), dtype=np.float32)
        if rollout.shape == logdiv.shape and rollout.size:
            rollout_valid_count += int(rollout.sum())
            logdiv_finite += int((rollout & np.isfinite(logdiv)).sum())

    summary = _summary(Path(overlay_dir))
    return {
        "raw_dir": raw_dir,
        "overlay_dir": overlay_dir,
        "raw_files": len(raw),
        "overlay_files": len(overlay),
        "sample_requested": len(indices),
        "sample_scanned": scanned,
        "missing_overlay_files_in_sample": missing_overlay,
        "read_errors": read_errors,
        "missing_required_key_counts": missing_counts,
        "v15_label_key_counts": v15_counts,
        "critical_valid": critical_valid,
        "critical_unmapped": critical_bad,
        "critical_unmapped_rate": critical_bad / max(critical_valid, 1),
        "response_valid_slots": response_slots,
        "response_root_out_of_range": root_bad,
        "response_root_out_of_range_rate": root_bad / max(response_slots, 1),
        "rollout_valid_count": rollout_valid_count,
        "finite_logdiv_count": logdiv_finite,
        "finite_logdiv_rate": logdiv_finite / max(rollout_valid_count, 1),
        "overlay_summary": summary,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Gate reuse of existing v9 transport caches for v15 model training.")
    ap.add_argument("--raw-train", required=True)
    ap.add_argument("--raw-val", required=True)
    ap.add_argument("--transport-train", required=True)
    ap.add_argument("--transport-val", required=True)
    ap.add_argument("--sample-scenes", type=int, default=512)
    ap.add_argument("--min-train-scenes", type=int, default=20000)
    ap.add_argument("--min-val-scenes", type=int, default=5000)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    train = _scan_split(args.raw_train, args.transport_train, args.sample_scenes)
    val = _scan_split(args.raw_val, args.transport_val, args.sample_scenes)

    overlap = len({p.name for p in COWPNpzDataset(args.raw_train).paths} & {p.name for p in COWPNpzDataset(args.raw_val).paths})

    reasons: list[str] = []
    for split_name, split, minimum in (("train", train, args.min_train_scenes), ("val", val, args.min_val_scenes)):
        if split["raw_files"] < minimum:
            reasons.append(f"{split_name}: raw scene count {split['raw_files']} < required {minimum}")
        if split["raw_files"] != split["overlay_files"]:
            reasons.append(f"{split_name}: raw/transport file count mismatch {split['raw_files']} != {split['overlay_files']}")
        if split["missing_overlay_files_in_sample"]:
            reasons.append(f"{split_name}: sampled files missing from transport overlay")
        if split["read_errors"]:
            reasons.append(f"{split_name}: sampled overlay files have read errors")
        missing = {k: v for k, v in split["missing_required_key_counts"].items() if v}
        if missing:
            reasons.append(f"{split_name}: required v9/model keys missing in sample: {missing}")
        if split["critical_unmapped_rate"] > 0.02:
            reasons.append(f"{split_name}: critical unmapped rate {split['critical_unmapped_rate']:.4f} > 0.02")
        if split["response_root_out_of_range_rate"] > 1e-4:
            reasons.append(f"{split_name}: response root out-of-range rate {split['response_root_out_of_range_rate']:.6f} > 1e-4")
        meta = split["overlay_summary"]
        if not meta.get("exists") or meta.get("read_error"):
            reasons.append(f"{split_name}: transport_augmentation_summary.json missing or unreadable")
        elif not bool(meta.get("complete", True)) or int(meta.get("error_count", 0)) != 0:
            reasons.append(f"{split_name}: transport overlay summary is incomplete")

    if overlap:
        reasons.append(f"train/val filename overlap={overlap}")

    v9_pass = not reasons
    v15_label_materialized = all(
        split["sample_scanned"] > 0 and all(split["v15_label_key_counts"][k] == split["sample_scanned"] for k in V15_LABEL_KEYS)
        for split in (train, val)
    )

    report = {
        "train": train,
        "val": val,
        "cross_split_filename_overlap": overlap,
        "decisions": {
            "reuse_for_v14_or_v15_model_with_v9_labels": {
                "pass": v9_pass,
                "reasons": reasons,
                "meaning": "Existing caches can train the v15 model and support real online Waymax evaluation, but labels remain the v9 protocol.",
            },
            "reuse_as_true_v15_causal_label_dataset": {
                "pass": bool(v9_pass and v15_label_materialized),
                "reasons": [] if v15_label_materialized else [
                    "v15 OBS contamination/map-compliance tensors are not materialized in the sampled cache; v9 natural/response/witness labels cannot validate the new label-generation claim."
                ],
            },
            "logdiv_supervision": {
                "pass": False,
                "reasons": ["Existing safety replay contains no finite log-divergence targets; keep outcome_logdiv and logdiv-based selection/reporting disabled."],
            },
        },
        "recommended_mode": "reuse_v9_for_next_model_run" if v9_pass else "repair_or_rebuild_cache_before_training",
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not v9_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
