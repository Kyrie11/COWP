from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from cowp.core.constants import ProposalSource
from cowp.data.dataset import COWPNpzDataset

FINGERPRINT_FILES = (
    "cowp/geometry/lane_graph.py",
    "cowp/label/trajectory_primitives.py",
    "cowp/label/ego_candidates.py",
    "cowp/label/label_engine.py",
    "cowp/data/cache_schema.py",
    "configs/label_cowp_v16_8.yaml",
    "configs/eval_cowp_v16_8.yaml",
)

FRESH_PROVENANCE_KEYS = (
    "cowp/candidates/proposal_source",
    "cowp/candidates/proposal_region_id",
    "cowp/candidates/proposal_target_time_s",
    "cowp/candidates/proposal_timing_side",
    "cowp/candidates/proposal_target_agent_index",
    "cowp/candidates/proposal_gap_s",
    "cowp/candidates/proposal_accel_mps2",
    "cowp/candidates/proposal_entry_distance_m",
    "cowp/candidates/proposal_target_tta_error_s",
)


def current_fingerprint(code_root: Path) -> str:
    h = hashlib.sha256()
    for name in FINGERPRINT_FILES:
        path = code_root / name
        h.update(name.encode("utf-8"))
        h.update(path.read_bytes())
    return h.hexdigest()


def _sample_indices(n: int, limit: int) -> list[int]:
    if limit <= 0 or limit >= n:
        return list(range(n))
    return sorted(set(np.linspace(0, n - 1, num=limit, dtype=np.int64).tolist()))


def _scan(cache_dir: str, sample_scenes: int) -> dict[str, Any]:
    ds = COWPNpzDataset(cache_dir)
    indices = _sample_indices(len(ds), sample_scenes)
    missing = {k: 0 for k in FRESH_PROVENANCE_KEYS}
    read_errors: list[dict[str, str]] = []
    robust_candidates = 0
    finite_timing_errors = 0
    wanted = set(FRESH_PROVENANCE_KEYS)
    wanted.add("cowp/candidates/valid")
    for i in indices:
        try:
            row = ds.load(i, wanted)
        except Exception as exc:
            if len(read_errors) < 20:
                read_errors.append({"file": ds.paths[i].name, "error": repr(exc)})
            continue
        for key in FRESH_PROVENANCE_KEYS:
            if key not in row:
                missing[key] += 1
        src = np.asarray(row.get("cowp/candidates/proposal_source", []), dtype=np.int64).reshape(-1)
        valid = np.asarray(row.get("cowp/candidates/valid", np.ones_like(src, dtype=bool)), dtype=bool).reshape(-1)[: len(src)]
        robust = valid & (src == int(ProposalSource.ROBUST_BCTE))
        robust_candidates += int(robust.sum())
        err = np.asarray(row.get("cowp/candidates/proposal_target_tta_error_s", []), dtype=np.float32).reshape(-1)
        if len(err) >= len(robust):
            finite_timing_errors += int(np.isfinite(err[: len(robust)][robust]).sum())
    summary_path = Path(cache_dir) / "transport_augmentation_summary.json"
    summary: dict[str, Any] = {"exists": summary_path.is_file(), "path": str(summary_path)}
    if summary_path.is_file():
        try:
            summary.update(json.loads(summary_path.read_text(encoding="utf-8")))
        except Exception as exc:
            summary["read_error"] = repr(exc)
    return {
        "cache_dir": str(Path(cache_dir).resolve()),
        "files": len(ds),
        "sample_requested": len(indices),
        "read_errors": read_errors,
        "missing_fresh_provenance_counts": missing,
        "sampled_robust_bcte_candidates": robust_candidates,
        "sampled_robust_bcte_finite_timing_errors": finite_timing_errors,
        "transport_summary": summary,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Strict preflight preventing stale v16.8 overlays from being used as v16.8.6 Priority-Commitment BCS-RMR-BCTE data."
    )
    ap.add_argument("--cowp-root", required=True)
    ap.add_argument("--raw-train", required=True)
    ap.add_argument("--raw-val", required=True)
    ap.add_argument("--transport-train", required=True)
    ap.add_argument("--transport-val", required=True)
    ap.add_argument("--sample-scenes", type=int, default=256)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    code_root = Path(__file__).resolve().parents[2]
    cowp_root = Path(args.cowp_root)
    expected = current_fingerprint(code_root)
    fingerprint_path = cowp_root / "build_fingerprint.sha256"
    manifest_path = cowp_root / "data_manifest_v16_8_6.json"
    stored = fingerprint_path.read_text(encoding="utf-8").strip() if fingerprint_path.is_file() else None
    manifest = None
    manifest_error = None
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            manifest_error = repr(exc)

    train = _scan(args.transport_train, args.sample_scenes)
    val = _scan(args.transport_val, args.sample_scenes)
    reasons: list[str] = []
    if stored is None:
        reasons.append("missing build_fingerprint.sha256: cache predates the v16.8.4 candidate/geometry build contract")
    elif stored != expected:
        reasons.append("build fingerprint does not match the current BCS-RMR-BCTE candidate/geometry implementation")
    if manifest is None:
        reasons.append("missing or unreadable data_manifest_v16_8_6.json")
    else:
        if manifest.get("schema_version") != "cowp_v16_8_6_priority_commitment_data_v1":
            reasons.append(f"unexpected manifest schema_version={manifest.get('schema_version')!r}")
        if manifest.get("build_fingerprint_sha256") != expected:
            reasons.append("manifest build_fingerprint_sha256 does not match current code")
        expected_paths = {
            "raw_train_cache": str(Path(args.raw_train)),
            "raw_val_cache": str(Path(args.raw_val)),
            "transport_train_cache": str(Path(args.transport_train)),
            "transport_val_cache": str(Path(args.transport_val)),
        }
        for key, expected_path in expected_paths.items():
            actual = manifest.get(key)
            if actual is None or Path(actual).resolve() != Path(expected_path).resolve():
                reasons.append(f"manifest {key} does not match requested cache: {actual!r} != {expected_path!r}")
    for split_name, split in (("train", train), ("val", val)):
        if split["read_errors"]:
            reasons.append(f"{split_name}: sampled cache read errors")
        missing = {k: v for k, v in split["missing_fresh_provenance_counts"].items() if v}
        if missing:
            reasons.append(f"{split_name}: sampled scenes are missing v16.8.4 provenance tensors: {missing}")
        meta = split["transport_summary"]
        if not meta.get("exists") or meta.get("read_error"):
            reasons.append(f"{split_name}: transport_augmentation_summary.json missing or unreadable")
        else:
            if str(meta.get("storage_mode", "")) != "overlay":
                reasons.append(f"{split_name}: transport storage_mode is not overlay")
            if str(meta.get("sidecar_subdir", "")) != ".transport_v16_8_6":
                reasons.append(f"{split_name}: sidecar_subdir={meta.get('sidecar_subdir')!r}, expected '.transport_v16_8_6'")
            if not bool(meta.get("complete", False)) or int(meta.get("error_count", 0)) != 0:
                reasons.append(f"{split_name}: transport augmentation is incomplete or has errors")

    report = {
        "schema_version": "cowp_v16_8_6_fresh_cache_protocol_gate_v1",
        "pass": not reasons,
        "cowp_root": str(cowp_root.resolve()),
        "current_build_fingerprint_sha256": expected,
        "stored_build_fingerprint_sha256": stored,
        "manifest_path": str(manifest_path),
        "manifest_error": manifest_error,
        "train": train,
        "val": val,
        "reasons": reasons,
        "interpretation": (
            "Fresh v16.8.6 cache identity/provenance passed; training may use these caches."
            if not reasons else
            "Do not train v16.8.6 on these caches. In particular, a transport overlay cannot retrofit a new ego proposal bank into stale raw cache files."
        ),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if reasons:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
