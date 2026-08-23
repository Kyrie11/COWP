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
    "cowp/core/types.py",
    "cowp/data/parse_scenario_proto.py",
    "cowp/geometry/lane_graph.py",
    "cowp/label/trajectory_primitives.py",
    "cowp/label/ego_candidates.py",
    "cowp/label/label_engine.py",
    "cowp/label/audit_relevance.py",
    "cowp/label/critical_agents.py",
    "cowp/label/natural_alternatives.py",
    "cowp/label/safe_budget_search.py",
    "cowp/label/safe_responses.py",
    "cowp/label/witness.py",
    "cowp/label/burden.py",
    "cowp/label/priority.py",
    "cowp/data/cache_schema.py",
    "cowp/data/dataset.py",
    "cowp/data/validation.py",
    "cowp/data/build_cache.py",
    "cowp/scripts/02_build_tensor_cache.py",
    "cowp/scripts/03_train.py",
    "cowp/scripts/09_build_waymax_rollout_dataset.py",
    "cowp/scripts/45_diagnose_proposal_ceiling.py",
    "cowp/scripts/57_diagnose_causal_audit.py",
    "cowp/scripts/58_screen_v16_8_9_causal_audit_probe.py",
    "cowp/scripts/60_verify_fresh_v16_8_9_cache.py",
    "cowp/scripts/62_audit_training_supervision.py",
    "cowp/scripts/63_validate_probe_manifest.py",
    "cowp/scripts/64_validate_womd_v131_contract.py",
    "cowp/scripts/65_audit_model_support.py",
    "cowp/scripts/69_audit_womd_split_layout.py",
    "cowp/scripts/74_audit_mechanism_contrast.py",
    "cowp/scripts/75_reaudit_v16_8_21_train_pilot.py",
    "cowp/scripts/76_reaudit_v16_8_21_strict_for_v16_8_22.py",
    "cowp/scripts/19_diagnose_waymax_cache_sufficiency.py",
    "configs/label_cowp_v16_8.yaml",
    "configs/label_cowp_v16_8_25_mcfc.yaml",
    "configs/model_cowp_v16_8.yaml",
    "configs/train_cowp_v16_8.yaml",
    "cowp/models/set_transport_head.py",
    "cowp/models/cowp_model.py",
    "cowp/models/losses.py",
    "cowp/waymax_eval/candidate_replay.py",
    "cowp/waymax_eval/metrics_cowp.py",
    "cowp/waymax_eval/rollout.py",
    "configs/eval_cowp_v16_8.yaml",
)

# Label-producing semantics are intentionally fingerprinted separately from the
# promotion/audit policy.  v16.8.16 used one monolithic fingerprint that also
# included scripts such as 65_audit_model_support.py; changing a statistical
# promotion threshold therefore forced an expensive label rebuild even when not
# one label tensor could change.  The semantic fingerprint below covers the
# actual Scenario->label transformation (including geometry and its configs) and
# permits a policy-only re-audit without weakening provenance.
LABEL_SEMANTIC_GLOBS = (
    "cowp/label/*.py",
    "cowp/geometry/*.py",
)
LABEL_SEMANTIC_FILES = (
    "cowp/core/constants.py",
    "cowp/core/types.py",
    "cowp/data/parse_scenario_proto.py",
    "configs/data.yaml",
    "configs/label_cowp_v16_8.yaml",
    "configs/label_cowp_v16_8_25_mcfc.yaml",
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


AUDIT_KEYS = (
    "cowp/audit/pair_relevant",
    "cowp/audit/relevance_mass",
    "cowp/audit/root_affected",
    "cowp/audit/root_unsafe",
    "cowp/audit/root_event_interval",
    "cowp/audit/root_direct_burden",
    "cowp/audit/root_budget_crossed",
    "cowp/audit/root_burden_only_affected",
    "cowp/audit/canonical_root_weight",
    "cowp/witness/pair_noncoercive_feasible",
    "cowp/witness/blocker_code",
    "cowp/candidates/audited_pair_count",
    "cowp/candidates/ncf_blocker_count",
)

INLINE_TRANSPORT_KEYS = (
    "cowp/transport/mode_valid",
    "cowp/transport/mode_conflict",
    "cowp/transport/mode_affected",
    "cowp/transport/mode_retained_low_safe",
    "cowp/transport/response_root_index",
    "cowp/transport/response_is_min_burden",
    "cowp/transport/root_recovery_mass",
    "cowp/transport/root_low_safe_score",
    "cowp/transport/root_target_confidence",
    "cowp/transport/root_min_safe_burden",
    "cowp/transport/transported_opr",
    "cowp/transport/canonical_root_weight",
)


def current_fingerprint(code_root: Path) -> str:
    h = hashlib.sha256()
    for name in FINGERPRINT_FILES:
        path = code_root / name
        h.update(name.encode("utf-8"))
        h.update(path.read_bytes())
    return h.hexdigest()


def current_label_semantic_fingerprint(code_root: Path) -> str:
    """Hash only files that can change Scenario-proto label tensors.

    This is deliberately narrower than :func:`current_fingerprint`.  It is used
    only to authorize reuse of an already-built label cache across a *policy*
    revision.  Training/model/tensor-cache code retains the broader fingerprint
    and full rebuild promotion still requires the current full fingerprint.
    """
    root = Path(code_root)
    paths = {root / name for name in LABEL_SEMANTIC_FILES}
    for pattern in LABEL_SEMANTIC_GLOBS:
        paths.update(root.glob(pattern))
    h = hashlib.sha256()
    for path in sorted(paths, key=lambda p: p.relative_to(root).as_posix()):
        rel = path.relative_to(root).as_posix()
        h.update(rel.encode("utf-8"))
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
    missing_transport = {k: 0 for k in INLINE_TRANSPORT_KEYS}
    missing_audit = {k: 0 for k in AUDIT_KEYS}
    read_errors: list[dict[str, str]] = []
    robust_candidates = 0
    finite_timing_errors = 0
    wanted = set(FRESH_PROVENANCE_KEYS)
    wanted.update(INLINE_TRANSPORT_KEYS)
    wanted.update(AUDIT_KEYS)
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
        for key in INLINE_TRANSPORT_KEYS:
            if key not in row:
                missing_transport[key] += 1
        for key in AUDIT_KEYS:
            if key not in row:
                missing_audit[key] += 1
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
        "missing_inline_transport_counts": missing_transport,
        "missing_audit_counts": missing_audit,
        "causal_audit_complete_in_sample": not any(missing_audit.values()),
        "inline_transport_complete_in_sample": not any(missing_transport.values()),
        "sampled_robust_bcte_candidates": robust_candidates,
        "sampled_robust_bcte_finite_timing_errors": finite_timing_errors,
        "transport_summary": summary,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Strict preflight for v16.8.9 candidate-conditioned causal-audit data."
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
    manifest_path = cowp_root / "data_manifest_v16_8_9.json"
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
        reasons.append("missing or unreadable data_manifest_v16_8_9.json")
    else:
        schema = manifest.get("schema_version")
        if schema != "cowp_v16_8_9_causal_audit_self_contained_data_v1":
            reasons.append(f"unexpected manifest schema_version={schema!r}")
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
            reasons.append(f"{split_name}: sampled scenes are missing proposal provenance tensors: {missing}")
        missing_audit = {k: v for k, v in split["missing_audit_counts"].items() if v}
        if missing_audit:
            reasons.append(f"{split_name}: sampled scenes are missing v16.8.9 causal-audit tensors: {missing_audit}")
        meta = split["transport_summary"]
        inline_ok = bool(split.get("inline_transport_complete_in_sample", False))
        if inline_ok:
            # Preferred v16.8.7 engineering contract: transport supervision is
            # serialized in each fresh NPZ. No overlay/backing-cache dependency.
            pass
        elif not meta.get("exists") or meta.get("read_error"):
            reasons.append(f"{split_name}: neither complete inline transport tensors nor a readable transport overlay are present")
        else:
            if str(meta.get("storage_mode", "")) != "overlay":
                reasons.append(f"{split_name}: transport storage_mode is neither inline nor overlay")
            if str(meta.get("sidecar_subdir", "")) not in {".transport_v16_8_6", ".transport_v16_8_4"}:
                reasons.append(f"{split_name}: unexpected sidecar_subdir={meta.get('sidecar_subdir')!r}")
            if not bool(meta.get("complete", False)) or int(meta.get("error_count", 0)) != 0:
                reasons.append(f"{split_name}: transport augmentation is incomplete or has errors")

    report = {
        "schema_version": "cowp_v16_8_9_fresh_cache_protocol_gate_v1",
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
            "Fresh v16.8.9 cache identity, causal-audit supervision, and self-contained transport passed."
            if not reasons else
            "Do not train v16.8.9 on these caches. Missing causal-audit/affected-root labels cannot be retrofitted from stale tensor caches."
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
