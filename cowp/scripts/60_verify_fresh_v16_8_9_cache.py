from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from cowp.core.constants import ProposalSource

REQUIRED_KEYS = (
    "scenario/id",
    "cowp/candidates/valid",
    "cowp/candidates/conventional_safe",
    "cowp/candidates/noncoercive_feasible",
    "cowp/candidates/proposal_source",
    "cowp/candidates/proposal_region_id",
    "cowp/candidates/proposal_target_time_s",
    "cowp/candidates/proposal_timing_side",
    "cowp/candidates/proposal_target_agent_index",
    "cowp/candidates/proposal_gap_s",
    "cowp/candidates/proposal_accel_mps2",
    "cowp/candidates/proposal_entry_distance_m",
    "cowp/candidates/proposal_target_tta_error_s",
    "cowp/candidates/audited_pair_count",
    "cowp/candidates/ncf_blocker_count",
    "cowp/critical/valid",
    "cowp/audit/pair_relevant",
    "cowp/audit/relevance_mass",
    "cowp/audit/root_affected",
    "cowp/audit/root_unsafe",
    "cowp/audit/root_direct_burden",
    "cowp/audit/root_budget_crossed",
    "cowp/audit/root_burden_only_affected",
    "cowp/audit/canonical_root_weight",
    "cowp/witness/exists",
    "cowp/witness/pair_noncoercive_feasible",
    "cowp/witness/blocker_code",
    "cowp/witness/rho",
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

WAYMAX_READY_ANY = (
    "womd/state/is_sdc",
    "state/is_sdc",
)


def _restore(k: str) -> str:
    return k.replace("__", "/")


def _read_allowlist(path: str | None) -> set[str] | None:
    if not path:
        return None
    return {x.strip().split()[0] for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()}


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify fresh v16.8.9 causal-audit self-contained cache before training/Waymax experiments.")
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--allowlist", default=None)
    ap.add_argument("--sample-scenes", type=int, default=0, help="0 verifies every scenario file.")
    ap.add_argument("--require-sdc-paths", action="store_true", help="Require every inspected tensor-cache item to carry a positive WOMD-1.3.1 SDC-path contract marker.")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    root = Path(args.cache_dir)
    paths = sorted(p for p in root.glob("*.npz") if not p.name.startswith("."))
    if not paths:
        raise FileNotFoundError(f"no NPZ files in {root}")
    broken = [p for p in paths if p.is_symlink() and not p.exists()]
    non_regular = [p for p in paths if p.is_symlink()]
    allow = _read_allowlist(args.allowlist)
    stems = {p.stem for p in paths if p.exists()}
    missing_allow = sorted((allow or set()) - stems)
    extra_allow = sorted(stems - (allow or stems)) if allow is not None else []

    inspect = paths
    if args.sample_scenes > 0 and len(paths) > args.sample_scenes:
        idx = np.linspace(0, len(paths) - 1, num=int(args.sample_scenes), dtype=np.int64)
        inspect = [paths[int(i)] for i in idx]

    missing = Counter()
    read_errors: list[dict[str, str]] = []
    sid_mismatch: list[dict[str, str]] = []
    source_counts = Counter()
    ncf_scenes = valid_scenes = 0
    silent_blockers = irrelevant_blockers = affected_mismatch = 0
    conflict_mismatch = retain_mismatch = canonical_weight_mismatch = 0
    affected_definition_error = burden_only_definition_error = 0
    sdc_paths_missing = sdc_paths_not_ready = 0
    for p in inspect:
        if not p.exists():
            continue
        try:
            with np.load(p, allow_pickle=True) as z:
                files = {_restore(k): k for k in z.files}
                for key in REQUIRED_KEYS:
                    if key not in files:
                        missing[key] += 1
                if not any(k in files for k in WAYMAX_READY_ANY):
                    missing["waymax_ready:is_sdc"] += 1
                if args.require_sdc_paths:
                    sdc_key = files.get("cache/meta/sdc_paths_ready")
                    if sdc_key is None:
                        sdc_paths_missing += 1
                    else:
                        ready_arr = np.asarray(z[sdc_key]).reshape(-1)
                        if ready_arr.size != 1 or not bool(ready_arr[0]):
                            sdc_paths_not_ready += 1
                sid_key = files.get("scenario/id")
                if sid_key is not None:
                    arr = np.asarray(z[sid_key])
                    sid = str(arr.item()) if arr.size == 1 else str(arr.reshape(-1)[0])
                    if sid != p.stem:
                        sid_mismatch.append({"file": p.name, "scenario_id": sid})
                vkey = files.get("cowp/candidates/valid")
                skey = files.get("cowp/candidates/proposal_source")
                nkey = files.get("cowp/candidates/noncoercive_feasible")
                if vkey is not None:
                    valid = np.asarray(z[vkey], dtype=bool).reshape(-1)
                    valid_scenes += int(valid.any())
                    if skey is not None:
                        src = np.asarray(z[skey], dtype=np.int64).reshape(-1)[: len(valid)]
                        for value, count in zip(*np.unique(src[valid], return_counts=True)):
                            try:
                                name = ProposalSource(int(value)).name
                            except Exception:
                                name = f"UNKNOWN_{int(value)}"
                            source_counts[name] += int(count)
                    if nkey is not None:
                        ncf = np.asarray(z[nkey], dtype=bool).reshape(-1)[: len(valid)] & valid
                        ncf_scenes += int(ncf.any())
                # v16.8.9 semantic integrity: irrelevant pairs are vacuously
                # noncoercive and every relevant non-NCF pair must carry a witness.
                req = {k: files.get(k) for k in (
                    "cowp/candidates/valid", "cowp/critical/valid", "cowp/audit/pair_relevant",
                    "cowp/witness/exists", "cowp/witness/pair_noncoercive_feasible",
                    "cowp/audit/root_affected", "cowp/audit/root_unsafe",
                    "cowp/audit/root_budget_crossed", "cowp/audit/root_burden_only_affected",
                    "cowp/audit/canonical_root_weight",
                    "cowp/transport/mode_valid", "cowp/transport/mode_conflict",
                    "cowp/transport/mode_affected", "cowp/transport/mode_retained_low_safe",
                    "cowp/transport/canonical_root_weight",
                )}
                if all(v is not None for v in req.values()):
                    cv = np.asarray(z[req["cowp/candidates/valid"]], dtype=bool)
                    av = np.asarray(z[req["cowp/critical/valid"]], dtype=bool)
                    base = cv[:, None] & av[None, :]
                    rel = np.asarray(z[req["cowp/audit/pair_relevant"]], dtype=bool)
                    ex = np.asarray(z[req["cowp/witness/exists"]], dtype=bool)
                    pn = np.asarray(z[req["cowp/witness/pair_noncoercive_feasible"]], dtype=bool)
                    silent_blockers += int((base & rel & ~pn & ~ex).sum())
                    irrelevant_blockers += int((base & ~rel & ~pn).sum())
                    ra = np.asarray(z[req["cowp/audit/root_affected"]], dtype=bool)
                    ru = np.asarray(z[req["cowp/audit/root_unsafe"]], dtype=bool)
                    rb = np.asarray(z[req["cowp/audit/root_budget_crossed"]], dtype=bool)
                    rbo = np.asarray(z[req["cowp/audit/root_burden_only_affected"]], dtype=bool)
                    tv = np.asarray(z[req["cowp/transport/mode_valid"]], dtype=bool)
                    tc = np.asarray(z[req["cowp/transport/mode_conflict"]], dtype=bool)
                    ta = np.asarray(z[req["cowp/transport/mode_affected"]], dtype=bool)
                    tr = np.asarray(z[req["cowp/transport/mode_retained_low_safe"]], dtype=bool)
                    aw = np.asarray(z[req["cowp/audit/canonical_root_weight"]], dtype=np.float32)
                    tw = np.asarray(z[req["cowp/transport/canonical_root_weight"]], dtype=np.float32)
                    if ra.shape == ta.shape:
                        affected_mismatch += int(np.logical_xor(ra, ta).sum())
                    else:
                        affected_mismatch += 1
                    if ru.shape == tc.shape:
                        conflict_mismatch += int(np.logical_xor(ru, tc).sum())
                    else:
                        conflict_mismatch += 1
                    if ra.shape == tv.shape == tr.shape:
                        retain_mismatch += int(np.logical_xor(tr, tv & ~ra).sum())
                    else:
                        retain_mismatch += 1
                    if aw.shape == tw.shape:
                        canonical_weight_mismatch += int(np.sum(np.abs(aw - tw) > 1.0e-6))
                    else:
                        canonical_weight_mismatch += 1
                    if ra.shape == ru.shape == rb.shape:
                        affected_definition_error += int(np.logical_xor(ra, ru | rb).sum())
                    else:
                        affected_definition_error += 1
                    if rbo.shape == ru.shape == rb.shape:
                        burden_only_definition_error += int(np.logical_xor(rbo, rb & ~ru).sum())
                    else:
                        burden_only_definition_error += 1
        except Exception as exc:
            read_errors.append({"file": p.name, "error": repr(exc)})
            if len(read_errors) >= 20:
                break

    reasons: list[str] = []
    if broken:
        reasons.append(f"broken symlinks={len(broken)}")
    if non_regular:
        reasons.append(f"fresh cache must be self-contained regular NPZ files; symlinks={len(non_regular)}")
    if missing_allow or extra_allow:
        reasons.append(f"allowlist mismatch missing={len(missing_allow)} extra={len(extra_allow)}")
    if missing:
        reasons.append(f"required key omissions in inspected scenes: {dict(missing)}")
    if read_errors:
        reasons.append(f"read errors={len(read_errors)}")
    if sid_mismatch:
        reasons.append(f"scenario-id/filename mismatch={len(sid_mismatch)}")
    if valid_scenes < len(inspect):
        reasons.append(f"zero-valid scenes in cache sample={len(inspect)-valid_scenes}")
    if silent_blockers:
        reasons.append(f"silent relevant blockers={silent_blockers}")
    if irrelevant_blockers:
        reasons.append(f"irrelevant pair blockers={irrelevant_blockers}")
    if affected_mismatch:
        reasons.append(f"audit/transport affected-root mismatches={affected_mismatch}")
    if conflict_mismatch:
        reasons.append(f"audit/transport conflict-root mismatches={conflict_mismatch}")
    if retain_mismatch:
        reasons.append(f"transport retained-root definition mismatches={retain_mismatch}")
    if canonical_weight_mismatch:
        reasons.append(f"audit/transport canonical-root-weight mismatches={canonical_weight_mismatch}")
    if affected_definition_error:
        reasons.append(f"affected-root definition mismatches={affected_definition_error}")
    if burden_only_definition_error:
        reasons.append(f"burden-only definition mismatches={burden_only_definition_error}")
    if args.require_sdc_paths and sdc_paths_missing:
        reasons.append(f"missing cache/meta/sdc_paths_ready={sdc_paths_missing}")
    if args.require_sdc_paths and sdc_paths_not_ready:
        reasons.append(f"SDC-path contract not ready={sdc_paths_not_ready}")

    result = {
        "schema_version": "cowp_v16_8_9_self_contained_cache_integrity_v2",
        "pass": not reasons,
        "cache_dir": str(root.resolve()),
        "files": len(paths),
        "inspected": len(inspect),
        "allowlist_size": len(allow) if allow is not None else None,
        "missing_allowlist_count": len(missing_allow),
        "extra_allowlist_count": len(extra_allow),
        "broken_symlink_count": len(broken),
        "symlink_count": len(non_regular),
        "missing_required_key_counts": dict(missing),
        "read_errors": read_errors,
        "scenario_id_mismatch_examples": sid_mismatch[:20],
        "valid_scene_rate_inspected": valid_scenes / max(len(inspect), 1),
        "any_ncf_scene_rate_inspected": ncf_scenes / max(len(inspect), 1),
        "proposal_source_candidate_counts_inspected": dict(source_counts),
        "silent_blocker_count": silent_blockers,
        "irrelevant_blocker_count": irrelevant_blockers,
        "affected_root_mismatch_count": affected_mismatch,
        "conflict_root_mismatch_count": conflict_mismatch,
        "retained_root_mismatch_count": retain_mismatch,
        "canonical_root_weight_mismatch_count": canonical_weight_mismatch,
        "affected_definition_error_count": affected_definition_error,
        "burden_only_definition_error_count": burden_only_definition_error,
        "require_sdc_paths": bool(args.require_sdc_paths),
        "sdc_paths_missing_marker_count": sdc_paths_missing,
        "sdc_paths_not_ready_count": sdc_paths_not_ready,
        "reasons": reasons,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if reasons:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
