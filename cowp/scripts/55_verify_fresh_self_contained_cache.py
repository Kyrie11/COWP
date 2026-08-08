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
    "cowp/witness/exists",
    "cowp/witness/rho",
    "cowp/transport/mode_valid",
    "cowp/transport/mode_conflict",
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
    ap = argparse.ArgumentParser(description="Verify a fresh self-contained COWP cache before training/Waymax experiments.")
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--allowlist", default=None)
    ap.add_argument("--sample-scenes", type=int, default=0, help="0 verifies every scenario file.")
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

    result = {
        "schema_version": "cowp_v16_8_7_self_contained_cache_integrity_v1",
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
