from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


DEFAULT_ALLOWED_EXTRA = {"cowp/audit/root_event_interval"}


def _npz_map(directory: Path) -> dict[str, Path]:
    return {p.stem: p for p in sorted(directory.glob("*.npz")) if p.is_file()}


def _eq(a: np.ndarray, b: np.ndarray) -> bool:
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    try:
        return bool(np.array_equal(a, b, equal_nan=True))
    except TypeError:
        return bool(np.array_equal(a, b))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Bitwise/exact semantic parity check for an optimized fresh label build. "
            "All reference keys must exist and match exactly; only explicitly allowed new keys may be added."
        )
    )
    ap.add_argument("--reference-dir", required=True)
    ap.add_argument("--candidate-dir", required=True)
    ap.add_argument("--scene-ids", default=None)
    ap.add_argument("--max-scenes", type=int, default=0)
    ap.add_argument("--allow-extra-key", action="append", default=[])
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    ref_dir = Path(args.reference_dir)
    cand_dir = Path(args.candidate_dir)
    refs, cands = _npz_map(ref_dir), _npz_map(cand_dir)
    if args.scene_ids:
        wanted = [x.strip().split()[0] for x in Path(args.scene_ids).read_text(encoding="utf-8").splitlines() if x.strip()]
    else:
        wanted = sorted(set(refs) & set(cands))
    if args.max_scenes > 0:
        wanted = wanted[: int(args.max_scenes)]
    if not wanted:
        raise FileNotFoundError("no overlapping/requested NPZ scenes to compare")

    allowed_extra = set(DEFAULT_ALLOWED_EXTRA) | set(args.allow_extra_key)
    missing_reference: list[str] = []
    missing_candidate: list[str] = []
    mismatches: list[dict[str, object]] = []
    unexpected_extra: list[dict[str, object]] = []
    compared = 0
    for sid in wanted:
        rp, cp = refs.get(sid), cands.get(sid)
        if rp is None:
            missing_reference.append(sid)
            continue
        if cp is None:
            missing_candidate.append(sid)
            continue
        compared += 1
        with np.load(rp, allow_pickle=True) as r, np.load(cp, allow_pickle=True) as c:
            rkeys, ckeys = set(r.files), set(c.files)
            missing_keys = sorted(rkeys - ckeys)
            extra_keys = sorted(ckeys - rkeys)
            if missing_keys and len(mismatches) < 50:
                mismatches.append({"scenario_id": sid, "kind": "missing_keys", "keys": missing_keys[:20]})
            bad_extra = [k for k in extra_keys if k.replace("__", "/") not in allowed_extra and k not in allowed_extra]
            if bad_extra and len(unexpected_extra) < 50:
                unexpected_extra.append({"scenario_id": sid, "keys": bad_extra[:20]})
            for key in sorted(rkeys & ckeys):
                a, b = np.asarray(r[key]), np.asarray(c[key])
                if not _eq(a, b):
                    if len(mismatches) < 50:
                        rec: dict[str, object] = {
                            "scenario_id": sid,
                            "kind": "array_mismatch",
                            "key": key,
                            "reference_shape": list(a.shape),
                            "candidate_shape": list(b.shape),
                            "reference_dtype": str(a.dtype),
                            "candidate_dtype": str(b.dtype),
                        }
                        if a.shape == b.shape and np.issubdtype(a.dtype, np.number) and np.issubdtype(b.dtype, np.number):
                            try:
                                d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
                                finite = np.isfinite(d)
                                rec["max_abs_diff"] = float(np.max(np.abs(d[finite]))) if np.any(finite) else None
                            except Exception:
                                pass
                        mismatches.append(rec)

    passed = bool(compared) and not missing_reference and not missing_candidate and not mismatches and not unexpected_extra
    report = {
        "schema_version": "cowp_label_semantic_equivalence_v1",
        "pass": passed,
        "reference_dir": str(ref_dir.resolve()),
        "candidate_dir": str(cand_dir.resolve()),
        "requested_scenes": len(wanted),
        "compared_scenes": compared,
        "allowed_candidate_extra_keys": sorted(allowed_extra),
        "missing_reference_scenes": missing_reference[:50],
        "missing_candidate_scenes": missing_candidate[:50],
        "mismatches": mismatches,
        "unexpected_extra_keys": unexpected_extra,
        "interpretation": (
            "PASS: optimization preserved every reference label tensor exactly; only explicitly allowed new audit keys were added."
            if passed else
            "FAIL: at least one pre-existing label tensor changed or a requested scene/key is missing. Treat this as a semantic change and investigate before full rebuild."
        ),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
