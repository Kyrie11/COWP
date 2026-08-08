from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import yaml


def _stored(key: str) -> str:
    return key.replace("/", "__")


def _atomic_save_npz(path: Path, arrays: dict[str, np.ndarray], *, compress: bool) -> None:
    tmp = path.with_name(path.name + ".repair_tmp")
    with tmp.open("wb") as fh:
        if compress:
            np.savez_compressed(fh, **arrays)
        else:
            np.savez(fh, **arrays)
    os.replace(tmp, path)


def _get(arrays: dict[str, np.ndarray], key: str) -> np.ndarray:
    skey = _stored(key)
    if skey not in arrays:
        raise KeyError(key)
    return np.asarray(arrays[skey])


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Repair the v16.8.9 root-level audit/transport serialization contract "
            "without recomputing Scenario-proto labels. Candidate NCF/witness/response "
            "semantics are intentionally left unchanged."
        )
    )
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--label-config", default="configs/label_cowp_v16_8.yaml")
    ap.add_argument("--output", required=True, help="JSON repair manifest")
    ap.add_argument("--compress", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = Path(args.cache_dir)
    paths = sorted(p for p in root.glob("*.npz") if p.is_file())
    if not paths:
        raise FileNotFoundError(f"no NPZ files in {root}")
    cfg = yaml.safe_load(Path(args.label_config).read_text(encoding="utf-8")) or {}
    direct_margin = float(cfg.get("ncf", {}).get("audit_direct_burden_margin", 0.0))

    totals = {
        "files": 0,
        "affected_before_mismatch": 0,
        "conflict_before_mismatch": 0,
        "retain_before_mismatch": 0,
        "budget_crossed_roots": 0,
        "burden_only_roots": 0,
        "files_changed": 0,
    }
    errors: list[dict[str, str]] = []

    for path in paths:
        try:
            with np.load(path, allow_pickle=True) as z:
                arrays = {k: np.asarray(z[k]) for k in z.files}

            affected = _get(arrays, "cowp/audit/root_affected").astype(bool, copy=False)
            unsafe = _get(arrays, "cowp/audit/root_unsafe").astype(bool, copy=False)
            direct = _get(arrays, "cowp/audit/root_direct_burden").astype(np.float32, copy=False)
            beta = _get(arrays, "cowp/natural/beta").astype(np.float32, copy=False)
            mode_valid = _get(arrays, "cowp/transport/mode_valid").astype(bool, copy=False)
            old_aff = _get(arrays, "cowp/transport/mode_affected").astype(bool, copy=False)
            old_conf = _get(arrays, "cowp/transport/mode_conflict").astype(bool, copy=False)
            old_retain = _get(arrays, "cowp/transport/mode_retained_low_safe").astype(bool, copy=False)

            if affected.shape != unsafe.shape or affected.shape != mode_valid.shape:
                raise ValueError(
                    f"root shape mismatch affected={affected.shape} unsafe={unsafe.shape} mode_valid={mode_valid.shape}"
                )
            if direct.shape != affected.shape:
                raise ValueError(f"direct burden shape {direct.shape} != {affected.shape}")
            if beta.ndim != 1 or beta.shape[0] != affected.shape[1]:
                raise ValueError(f"beta shape {beta.shape} incompatible with {affected.shape}")

            budget = direct > (beta[None, :, None] + direct_margin)
            # Audit only evaluates neutral-low-burden natural roots; affected is
            # the source of truth for whether a root participates at all.  Limit
            # the derived budget flag to the same support to avoid turning padded
            # direct-burden defaults into semantic labels.
            budget &= affected | unsafe | (direct > 0.0)
            burden_only = budget & ~unsafe

            # The v16.8.9 affected definition is exactly unsafe OR direct budget
            # crossing. Existing smoke labels already store the authoritative
            # affected tensor; assert rather than silently change NCF semantics.
            identity_error = np.logical_xor(affected, unsafe | budget)
            if np.any(identity_error):
                raise ValueError(
                    f"existing audit/root_affected violates v16.8.9 identity in {int(identity_error.sum())} roots; "
                    "requires fresh Scenario-proto rebuild rather than contract repair"
                )

            totals["files"] += 1
            totals["affected_before_mismatch"] += int(np.logical_xor(old_aff, affected).sum())
            totals["conflict_before_mismatch"] += int(np.logical_xor(old_conf, unsafe).sum())
            totals["retain_before_mismatch"] += int(np.logical_xor(old_retain, mode_valid & ~affected).sum())
            totals["budget_crossed_roots"] += int(budget.sum())
            totals["burden_only_roots"] += int(burden_only.sum())

            canonical_key = _stored("cowp/audit/canonical_root_weight")
            transport_weight = _get(arrays, "cowp/transport/canonical_root_weight").astype(np.float32, copy=False)

            changed = (
                np.any(old_aff != affected)
                or np.any(old_conf != unsafe)
                or np.any(old_retain != (mode_valid & ~affected))
                or _stored("cowp/audit/root_budget_crossed") not in arrays
                or _stored("cowp/audit/root_burden_only_affected") not in arrays
                or canonical_key not in arrays
            )
            if changed:
                totals["files_changed"] += 1
            if not args.dry_run and changed:
                arrays[_stored("cowp/audit/root_budget_crossed")] = budget.astype(np.bool_)
                arrays[_stored("cowp/audit/root_burden_only_affected")] = burden_only.astype(np.bool_)
                arrays[canonical_key] = transport_weight.astype(np.float32)
                arrays[_stored("cowp/transport/mode_conflict")] = unsafe.astype(np.bool_)
                arrays[_stored("cowp/transport/mode_affected")] = affected.astype(np.bool_)
                arrays[_stored("cowp/transport/mode_retained_low_safe")] = (mode_valid & ~affected).astype(np.bool_)
                _atomic_save_npz(path, arrays, compress=bool(args.compress))
        except Exception as exc:
            if len(errors) < 50:
                errors.append({"file": path.name, "error": repr(exc)})

    result = {
        "schema_version": "cowp_v16_8_9_audit_transport_contract_repair_v1",
        "cache_dir": str(root.resolve()),
        "dry_run": bool(args.dry_run),
        **totals,
        "errors": errors,
        "pass": not errors,
        "semantic_note": (
            "This repair only makes root-level audit/transport serialization self-consistent and adds explicit "
            "budget/burden-only/canonical-weight fields. It does not change candidate trajectories, pair relevance, "
            "responses, witnesses, pair NCF, or candidate NCF."
        ),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
