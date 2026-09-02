from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np

TRANSPORT_KEYS = (
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
)


def _stored(key: str) -> str:
    return key.replace("/", "__")


def _read_summary(root: Path) -> dict:
    p = root / "transport_augmentation_summary.json"
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _check_sidecar(path: Path) -> tuple[bool, list[str]]:
    try:
        with np.load(path, allow_pickle=True) as z:
            files = set(z.files)
    except Exception as exc:
        return False, [f"unreadable:{type(exc).__name__}"]
    missing = [k for k in TRANSPORT_KEYS if k not in files and _stored(k) not in files]
    return not missing, missing


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Rebase a legacy transport overlay whose top-level NPZ symlinks point to a deleted "
            "Waymax-attached cache. Existing transport sidecars are copied, while new top-level "
            "links point to the surviving base tensor cache. No Waymax outcome fields are invented."
        )
    )
    ap.add_argument("--base-cache", required=True)
    ap.add_argument("--old-overlay", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--sidecar-subdir", default=None)
    ap.add_argument("--verify-all-sidecars", action="store_true")
    ap.add_argument("--verify-sample", type=int, default=256)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    base = Path(args.base_cache).resolve()
    old = Path(args.old_overlay).resolve()
    out = Path(args.output_dir).resolve()
    if not base.is_dir():
        raise FileNotFoundError(f"base cache does not exist: {base}")
    if not old.is_dir():
        raise FileNotFoundError(f"old overlay does not exist: {old}")
    if out == base or out == old:
        raise ValueError("output-dir must be a new directory; do not mutate the surviving base or broken overlay in place")

    old_meta = _read_summary(old)
    side_name = str(args.sidecar_subdir or old_meta.get("sidecar_subdir") or ".transport_v16_8")
    old_side = old / side_name
    if not old_side.is_dir():
        raise FileNotFoundError(
            f"transport sidecar directory missing: {old_side}. Recompute transport from --base-cache with "
            "cowp.scripts.26_augment_transport_labels instead."
        )

    base_files = {p.name: p for p in base.glob("*.npz") if not p.name.startswith(".") and p.is_file()}
    side_files = {p.name: p for p in old_side.glob("*.npz") if not p.name.startswith(".") and p.is_file()}
    missing_side = sorted(set(base_files) - set(side_files))
    extra_side = sorted(set(side_files) - set(base_files))
    if missing_side:
        raise RuntimeError(
            f"legacy sidecars do not cover the surviving base cache: missing={len(missing_side)} examples={missing_side[:5]}. "
            "For a clean legacy cache, recompute transport from tensor_cache_train/val instead of partially rebasing."
        )

    verify_names = sorted(side_files)
    if not args.verify_all_sidecars and args.verify_sample > 0 and len(verify_names) > args.verify_sample:
        idx = np.linspace(0, len(verify_names) - 1, num=int(args.verify_sample), dtype=np.int64)
        verify_names = [verify_names[int(i)] for i in idx]
    bad: list[dict[str, object]] = []
    for name in verify_names:
        ok, missing = _check_sidecar(side_files[name])
        if not ok:
            bad.append({"file": name, "missing_or_error": missing})
            if len(bad) >= 20:
                break
    if bad:
        raise RuntimeError(f"transport sidecar verification failed; examples={bad[:5]}")

    if out.exists():
        if not args.force:
            raise FileExistsError(f"output already exists: {out}; pass --force only if you intend to replace it")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    out_side = out / side_name
    out_side.mkdir(parents=True, exist_ok=True)

    for name, src in base_files.items():
        os.symlink(str(src.resolve()), str(out / name))
    for name, src in side_files.items():
        shutil.copy2(src, out_side / name)

    summary = {
        "schema_version": "cowp_v16_8_7_rebased_legacy_transport_overlay_v1",
        "input_dir": str(base),
        "output_dir": str(out),
        "storage_mode": "overlay",
        "sidecar_subdir": side_name,
        "files_total": len(base_files),
        "files_completed": len(base_files),
        "error_count": 0,
        "complete": True,
        "rebased_from_overlay": str(old),
        "old_overlay_declared_input_dir": old_meta.get("input_dir"),
        "old_overlay_extra_sidecars_ignored": len(extra_side),
        "transport_sidecars_recomputed": False,
        "waymax_outcome_fields_available": False,
        "lineage_note": (
            "The original Waymax-attached backing cache was a copy of the base tensor cache plus waymax/* fields. "
            "Transport augmentation does not consume waymax/* fields, so existing sidecars can be safely paired with "
            "the surviving base cache when filenames match exactly. This rebased cache is for legacy RCOT/BCOT diagnostics/training only; "
            "it does not retrofit v16.8.6 proposal tensors and is not paper-grade v16.8.6 data."
        ),
    }
    (out / "transport_augmentation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
