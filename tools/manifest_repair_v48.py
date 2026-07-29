#!/usr/bin/env python3
"""Metadata-only reconstruction of OC-RAP ``manifest.csv`` files.

Dataset builders write NPZ samples atomically and normally write manifest.csv at
clean shutdown.  An interrupted build can therefore leave valid samples without
a manifest.  This module reconstructs only the CSV metadata; it never regenerates
or modifies sample NPZ files.
"""
from __future__ import annotations

import csv
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


MANIFEST_FIELDS = [
    "path",
    "scene_id",
    "original_scenario_id",
    "time_index",
    "candidate_index",
    "split_id",
    "is_nominal",
    "r_orc_star",
    "r_dep_star",
    "oracle_gap_star",
    "i_art_star",
    "regime_label",
]
_FILENAME_RE = re.compile(r"_t(?P<time>\d+)_a(?P<candidate>\d+)\.npz$")


@dataclass(frozen=True)
class ManifestRepairResult:
    dataset_root: str
    manifest: str
    sample_count: int
    previous_manifest_rows: int | None
    action: str
    dataset_summary_present: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset_root": self.dataset_root,
            "manifest": self.manifest,
            "sample_count": self.sample_count,
            "previous_manifest_rows": self.previous_manifest_rows,
            "action": self.action,
            "dataset_summary_present": self.dataset_summary_present,
        }


def _sample_paths(root: Path) -> list[Path]:
    sample_dir = root / "samples"
    paths = sorted(sample_dir.glob("*.npz")) if sample_dir.is_dir() else sorted(root.glob("*.npz"))
    return [p for p in paths if p.is_file()]


def _count_manifest_rows(path: Path) -> int:
    with path.open("r", newline="", encoding="utf-8") as f:
        return sum(1 for _ in csv.DictReader(f))


def _scalar(z: np.lib.npyio.NpzFile, key: str, default: Any = "") -> Any:
    if key not in z.files:
        return default
    arr = np.asarray(z[key])
    if arr.shape == ():
        value = arr.item()
    elif arr.size == 1:
        value = arr.reshape(-1)[0].item()
    else:
        value = arr.tolist()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _regime_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, dict):
        return ";".join(str(k) for k, enabled in value.items() if enabled)
    if isinstance(value, (list, tuple)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    text = str(value)
    try:
        parsed = json.loads(text)
    except Exception:
        return text
    if isinstance(parsed, dict):
        return ";".join(str(k) for k, enabled in parsed.items() if enabled)
    return text


def _fallback_split(root: Path) -> str:
    name = root.name.lower()
    if name.startswith("train"):
        return "train"
    if name.startswith("val") or name.startswith("dev"):
        return "val"
    if name.startswith("calibration"):
        return "calibration"
    if name.startswith("test"):
        return "test"
    return ""


def _row_from_npz(path: Path, root: Path) -> dict[str, Any]:
    match = _FILENAME_RE.search(path.name)
    fallback_time = int(match.group("time")) if match else ""
    fallback_candidate = int(match.group("candidate")) if match else ""
    try:
        with np.load(path, allow_pickle=True) as z:
            scene = str(_scalar(z, "scene_id", "")).strip()
            original = str(_scalar(z, "original_scenario_id", "")).strip() or scene
            if not scene:
                raise ValueError("missing scene_id")
            return {
                "path": str(path.relative_to(root)),
                "scene_id": scene,
                "original_scenario_id": original,
                "time_index": _scalar(z, "time_index", fallback_time),
                "candidate_index": _scalar(z, "candidate_index", fallback_candidate),
                "split_id": str(_scalar(z, "split_id", _fallback_split(root))),
                "is_nominal": int(float(_scalar(z, "is_nominal", 0) or 0)),
                "r_orc_star": _scalar(z, "r_orc_star", ""),
                "r_dep_star": _scalar(z, "r_dep_star", ""),
                "oracle_gap_star": _scalar(z, "oracle_gap_star", ""),
                "i_art_star": int(float(_scalar(z, "i_art_star", 0) or 0)),
                "regime_label": _regime_text(_scalar(z, "regime_label", "")),
            }
    except Exception as exc:
        raise RuntimeError(f"cannot read manifest metadata from {path}: {exc}") from exc


def ensure_manifest(
    root: str | Path,
    *,
    workers: int = 4,
    rebuild_if_stale: bool = False,
    force: bool = False,
) -> ManifestRepairResult:
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"dataset root does not exist: {root}")
    manifest = root / "manifest.csv"
    paths_before = _sample_paths(root)
    if not paths_before:
        raise FileNotFoundError(f"no NPZ samples found under {root}")

    previous_rows: int | None = None
    if manifest.exists():
        previous_rows = _count_manifest_rows(manifest)
        if not force and (not rebuild_if_stale or previous_rows == len(paths_before)):
            return ManifestRepairResult(
                dataset_root=str(root),
                manifest=str(manifest),
                sample_count=len(paths_before),
                previous_manifest_rows=previous_rows,
                action="kept_existing",
                dataset_summary_present=(root / "dataset_summary.json").exists(),
            )

    rows: list[dict[str, Any] | None] = [None] * len(paths_before)
    worker_count = max(1, int(workers))
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        futures = {pool.submit(_row_from_npz, p, root): i for i, p in enumerate(paths_before)}
        for future in as_completed(futures):
            rows[futures[future]] = future.result()

    paths_after = _sample_paths(root)
    before_names = [p.name for p in paths_before]
    after_names = [p.name for p in paths_after]
    if before_names != after_names:
        raise RuntimeError(
            f"dataset changed while reconstructing manifest: {root}. "
            "A dataset builder may still be writing samples; stop/wait for it and retry."
        )

    tmp = root / f".manifest.csv.tmp.{os.getpid()}"
    try:
        with tmp.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(row for row in rows if row is not None)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, manifest)
    finally:
        if tmp.exists():
            tmp.unlink()

    action = "rebuilt_stale" if previous_rows is not None else "created_missing"
    provenance = {
        "event": "manifest_metadata_reconstructed",
        "dataset_root": str(root),
        "manifest": str(manifest),
        "sample_count": len(paths_before),
        "previous_manifest_rows": previous_rows,
        "action": action,
        "sample_npz_modified": False,
        "dataset_summary_present": (root / "dataset_summary.json").exists(),
        "note": (
            "dataset_summary.json is absent; this manifest represents the stable NPZ snapshot, "
            "not proof that the original dataset build completed"
            if not (root / "dataset_summary.json").exists()
            else "dataset_summary.json is present"
        ),
    }
    (root / "manifest_repair_v48.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return ManifestRepairResult(
        dataset_root=str(root),
        manifest=str(manifest),
        sample_count=len(paths_before),
        previous_manifest_rows=previous_rows,
        action=action,
        dataset_summary_present=(root / "dataset_summary.json").exists(),
    )


def ensure_many(
    roots: Iterable[str | Path],
    *,
    workers: int = 4,
    rebuild_if_stale: bool = False,
    force: bool = False,
) -> list[ManifestRepairResult]:
    return [
        ensure_manifest(root, workers=workers, rebuild_if_stale=rebuild_if_stale, force=force)
        for root in roots
    ]
