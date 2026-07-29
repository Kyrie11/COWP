from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
from manifest_repair_v48 import ensure_manifest  # noqa: E402


def _write_sample(root: Path, scene: str, original: str, t: int, a: int) -> None:
    sample_dir = root / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        sample_dir / f"{scene}_t{t:04d}_a{a:02d}.npz",
        scene_id=np.asarray(scene),
        original_scenario_id=np.asarray(original),
        time_index=np.asarray(t),
        candidate_index=np.asarray(a),
        split_id=np.asarray("val"),
        is_nominal=np.asarray(int(a == 0)),
        r_orc_star=np.asarray(1.0),
        r_dep_star=np.asarray(0.5),
        oracle_gap_star=np.asarray(0.5),
        i_art_star=np.asarray(0),
        regime_label=np.asarray('{"near_contact": true}'),
    )


def test_reconstruct_missing_manifest_and_keep_existing(tmp_path: Path) -> None:
    root = tmp_path / "val_contact"
    _write_sample(root, "derived_a", "source_1", 4, 0)
    _write_sample(root, "derived_a", "source_1", 4, 1)

    result = ensure_manifest(root, workers=2, rebuild_if_stale=True)
    assert result.action == "created_missing"
    with (root / "manifest.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert {row["original_scenario_id"] for row in rows} == {"source_1"}
    assert {row["path"] for row in rows} == {
        "samples/derived_a_t0004_a00.npz",
        "samples/derived_a_t0004_a01.npz",
    }

    kept = ensure_manifest(root, workers=1, rebuild_if_stale=True)
    assert kept.action == "kept_existing"


def test_rebuild_stale_manifest(tmp_path: Path) -> None:
    root = tmp_path / "val_contact"
    _write_sample(root, "s1", "o1", 1, 0)
    ensure_manifest(root)
    _write_sample(root, "s2", "o2", 2, 0)
    result = ensure_manifest(root, rebuild_if_stale=True)
    assert result.action == "rebuilt_stale"
    with (root / "manifest.csv").open(newline="", encoding="utf-8") as f:
        assert len(list(csv.DictReader(f))) == 2
