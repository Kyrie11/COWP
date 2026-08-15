from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from cowp.data.build_cache import _scenario_records_for_allowlist_from_index


def _minimal_label(path: Path, sid: str) -> None:
    np.savez(
        path,
        **{
            "scenario/id": np.asarray(sid),
            "cowp/candidates/trajectory": np.zeros((1, 2, 2), dtype=np.float32),
            "cowp/candidates/valid": np.ones((1,), dtype=bool),
            "cowp/critical/track_index": np.zeros((1,), dtype=np.int64),
            "cowp/natural/traj": np.zeros((1, 1, 2, 2), dtype=np.float32),
            "cowp/response/traj": np.zeros((1, 1, 1, 2, 2), dtype=np.float32),
            "cowp/witness/exists": np.zeros((1, 1), dtype=bool),
        },
    )


def test_location_index_resolves_sparse_allowlist(tmp_path: Path) -> None:
    idx = tmp_path / "locations.jsonl"
    rows = [
        {"scenario_id": "a", "file": "/x/000.tfrecord", "record_index": 3},
        {"scenario_id": "b", "file": "/x/001.tfrecord", "record_index": 7},
        {"scenario_id": "c", "file": "/x/001.tfrecord", "record_index": 9},
    ]
    idx.write_text("\n".join(json.dumps(x) for x in rows) + "\n", encoding="utf-8")
    by_file, stats = _scenario_records_for_allowlist_from_index(idx, {"a", "c", "missing"})
    assert stats["location_capable"] is True
    assert stats["matched_ids"] == 2
    assert stats["missing_ids"] == ["missing"]
    assert by_file["/x/000.tfrecord"] == {3: "a"}
    assert by_file["/x/001.tfrecord"] == {9: "c"}


def test_sparse_validator_distinguishes_interrupted_build(tmp_path: Path) -> None:
    mod = importlib.import_module("cowp.scripts.71_validate_sparse_label_build")
    labels = tmp_path / "labels"
    labels.mkdir()
    ids = tmp_path / "ids.txt"
    ids.write_text("a\nb\n", encoding="utf-8")
    profile = tmp_path / "profile.jsonl"
    profile.write_text("", encoding="utf-8")
    # Call the helper pieces directly: neither id is in the profile and neither NPZ exists.
    latest, errors = mod._latest_profile(profile)
    assert latest == {}
    assert errors == []
    assert mod._check_npz(labels / "a.npz", "a") == (False, "missing")


def test_sparse_validator_accepts_complete_requested_set(tmp_path: Path) -> None:
    labels = tmp_path / "labels"
    labels.mkdir()
    ids = tmp_path / "ids.txt"
    ids.write_text("a\nb\n", encoding="utf-8")
    profile = tmp_path / "profile.jsonl"
    profile.write_text(
        json.dumps({"status": "written", "scenario_id": "a"}) + "\n" +
        json.dumps({"status": "written", "scenario_id": "b"}) + "\n",
        encoding="utf-8",
    )
    _minimal_label(labels / "a.npz", "a")
    _minimal_label(labels / "b.npz", "b")
    out = tmp_path / "report.json"
    proc = subprocess.run(
        [
            sys.executable, "-m", "cowp.scripts.71_validate_sparse_label_build",
            "--labels-dir", str(labels), "--scene-ids", str(ids),
            "--profile-jsonl", str(profile), "--output", str(out),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["pass"] is True
    assert report["counts"]["requested"] == 2
    assert report["counts"]["missing_npz"] == 0


def test_semantic_compare_missing_reference_is_precondition_not_semantic_change(tmp_path: Path) -> None:
    ref = tmp_path / "ref"
    cand = tmp_path / "cand"
    ref.mkdir(); cand.mkdir()
    ids = tmp_path / "ids.txt"
    ids.write_text("a\n", encoding="utf-8")
    _minimal_label(cand / "a.npz", "a")
    out = tmp_path / "eq.json"
    proc = subprocess.run(
        [
            sys.executable, "-m", "cowp.scripts.66_compare_label_semantic_equivalence",
            "--reference-dir", str(ref), "--candidate-dir", str(cand),
            "--scene-ids", str(ids), "--output", str(out),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["failure_class"] == "reference_build_incomplete"
    assert report["semantic_change_detected"] is False
    assert report["precondition_failure"] is True
    assert report["compared_scenes"] == 0
