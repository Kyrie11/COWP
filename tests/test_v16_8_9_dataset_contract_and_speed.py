from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from cowp.data.build_cache import _sdc_path_contract_errors


def _path_example(split_xyz: bool = False):
    npaths, npoints = 2, 3
    xyz = np.arange(npaths * npoints * 3, dtype=np.float32).reshape(npaths, npoints, 3)
    ex = {
        "path_samples/valid": np.ones((npaths, npoints), dtype=np.int64),
        "path_samples/id": np.arange(npaths * npoints, dtype=np.int64).reshape(npaths, npoints),
        "path_samples/arc_length": np.arange(npaths * npoints, dtype=np.float32).reshape(npaths, npoints),
        "path_samples/on_route": np.asarray([[1], [0]], dtype=np.int64),
    }
    if split_xyz:
        ex["path_samples/x"] = xyz[..., 0]
        ex["path_samples/y"] = xyz[..., 1]
        ex["path_samples/z"] = xyz[..., 2]
    else:
        ex["path_samples/xyz"] = xyz
    return ex


def test_sdc_path_contract_accepts_xyz_and_split_xyz():
    assert _sdc_path_contract_errors(_path_example(False)) == []
    assert _sdc_path_contract_errors(_path_example(True)) == []


def test_sdc_path_contract_preserves_off_route_only_official_scenes():
    ex = _path_example(False)
    ex["path_samples/on_route"][:] = 0
    assert _sdc_path_contract_errors(ex) == []


def test_sdc_path_contract_rejects_shape_mismatch():
    ex = _path_example(False)
    ex["path_samples/arc_length"] = ex["path_samples/arc_length"][:, :-1]
    assert any(x.startswith("shape:arc_length=") for x in _sdc_path_contract_errors(ex))


def test_probe_manifest_requires_target_1200_union(tmp_path: Path):
    hard = tmp_path / "hard.txt"
    rand = tmp_path / "random.txt"
    union = tmp_path / "union.txt"
    ceiling = tmp_path / "ceiling.json"
    out = tmp_path / "report.json"
    hard.write_text("h1\nh2\n")
    rand.write_text("r1\nr2\nr3\n")
    union.write_text("h1\nh2\nr1\nr2\nr3\n")
    ceiling.write_text(json.dumps({
        "hard_scene_ids_path": str(hard),
        "representative_random_scene_ids_path": str(rand),
        "hard_scene_probe_count": 2,
        "representative_random_scene_count": 3,
    }))
    subprocess.check_call([
        sys.executable, "-m", "cowp.scripts.63_validate_probe_manifest",
        "--ceiling-json", str(ceiling), "--hard-ids", str(hard), "--random-ids", str(rand),
        "--union-ids", str(union), "--expected-hard", "2", "--expected-random", "3", "--output", str(out),
    ])
    rep = json.loads(out.read_text())
    assert rep["pass"] is True
    assert rep["checks"]["union_target_count"] is True


def test_hash_holdout_manifest_is_deterministic_and_excludes(tmp_path):
    import json
    import subprocess
    import sys

    index = tmp_path / "index.jsonl"
    index.write_text("".join(json.dumps({"scenario_id": f"s{i}"}) + "\n" for i in range(20)), encoding="utf-8")
    excluded = tmp_path / "excluded.txt"
    excluded.write_text("s1\ns2\ns3\n", encoding="utf-8")
    out1, man1 = tmp_path / "a.txt", tmp_path / "a.json"
    out2, man2 = tmp_path / "b.txt", tmp_path / "b.json"
    cmd = [
        sys.executable, "-m", "cowp.scripts.67_make_hash_holdout_manifest",
        "--index-jsonl", str(index), "--count", "7", "--seed", "fixed",
        "--exclude", str(excluded),
    ]
    subprocess.check_call(cmd + ["--output-ids", str(out1), "--output-manifest", str(man1)])
    subprocess.check_call(cmd + ["--output-ids", str(out2), "--output-manifest", str(man2)])
    ids = out1.read_text(encoding="utf-8").splitlines()
    assert ids == out2.read_text(encoding="utf-8").splitlines()
    assert len(ids) == 7 and len(set(ids)) == 7
    assert not (set(ids) & {"s1", "s2", "s3"})
    report = json.loads(man1.read_text(encoding="utf-8"))
    assert report["selected_count"] == 7
    assert report["leakage_checks"]["selected_intersects_excluded"] is False
