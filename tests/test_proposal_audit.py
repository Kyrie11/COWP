import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from cache_io import infer_scene_record, load_npz


def test_alias_resolution(tmp_path):
    p = tmp_path / "s1.npz"
    np.savez(
        p,
        candidate_valid=np.array([1, 1, 0]),
        candidate_conventional_safe=np.array([1, 0, 1]),
        candidate_ncf=np.array([0, 1, 1]),
        candidate_priority_eligible=np.array([1, 1, 0]),
        candidate_priority_ncf=np.array([0, 1, 0]),
        candidate_source=np.array(["base", "rmr_bcte", "base"]),
    )
    rec = infer_scene_record(p, load_npz(p))
    assert rec.scenario_id == "s1"
    assert rec.any_conventional_safe()
    assert rec.any_ncf()
    assert rec.any_priority_eligible()
    assert rec.any_priority_ncf()
    assert rec.ncf_sources() == {"rmr_bcte"}
