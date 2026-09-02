from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from ocrap.models.data import MODEL_SAMPLE_NPZ_KEYS, TEACHER_PCD_NPZ_KEYS

ROOT = Path(__file__).resolve().parents[1]


def _teacher_builder_module():
    p = ROOT / "tools" / "build_teacher_pcd_index_v48.py"
    spec = importlib.util.spec_from_file_location("v4856_teacher_builder", p)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_teacher_index_reads_strict_minimal_npz_subset() -> None:
    assert TEACHER_PCD_NPZ_KEYS < MODEL_SAMPLE_NPZ_KEYS
    for key in ("bev_occ", "agent_history", "map_polylines", "dynamic_map", "route", "prefix_states", "prefix_controls", "root_signature", "root_future_signature"):
        assert key in MODEL_SAMPLE_NPZ_KEYS
        assert key not in TEACHER_PCD_NPZ_KEYS
    for key in ("m_star", "root_probs", "c_star", "r_dep_star", "r_orc_star", "scene_id", "time_index", "candidate_index", "is_nominal"):
        assert key in TEACHER_PCD_NPZ_KEYS


def test_teacher_components_ignore_model_only_payload() -> None:
    mod = _teacher_builder_module()
    rng = np.random.default_rng(7)
    k, l = 8, 12
    p = rng.random(k).astype(np.float32); p /= p.sum()
    base = {
        "m_star": rng.normal(size=(k, l)).astype(np.float32),
        "root_probs": p,
        "c_star": np.eye(k, dtype=np.float32),
        "root_valid": np.ones(k, dtype=bool),
        "option_valid": np.ones(l, dtype=bool),
        "r_dep_star": np.asarray(0.2, dtype=np.float32),
        "r_orc_star": np.asarray(0.35, dtype=np.float32),
        "hard_violation": np.asarray(0.0, dtype=np.float32),
        "harm_proxy": np.asarray(0.0, dtype=np.float32),
    }
    full = dict(base)
    full.update({
        "bev_occ": rng.integers(0, 255, size=(7, 64, 64), dtype=np.uint8),
        "agent_history": rng.normal(size=(32, 8)).astype(np.float32),
        "map_polylines": rng.normal(size=(64, 20, 4)).astype(np.float32),
    })
    a = mod.teacher_components(full, alpha=0.2, beta=0.2, top_m=8, option_execution_semantics="observation_class")
    b = mod.teacher_components(base, alpha=0.2, beta=0.2, top_m=8, option_execution_semantics="observation_class")
    assert a == b


def test_v4856_launcher_reuses_raw_teacher_coordinates_and_emits_timings() -> None:
    text = (ROOT / "scripts" / "run_v48_56_dcp_drfc_bcde_drac_two_gpu.sh").read_text()
    assert ".v48_56_raw_teacher_cache" in text
    assert "V4856_RAW_TEACHER_INDEX" in text
    assert '"$A_RUN/evidence_adapt_teacher_pcd_index.jsonl"' in text
    assert "OC-RAP-v48.56-stage-timing.jsonl" in text
    assert "OC-RAP-v48.56-stage-timing-summary.json" in text
