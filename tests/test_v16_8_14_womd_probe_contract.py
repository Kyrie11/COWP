from __future__ import annotations

from importlib import import_module
from pathlib import Path

import numpy as np
import pytest

from cowp.data.build_cache import _assert_all_label_caches_present


preflight = import_module("cowp.scripts.64_validate_womd_v131_contract")


def test_womd_shard_manifest_detects_partial_readable_training_download() -> None:
    files = [f"/x/training_tfexample.tfrecord-{i:05d}-of-01000" for i in range(557)]
    report = preflight._shard_manifest(files, expected=1000)
    assert report["present_shards"] == 557
    assert report["encoded_total"] == 1000
    assert report["missing_shard_count"] == 443
    assert report["complete"] is False


def test_womd_shard_manifest_accepts_complete_validation_download() -> None:
    files = [f"/x/validation_tfexample.tfrecord-{i:05d}-of-00150" for i in range(150)]
    report = preflight._shard_manifest(files, expected=150)
    assert report["missing_shard_count"] == 0
    assert report["complete"] is True


def test_tensor_cache_completeness_guard_rejects_missing_label_match(tmp_path: Path) -> None:
    labels = {"scene_a": tmp_path / "label_a.npz", "scene_b": tmp_path / "label_b.npz"}
    np.savez(tmp_path / "scene_a.npz", x=np.asarray([1]))
    with pytest.raises(RuntimeError, match="1 / 2 label scenario ids"):
        _assert_all_label_caches_present(labels, tmp_path)


def test_smoke_manifest_is_derived_from_current_cache_not_legacy_manifest() -> None:
    script = Path("NEXT_RUN_COMMANDS_V16_8_14_CAUSAL_AUDIT_SMOKE_CN.sh").read_text(encoding="utf-8")
    assert "SOURCE_PROBE_ROOT" not in script
    assert "cowp.scripts.45_diagnose_proposal_ceiling" in script
    assert "--probe-total-count" in script
    assert "--min-hard" in script


def test_master_preflight_requires_complete_primary_splits() -> None:
    script = Path("NEXT_EXECUTION_V16_8_14_CN.sh").read_text(encoding="utf-8")
    assert "--require-complete-primary-splits" in script
    assert "split-audit" in script
