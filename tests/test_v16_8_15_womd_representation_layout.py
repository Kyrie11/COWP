from __future__ import annotations

from importlib import import_module
from pathlib import Path

layout = import_module("cowp.scripts.69_audit_womd_split_layout")


def test_training_20s_is_scenario_only_not_tfexample() -> None:
    assert "training_20s" in layout.SCENARIO_SPLITS
    assert "training_20s" not in layout.TFEXAMPLE_SPLITS
    row = layout._representation_row(
        root=Path("/dataset"),
        representation="tf_example",
        split="training_20s",
        sample_scenario_shards=1,
    )
    assert row["applicable"] is False
    assert "glob" not in row


def test_visualization_is_scenario_only_not_tfexample() -> None:
    assert "visualization" in layout.SCENARIO_SPLITS
    assert "visualization" not in layout.TFEXAMPLE_SPLITS
    row = layout._representation_row(
        root=Path("/dataset"),
        representation="tf_example",
        split="visualization",
        sample_scenario_shards=1,
    )
    assert row["applicable"] is False
    assert "glob" not in row


def test_primary_cowp_pairs_are_9s_training_validation_only() -> None:
    for split in ("training", "validation"):
        assert layout.SCENARIO_SPLITS[split]["required_primary"] is True
        assert layout.TFEXAMPLE_SPLITS[split]["required_primary"] is True
    assert layout.SCENARIO_SPLITS["testing"]["required_primary"] is False
    assert layout.TFEXAMPLE_SPLITS["testing"]["required_primary"] is False


def test_v15_master_uses_only_primary_train_val_for_build() -> None:
    script = Path("NEXT_EXECUTION_V16_8_15_CN.sh").read_text(encoding="utf-8")
    assert 'SCENARIO_TRAIN="$WOMD_ROOT/uncompressed/scenario/training/*.tfrecord*"' in script
    assert 'SCENARIO_VAL="$WOMD_ROOT/uncompressed/scenario/validation/*.tfrecord*"' in script
    assert 'TFEXAMPLE_TRAIN="$WOMD_ROOT/uncompressed/tf_example/training/*.tfrecord*"' in script
    assert 'TFEXAMPLE_VAL="$WOMD_ROOT/uncompressed/tf_example/validation/*.tfrecord*"' in script
    assert "tf_example/training_20s" not in script
