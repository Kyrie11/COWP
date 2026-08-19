from __future__ import annotations

import json
import sys
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


def test_v24_primary_only_preflight_does_not_touch_optional_womd_splits(monkeypatch, tmp_path: Path) -> None:
    calls: list[tuple[str, str]] = []

    def fake_representation_row(*, root: Path, representation: str, split: str, sample_scenario_shards: int):
        calls.append((representation, split))
        return {
            "applicable": True,
            "released_for_this_representation": True,
            "glob": str(root / "uncompressed" / representation / split / "*.tfrecord*"),
            "shards": {"complete": True},
        }

    out = tmp_path / "layout.json"
    monkeypatch.setattr(layout, "_representation_row", fake_representation_row)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "69_audit_womd_split_layout.py",
            "--womd-root",
            str(tmp_path / "womd"),
            "--primary-only",
            "--output",
            str(out),
        ],
    )

    layout.main()

    assert calls == [
        ("scenario", "training"),
        ("tf_example", "training"),
        ("scenario", "validation"),
        ("tf_example", "validation"),
    ]
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["primary_only"] is True
    assert payload["audit_scope"] == ["training", "validation"]
    assert set(payload["splits"]) == {"training", "validation"}


def test_v24_launcher_enables_primary_only_womd_preflight() -> None:
    script = Path("NEXT_EXECUTION_V16_8_24_CN.sh").read_text(encoding="utf-8")
    assert "cowp.scripts.69_audit_womd_split_layout" in script
    assert "--primary-only" in script
