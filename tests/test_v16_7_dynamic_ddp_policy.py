from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "cowp" / "scripts" / "03_train.py"
WRAPPER = ROOT / "NEXT_RUN_COMMANDS_V16_7_MECHANISM_CN.sh"


def _ddp_block() -> str:
    source = TRAIN.read_text(encoding="utf-8")
    start = source.index("    model = _maybe_compile_model")
    end = source.index("    epochs = args.epochs", start)
    return source[start:end]


def test_witness_and_planner_are_not_promoted_to_static_graph() -> None:
    block = _ddp_block()
    assert 'or (no_freeze_transition and stage in {"witness", "planner"})' not in block
    assert 'permanent_natural_freeze and stage in {"natural", "representation"}' in block
    assert '"find_unused_parameters": not static_stage_ddp' in block


def test_dynamic_ddp_keeps_real_unused_parameter_semantics() -> None:
    block = _ddp_block()
    assert 'ddp_kwargs["static_graph"] = True' in block
    assert 'if static_stage_ddp:' in block
    assert '"gradient_as_bucket_view": True' in block
    assert 'synthetic zero loss' in block


def test_mechanism_wrapper_has_distinct_one_shot_provenance_marker() -> None:
    wrapper = WRAPPER.read_text(encoding="utf-8")
    assert '.v16_7_dynamic_ddp_hotfix_v3_applied' in wrapper
    assert 'disable invalid static DDP for dynamic witness/planner supervision' in wrapper
