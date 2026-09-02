import numpy as np

from ocrap.cli.main import make_parser, build_cfg
from ocrap.evaluation.metrics import deployable_recovery_success


def test_common_set_before_subcommand_is_preserved():
    parser = make_parser()
    args = parser.parse_args([
        "--set", "num_candidate_prefixes=4",
        "build-dataset",
        "--output", "/tmp/out",
    ])
    cfg = build_cfg(args)
    assert cfg["num_candidate_prefixes"] == 4


def test_common_set_after_subcommand_is_preserved():
    parser = make_parser()
    args = parser.parse_args([
        "build-dataset",
        "--output", "/tmp/out",
        "--set", "num_candidate_prefixes=5",
    ])
    cfg = build_cfg(args)
    assert cfg["num_candidate_prefixes"] == 5


def test_drs_masks_invalid_roots():
    # Root 1 is invalid/padded and has a negative selected margin.  The DRS should
    # be 1.0 because the only valid root succeeds.
    m_star = np.array([[1.0, -1.0], [-1.0, -1.0]], dtype=float)
    root_probs = np.array([0.5, 0.5], dtype=float)
    root_valid = np.array([1.0, 0.0], dtype=float)
    selected_options = np.array([0, 0], dtype=int)
    assert deployable_recovery_success(m_star, root_probs, selected_options, root_valid) == 1.0


def test_empty_cli_override_is_preserved_as_empty_string():
    parser = make_parser()
    args = parser.parse_args([
        "train",
        "--dataset", "/tmp/train",
        "--output", "/tmp/out",
        "--set", "training.init_checkpoint=",
        "--set", "model.direct_recovery_evidence_component_reliability=",
        "--set", "training.direct_value_ordinal_evidence_component_reliability=",
    ])
    cfg = build_cfg(args)
    assert cfg["training"]["init_checkpoint"] == ""
    assert cfg["model"]["direct_recovery_evidence_component_reliability"] == ""
    assert cfg["training"]["direct_value_ordinal_evidence_component_reliability"] == ""


def test_cli_yaml_null_remains_available_explicitly():
    parser = make_parser()
    args = parser.parse_args([
        "train",
        "--dataset", "/tmp/train",
        "--output", "/tmp/out",
        "--set", "training.init_checkpoint=null",
    ])
    cfg = build_cfg(args)
    assert cfg["training"]["init_checkpoint"] is None
