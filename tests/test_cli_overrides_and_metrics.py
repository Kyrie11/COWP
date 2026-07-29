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
