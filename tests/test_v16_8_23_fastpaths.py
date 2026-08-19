from __future__ import annotations

import copy
import numpy as np

from cowp.core.config import load_config
from cowp.core.constants import PriorityRelation
from cowp.geometry.collision import unsafe_between, unsafe_between_bool
from cowp.label.safe_budget_search import (
    build_safe_budget_trajectory_bank,
    prepare_safe_budget_trajectory_bank,
    typed_safe_budget_search_evaluated,
)
from cowp.label.safe_responses import (
    build_root_recovery_trajectory_bank,
    prepare_root_recovery_burden_bank,
    root_conditioned_recovery_search,
)


def _cfg():
    return load_config("configs/label_cowp_v16_8.yaml", "configs/data.yaml")


def _traj(x0: float, y0: float, speed: float, heading: float = 0.0, n: int = 80) -> np.ndarray:
    t = np.arange(1, n + 1, dtype=np.float32) * 0.1
    out = np.zeros((n, 7), dtype=np.float32)
    out[:, 0] = x0 + np.cos(heading) * speed * t
    out[:, 1] = y0 + np.sin(heading) * speed * t
    out[:, 2] = heading
    out[:, 3] = np.cos(heading) * speed
    out[:, 4] = np.sin(heading) * speed
    out[:, 5] = 4.8
    out[:, 6] = 1.9
    return out


def test_unsafe_between_bool_matches_full_predicate():
    cfg = _cfg()
    cases = [
        (_traj(0, 0, 8), _traj(50, 0, 8)),
        (_traj(0, 0, 8), _traj(8, 0, 3)),
        (_traj(0, 0, 8), _traj(0, 3.2, 8)),
        (_traj(0, 0, 5), _traj(15, 0, 0)),
        (_traj(0, 0, 6, 0.2), _traj(10, 4, 4, -0.1)),
    ]
    rng = np.random.default_rng(7)
    for _ in range(20):
        cases.append((
            _traj(float(rng.uniform(-5, 5)), float(rng.uniform(-5, 5)), float(rng.uniform(0, 12)), float(rng.uniform(-.4, .4))),
            _traj(float(rng.uniform(-5, 20)), float(rng.uniform(-5, 5)), float(rng.uniform(0, 12)), float(rng.uniform(-.4, .4))),
        ))
    for ego, agent in cases:
        assert unsafe_between_bool(ego, agent, cfg, agent_type=1) == unsafe_between(ego, agent, cfg, agent_type=1).unsafe


def test_root_recovery_min_only_preserves_best_and_low_ok():
    cfg = _cfg()
    root = _traj(0, 0, 7)
    bank = build_root_recovery_trajectory_bank(root, cfg)
    static = prepare_root_recovery_burden_bank(root, bank, cfg, object_type=1, rho=PriorityRelation.AGENT_PRIORITY)
    for ego in (_traj(20, 0, 3), _traj(4, 0, 2), _traj(0, 4, 6), _traj(30, 0, 10)):
        full = root_conditioned_recovery_search(
            root, ego, cfg, object_type=1, beta=.55, rho=PriorityRelation.AGENT_PRIORITY,
            trajectory_bank=bank, static_burden_bank=static, min_only=False,
        )
        fast = root_conditioned_recovery_search(
            root, ego, cfg, object_type=1, beta=.55, rho=PriorityRelation.AGENT_PRIORITY,
            trajectory_bank=bank, static_burden_bank=static, min_only=True,
        )
        assert full[0] == fast[0]
        assert full[1] == fast[1]


def test_safe_budget_prepared_early_stop_preserves_returned_rows():
    cfg_fast = _cfg()
    cfg_ref = copy.deepcopy(cfg_fast)
    cfg_ref.setdefault("engineering", {})["unsafe_bool_fastpath"] = False
    cfg_ref["engineering"]["safe_budget_early_stop_fastpath"] = False
    current = np.asarray([0, 0, 0, 7, 0, 7, 0, 4.8, 1.9], dtype=np.float32)
    natural = _traj(0, 0, 7)
    raw = build_safe_budget_trajectory_bank(current, 80, .1, cfg_fast)
    prepared = prepare_safe_budget_trajectory_bank(
        raw, object_type=1, cfg=cfg_fast, natural_ref=natural, rho=PriorityRelation.AGENT_PRIORITY,
    )
    for ego in (_traj(25, 0, 2), _traj(3, 0, 1), _traj(0, 4, 8)):
        ref = typed_safe_budget_search_evaluated(
            current, 80, .1, ego, 1, cfg_ref, natural_ref=natural,
            rho=PriorityRelation.AGENT_PRIORITY, trajectory_bank=raw,
        )
        fast = typed_safe_budget_search_evaluated(
            current, 80, .1, ego, 1, cfg_fast, natural_ref=natural,
            rho=PriorityRelation.AGENT_PRIORITY, trajectory_bank=prepared,
        )
        assert len(ref) == len(fast)
        for a, b in zip(ref, fast):
            assert np.array_equal(a[0], b[0])
            assert a[1] == b[1]
            assert a[2] == b[2]
            assert a[3] == b[3]
            assert np.array_equal(np.asarray(a[4]), np.asarray(b[4]))
