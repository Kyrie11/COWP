from __future__ import annotations

from cowp.planning.cowp_planner import COWPPlanner


def ablation_for_method(method: str) -> dict:
    table = {
        "cowp": {},
        "idm_lattice": {"use_hard_witness_rejection": False, "use_option_preservation": False, "soft_burden_cost_only": False},
        "cowp_wo_counterfactual": {"use_neutral_branch": False, "use_priority_branch": False},
        "cowp_wo_option_preservation": {"use_option_preservation": False},
        "cowp_wo_witness_rejection": {"use_hard_witness_rejection": False},
        "soft_burden_cost_only": {"use_hard_witness_rejection": False, "soft_burden_cost_only": True},
        "cowp_wo_dual_edge": {"use_dual_edge": False},
        "cowp_wo_conflict_query": {"use_conflict_query": False},
    }
    if method not in table:
        raise ValueError(f"Unknown method/ablation: {method}")
    return table[method]


def planner_for_method(method: str, cfg: dict) -> COWPPlanner:
    return COWPPlanner(cfg, ablation=ablation_for_method(method))
