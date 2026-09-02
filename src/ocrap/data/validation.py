from __future__ import annotations

REQUIRED_SAMPLE_FIELDS = [
    "scene_id", "original_scenario_id", "time_index", "candidate_index", "split_id", "is_nominal",
    "agent_history", "agent_valid", "map_polylines", "map_valid", "dynamic_map", "route", "bev_occ",
    "prefix_states", "prefix_controls", "prefix_macro_id", "prefix_macro_name", "prefix_param", "utility",
    "hard_violation", "harm_proxy", "feasible", "prefix_diagnostics", "future_probs", "future_sources",
    "future_metadata", "future_valid", "root_assignments", "root_probs", "root_signature", "root_future_signature",
    "root_valid", "root_representative_future_id", "future_to_root_weight", "within_root_obs_dispersion",
    "obs_distance", "y_obs", "c_star", "m_star", "option_valid", "recovery_modes", "recovery_params",
    "r_orc_star", "r_dep_star", "oracle_gap_star", "i_art_star", "regime_label", "valid_masks",
    "teacher_diagnostics",
]


def missing_fields(keys: set[str]) -> list[str]:
    return [k for k in REQUIRED_SAMPLE_FIELDS if k not in keys]
