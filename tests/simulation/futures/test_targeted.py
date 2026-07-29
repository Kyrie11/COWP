from ocrap.data.build.synthetic import make_synthetic_scenario
from ocrap.data.build.history import construct_history
from ocrap.planning.prefix_generation import generate_candidate_prefixes
from ocrap.simulation.futures.targeted import targeted_perturbation
from ocrap.config.defaults import DEFAULT_CONFIG


def test_hidden_vehicle_pair_starts_after_prefix_and_from_unknown():
    cfg = DEFAULT_CONFIG.copy()
    cfg.update({"bev_resolution_m": 4.0, "num_candidate_prefixes": 2})
    raw = make_synthetic_scenario(0, cfg, artifact=True)
    hist = construct_history(raw, 10, cfg)
    prefix = generate_candidate_prefixes(hist, cfg)[0]
    fut = targeted_perturbation(hist, prefix, 50, "hidden_vehicle_yields", 5, 0.1, cfg)
    assert fut is not None
    assert fut.metadata["hidden_emergence"] is True
    assert fut.metadata["from_unknown_mask"] is True
    start = prefix.prefix_states.shape[0] + int(cfg["hidden_emergence_delay_steps"])
    slot_valid = fut.agent_valid[:, -1] if fut.agent_valid[:, -1].any() else fut.agent_valid.any(axis=1)
    assert slot_valid[:start].sum() == 0 or fut.metadata["hidden_spawn_cell"]
