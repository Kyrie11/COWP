from __future__ import annotations

from ocrap.data.schema import CandidatePrefix, CounterfactualFuture, SceneHistory
from ocrap.utils.seed import stable_seed

from .priors import normalize_future_priors
from .reactive import reactive_future, ReactivePolicy
from .replay import replay_future
from .targeted import TARGETED_KINDS, mine_artifact_futures, targeted_perturbation


def generate_counterfactual_futures(history: SceneHistory, prefix: CandidatePrefix, cfg: dict) -> list[CounterfactualFuture]:
    if str(cfg.get("simulation_backend", "ocrap_surrogate")) == "waymax_closed_loop":
        from ocrap.simulation.waymax_rollout import generate_waymax_counterfactual_futures

        return generate_waymax_counterfactual_futures(history, prefix, cfg)
    total_steps = int(round((float(cfg.get("prefix_horizon_s", 1.0)) + float(cfg.get("recovery_horizon_s", 4.0))) * float(cfg.get("sample_rate_hz", 10))))
    futures: list[CounterfactualFuture] = []
    priors = cfg.get("future_priors", {})
    replay_prior = float(priors.get("replay", 0.25))
    reactive_total = float(priors.get("reactive", 0.35))
    targeted_total = float(priors.get("targeted", 0.40))
    futures.append(replay_future(history, prefix, total_steps, replay_prior))
    n_reactive = int(cfg.get("num_reactive_futures", 4))
    for i in range(n_reactive):
        futures.append(reactive_future(history, prefix, total_steps, len(futures), reactive_total / max(n_reactive, 1), cfg))
    n_targeted = int(cfg.get("num_targeted_futures", 8))
    targeted_added = 0
    artifact_cfg = cfg.get("artifact", {}) if isinstance(cfg.get("artifact", {}), dict) else {}
    mine_prob = float(artifact_cfg.get("mine_probability", 1.0 if artifact_cfg.get("force_mine", True) else 0.0))
    mine_rng = __import__("numpy").random.default_rng(stable_seed("surrogate-mine", history.scene_id, history.time_index, prefix.macro_id))
    do_mine = bool(artifact_cfg.get("force_mine", True)) and float(mine_rng.random()) < max(0.0, min(1.0, mine_prob))
    mined = mine_artifact_futures(history, prefix, total_steps, len(futures), targeted_total / max(n_targeted, 1), cfg) if do_mine else []
    for f in mined:
        futures.append(f)
        targeted_added += 1
    kinds = cfg.get("targeted_future_kinds", TARGETED_KINDS)
    if not isinstance(kinds, (list, tuple)) or not kinds:
        kinds = TARGETED_KINDS
    kinds = [str(k) for k in kinds]
    kind_cursor = 0
    while targeted_added < n_targeted:
        kind = kinds[kind_cursor % len(kinds)]
        kind_cursor += 1
        # Avoid duplicating the mined pair too heavily; stress futures create non-degenerate observations.
        if mined and kind in {"hidden_vehicle_yields", "hidden_vehicle_accelerates"}:
            continue
        fut = targeted_perturbation(history, prefix, total_steps, kind, len(futures), targeted_total / max(n_targeted, 1), cfg)
        if fut is not None:
            futures.append(fut)
            targeted_added += 1
        if kind_cursor > len(kinds) * 4 and targeted_added == 0:
            break
    normalize_future_priors(futures)
    return futures
