from __future__ import annotations

from pathlib import Path

import numpy as np

from cowp.waymax_eval.baselines import planner_for_method
from cowp.waymax_eval.metrics_cowp import metrics_from_labels


def offline_candidate_eval(labels_dir: str | Path, cfg: dict, method: str = "cowp") -> dict[str, float]:
    labels = []
    selected = []
    planner = planner_for_method(method, cfg)
    for path in sorted(Path(labels_dir).glob("*.npz")):
        data = np.load(path, allow_pickle=True)
        label = {k: data[k] for k in data.files}
        decision = planner.select_from_labels(label)
        labels.append(label)
        selected.append(decision.candidate_index)
    return metrics_from_labels(selected, labels)


def waymax_closed_loop_rollout(data_config, policy_fn, num_scenarios: int | None = None):
    from cowp.waymax_eval.dataloader import waymax_state_generator

    gen = waymax_state_generator(data_config)
    states = []
    count = 0
    for state in gen:
        rollout_state = policy_fn(state)
        states.append(rollout_state)
        count += 1
        if num_scenarios is not None and count >= num_scenarios:
            break
    return states
