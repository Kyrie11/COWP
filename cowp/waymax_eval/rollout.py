from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from typing import Callable

import numpy as np

from cowp.utils.progress import tqdm_iter
from cowp.waymax_eval.baselines import planner_for_method
from cowp.waymax_eval.metrics_cowp import metrics_from_labels
from cowp.waymax_eval.metrics_standard import waymax_metric_dict


def offline_candidate_eval(labels_dir: str | Path, cfg: dict, method: str = "cowp", progress: bool = True) -> dict[str, float]:
    labels = []
    selected = []
    planner = planner_for_method(method, cfg)
    paths = sorted(Path(labels_dir).glob("*.npz"))
    iterator = tqdm_iter(paths, enabled=progress, total=len(paths), desc=f"Offline COWP label eval ({method})", unit="file")
    for path in iterator:
        with np.load(path, allow_pickle=True) as data:
            label = {k: data[k] for k in data.files}
        decision = planner.select_from_labels(label)
        labels.append(label)
        selected.append(decision.candidate_index)
        if hasattr(iterator, "set_postfix"):
            iterator.set_postfix(selected=decision.candidate_index, reason=decision.reason[:24], refresh=True)
    return metrics_from_labels(selected, labels)


def import_policy_fn(spec: str) -> Callable:
    if ":" not in spec:
        raise ValueError("Policy spec must be 'module.submodule:function_name'.")
    module_name, fn_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    fn = getattr(module, fn_name)
    if not callable(fn):
        raise TypeError(f"Imported policy {spec} is not callable.")
    return fn


def _call_policy(policy_fn: Callable, state, step: int, scenario_index: int):
    try:
        sig = inspect.signature(policy_fn)
        kwargs = {}
        if "step" in sig.parameters:
            kwargs["step"] = step
        if "scenario_index" in sig.parameters:
            kwargs["scenario_index"] = scenario_index
        return policy_fn(state, **kwargs)
    except (TypeError, ValueError):
        return policy_fn(state)


def _make_waymax_environment():
    try:
        from waymax import dynamics  # type: ignore
        from waymax import env as waymax_env  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError("waymax.env and waymax.dynamics are required for real closed-loop rollout.") from exc

    dynamics_cls = getattr(dynamics, "StateDynamics", None) or getattr(dynamics, "DeltaGlobal", None)
    env_cls = getattr(waymax_env, "MultiAgentEnvironment", None)
    if dynamics_cls is None or env_cls is None:
        raise RuntimeError("Unsupported Waymax API: expected waymax.dynamics.StateDynamics/DeltaGlobal and waymax.env.MultiAgentEnvironment.")
    dyn = dynamics_cls()
    try:
        return env_cls(dynamics_model=dyn)
    except TypeError:
        try:
            return env_cls(dynamics=dyn)
        except TypeError:
            return env_cls(dyn)


def _env_step(env, state, action):
    result = env.step(state, action)
    if isinstance(result, tuple):
        return result[0]
    return result


def _state_done(state) -> bool:
    for attr in ("is_done", "done"):
        val = getattr(state, attr, None)
        if val is None:
            continue
        try:
            return bool(np.asarray(val).all())
        except Exception:
            try:
                return bool(val)
            except Exception:
                return False
    return False


def waymax_closed_loop_rollout(
    data_config,
    policy_fn: Callable,
    num_scenarios: int | None = None,
    horizon_steps: int | None = None,
    progress: bool = True,
    compute_standard_metrics: bool = False,
):
    """Run real Waymax closed-loop simulation by stepping a Waymax environment.

    ``policy_fn`` must return a Waymax-compatible action for the current
    SimulatorState.  This function intentionally does not use proto-derived COWP
    labels; it is the closed-loop path for validating a planner in Waymax.
    """
    from cowp.waymax_eval.dataloader import waymax_state_generator

    env = _make_waymax_environment()
    horizon = int(horizon_steps) if horizon_steps is not None else 80
    gen = waymax_state_generator(data_config)
    total = num_scenarios
    iterator = tqdm_iter(gen, enabled=progress, total=total, desc="Waymax closed-loop rollout", unit="scenario")
    outputs = []
    for scenario_index, init_state in enumerate(iterator):
        state = init_state
        steps = 0
        for step in range(horizon):
            action = _call_policy(policy_fn, state, step=step, scenario_index=scenario_index)
            state = _env_step(env, state, action)
            steps += 1
            if _state_done(state):
                break
        item = {"state": state, "steps": steps}
        if compute_standard_metrics:
            item["standard_metrics"] = waymax_metric_dict(state)
        outputs.append(item)
        if hasattr(iterator, "set_postfix"):
            iterator.set_postfix(done=len(outputs), steps=steps, refresh=True)
        if num_scenarios is not None and len(outputs) >= num_scenarios:
            break
    return outputs
