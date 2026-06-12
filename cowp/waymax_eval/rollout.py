from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from typing import Callable

import numpy as np

from cowp.utils.progress import tqdm_iter
from cowp.waymax_eval.baselines import planner_for_method
from cowp.waymax_eval.metrics_cowp import metrics_from_labels, witness_quality
from cowp.waymax_eval.metrics_standard import waymax_metric_dict


def _label_from_batch_item(batch, i: int) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for k, v in batch.items():
        if k.startswith("cowp/") or k.startswith("map/") or k.startswith("scenario/") or k.startswith("dataset/"):
            try:
                out[k] = v[i].detach().cpu().numpy()
            except Exception:
                pass
    return out


def _select_from_learned(batch, pred, *, witness_threshold: float = 0.5) -> tuple[list[int], list[np.ndarray]]:
    import torch

    scores = pred["planner_score"].detach()
    witness_prob = torch.sigmoid(pred["witness"]["exist_logits"]).detach()
    opr = pred["witness"]["opr"].detach()
    cand_valid = batch["cowp/candidates/valid"].bool()
    conventional = batch.get("cowp/candidates/conventional_safe", cand_valid).bool()
    # Learned COWP selection: conventional geometric safety remains a hard rule,
    # but witness rejection and ranking use model predictions rather than label certificates.
    predicted_bad = (witness_prob >= witness_threshold).any(dim=-1)
    accepted = cand_valid & conventional & ~predicted_bad
    # Option preservation is a soft predicted gate; do not discard all candidates solely
    # due to a noisy OPR estimate in early checkpoints.
    accepted = accepted & (opr.min(dim=-1).values >= 0.05)
    selected: list[int] = []
    masks: list[np.ndarray] = []
    B = scores.shape[0]
    for b in range(B):
        mask = accepted[b]
        if mask.any():
            masked = torch.where(mask, scores[b], torch.full_like(scores[b], float("inf")))
            selected.append(int(torch.argmin(masked).item()))
        else:
            fallback = cand_valid[b] & conventional[b]
            if fallback.any():
                masked = torch.where(fallback, scores[b], torch.full_like(scores[b], float("inf")))
                selected.append(int(torch.argmin(masked).item()))
            elif cand_valid[b].any():
                masked = torch.where(cand_valid[b], scores[b], torch.full_like(scores[b], float("inf")))
                selected.append(int(torch.argmin(masked).item()))
            else:
                selected.append(-1)
        masks.append(mask.detach().cpu().numpy())
    return selected, masks


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


def learned_offline_candidate_eval(
    cache_dir: str | Path,
    checkpoint: str | Path,
    cfg: dict,
    *,
    batch_size: int = 8,
    device: str = "auto",
    witness_threshold: float = 0.5,
    progress: bool = True,
) -> dict[str, object]:
    import torch
    from torch.utils.data import DataLoader

    from cowp.data.dataset import TorchCOWPDataset, collate_torch
    from cowp.models.cowp_model import COWPModel

    dev = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device))
    ckpt = torch.load(checkpoint, map_location=dev)
    model_cfg = ckpt.get("cfg", cfg)
    model = COWPModel(model_cfg).to(dev)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ds = TorchCOWPDataset(cache_dir)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_torch)
    labels: list[dict[str, np.ndarray]] = []
    selected: list[int] = []
    witness_rows: list[dict[str, float]] = []
    iterator = tqdm_iter(dl, enabled=progress, total=len(dl), desc="Learned offline COWP eval", unit="batch")
    with torch.no_grad():
        for batch in iterator:
            batch = {k: v.to(dev) for k, v in batch.items() if torch.is_tensor(v)}
            pred = model(batch)
            batch_selected, _ = _select_from_learned(batch, pred, witness_threshold=witness_threshold)
            selected.extend(batch_selected)
            B = len(batch_selected)
            for i in range(B):
                labels.append(_label_from_batch_item(batch, i))
            cand_mask = batch["cowp/candidates/valid"].bool()
            crit_mask = batch["cowp/critical/valid"].bool()
            pair_mask = (cand_mask[:, :, None] & crit_mask[:, None, :]).detach().cpu().numpy()
            pred_exists = (torch.sigmoid(pred["witness"]["exist_logits"]) >= witness_threshold).detach().cpu().numpy()
            pred_token = pred["witness"]["token_logits"].argmax(dim=-1).detach().cpu().numpy()
            pred_interval = pred["witness"]["conflict_interval"].round().detach().cpu().numpy()
            gt_exists = batch["cowp/witness/exists"].bool().detach().cpu().numpy()
            gt_token = batch["cowp/witness/token"].long().detach().cpu().numpy()
            gt_interval = batch["cowp/witness/conflict_interval"].detach().cpu().numpy()
            for i in range(B):
                witness_rows.append(witness_quality(pred_exists[i], pred_token[i], pred_interval[i], gt_exists[i], gt_token[i], gt_interval[i], pair_mask[i]))
            if hasattr(iterator, "set_postfix"):
                iterator.set_postfix(done=len(selected), last=batch_selected[-1] if batch_selected else -1, refresh=True)
    metrics = metrics_from_labels(selected, labels)
    if witness_rows:
        keys = sorted(witness_rows[0])
        metrics.update({f"WitnessQuality/{k}": float(np.mean([r[k] for r in witness_rows])) for k in keys})
    metrics["num_scenes"] = len(selected)
    metrics["mode"] = "learned_offline"
    metrics["witness_threshold"] = float(witness_threshold)
    return metrics


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


def _make_waymax_environment(max_num_objects: int | None = None):
    try:
        import dataclasses
        from waymax import config as _config  # type: ignore
        from waymax import dynamics  # type: ignore
        from waymax import env as waymax_env  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError("waymax.env, waymax.config and waymax.dynamics are required for real closed-loop rollout.") from exc

    dynamics_cls = getattr(dynamics, "StateDynamics", None) or getattr(dynamics, "DeltaGlobal", None)
    env_cls = getattr(waymax_env, "MultiAgentEnvironment", None)
    if dynamics_cls is None or env_cls is None:
        raise RuntimeError("Unsupported Waymax API: expected waymax.dynamics.StateDynamics/DeltaGlobal and waymax.env.MultiAgentEnvironment.")
    dyn = dynamics_cls()
    env_config = getattr(_config, "EnvironmentConfig", lambda: None)()
    kwargs = {}
    if hasattr(_config, "ObjectType") and hasattr(_config.ObjectType, "VALID"):
        kwargs["controlled_object"] = _config.ObjectType.VALID
    if max_num_objects is not None:
        kwargs["max_num_objects"] = int(max_num_objects)
    if env_config is not None and kwargs:
        try:
            env_config = dataclasses.replace(env_config, **kwargs)
        except Exception:
            pass
    try:
        return env_cls(dynamics_model=dyn, config=env_config)
    except TypeError:
        try:
            return env_cls(dynamics_model=dyn)
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

    horizon = int(horizon_steps) if horizon_steps is not None else 80
    gen = waymax_state_generator(data_config)
    total = num_scenarios
    iterator = tqdm_iter(gen, enabled=progress, total=total, desc="Waymax closed-loop rollout", unit="scenario")
    outputs = []
    for scenario_index, init_state in enumerate(iterator):
        env = _make_waymax_environment(max_num_objects=getattr(init_state, "num_objects", None))
        state = env.reset(init_state)
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
