from __future__ import annotations

import importlib
import inspect
import gc
from pathlib import Path
from typing import Callable

import numpy as np

from cowp.utils.progress import tqdm_iter
from cowp.waymax_eval.baselines import planner_for_method
from cowp.waymax_eval.metrics_cowp import metrics_from_labels, witness_quality
from cowp.waymax_eval.metrics_standard import WaymaxStandardMetricAccumulator


def _label_from_batch_item(batch, i: int) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for k, v in batch.items():
        if k.startswith("cowp/") or k.startswith("map/") or k.startswith("scenario/") or k.startswith("dataset/"):
            try:
                out[k] = v[i].detach().cpu().numpy()
            except Exception:
                pass
    return out


def _select_from_learned(batch, pred, *, witness_threshold: float = 0.5, alpha_opr: float = 0.35) -> tuple[list[int], list[np.ndarray]]:
    import torch

    scores = pred["planner_score"].detach()
    witness_prob = torch.sigmoid(pred["witness"]["exist_logits"]).detach()
    opr = pred["witness"]["opr"].detach()
    cand_valid = batch["cowp/candidates/valid"].bool()
    conventional = batch.get("cowp/candidates/conventional_safe", cand_valid).bool()
    crit_mask = batch.get("cowp/critical/valid")
    if crit_mask is not None and witness_prob.ndim == 3:
        cm = crit_mask.bool()[:, None, :]
        witness_prob = torch.where(cm, witness_prob, torch.zeros_like(witness_prob))
        opr = torch.where(cm, opr, torch.ones_like(opr))
    # Learned COWP selection: conventional geometric safety remains a hard rule,
    # but witness rejection and option preservation use model predictions rather
    # than label certificates.
    predicted_bad = (witness_prob >= witness_threshold).any(dim=-1)
    accepted = cand_valid & conventional & ~predicted_bad
    accepted = accepted & (opr.min(dim=-1).values >= float(alpha_opr))
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



def _average_precision_binary(score: np.ndarray, target: np.ndarray) -> float:
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=bool).reshape(-1)
    finite = np.isfinite(score)
    score, target = score[finite], target[finite]
    if score.size == 0 or int(target.sum()) == 0:
        return 0.0
    order = np.argsort(-score)
    y = target[order].astype(np.float64)
    tp = np.cumsum(y)
    precision = tp / np.maximum(np.arange(1, len(y) + 1, dtype=np.float64), 1.0)
    return float((precision * y).sum() / max(float(y.sum()), 1.0))


def _ranking_pair_accuracy(scores: np.ndarray, ncf: np.ndarray, false_safe: np.ndarray, valid: np.ndarray) -> tuple[int, int]:
    good = 0
    total = 0
    for b in range(scores.shape[0]):
        pos = np.where(valid[b] & ncf[b])[0]
        neg = np.where(valid[b] & false_safe[b])[0]
        if len(pos) == 0 or len(neg) == 0:
            continue
        total += int(len(pos) * len(neg))
        good += int((scores[b, pos[:, None]] < scores[b, neg[None, :]]).sum())
    return good, total


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
    selected_ncf = 0
    selected_false_safe = 0
    selected_conventional = 0
    accepted_total = 0
    valid_total = 0
    accepted_ncf = 0
    total_ncf = 0
    accepted_false_safe = 0
    total_false_safe = 0
    pair_scores: list[np.ndarray] = []
    pair_targets: list[np.ndarray] = []
    rank_good = 0
    rank_total = 0
    iterator = tqdm_iter(dl, enabled=progress, total=len(dl), desc="Learned offline COWP eval", unit="batch")
    with torch.no_grad():
        for batch in iterator:
            batch = {k: v.to(dev) for k, v in batch.items() if torch.is_tensor(v)}
            pred = model(batch, stage="planner")
            if "critical_mask" in pred:
                batch = dict(batch)
                batch["cowp/critical/valid"] = pred["critical_mask"].bool()
            alpha = float(cfg.get("planning", {}).get("alpha_opr_infer", cfg.get("ncf", {}).get("alpha_opr", 0.35)))
            batch_selected, batch_accepted_masks = _select_from_learned(batch, pred, witness_threshold=witness_threshold, alpha_opr=alpha)
            selected.extend(batch_selected)
            B = len(batch_selected)
            batch_labels = []
            for i in range(B):
                item = _label_from_batch_item(batch, i)
                batch_labels.append(item)
                labels.append(item)
                sel = int(batch_selected[i])
                if sel >= 0:
                    selected_ncf += int(bool(item.get("cowp/candidates/noncoercive_feasible", np.zeros(1, dtype=bool))[sel]))
                    selected_false_safe += int(bool(item.get("cowp/candidates/false_safe", np.zeros(1, dtype=bool))[sel]))
                    selected_conventional += int(bool(item.get("cowp/candidates/conventional_safe", np.zeros(1, dtype=bool))[sel]))
                    accepted_mask_np = np.asarray(batch_accepted_masks[i], dtype=bool)
                    valid_np = np.asarray(item["cowp/candidates/valid"], dtype=bool)
                    ncf_np = np.asarray(item.get("cowp/candidates/noncoercive_feasible", np.zeros_like(valid_np)), dtype=bool) & valid_np
                    fs_np = np.asarray(item.get("cowp/candidates/false_safe", np.zeros_like(valid_np)), dtype=bool) & valid_np
                    accepted_total += int((accepted_mask_np & valid_np).sum())
                    valid_total += int(valid_np.sum())
                    accepted_ncf += int((accepted_mask_np & ncf_np).sum())
                    total_ncf += int(ncf_np.sum())
                    accepted_false_safe += int((accepted_mask_np & fs_np).sum())
                    total_false_safe += int(fs_np.sum())
            cand_mask = batch["cowp/candidates/valid"].bool()
            crit_mask = batch["cowp/critical/valid"].bool()
            pair_mask = (cand_mask[:, :, None] & crit_mask[:, None, :]).detach().cpu().numpy()
            pred_exists = (torch.sigmoid(pred["witness"]["exist_logits"]) >= witness_threshold).detach().cpu().numpy()
            pred_token = pred["witness"]["token_logits"].argmax(dim=-1).detach().cpu().numpy()
            pred_interval = pred["witness"]["conflict_interval"].round().detach().cpu().numpy()
            gt_exists = batch["cowp/witness/exists"].bool().detach().cpu().numpy()
            gt_token = batch["cowp/witness/token"].long().detach().cpu().numpy()
            gt_interval = batch["cowp/witness/conflict_interval"].detach().cpu().numpy()
            pair_score_np = torch.sigmoid(pred["witness"]["exist_logits"]).detach().cpu().numpy()
            pair_scores.append(pair_score_np[pair_mask])
            pair_targets.append(gt_exists[pair_mask])
            score_np = pred["planner_score"].detach().cpu().numpy()
            ncf_np = batch["cowp/candidates/noncoercive_feasible"].bool().detach().cpu().numpy()
            fs_np = batch["cowp/candidates/false_safe"].bool().detach().cpu().numpy()
            valid_np = cand_mask.detach().cpu().numpy()
            g, t = _ranking_pair_accuracy(score_np, ncf_np, fs_np, valid_np)
            rank_good += g
            rank_total += t
            for i in range(B):
                witness_rows.append(witness_quality(pred_exists[i], pred_token[i], pred_interval[i], gt_exists[i], gt_token[i], gt_interval[i], pair_mask[i]))
            if hasattr(iterator, "set_postfix"):
                iterator.set_postfix(done=len(selected), last=batch_selected[-1] if batch_selected else -1, refresh=True)
    metrics = metrics_from_labels(selected, labels)
    if witness_rows:
        keys = sorted(witness_rows[0])
        metrics.update({f"WitnessQuality/{k}": float(np.mean([r[k] for r in witness_rows])) for k in keys})
    metrics["SelectedNCFRate"] = float(selected_ncf / max(len(selected), 1))
    metrics["SelectedFalseSafeRate"] = float(selected_false_safe / max(len(selected), 1))
    metrics["SelectedConventionalSafeRate"] = float(selected_conventional / max(len(selected), 1))
    metrics["LearnedAcceptedCandidateRate"] = float(accepted_total / max(valid_total, 1))
    metrics["LearnedAcceptNCFRecall"] = float(accepted_ncf / max(total_ncf, 1))
    metrics["LearnedAcceptFalseSafeRate"] = float(accepted_false_safe / max(total_false_safe, 1))
    if pair_scores:
        metrics["WitnessQuality/AUPRC"] = _average_precision_binary(np.concatenate(pair_scores), np.concatenate(pair_targets))
    metrics["PlannerRankingPairAccuracy"] = float(rank_good / max(rank_total, 1)) if rank_total else 0.0
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


def _make_waymax_environment(max_num_objects: int | None = None, action_mode: str = "delta_xy_yaw"):
    try:
        import dataclasses
        from waymax import config as _config  # type: ignore
        from waymax import dynamics  # type: ignore
        from waymax import env as waymax_env  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError("waymax.env, waymax.config and waymax.dynamics are required for real closed-loop rollout.") from exc

    # Waymax has used both names in docs/examples.  Current public Waymax
    # exposes env.BaseEnvironment; some older/local builds exposed the alias
    # MultiAgentEnvironment.  Accept both.
    env_cls = getattr(waymax_env, "MultiAgentEnvironment", None) or getattr(waymax_env, "BaseEnvironment", None)
    if env_cls is None:
        try:
            from waymax.env.base_environment import BaseEnvironment as env_cls  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Unsupported Waymax API: expected waymax.env.BaseEnvironment or MultiAgentEnvironment.") from exc

    # The built-in COWP policy emits a 3-D delta action by default.  Therefore
    # DeltaGlobal is the correct dynamics for --waymax-action-mode delta_xy_yaw.
    # StateDynamics expects a 5-D absolute [x, y, yaw, vel_x, vel_y] action.
    if action_mode == "absolute_xy_yaw":
        dynamics_names = ("StateDynamics", "DeltaGlobal")
    else:
        dynamics_names = ("DeltaGlobal", "StateDynamics")
    dynamics_cls = next((getattr(dynamics, name, None) for name in dynamics_names if getattr(dynamics, name, None) is not None), None)
    if dynamics_cls is None:
        raise RuntimeError(f"Unsupported Waymax API: expected one of {dynamics_names} in waymax.dynamics.")
    dyn = dynamics_cls()

    env_config = getattr(_config, "EnvironmentConfig", lambda: None)()
    kwargs = {}
    # The built-in policy controls only the SDC/ego.  Do not set VALID here;
    # doing so freezes or overwrites every non-ego object with zero actions.
    if hasattr(_config, "ObjectType") and hasattr(_config.ObjectType, "SDC"):
        kwargs["controlled_object"] = _config.ObjectType.SDC
    if max_num_objects is not None:
        kwargs["max_num_objects"] = int(max_num_objects)
    if env_config is not None and kwargs:
        try:
            valid_fields = {f.name for f in dataclasses.fields(env_config)} if dataclasses.is_dataclass(env_config) else set(kwargs)
            env_config = dataclasses.replace(env_config, **{k: v for k, v in kwargs.items() if k in valid_fields})
        except Exception:
            pass
    try:
        return env_cls(dynamics_model=dyn, config=env_config)
    except TypeError:
        try:
            return env_cls(dyn, env_config)
        except TypeError:
            try:
                return env_cls(dynamics_model=dyn)
            except TypeError:
                return env_cls(dyn)



def _clear_accelerator_caches() -> None:
    """Best-effort cleanup between scenarios for mixed JAX + PyTorch rollout.

    The important memory fix is not retaining SimulatorState objects in the
    rollout output.  This helper only releases Python references/caches sooner.
    """
    gc.collect()
    try:
        import jax  # type: ignore

        clear = getattr(jax, "clear_caches", None)
        if callable(clear):
            clear()
    except Exception:
        pass
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _consume_policy_diagnostics(policy_fn: Callable) -> dict | None:
    consumer = getattr(policy_fn, "consume_diagnostics", None)
    if callable(consumer):
        try:
            row = consumer()
            return row if isinstance(row, dict) else None
        except Exception:
            return None
    return None

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
    action_mode: str = "delta_xy_yaw",
    keep_rollout_state: bool = False,
    clear_accelerator_cache: bool = False,
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
        max_objects = getattr(init_state, "num_objects", None)
        if max_objects is None and hasattr(init_state, "log_trajectory"):
            max_objects = getattr(init_state.log_trajectory, "num_objects", None)
        env = _make_waymax_environment(max_num_objects=max_objects, action_mode=action_mode)
        state = env.reset(init_state)
        steps = 0
        policy_diagnostics = []
        metric_acc = WaymaxStandardMetricAccumulator() if compute_standard_metrics else None
        for step in range(horizon):
            action = _call_policy(policy_fn, state, step=step, scenario_index=scenario_index)
            diag = _consume_policy_diagnostics(policy_fn)
            if diag is not None:
                policy_diagnostics.append(diag)
            state = _env_step(env, state, action)
            steps += 1
            if metric_acc is not None:
                # Waymax's built-in metrics are per-current-timestep metrics.  Update
                # after every simulator step so episode CR/offroad/wrong-way are
                # any-over-rollout events, not just the final frame.
                metric_acc.update(state)
            if _state_done(state):
                break
        if metric_acc is not None and steps == 0:
            metric_acc.update(state)
        # Do not retain SimulatorState by default.  A Waymax SimulatorState can keep
        # JAX device buffers alive; storing one per scenario causes closed-loop eval
        # memory to grow with --num-scenarios.  Metrics are computed before dropping
        # the state, so the JSON payload still contains the required evaluation info.
        item = {"steps": steps, "policy_diagnostics": policy_diagnostics}
        if compute_standard_metrics and metric_acc is not None:
            item["standard_metrics"] = metric_acc.finalize()
        if keep_rollout_state:
            item["state"] = state
        outputs.append(item)
        if clear_accelerator_cache:
            _clear_accelerator_caches()
        if hasattr(iterator, "set_postfix"):
            iterator.set_postfix(done=len(outputs), steps=steps, refresh=True)
        if num_scenarios is not None and len(outputs) >= num_scenarios:
            break
    return outputs
