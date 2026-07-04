from __future__ import annotations

import importlib
import inspect
import gc
from pathlib import Path
from typing import Callable

import numpy as np

from cowp.utils.progress import tqdm_iter
from cowp.waymax_eval.baselines import planner_for_method
from cowp.waymax_eval.metrics_cowp import metrics_from_labels, witness_quality, _progress_reference_m, _trajectory_progress_m
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


def _select_from_learned(
    batch,
    pred,
    *,
    witness_threshold: float = 0.5,
    alpha_opr: float = 0.35,
    gate_mode: str = "hard",
    secondary_witness_threshold: float = 0.85,
    secondary_opr_alpha: float = 0.10,
    soft_ncf_penalty: float = 1.5,
) -> tuple[list[int], list[np.ndarray]]:
    import torch

    scores = torch.nan_to_num(pred["planner_score"].detach().float(), nan=1e6, posinf=1e6, neginf=-1e6)
    witness_prob = torch.nan_to_num(torch.sigmoid(pred["witness"]["exist_logits"]).detach().float(), nan=1.0, posinf=1.0, neginf=0.0)
    opr = torch.nan_to_num(pred["witness"]["opr"].detach().float(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    cand_valid = batch["cowp/candidates/valid"].bool()
    conventional = batch.get("cowp/candidates/conventional_safe", cand_valid).bool()
    crit_mask = batch.get("cowp/critical/valid")
    if crit_mask is not None and witness_prob.ndim == 3:
        cm = crit_mask.bool()[:, None, :]
        witness_prob = torch.where(cm, witness_prob, torch.zeros_like(witness_prob))
        opr = torch.where(cm, opr, torch.ones_like(opr))
    gate_mode = str(gate_mode or "hard").lower()
    adjusted_scores = scores
    if gate_mode == "hard":
        predicted_bad = (witness_prob >= witness_threshold).any(dim=-1)
        accepted = cand_valid & conventional & ~predicted_bad
        accepted = accepted & (opr.min(dim=-1).values >= float(alpha_opr))
    else:
        # Offline proxy for priority-aware rejection: only hard-veto candidates
        # where a predicted witness coincides with poor option preservation.
        # Other witness signals are used as a soft ranking penalty.
        priority_proxy = (opr < float(alpha_opr)).float()
        primary_bad = ((witness_prob >= witness_threshold) & (priority_proxy > 0.0)).any(dim=-1)
        severe_bad = ((witness_prob >= float(secondary_witness_threshold)) & (opr <= float(secondary_opr_alpha))).any(dim=-1)
        penalty = (witness_prob * priority_proxy).amax(dim=-1) + torch.relu(float(alpha_opr) - opr).amax(dim=-1)
        adjusted_scores = scores + float(soft_ncf_penalty) * penalty
        if gate_mode == "soft":
            accepted = cand_valid & conventional
        else:
            accepted = cand_valid & conventional & ~primary_bad & ~severe_bad
    selected: list[int] = []
    masks: list[np.ndarray] = []
    B = scores.shape[0]
    for b in range(B):
        mask = accepted[b]
        if mask.any():
            masked = torch.where(mask, adjusted_scores[b], torch.full_like(adjusted_scores[b], float("inf")))
            selected.append(int(torch.argmin(masked).item()))
        else:
            fallback = cand_valid[b] & conventional[b]
            if fallback.any():
                masked = torch.where(fallback, adjusted_scores[b], torch.full_like(adjusted_scores[b], float("inf")))
                selected.append(int(torch.argmin(masked).item()))
            elif cand_valid[b].any():
                masked = torch.where(cand_valid[b], adjusted_scores[b], torch.full_like(adjusted_scores[b], float("inf")))
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



_EVAL_LABEL_KEYS = {
    "cowp/candidates/trajectory",
    "cowp/candidates/valid",
    "cowp/candidates/conventional_safe",
    "cowp/candidates/false_safe",
    "cowp/candidates/noncoercive_feasible",
    "cowp/critical/valid",
    "cowp/natural/beta",
    "cowp/witness/exists",
    "cowp/witness/token",
    "cowp/witness/burden_total",
    "cowp/witness/min_safe_burden",
    "cowp/witness/opr",
    "cowp/witness/conflict_interval",
    "waymax/candidate_rollout_valid",
    "waymax/candidate_collision",
    "waymax/candidate_offroad",
    "waymax/candidate_log_divergence",
}


def _slim_label_from_batch_item(batch, i: int) -> dict[str, np.ndarray]:
    """Copy only tensors needed by learned-offline metrics.

    The previous learned-offline evaluator used the generic label extractor while
    loading a non-stage-filtered dataset.  On real caches this can retain large
    response/natural/waymax tensors for every scene until the end of evaluation,
    which explains a host-RAM OOM kill even when GPU memory looks fine.
    """
    out: dict[str, np.ndarray] = {}
    for k in _EVAL_LABEL_KEYS:
        v = batch.get(k)
        if v is None:
            continue
        try:
            out[k] = v[i].detach().cpu().numpy()
        except Exception:
            pass
    return out


class _LabelMetricAccumulator:
    def __init__(self, *, beta_default: float = 0.65) -> None:
        self.beta_default = float(beta_default)
        self.n = 0
        self.cf_count = 0
        self.false_safe = 0
        self.cbs_sum = 0.0
        self.cbs_count = 0
        self.opr_sum = 0.0
        self.opr_count = 0
        self.hbcr = 0
        self.collision_or_offroad = 0
        self.fallback_count = 0
        self.progress_m_sum = 0.0
        self.progress_norm_sum = 0.0

    def add(self, k: int, label: dict[str, np.ndarray]) -> None:
        self.n += 1
        ref_progress = max(_progress_reference_m(label), 1e-6)
        if k < 0:
            self.fallback_count += 1
            return
        valid = np.asarray(label.get("cowp/candidates/valid", []), dtype=bool)
        if k >= len(valid) or not bool(valid[k]):
            self.fallback_count += 1
            return
        conv_arr = np.asarray(label.get("cowp/candidates/conventional_safe", valid), dtype=bool)
        conv = bool(conv_arr[k]) if k < len(conv_arr) else bool(valid[k])
        self.collision_or_offroad += int(not conv)
        traj = np.asarray(label["cowp/candidates/trajectory"])[k]
        p_m = _trajectory_progress_m(traj)
        self.progress_m_sum += float(p_m)
        self.progress_norm_sum += float(np.clip(p_m / ref_progress, 0.0, 1.0))

        crit = np.asarray(label.get("cowp/critical/valid", []), dtype=bool)
        if crit.size == 0:
            return
        wit = np.asarray(label.get("cowp/witness/exists", np.zeros((len(valid), len(crit)), dtype=bool)), dtype=bool)[k] & crit
        self.cf_count += int(conv)
        self.false_safe += int(conv and np.any(wit))
        if np.any(crit):
            burden = np.asarray(label.get("cowp/witness/burden_total", np.zeros((len(valid), len(crit)), dtype=np.float32)), dtype=np.float32)[k]
            opr = np.asarray(label.get("cowp/witness/opr", np.ones((len(valid), len(crit)), dtype=np.float32)), dtype=np.float32)[k]
            self.cbs_sum += float(np.nanmax(np.nan_to_num(burden[crit], nan=0.0, posinf=0.0, neginf=0.0)))
            self.cbs_count += 1
            self.opr_sum += float(np.nanmean(np.nan_to_num(opr[crit], nan=0.0, posinf=1.0, neginf=0.0)))
            self.opr_count += 1
            min_safe = np.asarray(label.get("cowp/witness/min_safe_burden", burden[None, ...].repeat(len(valid), axis=0)), dtype=np.float32)[k]
            beta = np.asarray(label.get("cowp/natural/beta", np.full(len(crit), self.beta_default, dtype=np.float32)), dtype=np.float32)
            if beta.shape[0] < crit.shape[0]:
                beta = np.pad(beta, (0, crit.shape[0] - beta.shape[0]), constant_values=self.beta_default)
            self.hbcr += int(np.any(min_safe[crit] > beta[: crit.shape[0]][crit]))

    def finish(self) -> dict[str, float]:
        return {
            "CR": float(self.collision_or_offroad / max(self.n, 1)),
            "EP": float(self.progress_norm_sum / max(self.n, 1)),
            "EP_m": float(self.progress_m_sum / max(self.n, 1)),
            "FallbackRate": float(self.fallback_count / max(self.n, 1)),
            "FSR": float(self.false_safe / max(self.cf_count, 1)),
            "CBS": float(self.cbs_sum / max(self.cbs_count, 1)),
            "OPR": float(self.opr_sum / max(self.opr_count, 1)),
            "HBCR": float(self.hbcr / max(self.n, 1)),
        }


class _LearnedMetricsAccumulator:
    def __init__(self, *, beta_default: float = 0.65) -> None:
        self.label_metrics = _LabelMetricAccumulator(beta_default=beta_default)
        self.witness_sums: dict[str, float] = {}
        self.witness_count = 0
        self.selected_total = 0
        self.selected_ncf = 0
        self.selected_false_safe = 0
        self.selected_conventional = 0
        self.accepted_total = 0
        self.valid_total = 0
        self.accepted_ncf = 0
        self.total_ncf = 0
        self.accepted_false_safe = 0
        self.total_false_safe = 0
        self.selected_waymax_valid = 0
        self.selected_waymax_collision = 0
        self.selected_waymax_offroad = 0
        self.selected_waymax_logdiv_sum = 0.0

    def add_selection(self, selected_idx: int, accepted_mask: np.ndarray, label: dict[str, np.ndarray]) -> None:
        self.selected_total += 1
        self.label_metrics.add(selected_idx, label)
        valid = np.asarray(label.get("cowp/candidates/valid", []), dtype=bool)
        if valid.size == 0:
            return
        ncf = np.asarray(label.get("cowp/candidates/noncoercive_feasible", np.zeros_like(valid)), dtype=bool) & valid
        fs = np.asarray(label.get("cowp/candidates/false_safe", np.zeros_like(valid)), dtype=bool) & valid
        conv = np.asarray(label.get("cowp/candidates/conventional_safe", valid), dtype=bool) & valid
        accepted = np.asarray(accepted_mask, dtype=bool) & valid
        if selected_idx >= 0 and selected_idx < len(valid):
            self.selected_ncf += int(bool(ncf[selected_idx]))
            self.selected_false_safe += int(bool(fs[selected_idx]))
            self.selected_conventional += int(bool(conv[selected_idx]))
        self.accepted_total += int(accepted.sum())
        self.valid_total += int(valid.sum())
        self.accepted_ncf += int((accepted & ncf).sum())
        self.total_ncf += int(ncf.sum())
        self.accepted_false_safe += int((accepted & fs).sum())
        self.total_false_safe += int(fs.sum())
        rollout_valid = label.get("waymax/candidate_rollout_valid")
        if selected_idx >= 0 and rollout_valid is not None:
            rv = np.asarray(rollout_valid, dtype=bool)
            if selected_idx < len(rv) and bool(rv[selected_idx]):
                self.selected_waymax_valid += 1
                collision = np.asarray(label.get("waymax/candidate_collision", np.zeros_like(rv)), dtype=bool)
                offroad = np.asarray(label.get("waymax/candidate_offroad", np.zeros_like(rv)), dtype=bool)
                logdiv = np.asarray(label.get("waymax/candidate_log_divergence", np.zeros_like(rv, dtype=np.float32)), dtype=np.float32)
                self.selected_waymax_collision += int(selected_idx < len(collision) and bool(collision[selected_idx]))
                self.selected_waymax_offroad += int(selected_idx < len(offroad) and bool(offroad[selected_idx]))
                if selected_idx < len(logdiv) and np.isfinite(logdiv[selected_idx]):
                    self.selected_waymax_logdiv_sum += float(logdiv[selected_idx])

    def add_witness_quality(self, row: dict[str, float]) -> None:
        self.witness_count += 1
        for k, v in row.items():
            self.witness_sums[k] = self.witness_sums.get(k, 0.0) + float(v)

    def finish(self, *, auprc: float, rank_good: int, rank_total: int, witness_threshold: float) -> dict[str, object]:
        metrics: dict[str, object] = self.label_metrics.finish()
        if self.witness_count:
            for k, v in self.witness_sums.items():
                metrics[f"WitnessQuality/{k}"] = float(v / max(self.witness_count, 1))
        metrics["SelectedNCFRate"] = float(self.selected_ncf / max(self.selected_total, 1))
        metrics["SelectedFalseSafeRate"] = float(self.selected_false_safe / max(self.selected_total, 1))
        metrics["SelectedConventionalSafeRate"] = float(self.selected_conventional / max(self.selected_total, 1))
        metrics["LearnedAcceptedCandidateRate"] = float(self.accepted_total / max(self.valid_total, 1))
        metrics["LearnedAcceptNCFRecall"] = float(self.accepted_ncf / max(self.total_ncf, 1))
        metrics["LearnedAcceptFalseSafeRate"] = float(self.accepted_false_safe / max(self.total_false_safe, 1))
        metrics["WitnessQuality/AUPRC"] = float(auprc)
        metrics["PlannerRankingPairAccuracy"] = float(rank_good / max(rank_total, 1)) if rank_total else 0.0
        if self.selected_waymax_valid > 0:
            metrics["SelectedWaymaxRolloutValid"] = int(self.selected_waymax_valid)
            metrics["SelectedWaymaxCollisionRate"] = float(self.selected_waymax_collision / max(self.selected_waymax_valid, 1))
            metrics["SelectedWaymaxOffroadRate"] = float(self.selected_waymax_offroad / max(self.selected_waymax_valid, 1))
            metrics["SelectedWaymaxUnsafeRate"] = float((self.selected_waymax_collision + self.selected_waymax_offroad) / max(self.selected_waymax_valid, 1))
            metrics["SelectedWaymaxMeanLogDivergence"] = float(self.selected_waymax_logdiv_sum / max(self.selected_waymax_valid, 1))
        metrics["num_scenes"] = int(self.selected_total)
        metrics["mode"] = "learned_offline"
        metrics["witness_threshold"] = float(witness_threshold)
        return metrics


def _learned_offline_candidate_eval_many(
    cache_dir: str | Path,
    checkpoint: str | Path,
    cfg: dict,
    *,
    batch_size: int = 8,
    device: str = "auto",
    witness_thresholds: list[float] | tuple[float, ...] = (0.5,),
    progress: bool = True,
    gate_mode: str = "hard",
    secondary_witness_threshold: float = 0.85,
    secondary_opr_alpha: float = 0.10,
    soft_ncf_penalty: float = 1.5,
) -> dict[float, dict[str, object]]:
    import torch
    from torch.utils.data import DataLoader

    from cowp.data.dataset import TorchCOWPDataset, collate_torch
    from cowp.models.cowp_model import COWPModel

    thresholds = sorted({float(t) for t in witness_thresholds})
    if not thresholds:
        thresholds = [0.5]
    dev = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device))
    ckpt = torch.load(checkpoint, map_location=dev)
    model_cfg = ckpt.get("cfg", cfg)
    model = COWPModel(model_cfg).to(dev)
    model.load_state_dict(ckpt["model"])
    model.eval()
    del ckpt

    # Use the stage-filtered evaluation view.  This avoids reading dense response
    # targets and broad waymax/* tensors that are irrelevant for planner eval.
    ds = TorchCOWPDataset(cache_dir, stage="planner_eval")
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_torch)
    beta_default = float(cfg.get("burden", {}).get("beta0_vehicle", 0.65))
    accs = {th: _LearnedMetricsAccumulator(beta_default=beta_default) for th in thresholds}
    pair_scores: list[np.ndarray] = []
    pair_targets: list[np.ndarray] = []
    rank_good = 0
    rank_total = 0
    iterator = tqdm_iter(dl, enabled=progress, total=len(dl), desc="Learned offline COWP eval", unit="batch")
    with torch.inference_mode():
        for batch in iterator:
            batch = {k: v.to(dev) for k, v in batch.items() if torch.is_tensor(v)}
            if not batch:
                continue
            pred = model(batch, stage="planner")
            if "critical_mask" in pred:
                batch = dict(batch)
                batch["cowp/critical/valid"] = pred["critical_mask"].bool()
            alpha = float(cfg.get("planning", {}).get("alpha_opr_infer", cfg.get("ncf", {}).get("alpha_opr", 0.35)))
            B = int(batch["cowp/candidates/valid"].shape[0])
            batch_labels = [_slim_label_from_batch_item(batch, i) for i in range(B)]
            cand_mask = batch["cowp/candidates/valid"].bool()
            crit_mask = batch["cowp/critical/valid"].bool()
            pair_mask_t = cand_mask[:, :, None] & crit_mask[:, None, :]
            pair_mask = pair_mask_t.detach().cpu().numpy()
            prob_t = torch.sigmoid(pred["witness"]["exist_logits"])
            pair_score_np = prob_t.detach().cpu().numpy()
            gt_exists = batch["cowp/witness/exists"].bool().detach().cpu().numpy()
            if pair_mask.any():
                pair_scores.append(pair_score_np[pair_mask])
                pair_targets.append(gt_exists[pair_mask])
            pred_token = pred["witness"]["token_logits"].argmax(dim=-1).detach().cpu().numpy()
            pred_interval = pred["witness"]["conflict_interval"].round().detach().cpu().numpy()
            gt_token = batch["cowp/witness/token"].long().detach().cpu().numpy()
            gt_interval = batch["cowp/witness/conflict_interval"].detach().cpu().numpy()

            score_np = pred["planner_score"].detach().cpu().numpy()
            ncf_np = batch["cowp/candidates/noncoercive_feasible"].bool().detach().cpu().numpy()
            fs_np = batch["cowp/candidates/false_safe"].bool().detach().cpu().numpy()
            valid_np = cand_mask.detach().cpu().numpy()
            g, t = _ranking_pair_accuracy(score_np, ncf_np, fs_np, valid_np)
            rank_good += g
            rank_total += t

            for th in thresholds:
                batch_selected, batch_accepted_masks = _select_from_learned(
                    batch,
                    pred,
                    witness_threshold=th,
                    alpha_opr=alpha,
                    gate_mode=gate_mode,
                    secondary_witness_threshold=secondary_witness_threshold,
                    secondary_opr_alpha=secondary_opr_alpha,
                    soft_ncf_penalty=soft_ncf_penalty,
                )
                pred_exists = (pair_score_np >= float(th))
                acc = accs[th]
                for i, item in enumerate(batch_labels):
                    acc.add_selection(int(batch_selected[i]), np.asarray(batch_accepted_masks[i], dtype=bool), item)
                    acc.add_witness_quality(witness_quality(pred_exists[i], pred_token[i], pred_interval[i], gt_exists[i], gt_token[i], gt_interval[i], pair_mask[i]))
            if hasattr(iterator, "set_postfix"):
                iterator.set_postfix(done=accs[thresholds[0]].selected_total, thresholds=len(thresholds), refresh=True)

    auprc = _average_precision_binary(np.concatenate(pair_scores), np.concatenate(pair_targets)) if pair_scores else 0.0
    return {th: accs[th].finish(auprc=auprc, rank_good=rank_good, rank_total=rank_total, witness_threshold=th) for th in thresholds}


def learned_offline_candidate_eval(
    cache_dir: str | Path,
    checkpoint: str | Path,
    cfg: dict,
    *,
    batch_size: int = 8,
    device: str = "auto",
    witness_threshold: float = 0.5,
    progress: bool = True,
    gate_mode: str = "hard",
    secondary_witness_threshold: float = 0.85,
    secondary_opr_alpha: float = 0.10,
    soft_ncf_penalty: float = 1.5,
) -> dict[str, object]:
    return _learned_offline_candidate_eval_many(
        cache_dir,
        checkpoint,
        cfg,
        batch_size=batch_size,
        device=device,
        witness_thresholds=[float(witness_threshold)],
        progress=progress,
        gate_mode=gate_mode,
        secondary_witness_threshold=secondary_witness_threshold,
        secondary_opr_alpha=secondary_opr_alpha,
        soft_ncf_penalty=soft_ncf_penalty,
    )[float(witness_threshold)]


def learned_offline_candidate_eval_sweep(
    cache_dir: str | Path,
    checkpoint: str | Path,
    cfg: dict,
    *,
    batch_size: int = 8,
    device: str = "auto",
    witness_thresholds: list[float] | tuple[float, ...] = (0.5,),
    progress: bool = True,
    gate_mode: str = "hard",
    secondary_witness_threshold: float = 0.85,
    secondary_opr_alpha: float = 0.10,
    soft_ncf_penalty: float = 1.5,
) -> list[dict[str, object]]:
    out = _learned_offline_candidate_eval_many(
        cache_dir,
        checkpoint,
        cfg,
        batch_size=batch_size,
        device=device,
        witness_thresholds=list(witness_thresholds),
        progress=progress,
        gate_mode=gate_mode,
        secondary_witness_threshold=secondary_witness_threshold,
        secondary_opr_alpha=secondary_opr_alpha,
        soft_ncf_penalty=soft_ncf_penalty,
    )
    return [out[th] for th in sorted(out)]

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
    # Replay scripts compute their own metric accumulator and never use rewards.
    # Keeping Waymax rewards on adds avoidable metric/reward work at every step
    # and can fail on releases where reward metrics require route observations.
    kwargs["compute_reward"] = False
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
