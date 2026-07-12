from __future__ import annotations

import importlib
import inspect
import gc
from pathlib import Path
from typing import Callable

import numpy as np

from cowp.core.constants import MacroType
from cowp.utils.progress import tqdm_iter
from cowp.waymax_eval.baselines import planner_for_method
from cowp.waymax_eval.metrics_cowp import metrics_from_labels, witness_quality, _progress_reference_m, _trajectory_progress_m
from cowp.waymax_eval.metrics_standard import WaymaxStandardMetricAccumulator


def _load_state_dict_compatible(model, state: dict) -> None:
    try:
        model.load_state_dict(state)
        return
    except Exception:
        pass
    if state and all(str(k).startswith("_orig_mod.") for k in state.keys()):
        state = {k[len("_orig_mod."):]: v for k, v in state.items()}
        try:
            model.load_state_dict(state)
            return
        except Exception:
            pass
    model_state = model.state_dict()
    compatible = {k: v for k, v in state.items() if k in model_state and tuple(model_state[k].shape) == tuple(v.shape)}
    model.load_state_dict(compatible, strict=False)


def _label_from_batch_item(batch, i: int) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for k, v in batch.items():
        if k.startswith("cowp/") or k.startswith("map/") or k.startswith("scenario/") or k.startswith("dataset/"):
            try:
                out[k] = v[i].detach().cpu().numpy()
            except Exception:
                pass
    return out


def _method_gate_defaults(method: str, gate_mode: str) -> tuple[str, str]:
    """Map method names to a learned-offline selection variant and gate mode."""
    m = str(method or "cowp").lower()
    g = str(gate_mode or "priority").lower()
    aliases = {
        "cowp_priority": "cowp",
        "priority_ncf": "cowp",
        "p_ncf": "cowp",
        "cowp_universal": "universal_ncf",
        "hard_ncf": "universal_ncf",
        "ego_utility": "idm_lattice",
        "utility_lattice": "idm_lattice",
        "safety_only": "conventional_safety",
        "planner_only": "planner_score_only",
        "no_ncf": "planner_score_only",
    }
    m = aliases.get(m, m)
    if m == "cowp" and g == "hard":
        # The paper/code now treats COWP as priority-aware NCF.  The original
        # universal veto remains available as method=universal_ncf.
        g = "priority"
    if m == "universal_ncf":
        g = "hard"
    elif m == "soft_burden_cost_only":
        g = "soft"
    elif m in {"idm_lattice", "conventional_safety", "planner_score_only", "outcome_oracle"}:
        g = "none"
    return m, g


def _stop_like_mask(batch, cand_valid, conventional):
    import torch

    macro = batch.get("cowp/candidates/macro_type")
    is_neutral = batch.get("cowp/candidates/is_neutral")
    mask = cand_valid.clone()
    if conventional is not None:
        mask = mask & conventional
    if macro is not None:
        stop_ids = torch.as_tensor(
            [int(MacroType.STOP_BEFORE_CONFLICT), int(MacroType.YIELD), int(MacroType.CREEP), int(MacroType.NEUTRAL_EGO)],
            device=macro.device,
            dtype=macro.dtype,
        )
        stop = (macro[..., None] == stop_ids).any(dim=-1)
        mask = mask & stop
    elif is_neutral is not None:
        mask = mask & is_neutral.bool()
    else:
        mask = torch.zeros_like(cand_valid)
    if is_neutral is not None:
        mask = mask | (cand_valid & conventional & is_neutral.bool())
    return mask


def _witness_probability_and_certificate(pred_witness, cfg=None):
    import torch
    cfg = cfg or {}
    pcfg = cfg.get("planning", {})
    temp = max(float(pcfg.get("witness_temperature", 1.0)), 1e-3)
    bias = float(pcfg.get("witness_logit_bias", 0.0))
    logit_prob = torch.sigmoid((pred_witness["exist_logits"].float() - bias) / temp).float()
    evidence_prob = pred_witness.get("evidential_prob")
    uncertainty = pred_witness.get("epistemic_uncertainty")
    source = str(pcfg.get("witness_probability_source", "mixed")).lower()
    if source == "logit" or not torch.is_tensor(evidence_prob):
        prob = logit_prob
    elif source == "evidential":
        prob = evidence_prob.float()
    else:
        mix = float(pcfg.get("evidential_probability_mix", 0.5))
        prob = (1.0 - mix) * logit_prob + mix * evidence_prob.float()
    unc = uncertainty.float().clamp(0.0, 1.0) if torch.is_tensor(uncertainty) else torch.zeros_like(prob)
    # Invalid numerical outputs are treated conservatively, but finite outputs are
    # no longer forced through the evidential head unless explicitly requested.
    prob = torch.nan_to_num(prob, nan=1.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    unc = torch.nan_to_num(unc, nan=1.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    ucb = float(pcfg.get("evidential_ucb_scale", 0.0 if source == "logit" else 0.15))
    return prob, (prob + ucb * unc).clamp(0.0, 1.0), unc


def _select_from_learned(
    batch,
    pred,
    *,
    witness_threshold: float = 0.5,
    alpha_opr: float = 0.35,
    gate_mode: str = "priority",
    secondary_witness_threshold: float = 0.85,
    secondary_opr_alpha: float = 0.10,
    priority_hard_threshold: float = 0.55,
    soft_ncf_penalty: float = 1.5,
    method: str = "cowp",
    offline_fallback: str = "stop_like",
    adaptive_frontier_margin: float = 0.20,
    outcome_risk_penalty: float = 0.0,
    outcome_risk_threshold: float = 1.10,
    cfg: dict | None = None,
) -> tuple[list[int], list[np.ndarray]]:
    import torch

    method, gate_mode = _method_gate_defaults(method, gate_mode)
    scores = torch.nan_to_num(pred["planner_score"].detach().float(), nan=1e6, posinf=1e6, neginf=-1e6)
    witness_prob, witness_cert, witness_uncertainty = _witness_probability_and_certificate(pred["witness"], cfg)
    witness_prob = witness_prob.detach()
    witness_cert = witness_cert.detach()
    opr = torch.nan_to_num(pred["witness"]["opr"].detach().float(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    cand_valid = batch["cowp/candidates/valid"].bool()
    conventional = batch.get("cowp/candidates/conventional_safe", cand_valid).bool()
    utility = batch.get("cowp/candidates/ego_utility_prior")
    if utility is not None:
        utility_scores = torch.nan_to_num(utility.detach().float(), nan=1e6, posinf=1e6, neginf=-1e6)
    else:
        utility_scores = scores
    outcome = pred.get("outcome", {})
    if isinstance(outcome, dict) and float(outcome_risk_penalty) > 0.0:
        col_r = torch.sigmoid(outcome.get("collision_logit", torch.zeros_like(scores)).detach().float())
        off_r = torch.sigmoid(outcome.get("offroad_logit", torch.zeros_like(scores)).detach().float())
        ld_r = outcome.get("logdiv", torch.zeros_like(scores)).detach().float().clamp_min(0.0) / 10.0
        outcome_risk = torch.nan_to_num(col_r + off_r + ld_r, nan=1.0, posinf=10.0, neginf=0.0)
    else:
        outcome_risk = torch.zeros_like(scores)
    crit_mask = batch.get("cowp/critical/valid")
    if crit_mask is not None and witness_prob.ndim == 3:
        cm = crit_mask.bool()[:, None, :]
        witness_prob = torch.where(cm, witness_prob, torch.zeros_like(witness_prob))
        witness_cert = torch.where(cm, witness_cert, torch.zeros_like(witness_cert))
        opr = torch.where(cm, opr, torch.ones_like(opr))

    # Offline upper-bound baseline when attached Waymax candidate outcomes exist.
    if method == "outcome_oracle":
        rv = batch.get("waymax/candidate_rollout_valid", cand_valid).bool()
        col = batch.get("waymax/candidate_collision")
        off = batch.get("waymax/candidate_offroad")
        ld = batch.get("waymax/candidate_log_divergence")
        cost = torch.zeros_like(scores)
        if col is not None:
            cost = cost + col.float() * 10.0
        if off is not None:
            cost = cost + off.float() * 10.0
        if ld is not None:
            cost = cost + torch.nan_to_num(ld.float(), nan=50.0, posinf=50.0, neginf=0.0) / 10.0
        accepted = cand_valid & conventional & rv
        adjusted_scores = cost + 0.01 * utility_scores
    # Internal baselines that do not use the coercion witness.
    elif method == "idm_lattice":
        accepted = cand_valid & conventional
        adjusted_scores = utility_scores
    elif method == "conventional_safety":
        accepted = cand_valid & conventional
        adjusted_scores = scores
    elif method == "planner_score_only":
        accepted = cand_valid
        adjusted_scores = scores
    else:
        adjusted_scores = scores
        if gate_mode in {"none", "off"}:
            accepted = cand_valid & conventional
        elif gate_mode == "hard":
            predicted_bad = (witness_cert >= witness_threshold).any(dim=-1)
            accepted = cand_valid & conventional & ~predicted_bad
            accepted = accepted & (opr.min(dim=-1).values >= float(alpha_opr))
        else:
            # Priority-aware P-NCF gate.  Low OPR is evidence of option collapse,
            # but it is not a protected-priority claim by itself.  The previous
            # implementation used ``priority_proxy > 0`` after mixing in an OPR
            # indicator, which collapsed priority-aware COWP into the universal NCF
            # veto and produced all-fallback learned-offline/Waymax behavior.
            if "priority_claim_logits" in pred and torch.is_tensor(pred["priority_claim_logits"]):
                learned_priority = torch.sigmoid(pred["priority_claim_logits"].detach().float())
            else:
                learned_priority = torch.zeros_like(witness_prob)
            # Offline eval has no live Waymax state, so use cached witness rho as a
            # physically grounded priority anchor.  Online evaluation uses the live
            # priority heuristic in policy_wrapper.py.  This keeps offline/online
            # COWP semantics aligned instead of relying on an uncalibrated learned
            # priority head alone.
            rho = batch.get("cowp/witness/rho")
            if rho is not None and torch.is_tensor(rho):
                # PriorityRelation.AGENT_PRIORITY == 2.
                rule_priority = (rho.long() == 2).float()
            else:
                rule_priority = torch.zeros_like(witness_prob)
            opr_collapse = (torch.relu(float(alpha_opr) - opr) / max(float(alpha_opr), 1e-6)).clamp(0.0, 1.0)
            priority_proxy = (0.45 * learned_priority + 0.45 * rule_priority + 0.10 * opr_collapse).clamp(0.0, 1.0)
            priority_claim = priority_proxy >= float(priority_hard_threshold)
            primary_bad = ((witness_cert >= witness_threshold) & priority_claim).any(dim=-1)
            option_bad = ((opr < float(alpha_opr)) & priority_claim).any(dim=-1)
            severe_bad = ((witness_cert >= float(secondary_witness_threshold)) & (opr <= float(secondary_opr_alpha)) & priority_claim).any(dim=-1)
            penalty = (witness_prob * priority_proxy).amax(dim=-1) + (torch.relu(float(alpha_opr) - opr) * priority_proxy).amax(dim=-1)
            adjusted_scores = scores + float(soft_ncf_penalty) * penalty + float(outcome_risk_penalty) * outcome_risk
            if gate_mode == "soft":
                accepted = cand_valid & conventional
            else:
                accepted = cand_valid & conventional & ~primary_bad & ~option_bad & ~severe_bad & (outcome_risk <= float(outcome_risk_threshold))

    if method == "cowp" and gate_mode in {"priority", "soft"}:
        frontier_base = cand_valid & conventional & (outcome_risk <= float(outcome_risk_threshold))
        # Relative certificate frontier: even when absolute witness probabilities are
        # imperfectly calibrated, COWP should choose from the least-coercive
        # conventional frontier in the same scene.  This is a set-valued NCF
        # relaxation, not an ad-hoc threshold: the threshold is the scene's own
        # minimum predicted coercion risk plus a margin.
        risk = penalty if "penalty" in locals() else witness_prob.amax(dim=-1) + torch.relu(float(alpha_opr) - opr).amax(dim=-1)
        for b in range(int(scores.shape[0])):
            base_b = frontier_base[b]
            if not base_b.any():
                continue
            best = torch.where(base_b, risk[b], torch.full_like(risk[b], float("inf"))).min()
            frontier = base_b & (risk[b] <= best + float(adaptive_frontier_margin))
            if frontier.any():
                accepted[b] = (accepted[b] & frontier) if accepted[b].any() else frontier

    selected: list[int] = []
    masks: list[np.ndarray] = []
    B = scores.shape[0]
    for b in range(B):
        mask = accepted[b]
        if mask.any():
            masked = torch.where(mask, adjusted_scores[b], torch.full_like(adjusted_scores[b], float("inf")))
            selected.append(int(torch.argmin(masked).item()))
        else:
            # Do not silently select the best conventional false-safe candidate
            # when the learned NCF filter rejects everything.  In closed loop this
            # corresponds to a conservative stop; in learned-offline metrics mark
            # it as fallback unless an explicit stop-like/neutral candidate exists
            # and the caller requested stop_like fallback.
            if str(offline_fallback).lower() == "stop_like":
                fallback = _stop_like_mask(batch, cand_valid, conventional)[b]
                if fallback.any():
                    masked = torch.where(fallback, adjusted_scores[b], torch.full_like(adjusted_scores[b], float("inf")))
                    selected.append(int(torch.argmin(masked).item()))
                else:
                    selected.append(-1)
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
    "cowp/candidates/macro_type",
    "cowp/candidates/ego_utility_prior",
    "cowp/candidates/is_neutral",
    "cowp/candidates/is_logged",
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
    priority_hard_threshold: float = 0.55,
    soft_ncf_penalty: float = 1.5,
    method: str = "cowp",
    offline_fallback: str = "stop_like",
    adaptive_frontier_margin: float = 0.20,
    outcome_risk_penalty: float = 0.0,
    outcome_risk_threshold: float = 1.10,
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
    _load_state_dict_compatible(model, ckpt["model"])
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
            prob_t, _, _ = _witness_probability_and_certificate(pred["witness"], cfg)
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
                    priority_hard_threshold=priority_hard_threshold,
                    soft_ncf_penalty=soft_ncf_penalty,
                    method=method,
                    offline_fallback=offline_fallback,
                    adaptive_frontier_margin=adaptive_frontier_margin,
                    outcome_risk_penalty=outcome_risk_penalty,
                    outcome_risk_threshold=outcome_risk_threshold,
                    cfg=cfg,
                )
                pred_exists = (pair_score_np >= float(th))
                acc = accs[th]
                for i, item in enumerate(batch_labels):
                    acc.add_selection(int(batch_selected[i]), np.asarray(batch_accepted_masks[i], dtype=bool), item)
                    acc.add_witness_quality(witness_quality(pred_exists[i], pred_token[i], pred_interval[i], gt_exists[i], gt_token[i], gt_interval[i], pair_mask[i]))
            if hasattr(iterator, "set_postfix"):
                iterator.set_postfix(done=accs[thresholds[0]].selected_total, thresholds=len(thresholds), refresh=True)

    all_pair_scores = np.concatenate(pair_scores) if pair_scores else np.asarray([], dtype=np.float32)
    auprc = _average_precision_binary(all_pair_scores, np.concatenate(pair_targets)) if pair_scores else 0.0
    out = {th: accs[th].finish(auprc=auprc, rank_good=rank_good, rank_total=rank_total, witness_threshold=th) for th in thresholds}
    if all_pair_scores.size:
        qs = np.quantile(all_pair_scores, [0.1, 0.5, 0.9, 0.99])
        for row in out.values():
            row["WitnessProb/p10"] = float(qs[0])
            row["WitnessProb/p50"] = float(qs[1])
            row["WitnessProb/p90"] = float(qs[2])
            row["WitnessProb/p99"] = float(qs[3])
    return out


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
    priority_hard_threshold: float = 0.55,
    soft_ncf_penalty: float = 1.5,
    method: str = "cowp",
    offline_fallback: str = "stop_like",
    adaptive_frontier_margin: float = 0.20,
    outcome_risk_penalty: float = 0.0,
    outcome_risk_threshold: float = 1.10,
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
        priority_hard_threshold=priority_hard_threshold,
        soft_ncf_penalty=soft_ncf_penalty,
        method=method,
        offline_fallback=offline_fallback,
        adaptive_frontier_margin=adaptive_frontier_margin,
        outcome_risk_penalty=outcome_risk_penalty,
        outcome_risk_threshold=outcome_risk_threshold,
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
    priority_hard_threshold: float = 0.55,
    soft_ncf_penalty: float = 1.5,
    method: str = "cowp",
    offline_fallback: str = "stop_like",
    adaptive_frontier_margin: float = 0.20,
    outcome_risk_penalty: float = 0.0,
    outcome_risk_threshold: float = 1.10,
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
        priority_hard_threshold=priority_hard_threshold,
        soft_ncf_penalty=soft_ncf_penalty,
        method=method,
        offline_fallback=offline_fallback,
        adaptive_frontier_margin=adaptive_frontier_margin,
        outcome_risk_penalty=outcome_risk_penalty,
        outcome_risk_threshold=outcome_risk_threshold,
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
    split: str | None = None,
    tfexample_glob: str | None = None,
    shard_index: int = 0,
    num_shards: int = 1,
):
    """Run real Waymax closed-loop simulation by stepping a Waymax environment.

    ``policy_fn`` must return a Waymax-compatible action for the current
    SimulatorState.  This function intentionally does not use proto-derived COWP
    labels; it is the closed-loop path for validating a planner in Waymax.
    """
    from cowp.waymax_eval.dataloader import waymax_state_generator

    horizon = int(horizon_steps) if horizon_steps is not None else 80
    gen = waymax_state_generator(data_config, split=split, tfexample_glob=tfexample_glob)
    total = num_scenarios
    num_shards = max(int(num_shards), 1)
    shard_index = int(shard_index) % num_shards
    iterator = tqdm_iter(gen, enabled=progress, total=total, desc=f"Waymax closed-loop rollout shard {shard_index}/{num_shards}", unit="scenario")
    outputs = []
    for raw_index, init_state in enumerate(iterator):
        if num_shards > 1 and (raw_index % num_shards) != shard_index:
            continue
        scenario_index = raw_index
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
