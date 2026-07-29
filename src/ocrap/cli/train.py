from __future__ import annotations

from pathlib import Path
from time import perf_counter
from collections import Counter
import json
import os
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler

from ocrap.algorithms.ocmero import oc_mero, torch_oc_mero
from ocrap.data.serialization import ensure_dir, load_npz, write_json
from ocrap.models.data import OCRAPSampleDataset, OPTION_FEATURE_DIM, bucket_id_for_path, iter_sample_paths_many, scalar_metadata_for_path, split_paths_by_npz_split, stable_scene_hash
from ocrap.models.losses import (
    anti_oracle_loss,
    artifact_gap_loss,
    best_shared_option_loss,
    deployability_classification_loss,
    shared_option_admission_loss,
    shared_option_q_regression_loss,
    shared_option_success_regression_loss,
    shared_option_success_bce_loss,
    groupwise_candidate_ranking_loss,
    groupwise_candidate_ce_loss,
    nominal_switch_consistency_loss,
    groupwise_score_distillation_loss,
    safe_nominal_preservation_loss,
    protective_macro_recovery_loss,
    deployability_dominance_calibration_loss,
    direct_teacher_pcd_loss,
    macro_shared_success_calibration_loss,
    observation_consistent_recovery_advantage_loss,
    direct_uncertainty_recovery_value_loss,
    _differentiable_shared_success,
    _exact_teacher_shared_success,
    _torch_pcd_score,
)
from ocrap.evaluation.metrics import (
    best_shared_option_index,
    deployable_recovery_success,
    post_contact_deployability_score,
)
from ocrap.models.ocrap import OCRAPModel
from ocrap.utils.seed import seed_everything


def _device(cfg: dict) -> torch.device:
    tcfg = cfg.get("training", {}) if isinstance(cfg.get("training", {}), dict) else {}
    requested = str(tcfg.get("device", "auto"))
    if requested == "auto":
        if bool(tcfg.get("require_cuda", False)) and not torch.cuda.is_available():
            raise RuntimeError("training.require_cuda=true, but torch.cuda.is_available() is false")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if bool(tcfg.get("require_cuda", False)) and device.type != "cuda":
        raise RuntimeError("training.require_cuda=true, but selected device is not CUDA")
    if bool(tcfg.get("require_cuda", False)) and not torch.cuda.is_available():
        raise RuntimeError("training.require_cuda=true, but torch.cuda.is_available() is false")
    return device


def _device_summary(device: torch.device) -> dict[str, object]:
    summary: dict[str, object] = {
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_version": str(torch.__version__),
    }
    if torch.cuda.is_available():
        idx = device.index if device.type == "cuda" and device.index is not None else torch.cuda.current_device()
        try:
            summary.update({
                "cuda_device_index": int(idx),
                "cuda_device_name": torch.cuda.get_device_name(idx),
                "cuda_device_count": int(torch.cuda.device_count()),
            })
        except Exception:
            pass
    return summary


def _collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in batch[0].keys()}


def _masked_smooth_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    target = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
    mask = mask.bool() & torch.isfinite(target) & (target > -1e8)
    if not bool(mask.any()):
        return pred.sum() * 0.0
    return F.smooth_l1_loss(pred[mask], target[mask])


def _root_signature_loss(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], key: str, valid_key: str = "root_valid") -> torch.Tensor:
    if key not in out or key not in batch or batch[key].shape[-1] == 0:
        ref = next(iter(out.values()))
        return ref.sum() * 0.0
    pred = out[key]
    target = batch[key].float()
    mask = batch[valid_key].bool().unsqueeze(-1).expand_as(target)
    return _masked_smooth_l1(pred, target, mask)


def _obs_bce(pred_c: torch.Tensor, target_y: torch.Tensor, root_valid: torch.Tensor, *, balanced: bool = True) -> torch.Tensor:
    B, K, _ = pred_c.shape
    eye = torch.eye(K, dtype=torch.bool, device=pred_c.device).unsqueeze(0)
    pair_mask = root_valid.unsqueeze(1) & root_valid.unsqueeze(2) & (~eye)
    if not bool(pair_mask.any()):
        return pred_c.sum() * 0.0
    pred = pred_c.clamp(1e-5, 1.0 - 1e-5)[pair_mask]
    target = target_y[pair_mask].float().clamp(0.0, 1.0)
    if not balanced:
        return F.binary_cross_entropy(pred, target)
    pos = target >= 0.5
    neg = ~pos
    if not bool(pos.any()) or not bool(neg.any()):
        return F.binary_cross_entropy(pred, target)
    weights = torch.where(pos, 0.5 / pos.float().mean().clamp_min(1e-6), 0.5 / neg.float().mean().clamp_min(1e-6))
    return F.binary_cross_entropy(pred, target, weight=weights)


def _progress_iter(loader: DataLoader, *, enabled: bool, desc: str):
    if not enabled:
        return loader
    try:
        from tqdm.auto import tqdm  # type: ignore
        return tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
    except Exception:
        return loader




def _parameter_anchor_loss(model: torch.nn.Module, attribute: str = "_encoder_anchor_tensors") -> torch.Tensor:
    """L2-SP penalty to the loaded encoder, normalized by parameter count."""
    anchors = getattr(model, attribute, None)
    if not isinstance(anchors, dict) or not anchors:
        ref = next(model.parameters())
        return ref.sum() * 0.0
    total = None
    count = 0
    for name, param in model.named_parameters():
        target = anchors.get(name)
        if target is None or not param.requires_grad:
            continue
        term = (param - target).pow(2).sum()
        total = term if total is None else total + term
        count += int(param.numel())
    if total is None or count <= 0:
        ref = next(model.parameters())
        return ref.sum() * 0.0
    return total / float(count)

def _keep_fully_frozen_modules_in_eval(model: torch.nn.Module) -> None:
    """Keep fully frozen subtrees deterministic while trainable heads train."""
    for module in list(model.modules())[1:]:
        params = list(module.parameters())
        if params and all(not p.requires_grad for p in params):
            module.eval()


def _dataset_root_name(p: Path) -> str:
    parts = list(p.parts)
    for i in range(len(parts) - 2, -1, -1):
        name = parts[i]
        low = name.lower()
        if any(tok in low for tok in ("safe", "near", "contact", "train_", "val_", "test_")):
            return name
    return p.parent.parent.name if p.parent.name == "samples" else p.parent.name




def _parse_int_tuple(value, default: tuple[int, ...]) -> tuple[int, ...]:
    if value is None or value == "":
        return tuple(default)
    if isinstance(value, (list, tuple, set)):
        out = []
        for x in value:
            try:
                out.append(int(x))
            except Exception:
                continue
        return tuple(out) if out else tuple(default)
    text = str(value).strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    out = []
    for part in text.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except Exception:
            continue
    return tuple(out) if out else tuple(default)


def _profile_paths(paths: list[Path], *, stage: str, max_scalar_scan: int | None = None) -> dict[str, object]:
    total = len(paths)
    roots = Counter(_dataset_root_name(p) for p in paths)
    limit = total if max_scalar_scan is None or max_scalar_scan <= 0 else min(total, int(max_scalar_scan))
    num_art = num_neg = num_safe_pos = 0
    r_sum = 0.0
    scanned = 0
    for p in paths[:limit]:
        try:
            is_art = float(np.asarray(scalar_metadata_for_path(p, "i_art_star", 0)).item()) > 0.5
            r_dep = float(np.asarray(scalar_metadata_for_path(p, "r_dep_star", 0)).item())
            is_neg = r_dep < 0.0
            root = _dataset_root_name(p).lower()
            is_safe_pos = ("safe" in root) and (not is_neg) and (not is_art)
            num_art += int(is_art)
            num_neg += int(is_neg)
            num_safe_pos += int(is_safe_pos)
            r_sum += r_dep
            scanned += 1
        except Exception:
            pass
    return {
        "event": "dataset_profile",
        "stage": stage,
        "num_paths": int(total),
        "roots": dict(sorted(roots.items())),
        "scalar_scanned": int(scanned),
        "artifact_fraction": float(num_art / max(scanned, 1)),
        "negative_deployable_fraction": float(num_neg / max(scanned, 1)),
        "safe_positive_fraction": float(num_safe_pos / max(scanned, 1)),
        "r_dep_mean": float(r_sum / max(scanned, 1)),
    }



def _direct_value_loss_from_outputs(
    out: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    tcfg: dict,
    model_cfg: dict,
    teacher_q: torch.Tensor,
    *,
    option_gamma: float,
    option_temperature: float,
) -> torch.Tensor:
    """Compute robust aggregate loss plus asymmetric expert specialization.

    Expert 0 is recovery-seeking (higher positive/opportunity emphasis); expert 1
    is harm-averse (higher harmful-switch and false-admission emphasis).  Both
    experts see every Near/Contact group, so diversity is risk-attitude based and
    never depends on a hidden regime label.
    """
    if "direct_recovery_value_logit" not in out:
        return batch["r_dep_star"].float().sum() * 0.0

    base_kwargs = dict(
        macro_ids=_parse_int_tuple(tcfg.get("direct_value_macro_ids", "2,3,5,7"), (2, 3, 5, 7)),
        bucket_ids=_parse_int_tuple(tcfg.get("direct_value_bucket_ids", "1,2"), (1, 2)),
        temperature=float(tcfg.get("direct_value_temperature", 0.12)),
        positive_gain=float(tcfg.get("direct_value_positive_gain", 0.03)),
        negative_gain=float(tcfg.get("direct_value_negative_gain", 0.02)),
        rank_margin=float(tcfg.get("direct_value_rank_margin", 0.04)),
        point_weight=float(tcfg.get("direct_value_point_weight", 0.15)),
        listwise_weight=float(tcfg.get("direct_value_listwise_weight", 0.35)),
        advantage_weight=float(tcfg.get("direct_value_advantage_weight", 1.0)),
        centered_weight=float(tcfg.get("direct_value_centered_weight", 1.0)),
        positive_group_weight=float(tcfg.get("direct_value_positive_group_weight", 4.0)),
        negative_group_weight=float(tcfg.get("direct_value_negative_group_weight", 1.0)),
        ambiguous_group_weight=float(tcfg.get("direct_value_ambiguous_group_weight", 0.25)),
        near_weight=float(tcfg.get("direct_value_near_weight", 1.5)),
        contact_weight=float(tcfg.get("direct_value_contact_weight", 1.0)),
        min_group_range=float(tcfg.get("direct_value_min_group_range", 0.01)),
        false_positive_weight=float(tcfg.get("direct_value_false_positive_weight", 1.0)),
        variance_floor=float(tcfg.get("direct_value_variance_floor", 0.0025)),
        output_mode=str(tcfg.get("direct_value_output_mode", model_cfg.get("direct_recovery_value_output", "probability"))),
        pairwise_weight=float(tcfg.get("direct_value_pairwise_weight", 0.0)),
        top_rank_weight=float(tcfg.get("direct_value_top_rank_weight", 0.0)),
        success_gamma=option_gamma,
        success_temperature=option_temperature,
        opportunity_weight=float(tcfg.get("direct_value_opportunity_weight", 0.0)),
        opportunity_pos_weight=float(tcfg.get("direct_value_opportunity_pos_weight", 6.0)),
        harm_weight=float(tcfg.get("direct_value_harm_weight", 0.0)),
        harm_pos_weight=float(tcfg.get("direct_value_harm_pos_weight", 4.0)),
        setwise_admission_weight=float(tcfg.get("direct_value_setwise_admission_weight", 0.0)),
        opportunity_admission_weight=float(tcfg.get("direct_value_opportunity_admission_weight", 0.35)),
        harm_admission_weight=float(tcfg.get("direct_value_harm_admission_weight", 0.75)),
        selective_risk_weight=float(tcfg.get("direct_value_selective_risk_weight", 0.0)),
        selective_harm_budget=float(tcfg.get("direct_value_selective_harm_budget", 0.05)),
        selective_coverage_weight=float(tcfg.get("direct_value_selective_coverage_weight", 0.0)),
        selective_coverage_target=float(tcfg.get("direct_value_selective_coverage_target", 0.65)),
        policy_distill_weight=float(tcfg.get("direct_value_policy_distill_weight", 0.0)),
        policy_teacher_temperature=float(tcfg.get("direct_value_policy_teacher_temperature", 0.08)),
        policy_regret_weight=float(tcfg.get("direct_value_policy_regret_weight", 0.0)),
        policy_regret_margin=float(tcfg.get("direct_value_policy_regret_margin", 0.0)),
        opportunity_soft_label_temperature=float(tcfg.get("direct_value_opportunity_soft_label_temperature", 0.0)),
        harm_soft_label_temperature=float(tcfg.get("direct_value_harm_soft_label_temperature", 0.0)),
        policy_decouple_admission=bool(tcfg.get("direct_value_policy_decouple_admission", True)),
        policy_admission_distill_weight=float(tcfg.get("direct_value_policy_admission_distill_weight", 0.0)),
        group_dro_weight=float(tcfg.get("direct_value_group_dro_weight", 0.0)),
        group_dro_temperature=float(tcfg.get("direct_value_group_dro_temperature", 0.35)),
        group_dro_severity_thresholds=tuple(
            float(x) for x in str(tcfg.get("direct_value_group_dro_severity_thresholds", "0.25,0.55")).split(",") if str(x).strip()
        ),
        exact_teacher_pcd=bool(tcfg.get("direct_value_exact_teacher_pcd", False)),
        preference_weight=float(tcfg.get("direct_value_preference_weight", 0.0)),
        preference_temperature=float(tcfg.get("direct_value_preference_temperature", 0.06)),
        preference_min_gap=float(tcfg.get("direct_value_preference_min_gap", 0.01)),
        preference_margin=float(tcfg.get("direct_value_preference_margin", 0.03)),
        preference_confidence_scale=float(tcfg.get("direct_value_preference_confidence_scale", 0.04)),
        preference_regret_weight=float(tcfg.get("direct_value_preference_regret_weight", 0.0)),
        preference_listwise_weight=float(tcfg.get("direct_value_preference_listwise_weight", 0.0)),
        preference_gap_weight=float(tcfg.get("direct_value_preference_gap_weight", 0.0)),
        preference_set_weight=float(tcfg.get("direct_value_preference_set_weight", 0.0)),
        preference_set_margin=float(tcfg.get("direct_value_preference_set_margin", 0.02)),
        preference_tie_epsilon_near=float(tcfg.get("direct_value_preference_tie_epsilon_near", 0.025)),
        preference_tie_epsilon_contact=float(tcfg.get("direct_value_preference_tie_epsilon_contact", 0.010)),
        preference_all_group_set_weight=float(tcfg.get("direct_value_preference_all_group_set_weight", 0.0)),
        preference_set_replace_singlewinner=bool(tcfg.get("direct_value_preference_set_replace_singlewinner", False)),
        preference_nominal_margin=float(tcfg.get("direct_value_preference_nominal_margin", 0.02)),
        preference_harm_margin=float(tcfg.get("direct_value_preference_harm_margin", 0.03)),
        preference_set_mass_loss=bool(tcfg.get("direct_value_preference_set_mass_loss", False)),
        preference_noop_nominal_only=bool(tcfg.get("direct_value_preference_noop_nominal_only", False)),
        preference_deadzone_margin=float(tcfg.get("direct_value_preference_deadzone_margin", 0.008)),
        preference_conditional_set_weight=float(tcfg.get("direct_value_preference_conditional_set_weight", 0.0)),
        preference_conditional_noop_weight=float(tcfg.get("direct_value_preference_conditional_noop_weight", 0.35)),
        preference_conditional_regret_weight=float(tcfg.get("direct_value_preference_conditional_regret_weight", 0.5)),
        preference_conditional_pairwise_weight=float(tcfg.get("direct_value_preference_conditional_pairwise_weight", 0.0)),
        preference_conditional_pairwise_min_gap=float(tcfg.get("direct_value_preference_conditional_pairwise_min_gap", 0.01)),
        preference_conditional_pairwise_margin=float(tcfg.get("direct_value_preference_conditional_pairwise_margin", 0.02)),
        delta_nll_weight=float(tcfg.get("direct_value_delta_nll_weight", 0.0)),
        delta_sign_weight=float(tcfg.get("direct_value_delta_sign_weight", 0.0)),
        delta_sign_temperature=float(tcfg.get("direct_value_delta_sign_temperature", 0.04)),
        certificate_policy_top1_weight=float(tcfg.get("direct_value_certificate_policy_top1_weight", 0.0)),
        certificate_policy_top1_sign_weight=float(tcfg.get("direct_value_certificate_policy_top1_sign_weight", 0.0)),
        certificate_policy_top1_temperature=float(tcfg.get("direct_value_certificate_policy_top1_temperature", 0.04)),
        ordinal_evidence_policy_top1_weight=float(tcfg.get("direct_value_ordinal_evidence_policy_top1_weight", 0.0)),
        ordinal_evidence_all_candidate_weight=float(tcfg.get("direct_value_ordinal_evidence_all_candidate_weight", 0.0)),
        ordinal_evidence_focal_gamma=float(tcfg.get("direct_value_ordinal_evidence_focal_gamma", 1.5)),
        ordinal_evidence_ordered_nll_top1_weight=float(tcfg.get("direct_value_ordinal_evidence_ordered_nll_top1_weight", 0.0)),
        ordinal_evidence_ordered_nll_all_weight=float(tcfg.get("direct_value_ordinal_evidence_ordered_nll_all_weight", 0.0)),
        ordinal_evidence_harm_class_weight=float(tcfg.get("direct_value_ordinal_evidence_harm_class_weight", 2.0)),
        ordinal_evidence_dead_class_weight=float(tcfg.get("direct_value_ordinal_evidence_dead_class_weight", 0.5)),
        ordinal_evidence_benefit_class_weight=float(tcfg.get("direct_value_ordinal_evidence_benefit_class_weight", 1.25)),
        ordinal_evidence_pairwise_benefit_weight=float(tcfg.get("direct_value_ordinal_evidence_pairwise_benefit_weight", 0.0)),
        ordinal_evidence_pairwise_harm_weight=float(tcfg.get("direct_value_ordinal_evidence_pairwise_harm_weight", 0.0)),
        ordinal_evidence_pairwise_margin=float(tcfg.get("direct_value_ordinal_evidence_pairwise_margin", 0.25)),
    )

    def compute(
        value_logit: torch.Tensor,
        value_logvar: torch.Tensor,
        opportunity_logit: torch.Tensor | None,
        harm_logit: torch.Tensor | None,
        rank_logit: torch.Tensor | None,
        delta_mean: torch.Tensor | None,
        delta_logvar: torch.Tensor | None,
        overrides: dict[str, float] | None = None,
    ) -> torch.Tensor:
        kwargs = dict(base_kwargs)
        if overrides:
            kwargs.update(overrides)
        return direct_uncertainty_recovery_value_loss(
            value_logit, value_logvar,
            batch["r_dep_star"].float(), batch["r_orc_star"].float(), teacher_q,
            batch["root_probs"].float(), batch["root_valid"], batch["option_valid"],
            batch["scene_hash"], batch["time_index"],
            batch.get("prefix_macro_type_id", batch.get("candidate_index", torch.zeros_like(batch["time_index"]))),
            batch["is_nominal"].float(),
            batch.get("bucket_id", torch.full_like(batch["time_index"], 3)),
            pred_opportunity_logit=opportunity_logit,
            pred_harm_logit=harm_logit,
            pred_rank_logit=rank_logit,
            pred_delta_mean=delta_mean,
            pred_delta_logvar=delta_logvar,
            teacher_m_star=batch["m_star"].float(),
            **kwargs,
        )

    aggregate = compute(
        out["direct_recovery_value_logit"], out["direct_recovery_value_logvar"],
        out.get("direct_recovery_opportunity_logit"), out.get("direct_recovery_harm_logit"),
        out.get("direct_recovery_rank_logit"),
        out.get("direct_recovery_delta_mean"), out.get("direct_recovery_delta_logvar"),
    )
    expert_outputs = out.get("direct_expert_outputs")
    specialist_weight = float(tcfg.get("direct_value_expert_specialization_weight", 0.35))
    if expert_outputs is None or expert_outputs.ndim != 3 or specialist_weight <= 0.0:
        return aggregate

    expert_losses: list[torch.Tensor] = []
    for expert_id in range(expert_outputs.shape[1]):
        eo = expert_outputs[:, expert_id]
        cursor = 2
        opp = eo[:, cursor] if bool(model_cfg.get("direct_recovery_opportunity_head", False)) else None
        cursor += int(bool(model_cfg.get("direct_recovery_opportunity_head", False)))
        harm = eo[:, cursor] if bool(model_cfg.get("direct_recovery_harm_head", False)) else None
        if expert_id % 2 == 0:
            overrides = {
                "positive_group_weight": base_kwargs["positive_group_weight"] * float(tcfg.get("direct_value_recovery_expert_positive_scale", 1.5)),
                "opportunity_weight": base_kwargs["opportunity_weight"] * float(tcfg.get("direct_value_recovery_expert_opportunity_scale", 1.5)),
                "harm_weight": base_kwargs["harm_weight"] * float(tcfg.get("direct_value_recovery_expert_harm_scale", 0.75)),
                "false_positive_weight": base_kwargs["false_positive_weight"] * float(tcfg.get("direct_value_recovery_expert_fp_scale", 0.8)),
            }
        else:
            overrides = {
                "positive_group_weight": base_kwargs["positive_group_weight"] * float(tcfg.get("direct_value_harm_expert_positive_scale", 0.75)),
                "negative_group_weight": base_kwargs["negative_group_weight"] * float(tcfg.get("direct_value_harm_expert_negative_scale", 1.5)),
                "opportunity_weight": base_kwargs["opportunity_weight"] * float(tcfg.get("direct_value_harm_expert_opportunity_scale", 0.75)),
                "harm_weight": base_kwargs["harm_weight"] * float(tcfg.get("direct_value_harm_expert_harm_scale", 1.75)),
                "harm_pos_weight": base_kwargs["harm_pos_weight"] * float(tcfg.get("direct_value_harm_expert_pos_scale", 1.5)),
                "false_positive_weight": base_kwargs["false_positive_weight"] * float(tcfg.get("direct_value_harm_expert_fp_scale", 1.75)),
            }
        overrides.update({
            "preference_weight": 0.0,
            "preference_regret_weight": 0.0,
            "preference_listwise_weight": 0.0,
            "preference_gap_weight": 0.0,
            "delta_nll_weight": 0.0,
        })
        expert_losses.append(compute(eo[:, 0], eo[:, 1], opp, harm, None, None, None, overrides))
    return aggregate + specialist_weight * torch.stack(expert_losses).mean()


def _direct_policy_batch_stats(
    out: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    tcfg: dict,
    teacher_q: torch.Tensor,
    *,
    option_gamma: float,
    option_temperature: float,
) -> dict[str, float]:
    """Exact-contract ranking and deployment-aligned certificate metrics.

    v48.7 separates two validation questions:
      1) preference risk: did the rank head choose an exact-PCD acceptable
         recovery candidate, allowing regime-specific near-ties;
      2) certificate risk: would the fixed validation certificate actually
         admit that candidate, and what harmful/false/missed-opportunity cost
         would result.

    The latter uses the same Gaussian delta CDF semantics as calibration rather
    than comparing a raw logit/delta to a probability threshold.
    """
    if "direct_recovery_value_logit" not in out:
        return {}
    value_score = out["direct_recovery_value_logit"].float().reshape(-1)
    output_mode = str(tcfg.get("direct_value_output_mode", "probability") or "probability").strip().lower()
    if output_mode != "score":
        value_score = torch.sigmoid(value_score)
    value_logvar = out.get("direct_recovery_value_logvar")
    if value_logvar is not None:
        value_logvar = value_logvar.float().reshape(-1).clamp(-7.0, 2.0)
    direct_delta = out.get("direct_recovery_delta_mean")
    direct_delta_logvar = out.get("direct_recovery_delta_logvar")
    opportunity_logit = out.get("direct_recovery_opportunity_logit")
    harm_logit = out.get("direct_recovery_harm_logit")
    if opportunity_logit is not None:
        opportunity_logit = opportunity_logit.float().reshape(-1)
    if harm_logit is not None:
        harm_logit = harm_logit.float().reshape(-1)
    if direct_delta is not None:
        direct_delta = direct_delta.float().reshape(-1)
    if direct_delta_logvar is not None:
        direct_delta_logvar = direct_delta_logvar.float().reshape(-1).clamp(-7.0, 2.0)
    rank_score = out.get("direct_recovery_rank_logit", out["direct_recovery_value_logit"]).float().reshape(-1)
    trd = torch.nan_to_num(batch["r_dep_star"].float().reshape(-1), nan=-20.0, posinf=20.0, neginf=-20.0)
    tro = torch.nan_to_num(batch["r_orc_star"].float().reshape(-1), nan=-20.0, posinf=20.0, neginf=-20.0)
    if bool(tcfg.get("direct_value_exact_teacher_pcd", False)):
        teacher_drs = _exact_teacher_shared_success(
            teacher_q, batch["m_star"].float(), batch["root_probs"].float(),
            batch["root_valid"], batch["option_valid"], gamma=option_gamma,
        ).reshape(-1)
    else:
        teacher_drs = _differentiable_shared_success(
            teacher_q, batch["root_probs"].float(), batch["root_valid"], batch["option_valid"],
            gamma=option_gamma, temperature=max(0.08, option_temperature * 0.5),
        ).reshape(-1)
    target = _torch_pcd_score(teacher_drs, trd, torch.clamp(tro - trd, min=0.0)).detach().clamp(0.0, 1.0)
    bid = batch.get("bucket_id", torch.full_like(batch["time_index"], 3)).reshape(-1)
    sh = batch["scene_hash"].reshape(-1)
    ti = batch["time_index"].reshape(-1)
    isn = batch["is_nominal"].reshape(-1) > 0.5
    mac = batch.get("prefix_macro_type_id", batch.get("candidate_index", torch.zeros_like(ti))).reshape(-1)
    allowed = torch.zeros_like(isn)
    for m in _parse_int_tuple(tcfg.get("direct_value_macro_ids", "2,3,5,6,7"), (2, 3, 5, 6, 7)):
        allowed |= mac == int(m)
    positive_gain = float(tcfg.get("direct_value_positive_gain", 0.03))
    negative_gain = float(tcfg.get("direct_value_negative_gain", 0.02))
    opp_threshold = float(tcfg.get("direct_policy_metric_opportunity_threshold", 0.65))
    harm_threshold = float(tcfg.get("direct_policy_metric_harm_threshold", 0.30))
    rank_margin_threshold = float(tcfg.get("direct_policy_metric_rank_margin_threshold", 0.02))
    min_delta_mean = float(tcfg.get("direct_policy_metric_min_delta_mean", 0.0))
    risk_source = str(tcfg.get("direct_policy_metric_risk_source", "gaussian_delta") or "gaussian_delta").strip().lower()
    conditional_preference = bool(tcfg.get("direct_value_preference_conditional_mode", False))
    stats: dict[str, float] = {}

    def add(name: str, value: float) -> None:
        stats[name] = stats.get(name, 0.0) + float(value)

    def normal_cdf(z: torch.Tensor) -> torch.Tensor:
        return 0.5 * (1.0 + torch.erf(z / (2.0 ** 0.5)))

    keys = torch.stack([bid, sh, ti], dim=1)
    for key in torch.unique(keys, dim=0):
        idx = torch.where((keys == key.unsqueeze(0)).all(dim=1))[0]
        noms = idx[isn[idx]]
        recs = idx[(~isn[idx]) & allowed[idx]]
        if noms.numel() == 0 or recs.numel() == 0:
            continue
        nom = noms[0]
        teacher_delta = target[recs] - target[nom]
        pred_rank_delta = rank_score[recs] - rank_score[nom]
        oracle_j = int(torch.argmax(teacher_delta).item())
        oracle_adv_raw = float(teacher_delta[oracle_j].item())
        oracle_adv = max(0.0, oracle_adv_raw)
        pred_j = int(torch.argmax(pred_rank_delta).item())
        chosen_teacher_adv = float(teacher_delta[pred_j].item())
        positive_group = oracle_adv >= positive_gain
        bucket = int(bid[nom].item())
        suffix = "near" if bucket == 1 else ("contact" if bucket == 2 else "other")
        tie_eps = float(
            tcfg.get(
                "direct_value_preference_tie_epsilon_near" if bucket == 1 else "direct_value_preference_tie_epsilon_contact",
                0.025 if bucket == 1 else 0.010,
            )
        )
        acceptable = bool(chosen_teacher_adv >= oracle_adv_raw - tie_eps)
        regret = max(0.0, oracle_adv - chosen_teacher_adv - tie_eps) if positive_group else 0.0
        rank_harmful = chosen_teacher_adv <= -negative_gain
        rank_switch_nonpositive = (not positive_group) and bool(float(pred_rank_delta.max().item()) > 0.0)

        recovery_scores = pred_rank_delta
        if recovery_scores.numel() > 1:
            sorted_scores = torch.topk(recovery_scores, k=2).values
            runner = sorted_scores[1] if conditional_preference else torch.maximum(
                sorted_scores[1], recovery_scores.new_zeros(())
            )
        else:
            runner = recovery_scores[pred_j] - 1.0 if conditional_preference else recovery_scores.new_zeros(())
        rank_margin = float((recovery_scores[pred_j] - runner).item())

        chosen_idx = recs[pred_j]
        if risk_source in {"heads", "ordinal_evidence"} and opportunity_logit is not None and harm_logit is not None:
            opp_delta = opportunity_logit[chosen_idx] - opportunity_logit[nom]
            harm_delta = harm_logit[chosen_idx] - harm_logit[nom]
            opportunity_prob = float(torch.sigmoid(opp_delta).item())
            harm_prob = float(torch.sigmoid(harm_delta).item())
            delta_mean = opp_delta.new_tensor(opportunity_prob - harm_prob)
        else:
            if direct_delta is not None:
                delta_mean = direct_delta[chosen_idx]
                if direct_delta_logvar is not None:
                    delta_std = torch.exp(0.5 * direct_delta_logvar[chosen_idx]).clamp_min(1.0e-3)
                else:
                    delta_std = delta_mean.new_tensor(0.10)
            else:
                delta_mean = value_score[chosen_idx] - value_score[nom]
                if value_logvar is not None:
                    delta_std = torch.sqrt(
                        torch.exp(value_logvar[chosen_idx]) + torch.exp(value_logvar[nom])
                    ).clamp_min(1.0e-3)
                else:
                    delta_std = delta_mean.new_tensor(0.10)
            opportunity_prob = float(normal_cdf((delta_mean - positive_gain) / delta_std).item())
            harm_prob = float(normal_cdf((-negative_gain - delta_mean) / delta_std).item())
        admitted = bool(
            opportunity_prob >= opp_threshold
            and harm_prob <= harm_threshold
            and rank_margin >= rank_margin_threshold
            and float(delta_mean.item()) >= min_delta_mean
        )
        admitted_harmful = admitted and rank_harmful
        false_intervention = admitted and (not positive_group)
        positive_admission = admitted and positive_group
        scene_fold = int(abs(int(sh[nom].item())) % 3)
        conditional_regret = max(0.0, oracle_adv_raw - chosen_teacher_adv - tie_eps)
        conditional_hit = bool(chosen_teacher_adv >= oracle_adv_raw - tie_eps)
        for name in ("all", suffix, f"{suffix}_fold{scene_fold}"):
            add(f"conditional_count_{name}", 1.0)
            add(f"conditional_regret_sum_{name}", conditional_regret)
            add(f"conditional_top1_hit_{name}", float(conditional_hit))
            add(f"group_count_{name}", 1.0)
            add(f"rank_harmful_{name}", float(rank_harmful))
            add(f"rank_switch_nonpositive_{name}", float(rank_switch_nonpositive))
            add(f"admitted_harmful_{name}", float(admitted_harmful))
            add(f"false_intervention_{name}", float(false_intervention))
            add(f"admission_count_{name}", float(admitted))
        if positive_group:
            for name in ("all", suffix, f"{suffix}_fold{scene_fold}"):
                add(f"positive_count_{name}", 1.0)
                add(f"positive_regret_sum_{name}", regret)
                add(f"positive_top1_hit_{name}", float(acceptable))
                add(f"positive_admission_hit_{name}", float(positive_admission))
                add(f"positive_rank_margin_sum_{name}", rank_margin)
    return stats


def _finalize_direct_policy_stats(stats: dict[str, float], tcfg: dict | None = None) -> dict[str, float]:
    tcfg = tcfg or {}
    out: dict[str, float] = {}
    harm_lambda = float(tcfg.get("direct_policy_metric_harm_weight", 0.35))
    false_lambda = float(tcfg.get("direct_policy_metric_false_intervention_weight", 0.15))
    miss_lambda = float(tcfg.get("direct_policy_metric_missed_opportunity_weight", 0.25))
    rank_miss_lambda = float(tcfg.get("direct_policy_metric_rank_miss_weight", 0.10))
    rank_harm_lambda = float(tcfg.get("direct_policy_metric_rank_harm_weight", 0.25))
    rank_false_lambda = float(tcfg.get("direct_policy_metric_rank_false_switch_weight", 0.15))
    min_fold_positive = int(tcfg.get("direct_policy_metric_min_fold_positive", 6))
    robust_top_k = max(1, int(tcfg.get("direct_policy_metric_robust_top_k", 2)))
    conditional_preference = bool(tcfg.get("direct_value_preference_conditional_mode", False))
    suffixes = ["all", "near", "contact"] + [f"{regime}_fold{fold}" for regime in ("near", "contact") for fold in range(3)]
    for suffix in suffixes:
        tag = "" if suffix == "all" else f"_{suffix}"
        count = stats.get(f"group_count_{suffix}", 0.0)
        pos_count = stats.get(f"positive_count_{suffix}", 0.0)
        if count > 0:
            # Preserve pre-v48.5 diagnostics consumed by historical tests and
            # dashboards when legacy aggregate keys are supplied.
            if f"regret_sum_{suffix}" in stats:
                out[f"direct_group_regret_mean{tag}"] = stats.get(f"regret_sum_{suffix}", 0.0) / count
            if f"top1_hit_{suffix}" in stats:
                out[f"direct_group_top1_accuracy{tag}"] = stats.get(f"top1_hit_{suffix}", 0.0) / count
            out[f"direct_rank_harmful_top1_rate{tag}"] = stats.get(f"rank_harmful_{suffix}", stats.get(f"harmful_switch_{suffix}", 0.0)) / count
            out[f"direct_rank_false_switch_rate{tag}"] = stats.get(f"rank_switch_nonpositive_{suffix}", 0.0) / count
            out[f"direct_harmful_switch_rate{tag}"] = stats.get(f"admitted_harmful_{suffix}", stats.get(f"harmful_switch_{suffix}", 0.0)) / count
            out[f"direct_false_intervention_rate{tag}"] = stats.get(f"false_intervention_{suffix}", 0.0) / count
            out[f"direct_raw_admission_rate{tag}"] = stats.get(f"admission_count_{suffix}", 0.0) / count
        conditional_count = stats.get(f"conditional_count_{suffix}", 0.0)
        if conditional_count > 0:
            conditional_regret = stats.get(f"conditional_regret_sum_{suffix}", 0.0) / conditional_count
            conditional_top1 = stats.get(f"conditional_top1_hit_{suffix}", 0.0) / conditional_count
            out[f"direct_conditional_recovery_regret_mean{tag}"] = conditional_regret
            out[f"direct_conditional_recovery_top1_accuracy{tag}"] = conditional_top1
            out[f"direct_conditional_group_count{tag}"] = float(conditional_count)
            if conditional_preference:
                out[f"direct_preference_risk_mean{tag}"] = (
                    conditional_regret + rank_miss_lambda * (1.0 - conditional_top1)
                )
        if pos_count > 0:
            regret = stats.get(f"positive_regret_sum_{suffix}", 0.0) / pos_count
            top1 = stats.get(f"positive_top1_hit_{suffix}", 0.0) / pos_count
            recall = stats.get(f"positive_admission_hit_{suffix}", 0.0) / pos_count
            out[f"direct_positive_group_regret_mean{tag}"] = regret
            out[f"direct_positive_group_top1_accuracy{tag}"] = top1
            out[f"direct_positive_admission_recall{tag}"] = recall
            out[f"direct_positive_rank_margin_mean{tag}"] = stats.get(f"positive_rank_margin_sum_{suffix}", 0.0) / pos_count
            if not conditional_preference:
                out[f"direct_preference_risk_mean{tag}"] = (
                    regret
                    + rank_miss_lambda * (1.0 - top1)
                    + rank_harm_lambda * out.get(f"direct_rank_harmful_top1_rate{tag}", 0.0)
                    + rank_false_lambda * out.get(f"direct_rank_false_switch_rate{tag}", 0.0)
                )
            out[f"direct_positive_group_count{tag}"] = float(pos_count)
            out[f"direct_group_count{tag}"] = float(count)
        if count > 0 and pos_count > 0:
            cert_risk = (
                out[f"direct_positive_group_regret_mean{tag}"]
                + harm_lambda * out[f"direct_harmful_switch_rate{tag}"]
                + false_lambda * out[f"direct_false_intervention_rate{tag}"]
                + miss_lambda * (1.0 - out[f"direct_positive_admission_recall{tag}"])
            )
            out[f"direct_certificate_risk_mean{tag}"] = cert_risk
            # Backward-compatible alias used by older dashboards.
            out[f"direct_policy_risk_mean{tag}"] = cert_risk
    legacy_regrets = [out[k] for k in ("direct_group_regret_mean_near", "direct_group_regret_mean_contact") if k in out]
    if legacy_regrets:
        out["direct_group_regret_mean_worst"] = max(legacy_regrets)
    for metric in ("direct_preference_risk_mean", "direct_certificate_risk_mean", "direct_policy_risk_mean"):
        regime_vals = [out[k] for k in (f"{metric}_near", f"{metric}_contact") if k in out]
        if regime_vals:
            out[f"{metric}_worst"] = max(regime_vals)
        fold_vals = [
            value for key, value in out.items()
            if key.startswith(f"{metric}_near_fold") or key.startswith(f"{metric}_contact_fold")
        ]
        if fold_vals:
            out[f"{metric.replace('_mean', '')}_fold_worst"] = max(fold_vals)
    # v48.8 SCOPE: sparse fold maxima made v48.7 Stage-P selection brittle.
    # Aggregate only folds with enough positive groups, then use the mean of the
    # worst K supported folds (CVaR-like) instead of a single noisy maximum.
    for metric in ("direct_preference_risk_mean", "direct_certificate_risk_mean"):
        supported: list[float] = []
        for regime in ("near", "contact"):
            for fold in range(3):
                key = f"{metric}_{regime}_fold{fold}"
                count_key = (
                    f"direct_conditional_group_count_{regime}_fold{fold}"
                    if conditional_preference and metric == "direct_preference_risk_mean"
                    else f"direct_positive_group_count_{regime}_fold{fold}"
                )
                if key in out and out.get(count_key, 0.0) >= float(min_fold_positive):
                    supported.append(float(out[key]))
        if supported:
            ordered = sorted(supported, reverse=True)
            k = min(robust_top_k, len(ordered))
            out[f"{metric.replace('_mean', '')}_fold_robust"] = sum(ordered[:k]) / float(k)
            out[f"{metric.replace('_mean', '')}_supported_fold_count"] = float(len(supported))

    # Exact names consumed by v48.7 staged training.
    if "direct_preference_risk_fold_worst" not in out:
        vals = [v for k, v in out.items() if k.startswith("direct_preference_risk_mean_") and "fold" in k]
        if vals:
            out["direct_preference_risk_fold_worst"] = max(vals)
    if "direct_certificate_risk_fold_worst" not in out:
        vals = [v for k, v in out.items() if k.startswith("direct_certificate_risk_mean_") and "fold" in k]
        if vals:
            out["direct_certificate_risk_fold_worst"] = max(vals)
    return out


def _epoch(
    model: OCRAPModel,
    loader: DataLoader,
    cfg: dict,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    stage: str = "train",
    epoch: int | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    lw = cfg.get("loss_weights", {}) if isinstance(cfg.get("loss_weights", {}), dict) else {}
    ocfg = cfg.get("ocmero", {}) if isinstance(cfg.get("ocmero", {}), dict) else {}
    art_cfg = cfg.get("artifact", {}) if isinstance(cfg.get("artifact", {}), dict) else {}
    tcfg = cfg.get("training", {}) if isinstance(cfg.get("training", {}), dict) else {}
    model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model", {}), dict) else {}
    if training and bool(tcfg.get("frozen_modules_eval", False)):
        _keep_fully_frozen_modules_in_eval(model)
    progress = bool(tcfg.get("progress", cfg.get("progress", True)))
    direct_only_fast = bool(tcfg.get("direct_only_fast_path", False))
    amp_enabled = bool(tcfg.get("amp", False)) and device.type == "cuda"
    amp_dtype_name = str(tcfg.get("amp_dtype", "bfloat16")).strip().lower()
    amp_dtype = torch.float16 if amp_dtype_name in {"float16", "fp16", "half"} else torch.bfloat16
    totals: dict[str, float] = {}
    policy_totals: dict[str, float] = {}
    n = 0
    desc = f"{stage} ep{epoch}" if epoch is not None else stage
    for batch in _progress_iter(loader, enabled=progress, desc=desc):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        if direct_only_fast:
            amp_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if amp_enabled else nullcontext()
            with amp_ctx:
                group_index = torch.stack([
                    batch.get("bucket_id", torch.zeros_like(batch["time_index"])),
                    batch["scene_hash"], batch["time_index"],
                ], dim=-1)
                out = model(
                    batch["x"].float(), batch.get("option_features"),
                    bucket_id=batch.get("bucket_id"), group_index=group_index,
                    is_nominal=batch.get("is_nominal"), direct_only=True,
                )
            root_valid = batch["root_valid"].bool()
            option_gamma = float(art_cfg.get("admission_gamma", 0.0))
            option_temperature = float(tcfg.get("option_success_temperature", 0.35))
            with torch.no_grad():
                _teacher_r_dep, _teacher_r_orc, _teacher_gap, teacher_q = torch_oc_mero(
                    batch["m_star"].float(), batch["root_probs"].float(), batch["c_star"].float(),
                    alpha=float(ocfg.get("alpha", 0.2)), beta=float(ocfg.get("beta", 0.2)),
                    option_valid=batch["option_valid"], root_valid=root_valid,
                    use_lcvar=not bool((cfg.get("ablation", {}) or {}).get("without_lower_tail", False)),
                    use_obs_kernel=not bool((cfg.get("ablation", {}) or {}).get("without_observation_kernel", False)),
                    top_m=int(ocfg.get("top_m", 8)),
                )
            loss_direct_value = _direct_value_loss_from_outputs(
                out, batch, tcfg, model_cfg, teacher_q,
                option_gamma=option_gamma, option_temperature=option_temperature,
            )
            if not training:
                batch_policy_stats = _direct_policy_batch_stats(
                    out, batch, tcfg, teacher_q,
                    option_gamma=option_gamma, option_temperature=option_temperature,
                )
                for key, value in batch_policy_stats.items():
                    policy_totals[key] = policy_totals.get(key, 0.0) + float(value)
            loss_encoder_anchor = _parameter_anchor_loss(model)
            total = (
                float(lw.get("direct_recovery_value", 1.0)) * loss_direct_value
                + float(tcfg.get("encoder_anchor_weight", 0.0)) * loss_encoder_anchor
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
                total.backward()
                grad_clip = float(tcfg.get("grad_clip", 5.0))
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            bsz = int(batch["x"].shape[0])
            n += bsz
            vals = {
                "loss": float(total.item()),
                "loss_direct_recovery_value": float(loss_direct_value.item()),
                "loss_encoder_anchor": float(loss_encoder_anchor.item()),
                "direct_score_mean": float(out["direct_recovery_value_logit"].float().mean().item()),
                "direct_opportunity_mean": float(torch.sigmoid(out["direct_recovery_opportunity_logit"]).float().mean().item()) if "direct_recovery_opportunity_logit" in out else 0.0,
                "direct_harm_mean": float(torch.sigmoid(out["direct_recovery_harm_logit"]).float().mean().item()) if "direct_recovery_harm_logit" in out else 0.0,
                "direct_expert_disagreement_mean": float(out["direct_expert_disagreement"][:, 0].float().mean().item()) if "direct_expert_disagreement" in out else 0.0,
            }
            for k, v in vals.items():
                totals[k] = totals.get(k, 0.0) + float(v) * bsz
            continue

        group_index = torch.stack([
            batch.get("bucket_id", torch.zeros_like(batch["time_index"])),
            batch["scene_hash"], batch["time_index"],
        ], dim=-1)
        out = model(
            batch["x"].float(), batch.get("option_features"), bucket_id=batch.get("bucket_id"),
            group_index=group_index, is_nominal=batch.get("is_nominal"),
        )
        root_valid = batch["root_valid"].bool()
        masked_logits = out["root_logits"].masked_fill(~root_valid, -1.0e4)
        root_p = torch.softmax(masked_logits, dim=-1)
        root_target = batch["root_probs"].float() * root_valid.float()
        root_target = root_target / root_target.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        margin_target = torch.clamp(batch["m_star"].float(), min=-float(cfg.get("margin_clip", 5.0)), max=float(cfg.get("margin_clip", 5.0)))
        margin_mask = batch["root_valid"].unsqueeze(-1) & batch["option_valid"].unsqueeze(1)

        loss_root = -(root_target * F.log_softmax(masked_logits, dim=-1)).sum(dim=-1).mean()
        loss_margin = _masked_smooth_l1(out["margins"], margin_target, margin_mask)
        loss_sig = _root_signature_loss(out, batch, "root_signature")
        loss_future_sig = _root_signature_loss(out, batch, "root_future_signature")
        loss_obs = _obs_bce(
            out["c_star"],
            batch["y_obs"],
            batch["root_valid"],
            balanced=bool(tcfg.get("balanced_obs_loss", True)),
        )
        r_dep, r_orc, gap, pred_q = torch_oc_mero(
            out["margins"],
            root_p,
            out["c_star"],
            alpha=float(ocfg.get("alpha", 0.2)),
            beta=float(ocfg.get("beta", 0.2)),
            option_valid=batch["option_valid"],
            root_valid=root_valid,
            use_lcvar=not bool((cfg.get("ablation", {}) or {}).get("without_lower_tail", False)),
            use_obs_kernel=not bool((cfg.get("ablation", {}) or {}).get("without_observation_kernel", False)),
            top_m=int(ocfg.get("top_m", 8)),
        )
        with torch.no_grad():
            _teacher_r_dep, _teacher_r_orc, _teacher_gap, teacher_q = torch_oc_mero(
                batch["m_star"].float(),
                batch["root_probs"].float(),
                batch["c_star"].float(),
                alpha=float(ocfg.get("alpha", 0.2)),
                beta=float(ocfg.get("beta", 0.2)),
                option_valid=batch["option_valid"],
                root_valid=root_valid,
                use_lcvar=not bool((cfg.get("ablation", {}) or {}).get("without_lower_tail", False)),
                use_obs_kernel=not bool((cfg.get("ablation", {}) or {}).get("without_observation_kernel", False)),
                top_m=int(ocfg.get("top_m", 8)),
            )
        loss_dep = F.smooth_l1_loss(r_dep, batch["r_dep_star"].float())
        loss_orc = F.smooth_l1_loss(r_orc, batch["r_orc_star"].float())
        loss_art = anti_oracle_loss(r_orc, r_dep, batch["i_art_star"].float(), delta_neg=float(art_cfg.get("delta_neg", 0.0)))
        loss_gap = artifact_gap_loss(gap, (batch["r_orc_star"].float() - batch["r_dep_star"].float()).clamp_min(0.0), batch["i_art_star"].float(), margin=float(art_cfg.get("gap_margin", 0.5)))
        loss_admit = deployability_classification_loss(r_dep, batch["r_dep_star"].float(), gamma=float(art_cfg.get("admission_gamma", 0.0)))
        option_gamma = float(art_cfg.get("admission_gamma", 0.0))
        option_temperature = float(tcfg.get("option_success_temperature", 0.35))
        loss_option_q = shared_option_q_regression_loss(pred_q, teacher_q, batch["root_valid"], batch["option_valid"])
        loss_option_admit = shared_option_admission_loss(
            pred_q,
            teacher_q,
            batch["root_probs"].float(),
            batch["root_valid"],
            batch["option_valid"],
            gamma=option_gamma,
        )
        loss_option_success = shared_option_success_regression_loss(
            pred_q,
            teacher_q,
            batch["root_probs"].float(),
            batch["root_valid"],
            batch["option_valid"],
            gamma=option_gamma,
            temperature=option_temperature,
        )
        loss_option_success_bce = shared_option_success_bce_loss(
            pred_q,
            teacher_q,
            batch["root_probs"].float(),
            batch["root_valid"],
            batch["option_valid"],
            gamma=option_gamma,
            temperature=option_temperature,
        )
        loss_option_best = best_shared_option_loss(
            pred_q,
            teacher_q,
            batch["root_probs"].float(),
            batch["root_valid"],
            batch["option_valid"],
            gamma=option_gamma,
            temperature=option_temperature,
        )
        loss_group_rank = groupwise_candidate_ranking_loss(
            r_dep,
            gap,
            batch["r_dep_star"].float(),
            batch["r_orc_star"].float(),
            batch["i_art_star"].float(),
            batch["scene_hash"],
            batch["time_index"],
            batch["candidate_index"],
            margin=float(tcfg.get("group_ranking_margin", 0.25)),
            gap_weight=float(tcfg.get("group_ranking_gap_weight", 0.25)),
            teacher_gap_weight=float(tcfg.get("group_ranking_teacher_gap_weight", 0.25)),
            artifact_only=bool(tcfg.get("group_ranking_artifact_only", True)),
        )
        loss_group_ce = groupwise_candidate_ce_loss(
            r_dep,
            gap,
            batch["utility"].float(),
            batch["r_dep_star"].float(),
            batch["r_orc_star"].float(),
            batch["scene_hash"],
            batch["time_index"],
            temperature=float(tcfg.get("group_ce_temperature", 0.35)),
            pred_gap_weight=float(tcfg.get("group_ce_pred_gap_weight", 0.35)),
            teacher_gap_weight=float(tcfg.get("group_ce_teacher_gap_weight", 0.35)),
            utility_weight=float(tcfg.get("group_ce_utility_weight", 0.03)),
            require_deployable_target=bool(tcfg.get("group_ce_require_deployable_target", True)),
        )
        loss_nominal_switch = nominal_switch_consistency_loss(
            r_dep,
            gap,
            batch["utility"].float(),
            batch["r_dep_star"].float(),
            batch["r_orc_star"].float(),
            batch["scene_hash"],
            batch["time_index"],
            batch["is_nominal"].float(),
            margin=float(tcfg.get("nominal_switch_margin", 0.12)),
            pred_gap_weight=float(tcfg.get("nominal_switch_pred_gap_weight", 0.30)),
            teacher_gap_weight=float(tcfg.get("nominal_switch_teacher_gap_weight", 0.35)),
            utility_weight=float(tcfg.get("nominal_switch_utility_weight", 0.03)),
            teacher_gain_margin=float(tcfg.get("nominal_switch_teacher_gain_margin", 0.06)),
            nominal_deployable_gamma=float(tcfg.get("nominal_switch_deployable_gamma", 0.0)),
            nominal_gap_max=float(tcfg.get("nominal_switch_gap_max", 0.30)),
        )
        loss_group_distill = groupwise_score_distillation_loss(
            r_dep,
            gap,
            batch["utility"].float(),
            batch["r_dep_star"].float(),
            batch["r_orc_star"].float(),
            batch["scene_hash"],
            batch["time_index"],
            pred_gap_weight=float(tcfg.get("group_distill_pred_gap_weight", 0.45)),
            teacher_gap_weight=float(tcfg.get("group_distill_teacher_gap_weight", 0.45)),
            utility_weight=float(tcfg.get("group_distill_utility_weight", 0.02)),
            teacher_temperature=float(tcfg.get("group_distill_teacher_temperature", 0.20)),
            pred_temperature=float(tcfg.get("group_distill_pred_temperature", 0.30)),
        )
        loss_safe_nominal = safe_nominal_preservation_loss(
            r_dep,
            gap,
            batch["utility"].float(),
            pred_q,
            batch["root_probs"].float(),
            batch["root_valid"],
            batch["option_valid"],
            batch["scene_hash"],
            batch["time_index"],
            batch["is_nominal"].float(),
            batch.get("bucket_id", torch.full_like(batch["time_index"], 3)),
            margin=float(tcfg.get("safe_nominal_margin", 0.18)),
            pred_gap_weight=float(tcfg.get("safe_nominal_pred_gap_weight", 0.35)),
            utility_weight=float(tcfg.get("safe_nominal_utility_weight", 0.03)),
            drs_weight=float(tcfg.get("safe_nominal_drs_weight", 0.30)),
            min_nominal_success=float(tcfg.get("safe_nominal_min_success", 0.90)),
            success_gamma=option_gamma,
            success_temperature=option_temperature,
        )
        loss_protective_macro = protective_macro_recovery_loss(
            r_dep,
            gap,
            batch["utility"].float(),
            pred_q,
            batch["r_dep_star"].float(),
            batch["r_orc_star"].float(),
            teacher_q,
            batch["root_probs"].float(),
            batch["root_valid"],
            batch["option_valid"],
            batch["scene_hash"],
            batch["time_index"],
            batch.get("prefix_macro_type_id", batch.get("candidate_index", torch.zeros_like(batch["time_index"]))),
            batch["is_nominal"].float(),
            batch.get("bucket_id", torch.full_like(batch["time_index"], 3)),
            macro_ids=_parse_int_tuple(tcfg.get("protective_macro_ids", "2,7"), (2, 7)),
            bucket_ids=_parse_int_tuple(tcfg.get("protective_macro_bucket_ids", "2"), (2,)),
            margin=float(tcfg.get("protective_macro_margin", 0.14)),
            min_teacher_r_dep=float(tcfg.get("protective_macro_min_teacher_r_dep", 0.0)),
            min_teacher_drs=float(tcfg.get("protective_macro_min_teacher_drs", 0.50)),
            min_teacher_pcd_gain=float(tcfg.get("protective_macro_min_teacher_pcd_gain", 0.02)),
            max_nominal_teacher_pcd=float(tcfg.get("protective_macro_max_nominal_teacher_pcd", 0.90)),
            pred_gap_weight=float(tcfg.get("protective_macro_pred_gap_weight", 0.18)),
            pred_drs_weight=float(tcfg.get("protective_macro_pred_drs_weight", 0.65)),
            utility_weight=float(tcfg.get("protective_macro_utility_weight", 0.02)),
            teacher_gap_weight=float(tcfg.get("protective_macro_teacher_gap_weight", 0.10)),
            teacher_drs_weight=float(tcfg.get("protective_macro_teacher_drs_weight", 0.70)),
            success_gamma=option_gamma,
            success_temperature=option_temperature,
            target_min_pred_drs=float(tcfg.get("protective_macro_target_min_pred_drs", 0.62)),
        )
        loss_macro_drs = macro_shared_success_calibration_loss(
            pred_q,
            teacher_q,
            batch["root_probs"].float(),
            batch["root_valid"],
            batch["option_valid"],
            batch.get("prefix_macro_type_id", batch.get("candidate_index", torch.zeros_like(batch["time_index"]))),
            batch.get("bucket_id", torch.full_like(batch["time_index"], 3)),
            macro_ids=_parse_int_tuple(tcfg.get("macro_drs_ids", "2,3,5,7"), (2, 3, 5, 7)),
            bucket_ids=_parse_int_tuple(tcfg.get("macro_drs_bucket_ids", "1,2"), (1, 2)),
            gamma=option_gamma,
            temperature=option_temperature,
            pos_threshold=float(tcfg.get("macro_drs_pos_threshold", 0.80)),
            neg_threshold=float(tcfg.get("macro_drs_neg_threshold", 0.05)),
            pos_weight=float(tcfg.get("macro_drs_pos_weight", 4.0)),
            neg_weight=float(tcfg.get("macro_drs_neg_weight", 1.0)),
        )
        loss_ddc = deployability_dominance_calibration_loss(
            r_dep,
            gap,
            batch["utility"].float(),
            pred_q,
            batch["r_dep_star"].float(),
            batch["r_orc_star"].float(),
            teacher_q,
            batch["root_probs"].float(),
            batch["root_valid"],
            batch["option_valid"],
            batch["scene_hash"],
            batch["time_index"],
            batch.get("prefix_macro_type_id", batch.get("candidate_index", torch.zeros_like(batch["time_index"]))),
            batch["is_nominal"].float(),
            batch.get("bucket_id", torch.full_like(batch["time_index"], 3)),
            macro_ids=_parse_int_tuple(tcfg.get("ddc_macro_ids", "2,3,5,7"), (2, 3, 5, 7)),
            bucket_ids=_parse_int_tuple(tcfg.get("ddc_bucket_ids", "1,2"), (1, 2)),
            margin=float(tcfg.get("ddc_margin", 0.12)),
            min_teacher_pcd_gain=float(tcfg.get("ddc_min_teacher_pcd_gain", 0.04)),
            min_teacher_best_pcd=float(tcfg.get("ddc_min_teacher_best_pcd", 0.50)),
            max_nominal_teacher_pcd=float(tcfg.get("ddc_max_nominal_teacher_pcd", 0.62)),
            pred_gap_weight=float(tcfg.get("ddc_pred_gap_weight", 0.20)),
            pred_drs_weight=float(tcfg.get("ddc_pred_drs_weight", 0.35)),
            utility_weight=float(tcfg.get("ddc_utility_weight", 0.00)),
            success_gamma=option_gamma,
            success_temperature=option_temperature,
            target_min_pred_pcd=float(tcfg.get("ddc_target_min_pred_pcd", 0.45)),
            nominal_max_pred_pcd=float(tcfg.get("ddc_nominal_max_pred_pcd", 0.55)),
        )
        loss_teacher_pcd_direct = direct_teacher_pcd_loss(
            r_dep,
            gap,
            batch["utility"].float(),
            pred_q,
            batch["r_dep_star"].float(),
            batch["r_orc_star"].float(),
            teacher_q,
            batch["root_probs"].float(),
            batch["root_valid"],
            batch["option_valid"],
            batch["scene_hash"],
            batch["time_index"],
            batch.get("prefix_macro_type_id", batch.get("candidate_index", torch.zeros_like(batch["time_index"]))),
            batch["is_nominal"].float(),
            batch.get("bucket_id", torch.full_like(batch["time_index"], 3)),
            macro_ids=_parse_int_tuple(tcfg.get("teacher_pcd_direct_macro_ids", "2,3,5,7"), (2, 3, 5, 7)),
            positive_macro_ids=_parse_int_tuple(tcfg.get("teacher_pcd_direct_positive_macro_ids", tcfg.get("teacher_pcd_direct_macro_ids", "2,3,5,7")), (2, 3, 5, 7)),
            bucket_ids=_parse_int_tuple(tcfg.get("teacher_pcd_direct_bucket_ids", "2"), (2,)),
            success_gamma=option_gamma,
            success_temperature=option_temperature,
            regression_weight=float(tcfg.get("teacher_pcd_direct_regression_weight", 1.0)),
            ranking_weight=float(tcfg.get("teacher_pcd_direct_ranking_weight", 2.5)),
            nominal_penalty_weight=float(tcfg.get("teacher_pcd_direct_nominal_penalty_weight", 1.0)),
            false_positive_weight=float(tcfg.get("teacher_pcd_direct_false_positive_weight", 1.5)),
            margin=float(tcfg.get("teacher_pcd_direct_margin", 0.18)),
            min_teacher_pcd_gain=float(tcfg.get("teacher_pcd_direct_min_teacher_pcd_gain", 0.015)),
            min_teacher_best_pcd=float(tcfg.get("teacher_pcd_direct_min_teacher_best_pcd", 0.50)),
            max_nominal_teacher_pcd=float(tcfg.get("teacher_pcd_direct_max_nominal_teacher_pcd", 0.68)),
            target_min_pred_pcd=float(tcfg.get("teacher_pcd_direct_target_min_pred_pcd", 0.52)),
            nominal_max_pred_pcd=float(tcfg.get("teacher_pcd_direct_nominal_max_pred_pcd", 0.50)),
            focus_non_nominal_weight=float(tcfg.get("teacher_pcd_direct_focus_non_nominal_weight", 2.0)),
            false_positive_margin=float(tcfg.get("teacher_pcd_direct_false_positive_margin", 0.03)),
            component_weight=float(tcfg.get("teacher_pcd_direct_component_weight", 0.0)),
            positive_component_weight=float(tcfg.get("teacher_pcd_direct_positive_component_weight", 0.0)),
            nominal_cap_weight=float(tcfg.get("teacher_pcd_direct_nominal_cap_weight", 1.0)),
            positive_rank_all_weight=float(tcfg.get("teacher_pcd_direct_positive_rank_all_weight", 0.0)),
            positive_floor_weight=float(tcfg.get("teacher_pcd_direct_positive_floor_weight", 0.0)),
            positive_min_pred_r_dep=float(tcfg.get("teacher_pcd_direct_positive_min_pred_r_dep", -1.0e9)),
            positive_max_pred_gap=float(tcfg.get("teacher_pcd_direct_positive_max_pred_gap", -1.0)),
            positive_min_pred_drs=float(tcfg.get("teacher_pcd_direct_positive_min_pred_drs", -1.0)),
        )
        loss_recovery_advantage = observation_consistent_recovery_advantage_loss(
            r_dep, gap, pred_q,
            batch["r_dep_star"].float(), batch["r_orc_star"].float(), teacher_q,
            batch["root_probs"].float(), batch["root_valid"], batch["option_valid"],
            batch["scene_hash"], batch["time_index"],
            batch.get("prefix_macro_type_id", batch.get("candidate_index", torch.zeros_like(batch["time_index"]))),
            batch["is_nominal"].float(),
            batch.get("bucket_id", torch.full_like(batch["time_index"], 3)),
            macro_ids=_parse_int_tuple(tcfg.get("recovery_advantage_macro_ids", "2,3,5,7"), (2, 3, 5, 7)),
            bucket_ids=_parse_int_tuple(tcfg.get("recovery_advantage_bucket_ids", "1,2"), (1, 2)),
            positive_gain=float(tcfg.get("recovery_advantage_positive_gain", 0.03)),
            negative_gain=float(tcfg.get("recovery_advantage_negative_gain", 0.03)),
            advantage_margin=float(tcfg.get("recovery_advantage_margin", 0.10)),
            regression_weight=float(tcfg.get("recovery_advantage_regression_weight", 1.0)),
            ranking_weight=float(tcfg.get("recovery_advantage_ranking_weight", 1.0)),
            component_inversion_weight=float(tcfg.get("recovery_advantage_component_weight", 0.5)),
            false_positive_weight=float(tcfg.get("recovery_advantage_false_positive_weight", 0.75)),
            nominal_failure_pcd_max=float(tcfg.get("recovery_advantage_nominal_failure_pcd_max", 0.20)),
            target_min_pred_pcd=float(tcfg.get("recovery_advantage_target_min_pred_pcd", 0.50)),
            nominal_max_pred_pcd=float(tcfg.get("recovery_advantage_nominal_max_pred_pcd", 0.48)),
            near_weight=float(tcfg.get("recovery_advantage_near_weight", 1.5)),
            contact_weight=float(tcfg.get("recovery_advantage_contact_weight", 1.0)),
            success_gamma=option_gamma,
            success_temperature=option_temperature,
        )
        if "direct_recovery_value_logit" in out:
            loss_direct_value = _direct_value_loss_from_outputs(
                out, batch, tcfg, model_cfg, teacher_q,
                option_gamma=option_gamma, option_temperature=option_temperature,
            )
        else:
            loss_direct_value = r_dep.sum() * 0.0
        if "direct_expert_weights" in out:
            # Small unsupervised anti-collapse regularizer.  It does not teach a
            # regime label; it only prevents one observation-conditioned expert
            # from receiving all traffic before the value loss differentiates
            # the experts.  Per-sample weights remain free to be sharp.
            mean_expert_weight = out["direct_expert_weights"].mean(dim=0)
            target_expert_weight = torch.full_like(
                mean_expert_weight, 1.0 / float(mean_expert_weight.numel())
            )
            loss_direct_router_balance = F.mse_loss(
                mean_expert_weight, target_expert_weight
            )
        else:
            loss_direct_router_balance = r_dep.sum() * 0.0
        if bool((cfg.get("ablation", {}) or {}).get("without_anti_oracle", False)):
            loss_art = loss_art * 0.0
            loss_gap = loss_gap * 0.0
        loss_util = F.smooth_l1_loss(out["utility"], batch["utility"].float())
        total = (
            float(lw.get("assign", 1.0)) * loss_root
            + float(lw.get("margin", 2.0)) * loss_margin
            + float(lw.get("sig", 0.5)) * (loss_sig + loss_future_sig)
            + float(lw.get("obs", 1.0)) * loss_obs
            + float(lw.get("dep", 0.5)) * loss_dep
            + float(lw.get("orc", 0.5)) * loss_orc
            + float(lw.get("anti_oracle", 1.0)) * loss_art
            + float(lw.get("artifact_gap", 0.5)) * loss_gap
            + float(lw.get("admission", 0.2)) * loss_admit
            + float(lw.get("option_q", 0.5)) * loss_option_q
            + float(lw.get("option_admission", 0.4)) * loss_option_admit
            + float(lw.get("option_success", 0.0)) * loss_option_success
            + float(lw.get("option_success_bce", 0.0)) * loss_option_success_bce
            + float(lw.get("option_best", 0.2)) * loss_option_best
            + float(lw.get("group_ranking", 0.0)) * loss_group_rank
            + float(lw.get("group_ce", 0.0)) * loss_group_ce
            + float(lw.get("nominal_switch", 0.0)) * loss_nominal_switch
            + float(lw.get("group_distill", 0.0)) * loss_group_distill
            + float(lw.get("safe_nominal", 0.0)) * loss_safe_nominal
            + float(lw.get("protective_macro", 0.0)) * loss_protective_macro
            + float(lw.get("macro_drs", 0.0)) * loss_macro_drs
            + float(lw.get("ddc", 0.0)) * loss_ddc
            + float(lw.get("teacher_pcd_direct", 0.0)) * loss_teacher_pcd_direct
            + float(lw.get("recovery_advantage", 0.0)) * loss_recovery_advantage
            + float(lw.get("direct_recovery_value", 0.0)) * loss_direct_value
            + float(lw.get("direct_router_balance", 0.0)) * loss_direct_router_balance
            + float(lw.get("utility", 0.2)) * loss_util
        )
        if training:
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            grad_clip = float(tcfg.get("grad_clip", 5.0))
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        bsz = int(batch["x"].shape[0])
        n += bsz
        vals = {
            "loss": total.item(),
            "loss_root": loss_root.item(),
            "loss_margin": loss_margin.item(),
            "loss_sig": loss_sig.item(),
            "loss_future_sig": loss_future_sig.item(),
            "loss_obs": loss_obs.item(),
            "loss_dep": loss_dep.item(),
            "loss_orc": loss_orc.item(),
            "loss_art": loss_art.item(),
            "loss_gap": loss_gap.item(),
            "loss_admission": loss_admit.item(),
            "loss_option_q": loss_option_q.item(),
            "loss_option_admission": loss_option_admit.item(),
            "loss_option_success": loss_option_success.item(),
            "loss_option_success_bce": loss_option_success_bce.item(),
            "loss_option_best": loss_option_best.item(),
            "loss_group_ranking": loss_group_rank.item(),
            "loss_group_ce": loss_group_ce.item(),
            "loss_nominal_switch": loss_nominal_switch.item(),
            "loss_group_distill": loss_group_distill.item(),
            "loss_safe_nominal": loss_safe_nominal.item(),
            "loss_protective_macro": loss_protective_macro.item(),
            "loss_macro_drs": loss_macro_drs.item(),
            "loss_ddc": loss_ddc.item(),
            "loss_teacher_pcd_direct": loss_teacher_pcd_direct.item(),
            "loss_recovery_advantage": loss_recovery_advantage.item(),
            "loss_direct_recovery_value": loss_direct_value.item(),
            "loss_direct_router_balance": loss_direct_router_balance.item(),
            "loss_utility": loss_util.item(),
            "pred_r_dep_mean": r_dep.mean().item(),
            "teacher_r_dep_mean": batch["r_dep_star"].float().mean().item(),
        }
        for k, v in vals.items():
            totals[k] = totals.get(k, 0.0) + float(v) * bsz
    epoch_metrics = {k: float(v / max(n, 1)) for k, v in totals.items()}
    epoch_metrics.update(_finalize_direct_policy_stats(policy_totals, tcfg))
    epoch_metrics.update({"num_samples": int(n), "num_batches": int(len(loader))})
    return epoch_metrics



class SceneTimeBatchSampler(Sampler[list[int]]):
    """Batch sampler that keeps scene-time candidate sets together.

    Group-wise ranking losses need multiple candidates from the same scene-time
    in the same mini-batch.  Standard random sampling destroys that structure.
    """

    def __init__(self, groups: list[list[int]], batch_size: int, *, group_weights: list[float] | None = None, replacement: bool = True, shuffle_within_group: bool = True, shuffle_groups: bool = True):
        self.groups = [list(g) for g in groups if g]
        self.batch_size = max(1, int(batch_size))
        self.replacement = bool(replacement)
        self.shuffle_within_group = bool(shuffle_within_group)
        self.shuffle_groups = bool(shuffle_groups)
        if group_weights is None or len(group_weights) != len(self.groups):
            self.group_weights = torch.ones((len(self.groups),), dtype=torch.double)
        else:
            self.group_weights = torch.as_tensor(group_weights, dtype=torch.double).clamp_min(1.0e-8)

    def __len__(self) -> int:
        if not self.groups:
            return 0
        total = sum(len(g) for g in self.groups)
        return max(1, int(np.ceil(total / float(self.batch_size))))

    def __iter__(self):
        if not self.groups:
            return
        if self.replacement:
            order = torch.multinomial(self.group_weights, num_samples=len(self.groups), replacement=True).tolist()
        elif self.shuffle_groups:
            order = torch.randperm(len(self.groups)).tolist()
        else:
            order = list(range(len(self.groups)))
        batch: list[int] = []
        for gi in order:
            inds = list(self.groups[int(gi)])
            if self.shuffle_within_group and len(inds) > 1:
                perm = torch.randperm(len(inds)).tolist()
                inds = [inds[i] for i in perm]
            if len(batch) + len(inds) > self.batch_size and batch:
                yield batch
                batch = []
            if len(inds) > self.batch_size:
                for j in range(0, len(inds), self.batch_size):
                    chunk = inds[j : j + self.batch_size]
                    if chunk:
                        yield chunk
            else:
                batch.extend(inds)
        if batch:
            yield batch


def _sampler_weight_for_path(p: Path, cfg: dict, root_counts: Counter, total: int) -> tuple[float, bool, bool, bool]:
    tcfg = cfg.get("training", {}) if isinstance(cfg.get("training", {}), dict) else {}
    weight_art = float(tcfg.get("artifact_sampler_weight", 0.25))
    weight_neg = float(tcfg.get("negative_deployable_sampler_weight", 0.75))
    weight_safe_pos = float(tcfg.get("safe_positive_sampler_weight", 0.25))
    regime_balance_power = float(tcfg.get("regime_balance_power", 0.0))
    try:
        is_art = float(np.asarray(scalar_metadata_for_path(p, "i_art_star", 0)).item()) > 0.5
        r_dep = float(np.asarray(scalar_metadata_for_path(p, "r_dep_star", 0)).item())
        is_neg = r_dep < 0.0
        root_name = _dataset_root_name(p).lower()
        is_safe_pos = ("safe" in root_name) and (not is_neg) and (not is_art)
        w = 1.0
        if regime_balance_power > 0:
            root_count = max(1, int(root_counts.get(_dataset_root_name(p), 1)))
            w *= float((total / max(len(root_counts), 1) / root_count) ** regime_balance_power)
        if is_art:
            w += weight_art
        if is_neg:
            w += weight_neg
        if is_safe_pos:
            w += weight_safe_pos
        return float(w), bool(is_art), bool(is_neg), bool(is_safe_pos)
    except Exception:
        return 1.0, False, False, False


def _sampler_teacher_pcd(path: Path, cfg: dict) -> float:
    """Compute the exact composite PCD target used by training/calibration."""
    d = load_npz(path)
    m = np.asarray(d["m_star"], dtype=np.float64)
    probs = np.asarray(d["root_probs"], dtype=np.float64)
    c = np.asarray(d.get("c_star", np.eye(m.shape[0])), dtype=np.float64)
    root_valid = np.asarray(d.get("root_valid", np.ones(m.shape[0])), dtype=bool)
    option_valid = np.asarray(d.get("option_valid", np.ones(m.shape[1])), dtype=bool)
    ocfg = cfg.get("ocmero", {}) if isinstance(cfg.get("ocmero", {}), dict) else {}
    res = oc_mero(
        m, probs, c,
        alpha=float(ocfg.get("alpha", 0.2)),
        beta=float(ocfg.get("beta", 0.2)),
        option_valid=option_valid,
        root_valid=root_valid,
        use_lcvar=bool(ocfg.get("use_lcvar", True)),
        use_obs_kernel=bool(ocfg.get("use_obs_kernel", True)),
        top_m=int(ocfg.get("top_m", 8)),
    )
    option = best_shared_option_index(
        res.q, probs, gamma=0.0, root_valid=root_valid, option_valid=option_valid
    )
    drs = deployable_recovery_success(m, probs, option, root_valid=root_valid)
    r_dep = float(np.asarray(d.get("r_dep_star", res.r_dep)).reshape(()))
    r_orc = float(np.asarray(d.get("r_orc_star", res.r_orc)).reshape(()))
    return float(post_contact_deployability_score(drs, r_dep, max(0.0, r_orc - r_dep)))


def _make_group_batch_sampler(ds: OCRAPSampleDataset, cfg: dict, batch_size: int) -> SceneTimeBatchSampler | None:
    tcfg = cfg.get("training", {}) if isinstance(cfg.get("training", {}), dict) else {}
    if not bool(tcfg.get("group_batching", False)):
        return None
    total = len(ds.paths)
    roots = [_dataset_root_name(p) for p in ds.paths]
    root_counts = Counter(roots)
    group_index_path = str(tcfg.get("group_index_path", "") or "").strip()
    group_index: dict[str, dict[str, object]] = {}
    if group_index_path:
        index_file = Path(group_index_path)
        if not index_file.exists():
            raise FileNotFoundError(f"training.group_index_path does not exist: {index_file}")
        with index_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                raw_path = str(row.get("path", ""))
                if raw_path:
                    group_index[os.path.abspath(raw_path)] = row
        print({"event": "group_index_loaded", "path": str(index_file), "rows": len(group_index)}, flush=True)
    groups_by_key: dict[tuple[int, int, int], list[int]] = {}
    sample_weights: list[float] = []
    num_artifacts = num_negative = num_safe_pos = 0
    for i, p in enumerate(ds.paths):
        try:
            idx_row = group_index.get(os.path.abspath(os.fspath(p)))
            if idx_row is not None:
                scene = str(idx_row.get("scene", ""))
                t = int(idx_row.get("time", 0))
                bid_for_key = int(idx_row.get("bucket", bucket_id_for_path(p)))
            else:
                scene = scalar_metadata_for_path(p, "scene_id", "")
                t = int(np.asarray(scalar_metadata_for_path(p, "time_index", 0)).item())
                bid_for_key = bucket_id_for_path(p)
            key = (bid_for_key, stable_scene_hash(scene), t)
        except Exception:
            key = (3, i, 0)
        groups_by_key.setdefault(key, []).append(i)
        w, is_art, is_neg, is_safe_pos = _sampler_weight_for_path(p, cfg, root_counts, total)
        sample_weights.append(w)
        num_artifacts += int(is_art)
        num_negative += int(is_neg)
        num_safe_pos += int(is_safe_pos)
    group_keys = list(groups_by_key.keys())
    groups = list(groups_by_key.values())
    scene_group_counts = Counter((k[0], k[1]) for k in group_keys)
    scene_balance_power = float(tcfg.get("group_batch_scene_balance_power", 0.0))
    mean_groups_per_scene = float(np.mean(list(scene_group_counts.values()))) if scene_group_counts else 1.0
    hard_macro_ids = set(_parse_int_tuple(tcfg.get("group_batch_hard_macro_ids", ""), ()))
    hard_bucket_ids = set(_parse_int_tuple(tcfg.get("group_batch_hard_bucket_ids", ""), ()))
    hard_r_dep_min = float(tcfg.get("group_batch_hard_min_r_dep", 0.35))
    hard_boost = float(tcfg.get("group_batch_hard_boost", 0.0))

    # v43: oversample groups that contain a recovery candidate whose teacher
    # deployable value improves over the group's nominal candidate. v42 boosted
    # high absolute r_dep, which included many groups where nominal was equally
    # good or better and therefore diluted the rare positive-advantage signal.
    # v48.2: the public training script has always used the explicit
    # ``positive_advantage_*`` names.  Accept the older v43 aliases as a
    # backwards-compatible fallback, but make the canonical names win.  The
    # previous mismatch silently disabled exact teacher-PCD positive-group
    # oversampling even when the command line requested it.
    positive_macro_ids = set(_parse_int_tuple(
        tcfg.get("group_batch_positive_advantage_macro_ids", tcfg.get("group_batch_positive_macro_ids", "")), ()
    ))
    positive_bucket_ids = set(_parse_int_tuple(
        tcfg.get("group_batch_positive_advantage_bucket_ids", tcfg.get("group_batch_positive_bucket_ids", "")), ()
    ))
    positive_gain_min = float(tcfg.get(
        "group_batch_positive_advantage_gain_min", tcfg.get("group_batch_positive_r_dep_gain", 0.025)
    ))
    positive_boost = float(tcfg.get("group_batch_positive_advantage_boost", 0.0))
    positive_best_macro_balance_power = float(tcfg.get("group_batch_positive_best_macro_balance_power", 0.0))
    require_positive_groups = bool(tcfg.get("group_batch_require_positive_advantage_groups", False))
    if positive_boost > 0.0 and not positive_macro_ids:
        raise ValueError(
            "positive group boost is enabled but no recovery macro ids were configured; "
            "set training.group_batch_positive_advantage_macro_ids"
        )

    group_weights = []
    positive_best_macros: list[int | None] = []
    hard_groups = 0
    positive_advantage_groups = 0
    for g in groups:
        gw = float(max(sample_weights[i] for i in g))
        if hard_boost > 0.0 and hard_macro_ids:
            is_hard = False
            for i in g:
                p = ds.paths[i]
                try:
                    mac = int(float(np.asarray(scalar_metadata_for_path(p, "prefix_macro_type_id", scalar_metadata_for_path(p, "prefix_macro_id", -1))).item()))
                    bid = bucket_id_for_path(p)
                    rdep = float(np.asarray(scalar_metadata_for_path(p, "r_dep_star", -99.0)).item())
                    if mac in hard_macro_ids and (not hard_bucket_ids or bid in hard_bucket_ids) and rdep >= hard_r_dep_min:
                        is_hard = True
                        break
                except Exception:
                    continue
            if is_hard:
                gw *= hard_boost
                hard_groups += 1

        positive_best_macro: int | None = None
        if (positive_boost > 0.0 or positive_best_macro_balance_power > 0.0) and positive_macro_ids:
            nominal_target = None
            best_recovery_target = None
            used_exact_pcd = bool(group_index)
            for i in g:
                p = ds.paths[i]
                try:
                    idx_row = group_index.get(os.path.abspath(os.fspath(p)))
                    bid = int(idx_row.get("bucket", bucket_id_for_path(p))) if idx_row is not None else bucket_id_for_path(p)
                    if positive_bucket_ids and bid not in positive_bucket_ids:
                        continue
                    if idx_row is not None:
                        is_nominal = bool(idx_row.get("nominal", False))
                        target_value = float(idx_row.get("teacher_pcd", -99.0))
                        mac = int(idx_row.get("macro", -1))
                    else:
                        used_exact_pcd = False
                        is_nominal = bool(float(np.asarray(scalar_metadata_for_path(p, "is_nominal", 0.0)).item()) > 0.5)
                        target_value = float(np.asarray(scalar_metadata_for_path(p, "r_dep_star", -99.0)).item())
                        mac = int(float(np.asarray(scalar_metadata_for_path(p, "prefix_macro_type_id", scalar_metadata_for_path(p, "prefix_macro_id", -1))).item()))
                    if is_nominal:
                        nominal_target = target_value if nominal_target is None else max(nominal_target, target_value)
                        continue
                    if mac in positive_macro_ids:
                        if best_recovery_target is None or target_value > best_recovery_target:
                            best_recovery_target = target_value
                            positive_best_macro = mac
                except Exception:
                    continue
            if nominal_target is not None and best_recovery_target is not None and (best_recovery_target - nominal_target) >= positive_gain_min:
                if positive_boost > 0.0:
                    gw *= positive_boost
                positive_advantage_groups += 1
            else:
                positive_best_macro = None
        if scene_balance_power > 0.0:
            group_key = group_keys[len(group_weights)]
            scene_count = max(1, int(scene_group_counts[(group_key[0], group_key[1])]))
            gw *= float((mean_groups_per_scene / scene_count) ** scene_balance_power)
        group_weights.append(gw)
        positive_best_macros.append(positive_best_macro)

    # v48.6: exact positive opportunities were concentrated in one recovery
    # macro, and calibrated near-miss rules selected that macro in >90% of
    # cases.  Reweight only positive groups by inverse teacher-best-macro
    # frequency.  This uses the training teacher index, not validation/test
    # statistics, and is substantially less noisy than minibatch GroupDRO.
    positive_macro_counts = Counter(m for m in positive_best_macros if m is not None)
    if positive_best_macro_balance_power > 0.0 and positive_macro_counts:
        mean_positive_per_macro = sum(positive_macro_counts.values()) / max(1, len(positive_macro_counts))
        for i, macro in enumerate(positive_best_macros):
            if macro is None:
                continue
            factor = (mean_positive_per_macro / max(1, positive_macro_counts[macro])) ** positive_best_macro_balance_power
            group_weights[i] *= float(np.clip(factor, 0.5, 3.0))
    if require_positive_groups and positive_boost > 0.0 and positive_advantage_groups == 0:
        raise RuntimeError(
            "exact teacher-PCD sampler found zero positive-advantage groups; "
            "check GROUP_INDEX path matching and positive-advantage configuration"
        )
    print({
        "event": "group_batch_sampler_stats",
        "num_groups": int(len(groups)),
        "num_samples": int(total),
        "mean_group_size": float(np.mean([len(g) for g in groups])) if groups else 0.0,
        "max_group_size": int(max([len(g) for g in groups], default=0)),
        "replacement": bool(tcfg.get("group_batching_replacement", True)),
        "num_artifacts": int(num_artifacts),
        "num_negative_deployable": int(num_negative),
        "num_safe_positive": int(num_safe_pos),
        "hard_group_boost": float(hard_boost),
        "hard_groups": int(hard_groups),
        "positive_advantage_boost": float(positive_boost),
        "positive_advantage_gain_min": float(positive_gain_min),
        "positive_advantage_groups": int(positive_advantage_groups),
        "positive_advantage_target": "teacher_pcd" if group_index else "r_dep_fallback",
        "positive_best_macro_balance_power": float(positive_best_macro_balance_power),
        "positive_best_macro_counts": dict(sorted(positive_macro_counts.items())),
        "group_index_rows": int(len(group_index)),
        "scene_balance_power": float(scene_balance_power),
    }, flush=True)
    return SceneTimeBatchSampler(groups, batch_size, group_weights=group_weights, replacement=bool(tcfg.get("group_batching_replacement", True)))

def _make_sampler(ds: OCRAPSampleDataset, cfg: dict) -> WeightedRandomSampler | None:
    tcfg = cfg.get("training", {}) if isinstance(cfg.get("training", {}), dict) else {}
    weight_art = float(tcfg.get("artifact_sampler_weight", 0.25))
    weight_neg = float(tcfg.get("negative_deployable_sampler_weight", 0.75))
    weight_safe_pos = float(tcfg.get("safe_positive_sampler_weight", 0.25))
    regime_balance_power = float(tcfg.get("regime_balance_power", 0.0))
    if max(weight_art, weight_neg, weight_safe_pos, regime_balance_power) <= 0:
        return None
    weights = []
    num_artifacts = 0
    num_negative = 0
    num_safe_pos = 0
    total = len(ds.paths)
    roots = [_dataset_root_name(p) for p in ds.paths]
    root_counts = Counter(roots)
    print({
        "event": "sampler_weight_config",
        "artifact_sampler_weight": weight_art,
        "negative_deployable_sampler_weight": weight_neg,
        "safe_positive_sampler_weight": weight_safe_pos,
        "regime_balance_power": regime_balance_power,
        "root_counts": dict(sorted(root_counts.items())),
    }, flush=True)
    for idx, p in enumerate(ds.paths, 1):
        if idx == 1 or idx % 5000 == 0 or idx == total:
            print({"event": "sampler_scan_progress", "seen": idx, "total": total}, flush=True)
        try:
            is_art = float(np.asarray(scalar_metadata_for_path(p, "i_art_star", 0)).item()) > 0.5
            r_dep = float(np.asarray(scalar_metadata_for_path(p, "r_dep_star", 0)).item())
            is_neg = r_dep < 0.0
            root_name = _dataset_root_name(p).lower()
            is_safe_pos = ("safe" in root_name) and (not is_neg) and (not is_art)
            num_artifacts += int(is_art)
            num_negative += int(is_neg)
            num_safe_pos += int(is_safe_pos)
            w = 1.0
            if regime_balance_power > 0:
                root_count = max(1, int(root_counts.get(_dataset_root_name(p), 1)))
                w *= float((total / max(len(root_counts), 1) / root_count) ** regime_balance_power)
            if is_art:
                w += weight_art
            if is_neg:
                w += weight_neg
            if is_safe_pos:
                w += weight_safe_pos
            weights.append(w)
        except Exception:
            weights.append(1.0)
    print({
        "event": "sampler_scan_stats",
        "num_artifacts": int(num_artifacts),
        "artifact_fraction": float(num_artifacts / max(total, 1)),
        "num_negative_deployable": int(num_negative),
        "negative_deployable_fraction": float(num_negative / max(total, 1)),
        "num_safe_positive": int(num_safe_pos),
        "safe_positive_fraction": float(num_safe_pos / max(total, 1)),
    }, flush=True)
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), num_samples=len(weights), replacement=True)


def train(dataset: str, output: str, cfg: dict, val_dataset: str | None = None) -> dict:
    seed_everything(int(cfg.get("seed", 7)))
    out = ensure_dir(output)
    print({"event": "dataset_scan_start", "dataset": str(dataset)}, flush=True)
    paths = iter_sample_paths_many(dataset)
    print({"event": "dataset_scan_done", "num_npz_paths": len(paths)}, flush=True)
    if not paths:
        raise ValueError(f"No OC-RAP sample .npz files found under {dataset}")
    tcfg_for_split = cfg.get("training", {}) if isinstance(cfg.get("training", {}), dict) else {}
    explicit_val_dataset = val_dataset or tcfg_for_split.get("val_dataset") or tcfg_for_split.get("validation_dataset")
    print({"event": "split_scan_start", "splits": ["train", "val"], "explicit_val_dataset": bool(explicit_val_dataset)}, flush=True)
    train_paths = split_paths_by_npz_split(paths, "train")
    if explicit_val_dataset:
        val_all_paths = iter_sample_paths_many(str(explicit_val_dataset))
        val_paths = split_paths_by_npz_split(val_all_paths, {"val", "calibration"})
        if not val_paths:
            val_paths = val_all_paths
    else:
        val_paths = split_paths_by_npz_split(paths, "val")
    print({"event": "split_scan_done", "num_train_paths": len(train_paths), "num_val_paths": len(val_paths)}, flush=True)
    profile_max = int(tcfg_for_split.get("dataset_profile_max_scalar_scan", 0))
    if bool(tcfg_for_split.get("dataset_profile", True)):
        print(_profile_paths(train_paths, stage="train", max_scalar_scan=profile_max), flush=True)
        print(_profile_paths(val_paths, stage="val", max_scalar_scan=profile_max), flush=True)
    if not train_paths:
        train_paths = paths
    if not val_paths:
        print({"event": "validation_fallback_warning", "reason": "no explicit validation split; using first 10pct of training paths"}, flush=True)
        val_paths = train_paths[: max(1, min(len(train_paths), len(train_paths) // 10 or 1))]

    train_ds = OCRAPSampleDataset(train_paths, cfg)
    val_ds = OCRAPSampleDataset(val_paths, cfg)
    first = train_ds[0]
    num_roots = int(train_ds.num_roots)
    num_options = int(train_ds.num_options)
    d_signature = int(train_ds.d_signature)
    d_future_signature = int(train_ds.d_future_signature)
    device = _device(cfg)
    device_info = _device_summary(device)
    model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model", {}), dict) else {}
    d_model = int(model_cfg.get("d_model", 128))
    d_obs = int(model_cfg.get("d_obs", 64))
    tau_obs = float(model_cfg.get("tau_obs", (cfg.get("ocmero", {}) or {}).get("tau_obs", 1.0)))
    encoder_type = str(model_cfg.get("encoder_type", "mlp"))
    feature_layout = {
        "prefix_param_dim": int(cfg.get("prefix_param_dim", 5)),
        "num_macros": int(model_cfg.get("num_macros", 16)),
        "prefix_flat_dim": int(model_cfg.get("feature_prefix_flat_dim", 80)),
        "control_flat_dim": int(model_cfg.get("feature_control_flat_dim", 40)),
        "feature_max_agents": int(model_cfg.get("feature_max_agents", 32)),
        "bev_channels": int(cfg.get("bev_channels", 7)),
        "route_flat_dim": int(model_cfg.get("feature_route_flat_dim", 64)),
        "map_flat_dim": int(model_cfg.get("feature_map_flat_dim", 64)),
        "dyn_flat_dim": int(model_cfg.get("feature_dynamic_map_flat_dim", 32)),
    }
    model = OCRAPModel(
        train_ds.feature_dim,
        num_roots=num_roots,
        num_options=num_options,
        d_model=d_model,
        d_obs=d_obs,
        tau_obs=tau_obs,
        encoder_type=encoder_type,
        feature_layout=feature_layout,
        num_layers=int(model_cfg.get("transformer_layers", 2)),
        num_heads=int(model_cfg.get("transformer_heads", 4)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        d_signature=d_signature,
        d_future_signature=d_future_signature,
        option_feature_dim=OPTION_FEATURE_DIM,
        direct_recovery_value_head=bool(model_cfg.get("direct_recovery_value_head", False)),
        direct_recovery_value_pooling=str(model_cfg.get("direct_recovery_value_pooling", "scene")),
        direct_recovery_value_output=str(model_cfg.get("direct_recovery_value_output", "probability")),
        direct_recovery_value_regime_conditioning=bool(model_cfg.get("direct_recovery_value_regime_conditioning", False)),
        direct_recovery_value_num_regimes=int(model_cfg.get("direct_recovery_value_num_regimes", 4)),
        direct_recovery_value_regime_dim=int(model_cfg.get("direct_recovery_value_regime_dim", 16)),
        direct_recovery_opportunity_head=bool(model_cfg.get("direct_recovery_opportunity_head", False)),
        direct_recovery_harm_head=bool(model_cfg.get("direct_recovery_harm_head", False)),
        direct_recovery_value_experts=bool(model_cfg.get("direct_recovery_value_experts", False)),
        direct_recovery_value_num_experts=int(model_cfg.get("direct_recovery_value_num_experts", 2)),
        direct_recovery_value_expert_routing=str(model_cfg.get("direct_recovery_value_expert_routing", "bucket")),
        direct_recovery_value_router_temperature=float(model_cfg.get("direct_recovery_value_router_temperature", 1.0)),
        direct_recovery_value_router_pooling=str(model_cfg.get("direct_recovery_value_router_pooling", "candidate")),
        direct_recovery_expert_disagreement_penalty=float(model_cfg.get("direct_recovery_expert_disagreement_penalty", 0.5)),
        direct_recovery_set_context=bool(model_cfg.get("direct_recovery_set_context", False)),
        direct_recovery_set_context_hidden=int(model_cfg.get("direct_recovery_set_context_hidden", d_model)),
        direct_recovery_set_context_dropout=float(model_cfg.get("direct_recovery_set_context_dropout", model_cfg.get("dropout", 0.1))),
        direct_recovery_preference_head=bool(model_cfg.get("direct_recovery_preference_head", False)),
        direct_recovery_preference_hidden=int(model_cfg.get("direct_recovery_preference_hidden", max(16, d_model // 2))),
        direct_recovery_preference_dropout=float(model_cfg.get("direct_recovery_preference_dropout", 0.05)),
        direct_recovery_preference_context=bool(model_cfg.get("direct_recovery_preference_context", False)),
        direct_recovery_preference_context_hidden=int(model_cfg.get("direct_recovery_preference_context_hidden", d_model)),
        direct_recovery_relative_features_include_absolute=bool(model_cfg.get("direct_recovery_relative_features_include_absolute", True)),
        direct_recovery_set_tournament=bool(model_cfg.get("direct_recovery_set_tournament", False)),
        direct_recovery_set_tournament_hidden=int(model_cfg.get("direct_recovery_set_tournament_hidden", 48)),
        direct_recovery_set_tournament_heads=int(model_cfg.get("direct_recovery_set_tournament_heads", 4)),
        direct_recovery_set_tournament_dropout=float(model_cfg.get("direct_recovery_set_tournament_dropout", 0.05)),
        direct_recovery_set_tournament_replace_base=bool(model_cfg.get("direct_recovery_set_tournament_replace_base", True)),
        direct_recovery_delta_head=bool(model_cfg.get("direct_recovery_delta_head", False)),
        direct_recovery_delta_regime_experts=bool(model_cfg.get("direct_recovery_delta_regime_experts", False)),
        direct_recovery_delta_policy_features=bool(model_cfg.get("direct_recovery_delta_policy_features", False)),
        direct_recovery_delta_hidden=int(model_cfg.get("direct_recovery_delta_hidden", d_model)),
        direct_recovery_delta_dropout=float(model_cfg.get("direct_recovery_delta_dropout", 0.05)),
        direct_recovery_delta_initial_logvar=float(model_cfg.get("direct_recovery_delta_initial_logvar", -4.605170186)),
        direct_recovery_delta_mode=str(model_cfg.get("direct_recovery_delta_mode", "gaussian")),
    ).to(device)
    tcfg = cfg.get("training", {}) if isinstance(cfg.get("training", {}), dict) else {}

    # v28: residual DDC should fine-tune an already calibrated selector model
    # rather than relearning all heads from scratch.  The v27 trained run showed
    # that scratch training can minimize train loss while destroying the runtime
    # calibration scale used by the selector.
    init_checkpoint = str(tcfg.get("init_checkpoint", "") or "").strip()
    init_load_info: dict[str, object] = {}
    if init_checkpoint:
        init_path = Path(init_checkpoint)
        if not init_path.exists():
            raise FileNotFoundError(f"training.init_checkpoint does not exist: {init_checkpoint}")
        ckpt = torch.load(init_path, map_location=device)
        state = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt
        if not isinstance(state, dict):
            raise TypeError(f"Unsupported checkpoint state type: {type(state)!r}")
        current = model.state_dict()
        compatible = {}
        shape_mismatch = {}
        for key, value in state.items():
            if key in current and hasattr(value, "shape") and tuple(value.shape) == tuple(current[key].shape):
                compatible[key] = value
            elif key in current:
                shape_mismatch[key] = {
                    "checkpoint": list(getattr(value, "shape", ())),
                    "model": list(current[key].shape),
                }
        missing, unexpected = model.load_state_dict(compatible, strict=False)
        unexpected_in_source = sorted(set(state) - set(current))
        init_load_info = {
            "event": "init_checkpoint_loaded",
            "path": str(init_path),
            "loaded_keys": len(compatible),
            "missing_keys": list(missing),
            "unexpected_keys": sorted(set(unexpected) | set(unexpected_in_source)),
            "shape_mismatch_keys": shape_mismatch,
        }
        print(init_load_info, flush=True)

        # v48.10 COPE: staged training must not silently discard the learned
        # preference adapter because Stage C was instantiated with a different
        # hidden width.  Prefixes listed here are an architecture contract: all
        # destination tensors under each prefix must load with identical shapes.
        strict_prefixes = tuple(
            x.strip() for x in str(tcfg.get("strict_init_prefixes", "") or "").split(",") if x.strip()
        )
        if strict_prefixes:
            failures: dict[str, object] = {}
            for prefix in strict_prefixes:
                required = sorted(key for key in current if key.startswith(prefix))
                loaded = sorted(key for key in compatible if key.startswith(prefix))
                mismatched = {key: value for key, value in shape_mismatch.items() if key.startswith(prefix)}
                source_missing = sorted(set(required) - set(loaded) - set(mismatched))
                if not required or len(loaded) != len(required) or mismatched or source_missing:
                    failures[prefix] = {
                        "required_keys": required,
                        "loaded_keys": loaded,
                        "shape_mismatch": mismatched,
                        "missing_from_source": source_missing,
                    }
            if failures:
                raise RuntimeError(
                    "strict staged-checkpoint architecture contract failed: "
                    + json.dumps(failures, sort_keys=True)
                )
            print({
                "event": "strict_init_prefixes_verified",
                "prefixes": list(strict_prefixes),
            }, flush=True)

    encoder_anchor_weight = float(tcfg.get("encoder_anchor_weight", 0.0))
    if encoder_anchor_weight > 0.0:
        if not init_checkpoint:
            raise ValueError("training.encoder_anchor_weight requires training.init_checkpoint")
        model._encoder_anchor_tensors = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if name.startswith("encoder.")
        }
        print({
            "event": "encoder_anchor_enabled",
            "weight": encoder_anchor_weight,
            "num_tensors": len(model._encoder_anchor_tensors),
            "num_params": int(sum(x.numel() for x in model._encoder_anchor_tensors.values())),
        }, flush=True)

    freeze_prefixes = tuple(
        x.strip() for x in str(tcfg.get("freeze_param_prefixes", "") or "").split(",") if x.strip()
    )
    trainable_prefixes = tuple(
        x.strip() for x in str(tcfg.get("trainable_param_prefixes", "") or "").split(",") if x.strip()
    )
    if freeze_prefixes and trainable_prefixes:
        raise ValueError("Configure either training.freeze_param_prefixes or training.trainable_param_prefixes, not both")
    if trainable_prefixes:
        frozen = 0
        trainable = 0
        for name, param in model.named_parameters():
            keep = any(name.startswith(prefix) for prefix in trainable_prefixes)
            param.requires_grad_(keep)
            if keep:
                trainable += int(param.numel())
            else:
                frozen += int(param.numel())
        print({
            "event": "trainable_param_prefixes",
            "prefixes": list(trainable_prefixes),
            "frozen_params": int(frozen),
            "trainable_params": int(trainable),
        }, flush=True)
    elif freeze_prefixes:
        frozen = 0
        trainable = 0
        for name, param in model.named_parameters():
            if any(name.startswith(prefix) for prefix in freeze_prefixes):
                param.requires_grad_(False)
                frozen += int(param.numel())
            else:
                trainable += int(param.numel())
        print({
            "event": "freeze_param_prefixes",
            "prefixes": list(freeze_prefixes),
            "frozen_params": int(frozen),
            "trainable_params": int(trainable),
        }, flush=True)
    named_trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if not named_trainable:
        raise ValueError("No trainable parameters remain after training.freeze_param_prefixes")
    base_lr = float(tcfg.get("lr", 1e-3))
    encoder_lr_scale = float(tcfg.get("encoder_lr_scale", 1.0))
    encoder_params = [p for name, p in named_trainable if name.startswith("encoder.")]
    other_params = [p for name, p in named_trainable if not name.startswith("encoder.")]
    param_groups = []
    if encoder_params:
        param_groups.append({"params": encoder_params, "lr": base_lr * encoder_lr_scale, "group_name": "encoder"})
    if other_params:
        param_groups.append({"params": other_params, "lr": base_lr, "group_name": "policy_heads"})
    opt = torch.optim.AdamW(param_groups, lr=base_lr, weight_decay=float(tcfg.get("weight_decay", 1e-4)))
    print({
        "event": "optimizer_parameter_groups",
        "base_lr": base_lr,
        "encoder_lr_scale": encoder_lr_scale,
        "encoder_params": int(sum(p.numel() for p in encoder_params)),
        "other_params": int(sum(p.numel() for p in other_params)),
    }, flush=True)
    batch_size = int(tcfg.get("batch_size", 32))
    num_workers = int(tcfg.get("num_workers", 0))
    if device.type == "cuda":
        torch.set_float32_matmul_precision(str(tcfg.get("matmul_precision", "high")))
        torch.backends.cuda.matmul.allow_tf32 = bool(tcfg.get("allow_tf32", True))
        torch.backends.cudnn.allow_tf32 = bool(tcfg.get("allow_tf32", True))
        torch.backends.cudnn.benchmark = bool(tcfg.get("cudnn_benchmark", True))
    loader_kwargs = {
        "num_workers": num_workers,
        "collate_fn": _collate,
        "pin_memory": bool(device.type == "cuda" and tcfg.get("pin_memory", True)),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(tcfg.get("persistent_workers", True))
        loader_kwargs["prefetch_factor"] = max(1, int(tcfg.get("prefetch_factor", 3)))
    group_batch_sampler = _make_group_batch_sampler(train_ds, cfg, batch_size)
    if group_batch_sampler is not None:
        print({"event": "sampler_scan_done", "sampler": "scene_time_group_batch"}, flush=True)
        train_loader = DataLoader(train_ds, batch_sampler=group_batch_sampler, **loader_kwargs)
    else:
        print({"event": "sampler_scan_start", "artifact_sampler_weight": float(tcfg.get("artifact_sampler_weight", 0.25))}, flush=True)
        sampler = _make_sampler(train_ds, cfg)
        print({"event": "sampler_scan_done", "sampler": "weighted" if sampler is not None else "none"}, flush=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=sampler is None, sampler=sampler, **loader_kwargs)
    if bool(tcfg.get("group_batching", False)):
        val_cfg = dict(cfg)
        val_tcfg = dict(tcfg)
        val_tcfg["group_batching_replacement"] = False
        val_tcfg["group_batch_hard_boost"] = 0.0
        val_cfg["training"] = val_tcfg
        val_group_sampler = _make_group_batch_sampler(val_ds, val_cfg, batch_size)
        if val_group_sampler is not None:
            val_group_sampler.shuffle_within_group = False
            val_group_sampler.shuffle_groups = False
            val_loader = DataLoader(val_ds, batch_sampler=val_group_sampler, **loader_kwargs)
        else:
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)
    else:
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)
    epochs = int(tcfg.get("epochs", 10))
    best_val = float("inf")
    best_metric_min_delta = max(0.0, float(tcfg.get("best_metric_min_delta", 1.0e-6)))
    best_epoch = 0
    no_improve_epochs = 0
    history = []
    best_path = out / "best.pt"
    latest_path = out / "latest.pt"
    ckpt_dir = ensure_dir(out / "checkpoints")

    def _checkpoint_payload(ep: int, val_loss: float) -> dict:
        return {
            "cfg": cfg,
            "input_dim": train_ds.feature_dim,
            "num_roots": num_roots,
            "num_options": num_options,
            "d_model": d_model,
            "d_obs": d_obs,
            "tau_obs": tau_obs,
            "encoder_type": encoder_type,
            "feature_layout": feature_layout,
            "d_signature": d_signature,
            "d_future_signature": d_future_signature,
            "option_feature_dim": OPTION_FEATURE_DIM,
            "direct_recovery_value_head": bool(model_cfg.get("direct_recovery_value_head", False)),
            "direct_recovery_value_pooling": str(model_cfg.get("direct_recovery_value_pooling", "scene")),
            "direct_recovery_value_output": str(model_cfg.get("direct_recovery_value_output", "probability")),
            "direct_recovery_value_regime_conditioning": bool(model_cfg.get("direct_recovery_value_regime_conditioning", False)),
            "direct_recovery_value_num_regimes": int(model_cfg.get("direct_recovery_value_num_regimes", 4)),
            "direct_recovery_value_regime_dim": int(model_cfg.get("direct_recovery_value_regime_dim", 16)),
            "direct_recovery_opportunity_head": bool(model_cfg.get("direct_recovery_opportunity_head", False)),
            "direct_recovery_harm_head": bool(model_cfg.get("direct_recovery_harm_head", False)),
            "direct_recovery_value_experts": bool(model_cfg.get("direct_recovery_value_experts", False)),
            "direct_recovery_value_num_experts": int(model_cfg.get("direct_recovery_value_num_experts", 2)),
            "direct_recovery_value_expert_routing": str(model_cfg.get("direct_recovery_value_expert_routing", "bucket")),
            "direct_recovery_value_router_temperature": float(model_cfg.get("direct_recovery_value_router_temperature", 1.0)),
            "direct_recovery_value_router_pooling": str(model_cfg.get("direct_recovery_value_router_pooling", "candidate")),
            "direct_recovery_expert_disagreement_penalty": float(model_cfg.get("direct_recovery_expert_disagreement_penalty", 0.5)),
            "direct_recovery_set_context": bool(model_cfg.get("direct_recovery_set_context", False)),
            "direct_recovery_set_context_hidden": int(model_cfg.get("direct_recovery_set_context_hidden", d_model)),
            "direct_recovery_set_context_dropout": float(model_cfg.get("direct_recovery_set_context_dropout", model_cfg.get("dropout", 0.1))),
            "direct_recovery_preference_head": bool(model_cfg.get("direct_recovery_preference_head", False)),
            "direct_recovery_preference_hidden": int(model_cfg.get("direct_recovery_preference_hidden", max(16, d_model // 2))),
            "direct_recovery_preference_dropout": float(model_cfg.get("direct_recovery_preference_dropout", 0.05)),
            "direct_recovery_preference_context": bool(model_cfg.get("direct_recovery_preference_context", False)),
            "direct_recovery_preference_context_hidden": int(model_cfg.get("direct_recovery_preference_context_hidden", d_model)),
            "direct_recovery_relative_features_include_absolute": bool(model_cfg.get("direct_recovery_relative_features_include_absolute", True)),
            "direct_recovery_set_tournament": bool(model_cfg.get("direct_recovery_set_tournament", False)),
            "direct_recovery_set_tournament_hidden": int(model_cfg.get("direct_recovery_set_tournament_hidden", 48)),
            "direct_recovery_set_tournament_heads": int(model_cfg.get("direct_recovery_set_tournament_heads", 4)),
            "direct_recovery_set_tournament_dropout": float(model_cfg.get("direct_recovery_set_tournament_dropout", 0.05)),
            "direct_recovery_set_tournament_replace_base": bool(model_cfg.get("direct_recovery_set_tournament_replace_base", True)),
            "direct_recovery_delta_head": bool(model_cfg.get("direct_recovery_delta_head", False)),
            "direct_recovery_delta_regime_experts": bool(model_cfg.get("direct_recovery_delta_regime_experts", False)),
            "direct_recovery_delta_policy_features": bool(model_cfg.get("direct_recovery_delta_policy_features", False)),
            "direct_recovery_delta_hidden": int(model_cfg.get("direct_recovery_delta_hidden", d_model)),
            "direct_recovery_delta_dropout": float(model_cfg.get("direct_recovery_delta_dropout", 0.05)),
            "direct_recovery_delta_initial_logvar": float(model_cfg.get("direct_recovery_delta_initial_logvar", -4.605170186)),
            "direct_recovery_delta_mode": str(model_cfg.get("direct_recovery_delta_mode", "gaussian")),
            "model_state": model.state_dict(),
            "optimizer_state": opt.state_dict(),
            "epoch": int(ep),
            "val_loss": float(val_loss),
            "device_info_at_train": device_info,
            "note": "OC-RAP neural checkpoint: predicts root probabilities, recovery margins, utility, and observation compatibility from scene-prefix features.",
            "init_checkpoint": init_checkpoint,
            "freeze_param_prefixes": list(freeze_prefixes),
            "trainable_param_prefixes": list(trainable_prefixes),
        }
    print({
        "event": "train_start",
        **device_info,
        "num_train_samples": len(train_paths),
        "num_val_samples": len(val_paths),
        "train_batches_per_epoch": len(train_loader),
        "val_batches_per_epoch": len(val_loader),
        "epochs": epochs,
        "batch_size": batch_size,
        "d_model": d_model,
        "d_obs": d_obs,
        "encoder_type": encoder_type,
        "init_checkpoint": init_checkpoint or None,
        "freeze_param_prefixes": list(freeze_prefixes),
    }, flush=True)
    t0 = perf_counter()
    for ep in range(1, epochs + 1):
        ep_t0 = perf_counter()
        tr = _epoch(model, train_loader, cfg, device, opt, stage="train", epoch=ep)
        with torch.no_grad():
            va = _epoch(model, val_loader, cfg, device, None, stage="val", epoch=ep)
        row = {"epoch": ep, "train": tr, "val": va, "seconds": float(perf_counter() - ep_t0)}
        history.append(row)
        best_metric_name = str(tcfg.get("best_metric", "loss") or "loss")
        best_metric_mode = str(tcfg.get("best_metric_mode", "min") or "min").lower()
        if best_metric_name not in va:
            raise KeyError(
                f"Configured training.best_metric={best_metric_name!r} was not produced by validation. "
                f"Available metrics: {sorted(va)}"
            )
        current_metric = float(va[best_metric_name])
        compare_metric = current_metric if best_metric_mode != "max" else -current_metric
        improved = compare_metric < (best_val - best_metric_min_delta)
        payload = _checkpoint_payload(ep, current_metric)
        save_every = bool(tcfg.get("save_every_epoch", True))
        if save_every:
            torch.save(payload, ckpt_dir / f"epoch_{ep:04d}.pt")
        if bool(tcfg.get("save_latest", True)):
            torch.save(payload, latest_path)
        if improved:
            best_val = compare_metric
            best_epoch = ep
            no_improve_epochs = 0
            payload["val_loss"] = float(va.get("loss", current_metric))
            payload["best_metric_name"] = best_metric_name
            payload["best_metric_value"] = float(current_metric)
            payload["is_best"] = True
            torch.save(payload, best_path)
        else:
            no_improve_epochs += 1
        print({
            "event": "epoch_end",
            "epoch": ep,
            "train_loss": round(float(tr.get("loss", 0.0)), 6),
            "val_loss": round(float(va.get("loss", 0.0)), 6),
            "best_val_loss": round(float(best_val), 6),
            "best_metric_name": best_metric_name,
            "current_best_metric": round(float(current_metric), 6),
            "best_epoch": int(best_epoch),
            "improved": bool(improved),
            "seconds": round(float(row["seconds"]), 2),
        }, flush=True)
        patience = int(tcfg.get("early_stop_patience", 0) or 0)
        if patience > 0 and no_improve_epochs >= patience:
            print({
                "event": "early_stop",
                "epoch": ep,
                "best_epoch": int(best_epoch),
                "best_val_loss": float(best_val),
                "best_metric": str(tcfg.get("best_metric", "loss") or "loss"),
                "epochs_completed": int(len(history)),
                "patience": patience,
            }, flush=True)
            break
    result = {
        "checkpoint": str(best_path),
        "latest_checkpoint": str(latest_path),
        "checkpoint_dir": str(ckpt_dir),
        "num_train_samples": len(train_paths),
        "num_val_samples": len(val_paths),
        "input_dim": train_ds.feature_dim,
        "num_roots": num_roots,
        "num_options": num_options,
        "d_model": d_model,
        "d_obs": d_obs,
        "encoder_type": encoder_type,
        "init_checkpoint": init_checkpoint,
        "freeze_param_prefixes": list(freeze_prefixes),
        "trainable_param_prefixes": list(trainable_prefixes),
        "best_metric_min_delta": float(best_metric_min_delta),
        "best_val_loss": float(best_val),
        "best_metric": str(tcfg.get("best_metric", "loss") or "loss"),
        "device_info": device_info,
        "train_batches_per_epoch": len(train_loader),
        "val_batches_per_epoch": len(val_loader),
        "best_epoch": int(best_epoch),
        "epochs_completed": int(len(history)),
        "total_train_steps": int(len(train_loader) * len(history)),
        "elapsed_seconds": float(perf_counter() - t0),
        "history": history,
    }
    write_json(result, out / "train_summary.json")
    return result
