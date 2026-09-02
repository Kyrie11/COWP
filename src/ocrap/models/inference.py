from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ocrap.algorithms.ocmero import oc_mero, torch_oc_mero
from ocrap.models.data import (
    OPTION_FEATURE_DIM,
    DIRECT_ABSOLUTE_PHYSICAL_HEADROOM_FEATURE_SCHEMA,
    DIRECT_EXECUTABLE_RECOVERY_WITNESS_FEATURE_SCHEMA,
    DIRECT_COMMON_RECOVERY_WITNESS_FEATURE_SCHEMA,
    DIRECT_SEMANTIC_RECOVERY_WITNESS_FEATURE_SCHEMA,
    DIRECT_ACTIVE_CONSTRAINT_RECOVERY_WITNESS_FEATURE_SCHEMA,
    DIRECT_PROJECTED_BOUNDARY_RECOVERY_WITNESS_FEATURE_SCHEMA,
    DIRECT_ROBUST_TRUST_RECOVERY_WITNESS_FEATURE_SCHEMA,
    DIRECT_DEMAND_TEMPERED_RECOVERY_WITNESS_FEATURE_SCHEMA,
    DIRECT_OCCUPANCY_TEMPERED_RECOVERY_WITNESS_FEATURE_SCHEMA,
    DIRECT_BOUNDARY_OCCUPANCY_REACHABILITY_WITNESS_FEATURE_SCHEMA,
    DIRECT_INTERACTION_ORIENTED_RECOVERY_WITNESS_FEATURE_SCHEMA,
    DIRECT_INTERACTION_RESPONSE_RECOVERY_WITNESS_FEATURE_SCHEMA,
    DIRECT_SIGNED_VIABILITY_RECOVERY_WITNESS_FEATURE_SCHEMA,
    direct_absolute_physical_headroom_features_from_sample,
    direct_executable_recovery_witness_features_from_sample,
    direct_common_recovery_witness_features_from_sample,
    direct_semantic_recovery_witness_features_from_sample,
    fix_sample_geometry,
    sample_to_feature,
    samples_to_feature_matrix,
)
from ocrap.models.ocrap import OCRAPModel
from ocrap.v48_74_signed_viability import (
    V48_74_SOURCE as _V48_74_SOURCE,
    enabled as _v48_74_signed_viability_enabled,
)


@dataclass
class Prediction:
    r_dep: float
    r_orc: float
    gap: float
    q: np.ndarray
    root_probs: np.ndarray
    c_star: np.ndarray
    margins: np.ndarray
    direct_recovery_value: float | None = None
    direct_recovery_std: float | None = None
    direct_recovery_opportunity: float | None = None
    direct_recovery_opportunity_logit: float | None = None
    direct_recovery_harm: float | None = None
    direct_recovery_harm_logit: float | None = None
    direct_recovery_absolute_feasibility: float | None = None
    direct_recovery_absolute_feasibility_logit: float | None = None
    # v48.63 diagnostic-only quantifier witness summaries.
    direct_recovery_quantifier_best_common_viability: float | None = None
    direct_recovery_quantifier_universal_failure: float | None = None
    direct_recovery_quantifier_positive_option_count: float | None = None
    direct_recovery_quantifier_max_common_support: float | None = None
    # v48.64 diagnostic-only semantics-aligned witness summaries.
    direct_recovery_semantic_best_common_viability: float | None = None
    direct_recovery_semantic_universal_failure: float | None = None
    direct_recovery_semantic_positive_option_count: float | None = None
    direct_recovery_semantic_max_common_support: float | None = None
    direct_recovery_semantic_best_barriers: np.ndarray | None = None
    direct_recovery_semantic_limiting_constraint: int | None = None
    # v48.65 diagnostic-only observation-class-local certificate summaries.
    direct_recovery_semantic_classlocal_lcvar_viability: float | None = None
    direct_recovery_semantic_classlocal_viable_root_mass: float | None = None
    direct_recovery_semantic_classlocal_selected_support_mean: float | None = None
    direct_recovery_rank: float | None = None
    direct_recovery_delta: float | None = None
    direct_recovery_delta_std: float | None = None
    direct_recovery_component_harm: np.ndarray | None = None
    direct_recovery_component_margins: np.ndarray | None = None
    # v48.50 diagnostic-only native OC-MERO coordinates: [hard DRS, dep, smooth boundary DRS, gap quality].
    direct_recovery_native_certificate: np.ndarray | None = None
    # v48.57 diagnostic-only recovery integration measure.  It is identical to
    # ``root_probs`` unless CMRI is active on a complete scene-time group.
    recovery_root_probs: np.ndarray | None = None


@dataclass
class ModelBundle:
    model: OCRAPModel
    cfg: dict[str, Any]
    device: torch.device


def _v48_74_schema10_checkpoint_contract(
    *,
    feature_schema: int,
    feature_source: str,
    interaction_anchor_support: bool,
    interaction_response_support: bool,
) -> bool:
    """Validate and identify the V48.74 schema-10 inference contract.

    Returns ``False`` for historical schema/source pairs so the caller can use
    the legacy selector-derived mapping.  If either half of the V48.74 pair is
    present, validation is fail-closed: schema, source, selector, and overlay
    mode must all agree.  This helper is intentionally callable from the
    runtime preflight so the exact post-training RC30 path is checked before
    any GPU work starts.
    """
    schema_match = int(feature_schema) == DIRECT_SIGNED_VIABILITY_RECOVERY_WITNESS_FEATURE_SCHEMA
    source_match = str(feature_source) == _V48_74_SOURCE
    if not (schema_match or source_match):
        return False
    if not (schema_match and source_match):
        raise RuntimeError(
            "incomplete V48.74 signed-viability checkpoint feature contract: "
            f"schema={feature_schema}, source={feature_source!r}; expected "
            f"schema={DIRECT_SIGNED_VIABILITY_RECOVERY_WITNESS_FEATURE_SCHEMA}, "
            f"source={_V48_74_SOURCE!r}."
        )
    if not (interaction_anchor_support or interaction_response_support):
        raise RuntimeError(
            "V48.74 signed-viability checkpoint requires the registered "
            "coordinate-20/21 selector flags."
        )
    if not _v48_74_signed_viability_enabled():
        raise RuntimeError(
            "V48.74 signed-viability checkpoint requires "
            "OCRAP_V48_74_SIGNED_VIABILITY=1 so inference feature "
            "materialization and raw-debt decoding match schema 10."
        )
    return True


def _device_from_cfg(cfg: dict) -> torch.device:
    requested = str((cfg.get("training", {}) or {}).get("device", "auto"))
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _merge_runtime_cfg_for_inference(ckpt_cfg: dict[str, Any], runtime_cfg: dict | None) -> dict[str, Any]:
    """Merge only inference-safe runtime settings into checkpoint config.

    The checkpoint owns model geometry and feature geometry.  A default runtime
    config must not silently overwrite model.d_model, feature dimensions, or
    other fields that affect the module shapes.  This was the cause of the
    256-vs-128 load_state_dict mismatch seen after training with
    ``--set model.d_model=256`` and calibrating with default config.
    """
    cfg = dict(ckpt_cfg or {})
    if not runtime_cfg:
        return cfg
    allow_sections = {"selection", "calibration", "evaluation", "ablation", "ocmero", "baselines", "metrics", "artifact"}
    for k, v in runtime_cfg.items():
        if k in allow_sections:
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                tmp = dict(cfg[k])
                tmp.update(v)
                cfg[k] = tmp
            else:
                cfg[k] = v
    # Only device-related training overrides are inference-safe.
    rt_train = runtime_cfg.get("training", {}) if isinstance(runtime_cfg.get("training", {}), dict) else {}
    if rt_train:
        train_cfg = dict(cfg.get("training", {}) or {})
        for key in ("device", "require_cuda"):
            if key in rt_train:
                train_cfg[key] = rt_train[key]
        cfg["training"] = train_cfg
    return cfg


def _infer_d_model(ckpt: dict[str, Any], cfg: dict[str, Any]) -> int:
    if "d_model" in ckpt:
        return int(ckpt["d_model"])
    state = ckpt.get("model_state", {}) or {}
    for key in ("encoder.net.0.weight", "root_logits.weight", "margin_head.weight"):
        w = state.get(key)
        if w is not None:
            shape = tuple(w.shape)
            if key == "encoder.net.0.weight" and len(shape) == 2:
                return int(shape[0])
            if len(shape) == 2:
                return int(shape[1])
    return int((cfg.get("model", {}) or {}).get("d_model", 128))


def regime_id_from_cfg(cfg: dict | None) -> int:
    """Map the runtime calibration bucket to a legacy integer id.

    Observation-conditioned soft expert routing ignores this id.  It is retained
    for old checkpoints, hard-routing ablations, and bucket-specific selector
    calibration envelopes; it is never a teacher-future label.
    """
    cfg = cfg or {}
    sel = cfg.get("selection", {}) if isinstance(cfg.get("selection", {}), dict) else {}
    raw = str(sel.get("active_bucket_name", sel.get("regime_name", "")) or "").lower().replace("-", "_")
    if "near" in raw:
        return 1
    if "contact" in raw or "post_contact" in raw:
        return 2
    if "safe" in raw or "normal" in raw or "background" in raw:
        return 0
    return 3


def _load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # older torch
        return torch.load(path, map_location="cpu")


def load_model_bundle(checkpoint: str | Path | None, runtime_cfg: dict | None = None) -> ModelBundle | None:
    if not checkpoint:
        return None
    path = Path(checkpoint)
    if not path.exists():
        return None
    ckpt = _load_checkpoint(path)
    if "model_state" not in ckpt:
        return None
    cfg = _merge_runtime_cfg_for_inference(dict(ckpt.get("cfg", {}) or {}), runtime_cfg)
    device = _device_from_cfg(cfg)
    d_model = _infer_d_model(ckpt, cfg)
    d_obs = int(ckpt.get("d_obs", (cfg.get("model", {}) or {}).get("d_obs", 64)))
    tau_obs = float(ckpt.get("tau_obs", (cfg.get("model", {}) or {}).get("tau_obs", (cfg.get("ocmero", {}) or {}).get("tau_obs", 1.0))))
    model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model", {}), dict) else {}
    cphr_enabled = bool(ckpt.get(
        "direct_recovery_absolute_physical_headroom_correction",
        model_cfg.get("direct_recovery_absolute_physical_headroom_correction", False),
    ))
    if cphr_enabled:
        feature_schema = int(ckpt.get("direct_recovery_absolute_physical_headroom_feature_schema", 0) or 0)
        if feature_schema != DIRECT_ABSOLUTE_PHYSICAL_HEADROOM_FEATURE_SCHEMA:
            raise RuntimeError(
                "legacy/unknown CPHR checkpoint feature semantics: "
                f"schema={feature_schema}; v48.60.1 requires full-prefix schema="
                f"{DIRECT_ABSOLUTE_PHYSICAL_HEADROOM_FEATURE_SCHEMA}. Rerun v48.60 CPHR training."
            )
    erwf_enabled = bool(ckpt.get(
        "direct_recovery_absolute_executable_witness_correction",
        model_cfg.get("direct_recovery_absolute_executable_witness_correction", False),
    ))
    if erwf_enabled:
        feature_schema = int(ckpt.get("direct_recovery_absolute_executable_witness_feature_schema", 0) or 0)
        if feature_schema != DIRECT_EXECUTABLE_RECOVERY_WITNESS_FEATURE_SCHEMA:
            raise RuntimeError(
                "legacy/unknown ERWF checkpoint feature semantics: "
                f"schema={feature_schema}; v48.61 requires option-resolved executable witness schema="
                f"{DIRECT_EXECUTABLE_RECOVERY_WITNESS_FEATURE_SCHEMA}. Rerun v48.61 ERWF training."
            )
    common_witness_enabled = bool(ckpt.get(
        "direct_recovery_absolute_common_witness_correction",
        model_cfg.get("direct_recovery_absolute_common_witness_correction", False),
    ))
    if common_witness_enabled:
        feature_schema = int(ckpt.get("direct_recovery_absolute_common_witness_feature_schema", 0) or 0)
        if feature_schema != DIRECT_COMMON_RECOVERY_WITNESS_FEATURE_SCHEMA:
            raise RuntimeError(
                "legacy/unknown OC-CWRF checkpoint feature semantics: "
                f"schema={feature_schema}; v48.62 requires common-witness schema="
                f"{DIRECT_COMMON_RECOVERY_WITNESS_FEATURE_SCHEMA}. Rerun v48.62 training."
            )
    quantifier_witness_enabled = bool(ckpt.get(
        "direct_recovery_absolute_quantifier_witness_correction",
        model_cfg.get("direct_recovery_absolute_quantifier_witness_correction", False),
    ))
    if quantifier_witness_enabled:
        feature_schema = int(ckpt.get("direct_recovery_absolute_quantifier_witness_feature_schema", 0) or 0)
        if feature_schema != DIRECT_COMMON_RECOVERY_WITNESS_FEATURE_SCHEMA:
            raise RuntimeError(
                "legacy/unknown OC-QARW checkpoint feature semantics: "
                f"schema={feature_schema}; v48.63 requires quantifier common-witness schema="
                f"{DIRECT_COMMON_RECOVERY_WITNESS_FEATURE_SCHEMA}. Rerun v48.63 training."
            )
    semantic_witness_enabled = bool(ckpt.get(
        "direct_recovery_absolute_semantic_witness_correction",
        model_cfg.get("direct_recovery_absolute_semantic_witness_correction", False),
    ))
    if semantic_witness_enabled:
        feature_schema = int(ckpt.get("direct_recovery_absolute_semantic_witness_feature_schema", 0) or 0)
        route_alignment = bool(ckpt.get(
            "direct_recovery_semantic_witness_route_alignment",
            model_cfg.get("direct_recovery_semantic_witness_route_alignment", False),
        ))
        reentry_alignment = bool(ckpt.get(
            "direct_recovery_semantic_witness_reentry_alignment",
            model_cfg.get("direct_recovery_semantic_witness_reentry_alignment", False),
        ))
        control_projection = bool(ckpt.get(
            "direct_recovery_semantic_witness_control_projection",
            model_cfg.get("direct_recovery_semantic_witness_control_projection", False),
        ))
        boundary_transport = bool(ckpt.get(
            "direct_recovery_semantic_witness_boundary_transport",
            model_cfg.get("direct_recovery_semantic_witness_boundary_transport", False),
        ))
        projection_fidelity = bool(ckpt.get(
            "direct_recovery_semantic_witness_projection_fidelity_weighting",
            model_cfg.get("direct_recovery_semantic_witness_projection_fidelity_weighting", False),
        ))
        demand_normalized_fidelity = bool(ckpt.get(
            "direct_recovery_semantic_witness_demand_normalized_fidelity",
            model_cfg.get("direct_recovery_semantic_witness_demand_normalized_fidelity", False),
        ))
        robust_occupancy = bool(ckpt.get(
            "direct_recovery_semantic_witness_robust_occupancy",
            model_cfg.get("direct_recovery_semantic_witness_robust_occupancy", False),
        ))
        soft_occupancy_disagreement = bool(ckpt.get(
            "direct_recovery_semantic_witness_soft_occupancy_disagreement",
            model_cfg.get("direct_recovery_semantic_witness_soft_occupancy_disagreement", False),
        ))
        boundary_localized_occupancy_trust = bool(ckpt.get(
            "direct_recovery_semantic_witness_boundary_localized_occupancy_trust",
            model_cfg.get("direct_recovery_semantic_witness_boundary_localized_occupancy_trust", False),
        ))
        history_occupancy_reachability = bool(ckpt.get(
            "direct_recovery_semantic_witness_history_occupancy_reachability",
            model_cfg.get("direct_recovery_semantic_witness_history_occupancy_reachability", False),
        ))
        interaction_box_support = bool(ckpt.get(
            "direct_recovery_semantic_witness_interaction_box_support",
            model_cfg.get("direct_recovery_semantic_witness_interaction_box_support", False),
        ))
        interaction_hull_support = bool(ckpt.get(
            "direct_recovery_semantic_witness_interaction_hull_support",
            model_cfg.get("direct_recovery_semantic_witness_interaction_hull_support", False),
        ))
        interaction_anchor_support = bool(ckpt.get(
            "direct_recovery_semantic_witness_interaction_anchor_support",
            model_cfg.get("direct_recovery_semantic_witness_interaction_anchor_support", False),
        ))
        interaction_response_support = bool(ckpt.get(
            "direct_recovery_semantic_witness_interaction_response_support",
            model_cfg.get("direct_recovery_semantic_witness_interaction_response_support", False),
        ))
        feature_source = str(ckpt.get(
            "direct_recovery_absolute_semantic_witness_feature_source", ""
        ) or "")
        if _v48_74_schema10_checkpoint_contract(
            feature_schema=feature_schema,
            feature_source=feature_source,
            interaction_anchor_support=interaction_anchor_support,
            interaction_response_support=interaction_response_support,
        ):
            # V48.74 deliberately reuses the V48.73 anchor/response selector
            # booleans to select coordinates 20/21 while upgrading the actual
            # feature contract to schema 10.  The helper above validates the
            # serialized pair before this historical selector mapping is used.
            expected_semantic_schema = DIRECT_SIGNED_VIABILITY_RECOVERY_WITNESS_FEATURE_SCHEMA
        else:
            expected_semantic_schema = (
                DIRECT_INTERACTION_RESPONSE_RECOVERY_WITNESS_FEATURE_SCHEMA
                if (interaction_anchor_support or interaction_response_support)
                else (DIRECT_INTERACTION_ORIENTED_RECOVERY_WITNESS_FEATURE_SCHEMA
                if (interaction_box_support or interaction_hull_support)
                else (DIRECT_BOUNDARY_OCCUPANCY_REACHABILITY_WITNESS_FEATURE_SCHEMA
                if (boundary_localized_occupancy_trust or history_occupancy_reachability)
                else (DIRECT_OCCUPANCY_TEMPERED_RECOVERY_WITNESS_FEATURE_SCHEMA
                if soft_occupancy_disagreement
                else (DIRECT_DEMAND_TEMPERED_RECOVERY_WITNESS_FEATURE_SCHEMA
                if demand_normalized_fidelity
                else (DIRECT_ROBUST_TRUST_RECOVERY_WITNESS_FEATURE_SCHEMA
                if (projection_fidelity or robust_occupancy)
                else (DIRECT_PROJECTED_BOUNDARY_RECOVERY_WITNESS_FEATURE_SCHEMA
                      if (control_projection or boundary_transport)
                      else (DIRECT_ACTIVE_CONSTRAINT_RECOVERY_WITNESS_FEATURE_SCHEMA
                            if (route_alignment or reentry_alignment)
                            else DIRECT_SEMANTIC_RECOVERY_WITNESS_FEATURE_SCHEMA)))))))
            )
        if feature_schema != expected_semantic_schema:
            raise RuntimeError(
                "legacy/unknown OC-SARW checkpoint feature semantics: "
                f"schema={feature_schema}; configuration requires semantic-witness schema="
                f"{expected_semantic_schema}. Rerun the matching training version."
            )
    encoder_type = str(ckpt.get("encoder_type", model_cfg.get("encoder_type", "mlp")))
    feature_layout = ckpt.get("feature_layout", None)
    model = OCRAPModel(
        int(ckpt["input_dim"]),
        num_roots=int(ckpt["num_roots"]),
        num_options=int(ckpt["num_options"]),
        d_model=d_model,
        d_obs=d_obs,
        tau_obs=tau_obs,
        encoder_type=encoder_type,
        feature_layout=feature_layout,
        num_layers=int(model_cfg.get("transformer_layers", 2)),
        num_heads=int(model_cfg.get("transformer_heads", 4)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        d_signature=int(ckpt.get("d_signature", 0)),
        d_future_signature=int(ckpt.get("d_future_signature", 0)),
        option_feature_dim=int(ckpt.get("option_feature_dim", OPTION_FEATURE_DIM)),
        direct_recovery_value_head=bool(ckpt.get("direct_recovery_value_head", model_cfg.get("direct_recovery_value_head", False))),
        direct_recovery_value_pooling=str(ckpt.get("direct_recovery_value_pooling", model_cfg.get("direct_recovery_value_pooling", "scene"))),
        direct_recovery_value_output=str(ckpt.get("direct_recovery_value_output", model_cfg.get("direct_recovery_value_output", "probability"))),
        direct_recovery_value_regime_conditioning=bool(ckpt.get("direct_recovery_value_regime_conditioning", model_cfg.get("direct_recovery_value_regime_conditioning", False))),
        direct_recovery_value_num_regimes=int(ckpt.get("direct_recovery_value_num_regimes", model_cfg.get("direct_recovery_value_num_regimes", 4))),
        direct_recovery_value_regime_dim=int(ckpt.get("direct_recovery_value_regime_dim", model_cfg.get("direct_recovery_value_regime_dim", 16))),
        direct_recovery_opportunity_head=bool(ckpt.get("direct_recovery_opportunity_head", model_cfg.get("direct_recovery_opportunity_head", False))),
        direct_recovery_harm_head=bool(ckpt.get("direct_recovery_harm_head", model_cfg.get("direct_recovery_harm_head", False))),
        direct_recovery_value_experts=bool(ckpt.get("direct_recovery_value_experts", model_cfg.get("direct_recovery_value_experts", False))),
        direct_recovery_value_num_experts=int(ckpt.get("direct_recovery_value_num_experts", model_cfg.get("direct_recovery_value_num_experts", 2))),
        direct_recovery_value_expert_routing=str(ckpt.get("direct_recovery_value_expert_routing", model_cfg.get("direct_recovery_value_expert_routing", "bucket"))),
        direct_recovery_value_router_temperature=float(ckpt.get("direct_recovery_value_router_temperature", model_cfg.get("direct_recovery_value_router_temperature", 1.0))),
        direct_recovery_value_router_pooling=str(ckpt.get("direct_recovery_value_router_pooling", model_cfg.get("direct_recovery_value_router_pooling", "candidate"))),
        direct_recovery_expert_disagreement_penalty=float(ckpt.get("direct_recovery_expert_disagreement_penalty", model_cfg.get("direct_recovery_expert_disagreement_penalty", 0.5))),
        direct_recovery_set_context=bool(ckpt.get("direct_recovery_set_context", model_cfg.get("direct_recovery_set_context", False))),
        direct_recovery_set_context_hidden=int(ckpt.get("direct_recovery_set_context_hidden", model_cfg.get("direct_recovery_set_context_hidden", d_model))),
        direct_recovery_set_context_dropout=float(ckpt.get("direct_recovery_set_context_dropout", model_cfg.get("direct_recovery_set_context_dropout", model_cfg.get("dropout", 0.1)))),
        direct_recovery_preference_head=bool(ckpt.get("direct_recovery_preference_head", model_cfg.get("direct_recovery_preference_head", False))),
        direct_recovery_preference_hidden=int(ckpt.get("direct_recovery_preference_hidden", model_cfg.get("direct_recovery_preference_hidden", max(16, d_model // 2)))),
        direct_recovery_preference_dropout=float(ckpt.get("direct_recovery_preference_dropout", model_cfg.get("direct_recovery_preference_dropout", 0.05))),
        direct_recovery_preference_context=bool(ckpt.get("direct_recovery_preference_context", model_cfg.get("direct_recovery_preference_context", False))),
        direct_recovery_preference_context_hidden=int(ckpt.get("direct_recovery_preference_context_hidden", model_cfg.get("direct_recovery_preference_context_hidden", d_model))),
        direct_recovery_relative_features_include_absolute=bool(ckpt.get("direct_recovery_relative_features_include_absolute", model_cfg.get("direct_recovery_relative_features_include_absolute", True))),
        direct_recovery_set_tournament=bool(ckpt.get("direct_recovery_set_tournament", model_cfg.get("direct_recovery_set_tournament", False))),
        direct_recovery_set_tournament_hidden=int(ckpt.get("direct_recovery_set_tournament_hidden", model_cfg.get("direct_recovery_set_tournament_hidden", 48))),
        direct_recovery_set_tournament_heads=int(ckpt.get("direct_recovery_set_tournament_heads", model_cfg.get("direct_recovery_set_tournament_heads", 4))),
        direct_recovery_set_tournament_dropout=float(ckpt.get("direct_recovery_set_tournament_dropout", model_cfg.get("direct_recovery_set_tournament_dropout", 0.05))),
        direct_recovery_set_tournament_replace_base=bool(ckpt.get("direct_recovery_set_tournament_replace_base", model_cfg.get("direct_recovery_set_tournament_replace_base", True))),
        direct_recovery_delta_head=bool(ckpt.get("direct_recovery_delta_head", model_cfg.get("direct_recovery_delta_head", False))),
        direct_recovery_delta_regime_experts=bool(ckpt.get("direct_recovery_delta_regime_experts", model_cfg.get("direct_recovery_delta_regime_experts", False))),
        direct_recovery_delta_policy_features=bool(ckpt.get("direct_recovery_delta_policy_features", model_cfg.get("direct_recovery_delta_policy_features", False))),
        direct_recovery_delta_hidden=int(ckpt.get("direct_recovery_delta_hidden", model_cfg.get("direct_recovery_delta_hidden", d_model))),
        direct_recovery_delta_dropout=float(ckpt.get("direct_recovery_delta_dropout", model_cfg.get("direct_recovery_delta_dropout", 0.05))),
        direct_recovery_delta_initial_logvar=float(ckpt.get("direct_recovery_delta_initial_logvar", model_cfg.get("direct_recovery_delta_initial_logvar", -4.605170186))),
        direct_recovery_delta_mode=str(ckpt.get("direct_recovery_delta_mode", model_cfg.get("direct_recovery_delta_mode", "gaussian"))),
        direct_recovery_evidence_calibrator=bool(ckpt.get("direct_recovery_evidence_calibrator", model_cfg.get("direct_recovery_evidence_calibrator", False))),
        direct_recovery_evidence_calibrator_hidden=int(ckpt.get("direct_recovery_evidence_calibrator_hidden", model_cfg.get("direct_recovery_evidence_calibrator_hidden", 8))),
        direct_recovery_evidence_calibrator_scale=float(ckpt.get("direct_recovery_evidence_calibrator_scale", model_cfg.get("direct_recovery_evidence_calibrator_scale", 0.25))),
        direct_recovery_evidence_calibrator_mode=str(ckpt.get("direct_recovery_evidence_calibrator_mode", model_cfg.get("direct_recovery_evidence_calibrator_mode", "center_width"))),
        direct_recovery_evidence_calibrator_context=bool(ckpt.get("direct_recovery_evidence_calibrator_context", model_cfg.get("direct_recovery_evidence_calibrator_context", False))),
        direct_recovery_evidence_calibrator_context_detach=bool(ckpt.get("direct_recovery_evidence_calibrator_context_detach", model_cfg.get("direct_recovery_evidence_calibrator_context_detach", True))),
        direct_recovery_evidence_calibrator_context_source=str(ckpt.get("direct_recovery_evidence_calibrator_context_source", model_cfg.get("direct_recovery_evidence_calibrator_context_source", "relative"))),
        direct_recovery_evidence_interaction_hidden=int(ckpt.get("direct_recovery_evidence_interaction_hidden", model_cfg.get("direct_recovery_evidence_interaction_hidden", 64))),
        direct_recovery_evidence_interaction_dropout=float(ckpt.get("direct_recovery_evidence_interaction_dropout", model_cfg.get("direct_recovery_evidence_interaction_dropout", 0.05))),
        direct_recovery_evidence_dual_interaction_bridge=bool(ckpt.get(
            "direct_recovery_evidence_dual_interaction_bridge",
            model_cfg.get("direct_recovery_evidence_dual_interaction_bridge", False),
        )),
        direct_recovery_evidence_factorized_harm_interaction=bool(ckpt.get(
            "direct_recovery_evidence_factorized_harm_interaction",
            model_cfg.get("direct_recovery_evidence_factorized_harm_interaction", False),
        )),
        direct_recovery_evidence_partial_pool_harm_residual=bool(ckpt.get(
            "direct_recovery_evidence_partial_pool_harm_residual",
            model_cfg.get("direct_recovery_evidence_partial_pool_harm_residual", False),
        )),
        direct_recovery_evidence_partial_pool_harm_residual_scale=float(ckpt.get(
            "direct_recovery_evidence_partial_pool_harm_residual_scale",
            model_cfg.get("direct_recovery_evidence_partial_pool_harm_residual_scale", 0.50),
        )),
        direct_recovery_evidence_rank_benefit_skip=bool(ckpt.get(
            "direct_recovery_evidence_rank_benefit_skip",
            model_cfg.get("direct_recovery_evidence_rank_benefit_skip", False),
        )),
        direct_recovery_evidence_rank_benefit_gain_init=float(ckpt.get(
            "direct_recovery_evidence_rank_benefit_gain_init",
            model_cfg.get("direct_recovery_evidence_rank_benefit_gain_init", 1.0),
        )),
        direct_recovery_evidence_postprefix_obs_transport_benefit=bool(ckpt.get(
            "direct_recovery_evidence_postprefix_obs_transport_benefit",
            model_cfg.get("direct_recovery_evidence_postprefix_obs_transport_benefit", False),
        )),
        direct_recovery_evidence_postprefix_obs_transport_harm=bool(ckpt.get(
            "direct_recovery_evidence_postprefix_obs_transport_harm",
            model_cfg.get("direct_recovery_evidence_postprefix_obs_transport_harm", False),
        )),
        direct_recovery_evidence_postprefix_obs_transport_scale=float(ckpt.get(
            "direct_recovery_evidence_postprefix_obs_transport_scale",
            model_cfg.get("direct_recovery_evidence_postprefix_obs_transport_scale", 1.0),
        )),
        direct_recovery_evidence_roct_benefit=bool(ckpt.get(
            "direct_recovery_evidence_roct_benefit",
            model_cfg.get("direct_recovery_evidence_roct_benefit", False),
        )),
        direct_recovery_evidence_roct_deployability=bool(ckpt.get(
            "direct_recovery_evidence_roct_deployability",
            model_cfg.get("direct_recovery_evidence_roct_deployability", False),
        )),
        direct_recovery_evidence_roct_scale=float(ckpt.get(
            "direct_recovery_evidence_roct_scale",
            model_cfg.get("direct_recovery_evidence_roct_scale", 1.0),
        )),
        direct_recovery_evidence_roct_alpha=float(ckpt.get(
            "direct_recovery_evidence_roct_alpha",
            model_cfg.get("direct_recovery_evidence_roct_alpha", 0.2),
        )),
        direct_recovery_evidence_roct_beta=float(ckpt.get(
            "direct_recovery_evidence_roct_beta",
            model_cfg.get("direct_recovery_evidence_roct_beta", 0.2),
        )),
        direct_recovery_evidence_roct_top_m=int(ckpt.get(
            "direct_recovery_evidence_roct_top_m",
            model_cfg.get("direct_recovery_evidence_roct_top_m", 8),
        )),
        direct_recovery_evidence_roct_option_temperature=float(ckpt.get(
            "direct_recovery_evidence_roct_option_temperature",
            model_cfg.get("direct_recovery_evidence_roct_option_temperature", 0.35),
        )),
        direct_recovery_evidence_common_measure_root_mass=bool(ckpt.get(
            "direct_recovery_evidence_common_measure_root_mass",
            model_cfg.get("direct_recovery_evidence_common_measure_root_mass", False),
        )),
        direct_recovery_absolute_feasibility_head=bool(ckpt.get(
            "direct_recovery_absolute_feasibility_head",
            model_cfg.get("direct_recovery_absolute_feasibility_head", False),
        )),
        direct_recovery_absolute_option_margin_correction=bool(ckpt.get(
            "direct_recovery_absolute_option_margin_correction",
            model_cfg.get("direct_recovery_absolute_option_margin_correction", False),
        )),
        direct_recovery_absolute_physical_headroom_correction=bool(ckpt.get(
            "direct_recovery_absolute_physical_headroom_correction",
            model_cfg.get("direct_recovery_absolute_physical_headroom_correction", False),
        )),
        direct_recovery_absolute_executable_witness_correction=bool(ckpt.get(
            "direct_recovery_absolute_executable_witness_correction",
            model_cfg.get("direct_recovery_absolute_executable_witness_correction", False),
        )),
        direct_recovery_absolute_common_witness_correction=bool(ckpt.get(
            "direct_recovery_absolute_common_witness_correction",
            model_cfg.get("direct_recovery_absolute_common_witness_correction", False),
        )),
        direct_recovery_absolute_quantifier_witness_correction=bool(ckpt.get(
            "direct_recovery_absolute_quantifier_witness_correction",
            model_cfg.get("direct_recovery_absolute_quantifier_witness_correction", False),
        )),
        direct_recovery_absolute_semantic_witness_correction=bool(ckpt.get(
            "direct_recovery_absolute_semantic_witness_correction",
            model_cfg.get("direct_recovery_absolute_semantic_witness_correction", False),
        )),
        direct_recovery_semantic_witness_active_set_alignment=bool(ckpt.get(
            "direct_recovery_semantic_witness_active_set_alignment",
            model_cfg.get("direct_recovery_semantic_witness_active_set_alignment", True),
        )),
        direct_recovery_semantic_witness_path_stop_alignment=bool(ckpt.get(
            "direct_recovery_semantic_witness_path_stop_alignment",
            model_cfg.get("direct_recovery_semantic_witness_path_stop_alignment", True),
        )),
        direct_recovery_semantic_witness_classlocal_transport=bool(ckpt.get(
            "direct_recovery_semantic_witness_classlocal_transport",
            model_cfg.get("direct_recovery_semantic_witness_classlocal_transport", False),
        )),
        direct_recovery_semantic_witness_route_alignment=bool(ckpt.get(
            "direct_recovery_semantic_witness_route_alignment",
            model_cfg.get("direct_recovery_semantic_witness_route_alignment", False),
        )),
        direct_recovery_semantic_witness_reentry_alignment=bool(ckpt.get(
            "direct_recovery_semantic_witness_reentry_alignment",
            model_cfg.get("direct_recovery_semantic_witness_reentry_alignment", False),
        )),
        direct_recovery_semantic_witness_control_projection=bool(ckpt.get(
            "direct_recovery_semantic_witness_control_projection",
            model_cfg.get("direct_recovery_semantic_witness_control_projection", False),
        )),
        direct_recovery_semantic_witness_boundary_transport=bool(ckpt.get(
            "direct_recovery_semantic_witness_boundary_transport",
            model_cfg.get("direct_recovery_semantic_witness_boundary_transport", False),
        )),
        direct_recovery_semantic_witness_projection_fidelity_weighting=bool(ckpt.get(
            "direct_recovery_semantic_witness_projection_fidelity_weighting",
            model_cfg.get("direct_recovery_semantic_witness_projection_fidelity_weighting", False),
        )),
        direct_recovery_semantic_witness_active_constraint_typed_source=bool(ckpt.get(
            "direct_recovery_semantic_witness_active_constraint_typed_source",
            model_cfg.get("direct_recovery_semantic_witness_active_constraint_typed_source", False),
        )),
        direct_recovery_semantic_witness_root_tail_source=bool(ckpt.get(
            "direct_recovery_semantic_witness_root_tail_source",
            model_cfg.get("direct_recovery_semantic_witness_root_tail_source", False),
        )),
        direct_recovery_semantic_witness_tail_localization=bool(ckpt.get(
            "direct_recovery_semantic_witness_tail_localization",
            model_cfg.get("direct_recovery_semantic_witness_tail_localization", False),
        )),
        direct_recovery_semantic_witness_structured_tail_field=bool(ckpt.get(
            "direct_recovery_semantic_witness_structured_tail_field",
            model_cfg.get("direct_recovery_semantic_witness_structured_tail_field", False),
        )),
        direct_recovery_semantic_witness_signed_tail_channels=bool(ckpt.get(
            "direct_recovery_semantic_witness_signed_tail_channels",
            model_cfg.get("direct_recovery_semantic_witness_signed_tail_channels", False),
        )),
        direct_recovery_semantic_witness_counterfactual_tail_response=bool(ckpt.get(
            "direct_recovery_semantic_witness_counterfactual_tail_response",
            model_cfg.get("direct_recovery_semantic_witness_counterfactual_tail_response", False),
        )),
        direct_recovery_semantic_witness_demand_normalized_fidelity=bool(ckpt.get(
            "direct_recovery_semantic_witness_demand_normalized_fidelity",
            model_cfg.get("direct_recovery_semantic_witness_demand_normalized_fidelity", False),
        )),
        direct_recovery_semantic_witness_robust_occupancy=bool(ckpt.get(
            "direct_recovery_semantic_witness_robust_occupancy",
            model_cfg.get("direct_recovery_semantic_witness_robust_occupancy", False),
        )),
        direct_recovery_semantic_witness_soft_occupancy_disagreement=bool(ckpt.get(
            "direct_recovery_semantic_witness_soft_occupancy_disagreement",
            model_cfg.get("direct_recovery_semantic_witness_soft_occupancy_disagreement", False),
        )),
        direct_recovery_semantic_witness_boundary_localized_occupancy_trust=bool(ckpt.get(
            "direct_recovery_semantic_witness_boundary_localized_occupancy_trust",
            model_cfg.get("direct_recovery_semantic_witness_boundary_localized_occupancy_trust", False),
        )),
        direct_recovery_semantic_witness_history_occupancy_reachability=bool(ckpt.get(
            "direct_recovery_semantic_witness_history_occupancy_reachability",
            model_cfg.get("direct_recovery_semantic_witness_history_occupancy_reachability", False),
        )),
        direct_recovery_semantic_witness_interaction_box_support=bool(ckpt.get(
            "direct_recovery_semantic_witness_interaction_box_support",
            model_cfg.get("direct_recovery_semantic_witness_interaction_box_support", False),
        )),
        direct_recovery_semantic_witness_interaction_hull_support=bool(ckpt.get(
            "direct_recovery_semantic_witness_interaction_hull_support",
            model_cfg.get("direct_recovery_semantic_witness_interaction_hull_support", False),
        )),
        direct_recovery_semantic_witness_interaction_anchor_support=bool(ckpt.get(
            "direct_recovery_semantic_witness_interaction_anchor_support",
            model_cfg.get("direct_recovery_semantic_witness_interaction_anchor_support", False),
        )),
        direct_recovery_semantic_witness_interaction_response_support=bool(ckpt.get(
            "direct_recovery_semantic_witness_interaction_response_support",
            model_cfg.get("direct_recovery_semantic_witness_interaction_response_support", False),
        )),
        direct_recovery_evidence_native_certificate_preservation=bool(ckpt.get(
            "direct_recovery_evidence_native_certificate_preservation",
            model_cfg.get("direct_recovery_evidence_native_certificate_preservation", False),
        )),
        direct_recovery_evidence_native_margin_complete_preservation=bool(ckpt.get(
            "direct_recovery_evidence_native_margin_complete_preservation",
            model_cfg.get("direct_recovery_evidence_native_margin_complete_preservation", False),
        )),
        direct_recovery_evidence_native_advantage_preservation=bool(ckpt.get(
            "direct_recovery_evidence_native_advantage_preservation",
            model_cfg.get("direct_recovery_evidence_native_advantage_preservation", False),
        )),
        direct_recovery_evidence_native_exact_advantage_preservation=bool(ckpt.get(
            "direct_recovery_evidence_native_exact_advantage_preservation",
            model_cfg.get("direct_recovery_evidence_native_exact_advantage_preservation", False),
        )),
        direct_recovery_evidence_native_boundary_complete_advantage_preservation=bool(ckpt.get(
            "direct_recovery_evidence_native_boundary_complete_advantage_preservation",
            model_cfg.get("direct_recovery_evidence_native_boundary_complete_advantage_preservation", False),
        )),
        direct_recovery_evidence_physical_student_drs=bool(ckpt.get(
            "direct_recovery_evidence_physical_student_drs",
            model_cfg.get("direct_recovery_evidence_physical_student_drs", False),
        )),
        direct_recovery_evidence_native_drs_tolerance=float(ckpt.get(
            "direct_recovery_evidence_native_drs_tolerance",
            model_cfg.get("direct_recovery_evidence_native_drs_tolerance", 0.05),
        )),
        direct_recovery_evidence_native_deployability_tolerance=float(ckpt.get(
            "direct_recovery_evidence_native_deployability_tolerance",
            model_cfg.get("direct_recovery_evidence_native_deployability_tolerance", 0.05),
        )),
        direct_recovery_evidence_native_dep_boundary_aligned=bool(ckpt.get(
            "direct_recovery_evidence_native_dep_boundary_aligned",
            model_cfg.get("direct_recovery_evidence_native_dep_boundary_aligned", False),
        )),
        direct_recovery_evidence_native_gap_tolerance=float(ckpt.get(
            "direct_recovery_evidence_native_gap_tolerance",
            model_cfg.get("direct_recovery_evidence_native_gap_tolerance", 0.05),
        )),
        direct_recovery_evidence_native_positive_gain=float(ckpt.get(
            "direct_recovery_evidence_native_positive_gain",
            model_cfg.get("direct_recovery_evidence_native_positive_gain", 0.015),
        )),
        direct_recovery_evidence_calibrator_shared=bool(ckpt.get("direct_recovery_evidence_calibrator_shared", model_cfg.get("direct_recovery_evidence_calibrator_shared", False))),
        direct_recovery_evidence_calibrator_regime_scale=float(ckpt.get("direct_recovery_evidence_calibrator_regime_scale", model_cfg.get("direct_recovery_evidence_calibrator_regime_scale", 0.25))),
        direct_recovery_evidence_unified_experts=bool(ckpt.get("direct_recovery_evidence_unified_experts", model_cfg.get("direct_recovery_evidence_unified_experts", False))),
        direct_recovery_evidence_component_heads=bool(ckpt.get("direct_recovery_evidence_component_heads", model_cfg.get("direct_recovery_evidence_component_heads", False))),
        direct_recovery_evidence_component_count=int(ckpt.get("direct_recovery_evidence_component_count", model_cfg.get("direct_recovery_evidence_component_count", 3))),
        direct_recovery_evidence_component_scale=float(ckpt.get("direct_recovery_evidence_component_scale", model_cfg.get("direct_recovery_evidence_component_scale", 6.0))),
        direct_recovery_evidence_benefit_residual_scale=float(ckpt.get(
            "direct_recovery_evidence_benefit_residual_scale",
            model_cfg.get("direct_recovery_evidence_benefit_residual_scale", 1.0),
        )),
        direct_recovery_evidence_unbounded_benefit_factor=bool(ckpt.get(
            "direct_recovery_evidence_unbounded_benefit_factor",
            model_cfg.get("direct_recovery_evidence_unbounded_benefit_factor", False),
        )),
        direct_recovery_evidence_unbounded_harm_factors=bool(ckpt.get(
            "direct_recovery_evidence_unbounded_harm_factors",
            model_cfg.get("direct_recovery_evidence_unbounded_harm_factors", False),
        )),
        direct_recovery_evidence_component_reliability=str(ckpt.get(
            "direct_recovery_evidence_component_reliability",
            model_cfg.get("direct_recovery_evidence_component_reliability", ""),
        ) or ""),
        direct_recovery_evidence_concord=bool(ckpt.get("direct_recovery_evidence_concord", model_cfg.get("direct_recovery_evidence_concord", False))),
        direct_recovery_evidence_consensus_disagreement_penalty=float(ckpt.get(
            "direct_recovery_evidence_consensus_disagreement_penalty",
            model_cfg.get("direct_recovery_evidence_consensus_disagreement_penalty", 0.15),
        )),
        direct_recovery_evidence_consensus_prior_scale=float(ckpt.get(
            "direct_recovery_evidence_consensus_prior_scale",
            model_cfg.get("direct_recovery_evidence_consensus_prior_scale", 1.0),
        )),
        direct_recovery_evidence_admission_head=bool(ckpt.get(
            "direct_recovery_evidence_admission_head",
            model_cfg.get("direct_recovery_evidence_admission_head", False),
        )),
        direct_recovery_evidence_admission_scale=float(ckpt.get(
            "direct_recovery_evidence_admission_scale",
            model_cfg.get("direct_recovery_evidence_admission_scale", 2.0),
        )),
        direct_recovery_evidence_admission_bounded=bool(ckpt.get(
            "direct_recovery_evidence_admission_bounded",
            model_cfg.get("direct_recovery_evidence_admission_bounded", True),
        )),
        direct_recovery_evidence_admission_prior_detach=bool(ckpt.get(
            "direct_recovery_evidence_admission_prior_detach",
            model_cfg.get("direct_recovery_evidence_admission_prior_detach", True),
        )),
        direct_recovery_evidence_admission_prior_mode=str(ckpt.get(
            "direct_recovery_evidence_admission_prior_mode",
            model_cfg.get("direct_recovery_evidence_admission_prior_mode", "risk_centered"),
        )),
        direct_recovery_evidence_slack_temperature=float(ckpt.get(
            "direct_recovery_evidence_slack_temperature",
            model_cfg.get("direct_recovery_evidence_slack_temperature", 0.025),
        )),
        direct_recovery_evidence_slack_penalty=float(ckpt.get(
            "direct_recovery_evidence_slack_penalty",
            model_cfg.get("direct_recovery_evidence_slack_penalty", 1.0),
        )),
        direct_recovery_evidence_frontier_cap_temperature=float(ckpt.get(
            "direct_recovery_evidence_frontier_cap_temperature",
            model_cfg.get("direct_recovery_evidence_frontier_cap_temperature", 0.10),
        )),
        direct_recovery_evidence_benefit_margin_temperature=float(ckpt.get(
            "direct_recovery_evidence_benefit_margin_temperature",
            model_cfg.get("direct_recovery_evidence_benefit_margin_temperature", 0.025),
        )),
        direct_recovery_evidence_joint_reserve_temperature=float(ckpt.get(
            "direct_recovery_evidence_joint_reserve_temperature",
            model_cfg.get("direct_recovery_evidence_joint_reserve_temperature", 0.025),
        )),
        direct_recovery_evidence_reserve_factor_alignment=bool(ckpt.get(
            "direct_recovery_evidence_reserve_factor_alignment",
            model_cfg.get("direct_recovery_evidence_reserve_factor_alignment", False),
        )),
        direct_recovery_evidence_frontier=bool(ckpt.get(
            "direct_recovery_evidence_frontier",
            model_cfg.get("direct_recovery_evidence_frontier", False),
        )),
        direct_recovery_evidence_component_prior_logit=float(ckpt.get(
            "direct_recovery_evidence_component_prior_logit",
            model_cfg.get("direct_recovery_evidence_component_prior_logit", -2.0),
        )),
    ).to(device)
    # Strict loading remains the default for checkpoints with matching geometry.
    # A v39 checkpoint can initialize v40 training through train.py's explicit
    # strict=False path; inference should never silently use an untrained head.
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    cfg.setdefault("model", {})
    cfg["model"]["d_model"] = d_model
    cfg["model"]["d_obs"] = d_obs
    cfg["model"]["tau_obs"] = tau_obs
    cfg["model"]["encoder_type"] = encoder_type
    cfg["model"]["direct_recovery_value_pooling"] = str(ckpt.get("direct_recovery_value_pooling", model_cfg.get("direct_recovery_value_pooling", "scene")))
    cfg["model"]["direct_recovery_value_output"] = str(ckpt.get("direct_recovery_value_output", model_cfg.get("direct_recovery_value_output", "probability")))
    cfg["model"]["direct_recovery_value_regime_conditioning"] = bool(ckpt.get("direct_recovery_value_regime_conditioning", model_cfg.get("direct_recovery_value_regime_conditioning", False)))
    cfg["model"]["direct_recovery_opportunity_head"] = bool(ckpt.get("direct_recovery_opportunity_head", model_cfg.get("direct_recovery_opportunity_head", False)))
    cfg["model"]["direct_recovery_harm_head"] = bool(ckpt.get("direct_recovery_harm_head", model_cfg.get("direct_recovery_harm_head", False)))
    cfg["model"]["direct_recovery_value_experts"] = bool(ckpt.get("direct_recovery_value_experts", model_cfg.get("direct_recovery_value_experts", False)))
    cfg["model"]["direct_recovery_value_num_experts"] = int(ckpt.get("direct_recovery_value_num_experts", model_cfg.get("direct_recovery_value_num_experts", 2)))
    cfg["model"]["direct_recovery_value_expert_routing"] = str(ckpt.get("direct_recovery_value_expert_routing", model_cfg.get("direct_recovery_value_expert_routing", "bucket")))
    cfg["model"]["direct_recovery_value_router_temperature"] = float(ckpt.get("direct_recovery_value_router_temperature", model_cfg.get("direct_recovery_value_router_temperature", 1.0)))
    cfg["model"]["direct_recovery_value_router_pooling"] = str(ckpt.get("direct_recovery_value_router_pooling", model_cfg.get("direct_recovery_value_router_pooling", "candidate")))
    cfg["model"]["direct_recovery_expert_disagreement_penalty"] = float(ckpt.get("direct_recovery_expert_disagreement_penalty", model_cfg.get("direct_recovery_expert_disagreement_penalty", 0.5)))
    cfg["model"]["direct_recovery_set_context"] = bool(ckpt.get("direct_recovery_set_context", model_cfg.get("direct_recovery_set_context", False)))
    cfg["model"]["direct_recovery_set_context_hidden"] = int(ckpt.get("direct_recovery_set_context_hidden", model_cfg.get("direct_recovery_set_context_hidden", d_model)))
    cfg["model"]["direct_recovery_set_context_dropout"] = float(ckpt.get("direct_recovery_set_context_dropout", model_cfg.get("direct_recovery_set_context_dropout", model_cfg.get("dropout", 0.1))))
    cfg["model"]["direct_recovery_preference_head"] = bool(ckpt.get("direct_recovery_preference_head", model_cfg.get("direct_recovery_preference_head", False)))
    cfg["model"]["direct_recovery_preference_hidden"] = int(ckpt.get("direct_recovery_preference_hidden", model_cfg.get("direct_recovery_preference_hidden", max(16, d_model // 2))))
    cfg["model"]["direct_recovery_preference_dropout"] = float(ckpt.get("direct_recovery_preference_dropout", model_cfg.get("direct_recovery_preference_dropout", 0.05)))
    cfg["model"]["direct_recovery_preference_context"] = bool(ckpt.get("direct_recovery_preference_context", model_cfg.get("direct_recovery_preference_context", False)))
    cfg["model"]["direct_recovery_preference_context_hidden"] = int(ckpt.get("direct_recovery_preference_context_hidden", model_cfg.get("direct_recovery_preference_context_hidden", d_model)))
    cfg["model"]["direct_recovery_relative_features_include_absolute"] = bool(ckpt.get("direct_recovery_relative_features_include_absolute", model_cfg.get("direct_recovery_relative_features_include_absolute", True)))
    cfg["model"]["direct_recovery_set_tournament"] = bool(ckpt.get("direct_recovery_set_tournament", model_cfg.get("direct_recovery_set_tournament", False)))
    cfg["model"]["direct_recovery_set_tournament_hidden"] = int(ckpt.get("direct_recovery_set_tournament_hidden", model_cfg.get("direct_recovery_set_tournament_hidden", 48)))
    cfg["model"]["direct_recovery_set_tournament_heads"] = int(ckpt.get("direct_recovery_set_tournament_heads", model_cfg.get("direct_recovery_set_tournament_heads", 4)))
    cfg["model"]["direct_recovery_set_tournament_dropout"] = float(ckpt.get("direct_recovery_set_tournament_dropout", model_cfg.get("direct_recovery_set_tournament_dropout", 0.05)))
    cfg["model"]["direct_recovery_set_tournament_replace_base"] = bool(ckpt.get("direct_recovery_set_tournament_replace_base", model_cfg.get("direct_recovery_set_tournament_replace_base", True)))
    cfg["model"]["direct_recovery_delta_head"] = bool(ckpt.get("direct_recovery_delta_head", model_cfg.get("direct_recovery_delta_head", False)))
    cfg["model"]["direct_recovery_delta_regime_experts"] = bool(ckpt.get("direct_recovery_delta_regime_experts", model_cfg.get("direct_recovery_delta_regime_experts", False)))
    cfg["model"]["direct_recovery_delta_policy_features"] = bool(ckpt.get("direct_recovery_delta_policy_features", model_cfg.get("direct_recovery_delta_policy_features", False)))
    cfg["model"]["direct_recovery_delta_hidden"] = int(ckpt.get("direct_recovery_delta_hidden", model_cfg.get("direct_recovery_delta_hidden", d_model)))
    cfg["model"]["direct_recovery_delta_dropout"] = float(ckpt.get("direct_recovery_delta_dropout", model_cfg.get("direct_recovery_delta_dropout", 0.05)))
    cfg["model"]["direct_recovery_delta_initial_logvar"] = float(ckpt.get("direct_recovery_delta_initial_logvar", model_cfg.get("direct_recovery_delta_initial_logvar", -4.605170186)))
    cfg["model"]["direct_recovery_delta_mode"] = str(ckpt.get("direct_recovery_delta_mode", model_cfg.get("direct_recovery_delta_mode", "gaussian")))
    cfg["model"]["direct_recovery_evidence_calibrator"] = bool(ckpt.get("direct_recovery_evidence_calibrator", model_cfg.get("direct_recovery_evidence_calibrator", False)))
    cfg["model"]["direct_recovery_evidence_calibrator_hidden"] = int(ckpt.get("direct_recovery_evidence_calibrator_hidden", model_cfg.get("direct_recovery_evidence_calibrator_hidden", 8)))
    cfg["model"]["direct_recovery_evidence_calibrator_scale"] = float(ckpt.get("direct_recovery_evidence_calibrator_scale", model_cfg.get("direct_recovery_evidence_calibrator_scale", 0.25)))
    cfg["model"]["direct_recovery_evidence_calibrator_mode"] = str(ckpt.get("direct_recovery_evidence_calibrator_mode", model_cfg.get("direct_recovery_evidence_calibrator_mode", "center_width")))
    cfg["model"]["direct_recovery_evidence_calibrator_context"] = bool(ckpt.get("direct_recovery_evidence_calibrator_context", model_cfg.get("direct_recovery_evidence_calibrator_context", False)))
    cfg["model"]["direct_recovery_evidence_calibrator_context_detach"] = bool(ckpt.get("direct_recovery_evidence_calibrator_context_detach", model_cfg.get("direct_recovery_evidence_calibrator_context_detach", True)))
    cfg["model"]["direct_recovery_evidence_calibrator_context_source"] = str(ckpt.get("direct_recovery_evidence_calibrator_context_source", model_cfg.get("direct_recovery_evidence_calibrator_context_source", "relative")))
    cfg["model"]["direct_recovery_evidence_interaction_hidden"] = int(ckpt.get("direct_recovery_evidence_interaction_hidden", model_cfg.get("direct_recovery_evidence_interaction_hidden", 64)))
    cfg["model"]["direct_recovery_evidence_interaction_dropout"] = float(ckpt.get("direct_recovery_evidence_interaction_dropout", model_cfg.get("direct_recovery_evidence_interaction_dropout", 0.05)))
    cfg["model"]["direct_recovery_evidence_dual_interaction_bridge"] = bool(
        model.direct_recovery_evidence_dual_interaction_bridge
    )
    cfg["model"]["direct_recovery_evidence_factorized_harm_interaction"] = bool(
        model.direct_recovery_evidence_factorized_harm_interaction
    )
    cfg["model"]["direct_recovery_evidence_partial_pool_harm_residual"] = bool(
        model.direct_recovery_evidence_partial_pool_harm_residual
    )
    cfg["model"]["direct_recovery_evidence_partial_pool_harm_residual_scale"] = float(
        model.direct_recovery_evidence_partial_pool_harm_residual_scale
    )
    cfg["model"]["direct_recovery_evidence_rank_benefit_skip"] = bool(
        model.direct_recovery_evidence_rank_benefit_skip
    )
    cfg["model"]["direct_recovery_evidence_rank_benefit_gain_init"] = float(
        model.direct_recovery_evidence_rank_benefit_gain_init
    )
    cfg["model"]["direct_recovery_evidence_postprefix_obs_transport_benefit"] = bool(
        model.direct_recovery_evidence_postprefix_obs_transport_benefit
    )
    cfg["model"]["direct_recovery_evidence_postprefix_obs_transport_harm"] = bool(
        model.direct_recovery_evidence_postprefix_obs_transport_harm
    )
    cfg["model"]["direct_recovery_evidence_postprefix_obs_transport_scale"] = float(
        model.direct_recovery_evidence_postprefix_obs_transport_scale
    )
    cfg["model"]["direct_recovery_evidence_roct_benefit"] = bool(
        model.direct_recovery_evidence_roct_benefit
    )
    cfg["model"]["direct_recovery_evidence_roct_deployability"] = bool(
        model.direct_recovery_evidence_roct_deployability
    )
    cfg["model"]["direct_recovery_evidence_roct_scale"] = float(
        model.direct_recovery_evidence_roct_scale
    )
    cfg["model"]["direct_recovery_evidence_roct_alpha"] = float(
        model.direct_recovery_evidence_roct_alpha
    )
    cfg["model"]["direct_recovery_evidence_roct_beta"] = float(
        model.direct_recovery_evidence_roct_beta
    )
    cfg["model"]["direct_recovery_evidence_roct_top_m"] = int(
        model.direct_recovery_evidence_roct_top_m
    )
    cfg["model"]["direct_recovery_evidence_roct_option_temperature"] = float(
        model.direct_recovery_evidence_roct_option_temperature
    )
    cfg["model"]["direct_recovery_evidence_common_measure_root_mass"] = bool(
        model.direct_recovery_evidence_common_measure_root_mass
    )
    cfg["model"]["direct_recovery_absolute_feasibility_head"] = bool(
        model.direct_recovery_absolute_feasibility_head
    )
    cfg["model"]["direct_recovery_absolute_option_margin_correction"] = bool(
        model.direct_recovery_absolute_option_margin_correction
    )
    cfg["model"]["direct_recovery_absolute_physical_headroom_correction"] = bool(
        model.direct_recovery_absolute_physical_headroom_correction
    )
    cfg["model"]["direct_recovery_absolute_executable_witness_correction"] = bool(
        model.direct_recovery_absolute_executable_witness_correction
    )
    cfg["model"]["direct_recovery_absolute_common_witness_correction"] = bool(
        model.direct_recovery_absolute_common_witness_correction
    )
    cfg["model"]["direct_recovery_absolute_quantifier_witness_correction"] = bool(
        model.direct_recovery_absolute_quantifier_witness_correction
    )
    cfg["model"]["direct_recovery_absolute_semantic_witness_correction"] = bool(
        model.direct_recovery_absolute_semantic_witness_correction
    )
    cfg["model"]["direct_recovery_semantic_witness_active_set_alignment"] = bool(
        model.direct_recovery_semantic_witness_active_set_alignment
    )
    cfg["model"]["direct_recovery_semantic_witness_path_stop_alignment"] = bool(
        model.direct_recovery_semantic_witness_path_stop_alignment
    )
    cfg["model"]["direct_recovery_semantic_witness_classlocal_transport"] = bool(
        model.direct_recovery_semantic_witness_classlocal_transport
    )
    cfg["model"]["direct_recovery_semantic_witness_route_alignment"] = bool(
        model.direct_recovery_semantic_witness_route_alignment
    )
    cfg["model"]["direct_recovery_semantic_witness_reentry_alignment"] = bool(
        model.direct_recovery_semantic_witness_reentry_alignment
    )
    cfg["model"]["direct_recovery_semantic_witness_control_projection"] = bool(
        model.direct_recovery_semantic_witness_control_projection
    )
    cfg["model"]["direct_recovery_semantic_witness_boundary_transport"] = bool(
        model.direct_recovery_semantic_witness_boundary_transport
    )
    cfg["model"]["direct_recovery_semantic_witness_projection_fidelity_weighting"] = bool(
        model.direct_recovery_semantic_witness_projection_fidelity_weighting
    )
    cfg["model"]["direct_recovery_semantic_witness_active_constraint_typed_source"] = bool(
        model.direct_recovery_semantic_witness_active_constraint_typed_source
    )
    cfg["model"]["direct_recovery_semantic_witness_root_tail_source"] = bool(
        model.direct_recovery_semantic_witness_root_tail_source
    )
    cfg["model"]["direct_recovery_semantic_witness_tail_localization"] = bool(
        model.direct_recovery_semantic_witness_tail_localization
    )
    cfg["model"]["direct_recovery_semantic_witness_structured_tail_field"] = bool(
        model.direct_recovery_semantic_witness_structured_tail_field
    )
    cfg["model"]["direct_recovery_semantic_witness_signed_tail_channels"] = bool(
        model.direct_recovery_semantic_witness_signed_tail_channels
    )
    cfg["model"]["direct_recovery_semantic_witness_counterfactual_tail_response"] = bool(
        model.direct_recovery_semantic_witness_counterfactual_tail_response
    )
    cfg["model"]["direct_recovery_semantic_witness_demand_normalized_fidelity"] = bool(
        model.direct_recovery_semantic_witness_demand_normalized_fidelity
    )
    cfg["model"]["direct_recovery_semantic_witness_robust_occupancy"] = bool(
        model.direct_recovery_semantic_witness_robust_occupancy
    )
    cfg["model"]["direct_recovery_semantic_witness_soft_occupancy_disagreement"] = bool(
        model.direct_recovery_semantic_witness_soft_occupancy_disagreement
    )
    cfg["model"]["direct_recovery_semantic_witness_boundary_localized_occupancy_trust"] = bool(
        model.direct_recovery_semantic_witness_boundary_localized_occupancy_trust
    )
    cfg["model"]["direct_recovery_semantic_witness_history_occupancy_reachability"] = bool(
        model.direct_recovery_semantic_witness_history_occupancy_reachability
    )
    cfg["model"]["direct_recovery_semantic_witness_interaction_box_support"] = bool(
        model.direct_recovery_semantic_witness_interaction_box_support
    )
    cfg["model"]["direct_recovery_semantic_witness_interaction_hull_support"] = bool(
        model.direct_recovery_semantic_witness_interaction_hull_support
    )
    cfg["model"]["direct_recovery_semantic_witness_interaction_anchor_support"] = bool(
        model.direct_recovery_semantic_witness_interaction_anchor_support
    )
    cfg["model"]["direct_recovery_semantic_witness_interaction_response_support"] = bool(
        model.direct_recovery_semantic_witness_interaction_response_support
    )
    cfg["model"]["direct_recovery_evidence_native_certificate_preservation"] = bool(
        model.direct_recovery_evidence_native_certificate_preservation
    )
    cfg["model"]["direct_recovery_evidence_native_margin_complete_preservation"] = bool(
        model.direct_recovery_evidence_native_margin_complete_preservation
    )
    cfg["model"]["direct_recovery_evidence_native_advantage_preservation"] = bool(
        model.direct_recovery_evidence_native_advantage_preservation
    )
    cfg["model"]["direct_recovery_evidence_native_exact_advantage_preservation"] = bool(
        model.direct_recovery_evidence_native_exact_advantage_preservation
    )
    cfg["model"]["direct_recovery_evidence_native_boundary_complete_advantage_preservation"] = bool(
        model.direct_recovery_evidence_native_boundary_complete_advantage_preservation
    )
    cfg["model"]["direct_recovery_evidence_physical_student_drs"] = bool(
        model.direct_recovery_evidence_physical_student_drs
    )
    cfg["model"]["direct_recovery_evidence_native_drs_tolerance"] = float(
        model.direct_recovery_evidence_native_drs_tolerance
    )
    cfg["model"]["direct_recovery_evidence_native_deployability_tolerance"] = float(
        model.direct_recovery_evidence_native_deployability_tolerance
    )
    cfg["model"]["direct_recovery_evidence_native_dep_boundary_aligned"] = bool(
        model.direct_recovery_evidence_native_dep_boundary_aligned
    )
    cfg["model"]["direct_recovery_evidence_native_gap_tolerance"] = float(
        model.direct_recovery_evidence_native_gap_tolerance
    )
    cfg["model"]["direct_recovery_evidence_native_positive_gain"] = float(
        model.direct_recovery_evidence_native_positive_gain
    )
    cfg["model"]["direct_recovery_evidence_calibrator_shared"] = bool(ckpt.get("direct_recovery_evidence_calibrator_shared", model_cfg.get("direct_recovery_evidence_calibrator_shared", False)))
    cfg["model"]["direct_recovery_evidence_calibrator_regime_scale"] = float(ckpt.get("direct_recovery_evidence_calibrator_regime_scale", model_cfg.get("direct_recovery_evidence_calibrator_regime_scale", 0.25)))
    cfg["model"]["direct_recovery_evidence_unified_experts"] = bool(ckpt.get("direct_recovery_evidence_unified_experts", model_cfg.get("direct_recovery_evidence_unified_experts", False)))
    cfg["model"]["direct_recovery_evidence_component_heads"] = bool(ckpt.get("direct_recovery_evidence_component_heads", model_cfg.get("direct_recovery_evidence_component_heads", False)))
    cfg["model"]["direct_recovery_evidence_component_count"] = int(ckpt.get("direct_recovery_evidence_component_count", model_cfg.get("direct_recovery_evidence_component_count", 3)))
    cfg["model"]["direct_recovery_evidence_component_scale"] = float(ckpt.get("direct_recovery_evidence_component_scale", model_cfg.get("direct_recovery_evidence_component_scale", 6.0)))
    cfg["model"]["direct_recovery_evidence_benefit_residual_scale"] = float(
        model.direct_recovery_evidence_benefit_residual_scale
    )
    cfg["model"]["direct_recovery_evidence_unbounded_benefit_factor"] = bool(
        model.direct_recovery_evidence_unbounded_benefit_factor
    )
    cfg["model"]["direct_recovery_evidence_unbounded_harm_factors"] = bool(
        model.direct_recovery_evidence_unbounded_harm_factors
    )
    cfg["model"]["direct_recovery_evidence_concord"] = bool(ckpt.get("direct_recovery_evidence_concord", model_cfg.get("direct_recovery_evidence_concord", False)))
    cfg["model"]["direct_recovery_evidence_consensus_disagreement_penalty"] = float(ckpt.get(
        "direct_recovery_evidence_consensus_disagreement_penalty",
        model_cfg.get("direct_recovery_evidence_consensus_disagreement_penalty", 0.15),
    ))
    cfg["model"]["direct_recovery_evidence_consensus_prior_scale"] = float(ckpt.get(
        "direct_recovery_evidence_consensus_prior_scale",
        model_cfg.get("direct_recovery_evidence_consensus_prior_scale", 1.0),
    ))
    cfg["model"]["direct_recovery_evidence_admission_head"] = bool(ckpt.get(
        "direct_recovery_evidence_admission_head",
        model_cfg.get("direct_recovery_evidence_admission_head", False),
    ))
    cfg["model"]["direct_recovery_evidence_admission_scale"] = float(ckpt.get(
        "direct_recovery_evidence_admission_scale",
        model_cfg.get("direct_recovery_evidence_admission_scale", 2.0),
    ))
    cfg["model"]["direct_recovery_evidence_admission_bounded"] = bool(
        model.direct_recovery_evidence_admission_bounded
    )
    cfg["model"]["direct_recovery_evidence_admission_prior_detach"] = bool(
        model.direct_recovery_evidence_admission_prior_detach
    )
    cfg["model"]["direct_recovery_evidence_admission_prior_mode"] = str(
        model.direct_recovery_evidence_admission_prior_mode
    )
    cfg["model"]["direct_recovery_evidence_slack_temperature"] = float(
        model.direct_recovery_evidence_slack_temperature
    )
    cfg["model"]["direct_recovery_evidence_slack_penalty"] = float(
        model.direct_recovery_evidence_slack_penalty
    )
    cfg["model"]["direct_recovery_evidence_frontier_cap_temperature"] = float(
        model.direct_recovery_evidence_frontier_cap_temperature
    )
    cfg["model"]["direct_recovery_evidence_benefit_margin_temperature"] = float(
        model.direct_recovery_evidence_benefit_margin_temperature
    )
    cfg["model"]["direct_recovery_evidence_joint_reserve_temperature"] = float(
        model.direct_recovery_evidence_joint_reserve_temperature
    )
    cfg["model"]["direct_recovery_evidence_reserve_factor_alignment"] = bool(
        model.direct_recovery_evidence_reserve_factor_alignment
    )
    cfg["model"]["direct_recovery_evidence_frontier"] = bool(
        model.direct_recovery_evidence_frontier
    )
    cfg["model"]["direct_recovery_evidence_component_prior_logit"] = float(
        model.direct_recovery_evidence_component_prior_logit
    )
    cfg["model"]["direct_recovery_evidence_component_reliability"] = ",".join(
        f"{x:.8g}" for x in model.direct_recovery_evidence_component_reliability
    )
    # Fail closed when checkpoint construction and runtime reporting diverge.
    raw_component_reliability = ckpt.get(
        "direct_recovery_evidence_component_reliability",
        model_cfg.get("direct_recovery_evidence_component_reliability", ""),
    )
    if raw_component_reliability is None:
        reliability_values = []
    elif isinstance(raw_component_reliability, str):
        reliability_text = raw_component_reliability.strip()
        if reliability_text.lower() in {"", "none", "null", "~"}:
            reliability_values = []
        else:
            reliability_values = [
                float(x.strip()) for x in reliability_text.split(",") if x.strip()
            ]
    else:
        reliability_values = [float(x) for x in raw_component_reliability]
    component_count = int(ckpt.get(
        "direct_recovery_evidence_component_count",
        model_cfg.get("direct_recovery_evidence_component_count", 3),
    ))
    if not reliability_values:
        reliability_values = [1.0] * component_count
    if len(reliability_values) < component_count:
        reliability_values.extend([1.0] * (component_count - len(reliability_values)))
    expected_component_reliability = ",".join(
        f"{min(1.0, max(0.0, x)):.8g}" for x in reliability_values[:component_count]
    )
    expected_contract = {
        "direct_recovery_evidence_calibrator_context": bool(ckpt.get(
            "direct_recovery_evidence_calibrator_context",
            model_cfg.get("direct_recovery_evidence_calibrator_context", False),
        )),
        "direct_recovery_evidence_calibrator_context_source": str(ckpt.get(
            "direct_recovery_evidence_calibrator_context_source",
            model_cfg.get("direct_recovery_evidence_calibrator_context_source", "relative"),
        )),
        "direct_recovery_evidence_interaction_hidden": int(ckpt.get(
            "direct_recovery_evidence_interaction_hidden",
            model_cfg.get("direct_recovery_evidence_interaction_hidden", 64),
        )),
        "direct_recovery_evidence_interaction_dropout": float(ckpt.get(
            "direct_recovery_evidence_interaction_dropout",
            model_cfg.get("direct_recovery_evidence_interaction_dropout", 0.05),
        )),
        "direct_recovery_evidence_admission_bounded": bool(ckpt.get(
            "direct_recovery_evidence_admission_bounded",
            model_cfg.get("direct_recovery_evidence_admission_bounded", True),
        )),
        "direct_recovery_evidence_admission_prior_detach": bool(ckpt.get(
            "direct_recovery_evidence_admission_prior_detach",
            model_cfg.get("direct_recovery_evidence_admission_prior_detach", True),
        )),
        "direct_recovery_evidence_admission_prior_mode": str(ckpt.get(
            "direct_recovery_evidence_admission_prior_mode",
            model_cfg.get("direct_recovery_evidence_admission_prior_mode", "risk_centered"),
        )),
        "direct_recovery_evidence_slack_temperature": float(ckpt.get(
            "direct_recovery_evidence_slack_temperature",
            model_cfg.get("direct_recovery_evidence_slack_temperature", 0.025),
        )),
        "direct_recovery_evidence_slack_penalty": float(ckpt.get(
            "direct_recovery_evidence_slack_penalty",
            model_cfg.get("direct_recovery_evidence_slack_penalty", 1.0),
        )),
        "direct_recovery_evidence_frontier_cap_temperature": float(ckpt.get(
            "direct_recovery_evidence_frontier_cap_temperature",
            model_cfg.get("direct_recovery_evidence_frontier_cap_temperature", 0.10),
        )),
        "direct_recovery_evidence_benefit_margin_temperature": float(ckpt.get(
            "direct_recovery_evidence_benefit_margin_temperature",
            model_cfg.get("direct_recovery_evidence_benefit_margin_temperature", 0.025),
        )),
        "direct_recovery_evidence_joint_reserve_temperature": float(ckpt.get(
            "direct_recovery_evidence_joint_reserve_temperature",
            model_cfg.get("direct_recovery_evidence_joint_reserve_temperature", 0.025),
        )),
        "direct_recovery_evidence_frontier": bool(ckpt.get(
            "direct_recovery_evidence_frontier",
            model_cfg.get("direct_recovery_evidence_frontier", False),
        )),
        "direct_recovery_evidence_component_prior_logit": float(ckpt.get(
            "direct_recovery_evidence_component_prior_logit",
            model_cfg.get("direct_recovery_evidence_component_prior_logit", -2.0),
        )),
        "direct_recovery_evidence_component_reliability": expected_component_reliability,
        "direct_recovery_evidence_consensus_prior_scale": float(ckpt.get(
            "direct_recovery_evidence_consensus_prior_scale",
            model_cfg.get("direct_recovery_evidence_consensus_prior_scale", 1.0),
        )),
        "direct_recovery_evidence_common_measure_root_mass": bool(ckpt.get(
            "direct_recovery_evidence_common_measure_root_mass",
            model_cfg.get("direct_recovery_evidence_common_measure_root_mass", False),
        )),
        "direct_recovery_absolute_feasibility_head": bool(ckpt.get(
            "direct_recovery_absolute_feasibility_head",
            model_cfg.get("direct_recovery_absolute_feasibility_head", False),
        )),
        "direct_recovery_absolute_option_margin_correction": bool(ckpt.get(
            "direct_recovery_absolute_option_margin_correction",
            model_cfg.get("direct_recovery_absolute_option_margin_correction", False),
        )),
        "direct_recovery_absolute_physical_headroom_correction": bool(ckpt.get(
            "direct_recovery_absolute_physical_headroom_correction",
            model_cfg.get("direct_recovery_absolute_physical_headroom_correction", False),
        )),
        "direct_recovery_absolute_executable_witness_correction": bool(ckpt.get(
            "direct_recovery_absolute_executable_witness_correction",
            model_cfg.get("direct_recovery_absolute_executable_witness_correction", False),
        )),
        "direct_recovery_absolute_common_witness_correction": bool(ckpt.get(
            "direct_recovery_absolute_common_witness_correction",
            model_cfg.get("direct_recovery_absolute_common_witness_correction", False),
        )),
        "direct_recovery_absolute_quantifier_witness_correction": bool(ckpt.get(
            "direct_recovery_absolute_quantifier_witness_correction",
            model_cfg.get("direct_recovery_absolute_quantifier_witness_correction", False),
        )),
        "direct_recovery_absolute_semantic_witness_correction": bool(ckpt.get(
            "direct_recovery_absolute_semantic_witness_correction",
            model_cfg.get("direct_recovery_absolute_semantic_witness_correction", False),
        )),
        "direct_recovery_semantic_witness_active_set_alignment": bool(ckpt.get(
            "direct_recovery_semantic_witness_active_set_alignment",
            model_cfg.get("direct_recovery_semantic_witness_active_set_alignment", True),
        )),
        "direct_recovery_semantic_witness_path_stop_alignment": bool(ckpt.get(
            "direct_recovery_semantic_witness_path_stop_alignment",
            model_cfg.get("direct_recovery_semantic_witness_path_stop_alignment", True),
        )),
        "direct_recovery_semantic_witness_classlocal_transport": bool(ckpt.get(
            "direct_recovery_semantic_witness_classlocal_transport",
            model_cfg.get("direct_recovery_semantic_witness_classlocal_transport", False),
        )),
        "direct_recovery_semantic_witness_route_alignment": bool(ckpt.get(
            "direct_recovery_semantic_witness_route_alignment",
            model_cfg.get("direct_recovery_semantic_witness_route_alignment", False),
        )),
        "direct_recovery_semantic_witness_reentry_alignment": bool(ckpt.get(
            "direct_recovery_semantic_witness_reentry_alignment",
            model_cfg.get("direct_recovery_semantic_witness_reentry_alignment", False),
        )),
        "direct_recovery_semantic_witness_control_projection": bool(ckpt.get(
            "direct_recovery_semantic_witness_control_projection",
            model_cfg.get("direct_recovery_semantic_witness_control_projection", False),
        )),
        "direct_recovery_semantic_witness_boundary_transport": bool(ckpt.get(
            "direct_recovery_semantic_witness_boundary_transport",
            model_cfg.get("direct_recovery_semantic_witness_boundary_transport", False),
        )),
        "direct_recovery_semantic_witness_projection_fidelity_weighting": bool(ckpt.get(
            "direct_recovery_semantic_witness_projection_fidelity_weighting",
            model_cfg.get("direct_recovery_semantic_witness_projection_fidelity_weighting", False),
        )),
        "direct_recovery_semantic_witness_active_constraint_typed_source": bool(ckpt.get(
            "direct_recovery_semantic_witness_active_constraint_typed_source",
            model_cfg.get("direct_recovery_semantic_witness_active_constraint_typed_source", False),
        )),
        "direct_recovery_semantic_witness_root_tail_source": bool(ckpt.get(
            "direct_recovery_semantic_witness_root_tail_source",
            model_cfg.get("direct_recovery_semantic_witness_root_tail_source", False),
        )),
        "direct_recovery_semantic_witness_tail_localization": bool(ckpt.get(
            "direct_recovery_semantic_witness_tail_localization",
            model_cfg.get("direct_recovery_semantic_witness_tail_localization", False),
        )),
        "direct_recovery_semantic_witness_structured_tail_field": bool(ckpt.get(
            "direct_recovery_semantic_witness_structured_tail_field",
            model_cfg.get("direct_recovery_semantic_witness_structured_tail_field", False),
        )),
        "direct_recovery_semantic_witness_signed_tail_channels": bool(ckpt.get(
            "direct_recovery_semantic_witness_signed_tail_channels",
            model_cfg.get("direct_recovery_semantic_witness_signed_tail_channels", False),
        )),
        "direct_recovery_semantic_witness_counterfactual_tail_response": bool(ckpt.get(
            "direct_recovery_semantic_witness_counterfactual_tail_response",
            model_cfg.get("direct_recovery_semantic_witness_counterfactual_tail_response", False),
        )),
        "direct_recovery_semantic_witness_demand_normalized_fidelity": bool(ckpt.get(
            "direct_recovery_semantic_witness_demand_normalized_fidelity",
            model_cfg.get("direct_recovery_semantic_witness_demand_normalized_fidelity", False),
        )),
        "direct_recovery_semantic_witness_robust_occupancy": bool(ckpt.get(
            "direct_recovery_semantic_witness_robust_occupancy",
            model_cfg.get("direct_recovery_semantic_witness_robust_occupancy", False),
        )),
        "direct_recovery_semantic_witness_soft_occupancy_disagreement": bool(ckpt.get(
            "direct_recovery_semantic_witness_soft_occupancy_disagreement",
            model_cfg.get("direct_recovery_semantic_witness_soft_occupancy_disagreement", False),
        )),
        "direct_recovery_semantic_witness_boundary_localized_occupancy_trust": bool(ckpt.get(
            "direct_recovery_semantic_witness_boundary_localized_occupancy_trust",
            model_cfg.get("direct_recovery_semantic_witness_boundary_localized_occupancy_trust", False),
        )),
        "direct_recovery_semantic_witness_history_occupancy_reachability": bool(ckpt.get(
            "direct_recovery_semantic_witness_history_occupancy_reachability",
            model_cfg.get("direct_recovery_semantic_witness_history_occupancy_reachability", False),
        )),
        "direct_recovery_semantic_witness_interaction_box_support": bool(ckpt.get(
            "direct_recovery_semantic_witness_interaction_box_support",
            model_cfg.get("direct_recovery_semantic_witness_interaction_box_support", False),
        )),
        "direct_recovery_semantic_witness_interaction_hull_support": bool(ckpt.get(
            "direct_recovery_semantic_witness_interaction_hull_support",
            model_cfg.get("direct_recovery_semantic_witness_interaction_hull_support", False),
        )),
        "direct_recovery_semantic_witness_interaction_anchor_support": bool(ckpt.get(
            "direct_recovery_semantic_witness_interaction_anchor_support",
            model_cfg.get("direct_recovery_semantic_witness_interaction_anchor_support", False),
        )),
        "direct_recovery_semantic_witness_interaction_response_support": bool(ckpt.get(
            "direct_recovery_semantic_witness_interaction_response_support",
            model_cfg.get("direct_recovery_semantic_witness_interaction_response_support", False),
        )),
    }
    actual_contract = {
        "direct_recovery_evidence_calibrator_context": bool(model.direct_recovery_evidence_calibrator_context),
        "direct_recovery_evidence_calibrator_context_source": str(model.direct_recovery_evidence_calibrator_context_source),
        "direct_recovery_evidence_interaction_hidden": int(model.direct_recovery_evidence_interaction_hidden),
        "direct_recovery_evidence_interaction_dropout": float(model.direct_recovery_evidence_interaction_dropout),
        "direct_recovery_evidence_admission_bounded": bool(model.direct_recovery_evidence_admission_bounded),
        "direct_recovery_evidence_admission_prior_detach": bool(model.direct_recovery_evidence_admission_prior_detach),
        "direct_recovery_evidence_admission_prior_mode": str(model.direct_recovery_evidence_admission_prior_mode),
        "direct_recovery_evidence_slack_temperature": float(model.direct_recovery_evidence_slack_temperature),
        "direct_recovery_evidence_slack_penalty": float(model.direct_recovery_evidence_slack_penalty),
        "direct_recovery_evidence_frontier_cap_temperature": float(model.direct_recovery_evidence_frontier_cap_temperature),
        "direct_recovery_evidence_benefit_margin_temperature": float(model.direct_recovery_evidence_benefit_margin_temperature),
        "direct_recovery_evidence_joint_reserve_temperature": float(model.direct_recovery_evidence_joint_reserve_temperature),
        "direct_recovery_evidence_frontier": bool(model.direct_recovery_evidence_frontier),
        "direct_recovery_evidence_component_prior_logit": float(model.direct_recovery_evidence_component_prior_logit),
        "direct_recovery_evidence_component_reliability": ",".join(
            f"{x:.8g}" for x in model.direct_recovery_evidence_component_reliability
        ),
        "direct_recovery_evidence_consensus_prior_scale": float(
            model.direct_recovery_evidence_consensus_prior_scale
        ),
        "direct_recovery_evidence_common_measure_root_mass": bool(
            model.direct_recovery_evidence_common_measure_root_mass
        ),
        "direct_recovery_absolute_feasibility_head": bool(
            model.direct_recovery_absolute_feasibility_head
        ),
        "direct_recovery_absolute_option_margin_correction": bool(
            model.direct_recovery_absolute_option_margin_correction
        ),
        "direct_recovery_absolute_physical_headroom_correction": bool(
            model.direct_recovery_absolute_physical_headroom_correction
        ),
        "direct_recovery_absolute_executable_witness_correction": bool(
            model.direct_recovery_absolute_executable_witness_correction
        ),
        "direct_recovery_absolute_common_witness_correction": bool(
            model.direct_recovery_absolute_common_witness_correction
        ),
        "direct_recovery_absolute_quantifier_witness_correction": bool(
            model.direct_recovery_absolute_quantifier_witness_correction
        ),
        "direct_recovery_absolute_semantic_witness_correction": bool(
            model.direct_recovery_absolute_semantic_witness_correction
        ),
        "direct_recovery_semantic_witness_active_set_alignment": bool(
            model.direct_recovery_semantic_witness_active_set_alignment
        ),
        "direct_recovery_semantic_witness_path_stop_alignment": bool(
            model.direct_recovery_semantic_witness_path_stop_alignment
        ),
        "direct_recovery_semantic_witness_classlocal_transport": bool(
            model.direct_recovery_semantic_witness_classlocal_transport
        ),
        "direct_recovery_semantic_witness_route_alignment": bool(
            model.direct_recovery_semantic_witness_route_alignment
        ),
        "direct_recovery_semantic_witness_reentry_alignment": bool(
            model.direct_recovery_semantic_witness_reentry_alignment
        ),
        "direct_recovery_semantic_witness_control_projection": bool(
            model.direct_recovery_semantic_witness_control_projection
        ),
        "direct_recovery_semantic_witness_boundary_transport": bool(
            model.direct_recovery_semantic_witness_boundary_transport
        ),
        "direct_recovery_semantic_witness_projection_fidelity_weighting": bool(
            model.direct_recovery_semantic_witness_projection_fidelity_weighting
        ),
        "direct_recovery_semantic_witness_active_constraint_typed_source": bool(
            model.direct_recovery_semantic_witness_active_constraint_typed_source
        ),
        "direct_recovery_semantic_witness_root_tail_source": bool(
            model.direct_recovery_semantic_witness_root_tail_source
        ),
        "direct_recovery_semantic_witness_tail_localization": bool(
            model.direct_recovery_semantic_witness_tail_localization
        ),
        "direct_recovery_semantic_witness_structured_tail_field": bool(
            model.direct_recovery_semantic_witness_structured_tail_field
        ),
        "direct_recovery_semantic_witness_signed_tail_channels": bool(
            model.direct_recovery_semantic_witness_signed_tail_channels
        ),
        "direct_recovery_semantic_witness_counterfactual_tail_response": bool(
            model.direct_recovery_semantic_witness_counterfactual_tail_response
        ),
        "direct_recovery_semantic_witness_demand_normalized_fidelity": bool(
            model.direct_recovery_semantic_witness_demand_normalized_fidelity
        ),
        "direct_recovery_semantic_witness_robust_occupancy": bool(
            model.direct_recovery_semantic_witness_robust_occupancy
        ),
        "direct_recovery_semantic_witness_soft_occupancy_disagreement": bool(
            model.direct_recovery_semantic_witness_soft_occupancy_disagreement
        ),
        "direct_recovery_semantic_witness_boundary_localized_occupancy_trust": bool(
            model.direct_recovery_semantic_witness_boundary_localized_occupancy_trust
        ),
        "direct_recovery_semantic_witness_history_occupancy_reachability": bool(
            model.direct_recovery_semantic_witness_history_occupancy_reachability
        ),
        "direct_recovery_semantic_witness_interaction_box_support": bool(
            model.direct_recovery_semantic_witness_interaction_box_support
        ),
        "direct_recovery_semantic_witness_interaction_hull_support": bool(
            model.direct_recovery_semantic_witness_interaction_hull_support
        ),
        "direct_recovery_semantic_witness_interaction_anchor_support": bool(
            model.direct_recovery_semantic_witness_interaction_anchor_support
        ),
        "direct_recovery_semantic_witness_interaction_response_support": bool(
            model.direct_recovery_semantic_witness_interaction_response_support
        ),
    }
    if expected_contract != actual_contract:
        raise RuntimeError(
            f"checkpoint/inference evidence contract mismatch: expected={expected_contract}, actual={actual_contract}"
        )
    cfg["model"]["inference_evidence_contract_verified"] = True
    return ModelBundle(model=model, cfg=cfg, device=device)


def teacher_prediction_from_sample(d: dict[str, Any], cfg: dict | None = None) -> Prediction:
    cfg = cfg or {}
    res = oc_mero(
        np.asarray(d["m_star"], dtype=np.float64),
        np.asarray(d["root_probs"], dtype=np.float64),
        np.asarray(d["c_star"], dtype=np.float64),
        alpha=float((cfg.get("ocmero", {}) or {}).get("alpha", 0.2)),
        beta=float((cfg.get("ocmero", {}) or {}).get("beta", 0.2)),
        option_valid=np.asarray(d["option_valid"], dtype=bool),
        root_valid=np.asarray(d["root_valid"], dtype=bool) if "root_valid" in d else None,
        use_lcvar=not bool((cfg.get("ablation", {}) or {}).get("without_lower_tail", False)),
        use_obs_kernel=not bool((cfg.get("ablation", {}) or {}).get("without_observation_kernel", False)),
        top_m=int((cfg.get("ocmero", {}) or {}).get("top_m", 8)),
    )
    return Prediction(
        r_dep=float(res.r_dep),
        r_orc=float(res.r_orc),
        gap=float(res.gap),
        q=np.asarray(res.q, dtype=np.float32),
        root_probs=np.asarray(d["root_probs"], dtype=np.float32),
        c_star=np.asarray(d["c_star"], dtype=np.float32),
        margins=np.asarray(d["m_star"], dtype=np.float32),
    )


@torch.inference_mode()
def predict_samples(
    ds: list[dict[str, Any]],
    bundle: ModelBundle | None,
    cfg: dict | None = None,
    *,
    shared_scene_features: bool = False,
    shared_geometry: bool = False,
) -> list[Prediction]:
    """Vectorized version of :func:`predict_sample`.

    ``shared_geometry`` is retained as a backward-compatible no-op flag; geometry
    canonicalization is already deterministic in ``fix_sample_geometry``.

    Closed-loop evaluation replans many times and scores every candidate prefix at
    each replan.  Calling ``predict_sample`` once per prefix is correct but pays a
    Python/GPU dispatch cost for every candidate.  This helper keeps the exact
    same inference path while batching all candidates from a single replan into
    one model call.
    """
    if not ds:
        return []
    if bundle is None:
        return [teacher_prediction_from_sample(d, cfg) for d in ds]

    xs = torch.from_numpy(samples_to_feature_matrix(ds, bundle.cfg, shared_scene=shared_scene_features)).float().to(bundle.device)
    fixed = [
        fix_sample_geometry(
            d,
            num_roots=bundle.model.num_roots,
            num_options=bundle.model.num_options,
            d_signature=int(getattr(bundle.model, "d_signature", 0)),
            d_future_signature=int(getattr(bundle.model, "d_future_signature", 0)),
        )
        for d in ds
    ]
    option_features = torch.from_numpy(np.stack([f["option_features"] for f in fixed], axis=0)).float().to(bundle.device)
    root_valid = torch.from_numpy(np.stack([f["root_valid"] for f in fixed], axis=0)).bool().to(bundle.device)
    option_valid = torch.from_numpy(np.stack([f["option_valid"] for f in fixed], axis=0)).bool().to(bundle.device)
    runtime_cfg = cfg or bundle.cfg
    bucket_ids = torch.full((len(ds),), regime_id_from_cfg(runtime_cfg), dtype=torch.long, device=bundle.device)
    # predict_samples is normally called on one complete scene-time candidate set
    # (closed-loop replan or calibration group), so all candidates share a group id.
    group_index = torch.zeros((len(ds), 1), dtype=torch.long, device=bundle.device)
    is_nominal = torch.tensor([
        1.0 if float(np.asarray(d.get("is_nominal", 0)).reshape(-1)[0]) > 0.5 else 0.0 for d in ds
    ], dtype=torch.float32, device=bundle.device)
    cphr_features = None
    if bool(getattr(bundle.model, "direct_recovery_absolute_physical_headroom_correction", False)):
        cphr_features = torch.from_numpy(np.stack([
            direct_absolute_physical_headroom_features_from_sample(d, bundle.cfg) for d in ds
        ], axis=0)).float().to(bundle.device)
    erwf_features = None
    if bool(getattr(bundle.model, "direct_recovery_absolute_executable_witness_correction", False)):
        erwf_features = torch.from_numpy(np.stack([
            direct_executable_recovery_witness_features_from_sample(
                d, bundle.cfg, num_options=bundle.model.num_options
            ) for d in ds
        ], axis=0)).float().to(bundle.device)
    common_witness_features = None
    if bool(
        getattr(bundle.model, "direct_recovery_absolute_common_witness_correction", False)
        or getattr(bundle.model, "direct_recovery_absolute_quantifier_witness_correction", False)
    ):
        common_witness_features = torch.from_numpy(np.stack([
            direct_common_recovery_witness_features_from_sample(
                d, bundle.cfg, num_options=bundle.model.num_options
            ) for d in ds
        ], axis=0)).float().to(bundle.device)
    semantic_witness_features = None
    if bool(getattr(bundle.model, "direct_recovery_absolute_semantic_witness_correction", False)):
        semantic_witness_features = torch.from_numpy(np.stack([
            direct_semantic_recovery_witness_features_from_sample(
                d, bundle.cfg, num_options=bundle.model.num_options
            ) for d in ds
        ], axis=0)).float().to(bundle.device)
    out = bundle.model(
        xs, option_features, bucket_id=bucket_ids, group_index=group_index, is_nominal=is_nominal,
        absolute_physical_headroom_features=cphr_features,
        absolute_executable_witness_features=erwf_features,
        absolute_common_witness_features=common_witness_features,
        absolute_semantic_witness_features=semantic_witness_features,
        root_valid=root_valid, option_valid=option_valid,
    )
    p = torch.softmax(out["root_logits"].masked_fill(~root_valid, -1.0e4), dim=-1)
    recovery_p = torch.softmax(
        out.get("recovery_root_logits", out["root_logits"]).masked_fill(~root_valid, -1.0e4),
        dim=-1,
    )
    r_dep, r_orc, gap, q = torch_oc_mero(
        out["margins"],
        p,
        out["c_star"],
        alpha=float((bundle.cfg.get("ocmero", {}) or {}).get("alpha", 0.2)),
        beta=float((bundle.cfg.get("ocmero", {}) or {}).get("beta", 0.2)),
        option_valid=option_valid,
        root_valid=root_valid,
        use_lcvar=not bool((bundle.cfg.get("ablation", {}) or {}).get("without_lower_tail", False)),
        use_obs_kernel=not bool((bundle.cfg.get("ablation", {}) or {}).get("without_observation_kernel", False)),
        top_m=int((bundle.cfg.get("ocmero", {}) or {}).get("top_m", 8)),
    )

    r_dep_np = r_dep.detach().cpu().numpy().astype(np.float32)
    r_orc_np = r_orc.detach().cpu().numpy().astype(np.float32)
    gap_np = gap.detach().cpu().numpy().astype(np.float32)
    q_np = q.detach().cpu().numpy().astype(np.float32)
    p_np = p.detach().cpu().numpy().astype(np.float32)
    recovery_p_np = recovery_p.detach().cpu().numpy().astype(np.float32)
    c_np = out["c_star"].detach().cpu().numpy().astype(np.float32)
    m_np = out["margins"].detach().cpu().numpy().astype(np.float32)
    direct_mean_np = None
    direct_std_np = None
    direct_opp_np = None
    direct_opp_logit_np = None
    direct_harm_np = None
    direct_harm_logit_np = None
    direct_abs_feas_np = None
    direct_abs_feas_logit_np = None
    direct_qw_best_np = None
    direct_qw_failure_np = None
    direct_qw_positive_count_np = None
    direct_qw_max_support_np = None
    direct_sw_best_np = None
    direct_sw_failure_np = None
    direct_sw_positive_count_np = None
    direct_sw_max_support_np = None
    direct_sw_best_barriers_np = None
    direct_sw_limiting_constraint_np = None
    direct_sw_classlocal_lcvar_np = None
    direct_sw_classlocal_viable_mass_np = None
    direct_sw_classlocal_support_mean_np = None
    direct_rank_np = None
    direct_delta_np = None
    direct_delta_std_np = None
    direct_component_harm_np = None
    direct_component_margins_np = None
    direct_native_certificate_np = None
    if "direct_recovery_value_logit" in out:
        direct_tensor = out["direct_recovery_value_logit"]
        if str(getattr(bundle.model, "direct_recovery_value_output", "probability")) != "score":
            direct_tensor = torch.sigmoid(direct_tensor)
        direct_mean_np = direct_tensor.detach().cpu().numpy().astype(np.float32)
        direct_std_np = torch.exp(0.5 * out["direct_recovery_value_logvar"]).detach().cpu().numpy().astype(np.float32)
        if "direct_recovery_opportunity_logit" in out:
            direct_opp_logit_np = out["direct_recovery_opportunity_logit"].detach().cpu().numpy().astype(np.float32)
            direct_opp_np = torch.sigmoid(out["direct_recovery_opportunity_logit"]).detach().cpu().numpy().astype(np.float32)
        if "direct_recovery_harm_logit" in out:
            direct_harm_logit_np = out["direct_recovery_harm_logit"].detach().cpu().numpy().astype(np.float32)
            direct_harm_np = torch.sigmoid(out["direct_recovery_harm_logit"]).detach().cpu().numpy().astype(np.float32)
        if "direct_recovery_absolute_feasibility_probability" in out:
            direct_abs_feas_np = out["direct_recovery_absolute_feasibility_probability"].detach().cpu().numpy().astype(np.float32)
            direct_abs_feas_logit_np = out["direct_recovery_absolute_feasibility_logit"].detach().cpu().numpy().astype(np.float32)
        if "direct_recovery_absolute_quantifier_best_common_viability" in out:
            direct_qw_best_np = out["direct_recovery_absolute_quantifier_best_common_viability"].detach().cpu().numpy().astype(np.float32)
            direct_qw_failure_np = out["direct_recovery_absolute_quantifier_universal_failure"].detach().cpu().numpy().astype(np.float32)
            direct_qw_positive_count_np = out["direct_recovery_absolute_quantifier_positive_option_count"].detach().cpu().numpy().astype(np.float32)
            direct_qw_max_support_np = out["direct_recovery_absolute_quantifier_max_common_support"].detach().cpu().numpy().astype(np.float32)
        if "direct_recovery_absolute_semantic_best_common_viability" in out:
            direct_sw_best_np = out["direct_recovery_absolute_semantic_best_common_viability"].detach().cpu().numpy().astype(np.float32)
            direct_sw_failure_np = out["direct_recovery_absolute_semantic_universal_failure"].detach().cpu().numpy().astype(np.float32)
            direct_sw_positive_count_np = out["direct_recovery_absolute_semantic_positive_option_count"].detach().cpu().numpy().astype(np.float32)
            direct_sw_max_support_np = out["direct_recovery_absolute_semantic_max_common_support"].detach().cpu().numpy().astype(np.float32)
            direct_sw_best_barriers_np = out["direct_recovery_absolute_semantic_best_barriers"].detach().cpu().numpy().astype(np.float32)
            direct_sw_limiting_constraint_np = out["direct_recovery_absolute_semantic_limiting_constraint"].detach().cpu().numpy().astype(np.int64)
            if "direct_recovery_absolute_semantic_classlocal_lcvar_viability" in out:
                direct_sw_classlocal_lcvar_np = out["direct_recovery_absolute_semantic_classlocal_lcvar_viability"].detach().cpu().numpy().astype(np.float32)
                direct_sw_classlocal_viable_mass_np = out["direct_recovery_absolute_semantic_classlocal_viable_root_mass"].detach().cpu().numpy().astype(np.float32)
                direct_sw_classlocal_support_mean_np = out["direct_recovery_absolute_semantic_classlocal_selected_support_mean"].detach().cpu().numpy().astype(np.float32)
        if "direct_recovery_rank_logit" in out:
            direct_rank_np = out["direct_recovery_rank_logit"].detach().cpu().numpy().astype(np.float32)
        if "direct_recovery_delta_mean" in out:
            direct_delta_np = out["direct_recovery_delta_mean"].detach().cpu().numpy().astype(np.float32)
            direct_delta_std_np = torch.exp(0.5 * out["direct_recovery_delta_logvar"]).detach().cpu().numpy().astype(np.float32)
        if "direct_recovery_evidence_component_harm_probabilities" in out:
            direct_component_harm_np = out[
                "direct_recovery_evidence_component_harm_probabilities"
            ].detach().cpu().numpy().astype(np.float32)
        if "direct_recovery_evidence_predicted_component_margins" in out:
            direct_component_margins_np = out[
                "direct_recovery_evidence_predicted_component_margins"
            ].detach().cpu().numpy().astype(np.float32)
        if "direct_recovery_evidence_native_certificate" in out:
            direct_native_certificate_np = out[
                "direct_recovery_evidence_native_certificate"
            ].detach().cpu().numpy().astype(np.float32)
    preds: list[Prediction] = []
    for i in range(len(ds)):
        preds.append(
            Prediction(
                r_dep=float(r_dep_np[i]),
                r_orc=float(r_orc_np[i]),
                gap=float(gap_np[i]),
                q=q_np[i],
                root_probs=p_np[i],
                c_star=c_np[i],
                margins=m_np[i],
                direct_recovery_value=(None if direct_mean_np is None else float(direct_mean_np[i])),
                direct_recovery_std=(None if direct_std_np is None else float(direct_std_np[i])),
                direct_recovery_opportunity=(None if direct_opp_np is None else float(direct_opp_np[i])),
                direct_recovery_opportunity_logit=(None if direct_opp_logit_np is None else float(direct_opp_logit_np[i])),
                direct_recovery_harm=(None if direct_harm_np is None else float(direct_harm_np[i])),
                direct_recovery_harm_logit=(None if direct_harm_logit_np is None else float(direct_harm_logit_np[i])),
                direct_recovery_absolute_feasibility=(None if direct_abs_feas_np is None else float(direct_abs_feas_np[i])),
                direct_recovery_absolute_feasibility_logit=(None if direct_abs_feas_logit_np is None else float(direct_abs_feas_logit_np[i])),
                direct_recovery_quantifier_best_common_viability=(None if direct_qw_best_np is None else float(direct_qw_best_np[i])),
                direct_recovery_quantifier_universal_failure=(None if direct_qw_failure_np is None else float(direct_qw_failure_np[i])),
                direct_recovery_quantifier_positive_option_count=(None if direct_qw_positive_count_np is None else float(direct_qw_positive_count_np[i])),
                direct_recovery_quantifier_max_common_support=(None if direct_qw_max_support_np is None else float(direct_qw_max_support_np[i])),
                direct_recovery_semantic_best_common_viability=(None if direct_sw_best_np is None else float(direct_sw_best_np[i])),
                direct_recovery_semantic_universal_failure=(None if direct_sw_failure_np is None else float(direct_sw_failure_np[i])),
                direct_recovery_semantic_positive_option_count=(None if direct_sw_positive_count_np is None else float(direct_sw_positive_count_np[i])),
                direct_recovery_semantic_max_common_support=(None if direct_sw_max_support_np is None else float(direct_sw_max_support_np[i])),
                direct_recovery_semantic_best_barriers=(None if direct_sw_best_barriers_np is None else direct_sw_best_barriers_np[i].copy()),
                direct_recovery_semantic_limiting_constraint=(None if direct_sw_limiting_constraint_np is None else int(direct_sw_limiting_constraint_np[i])),
                direct_recovery_semantic_classlocal_lcvar_viability=(None if direct_sw_classlocal_lcvar_np is None else float(direct_sw_classlocal_lcvar_np[i])),
                direct_recovery_semantic_classlocal_viable_root_mass=(None if direct_sw_classlocal_viable_mass_np is None else float(direct_sw_classlocal_viable_mass_np[i])),
                direct_recovery_semantic_classlocal_selected_support_mean=(None if direct_sw_classlocal_support_mean_np is None else float(direct_sw_classlocal_support_mean_np[i])),
                direct_recovery_rank=(None if direct_rank_np is None else float(direct_rank_np[i])),
                direct_recovery_delta=(None if direct_delta_np is None else float(direct_delta_np[i])),
                direct_recovery_delta_std=(None if direct_delta_std_np is None else float(direct_delta_std_np[i])),
                direct_recovery_component_harm=(
                    None if direct_component_harm_np is None else direct_component_harm_np[i].copy()
                ),
                direct_recovery_component_margins=(
                    None if direct_component_margins_np is None else direct_component_margins_np[i].copy()
                ),
                direct_recovery_native_certificate=(
                    None if direct_native_certificate_np is None else direct_native_certificate_np[i].copy()
                ),
                recovery_root_probs=recovery_p_np[i].copy(),
            )
        )
    return preds


@torch.inference_mode()
def predict_sample(d: dict[str, Any], bundle: ModelBundle | None, cfg: dict | None = None) -> Prediction:
    if bundle is None:
        return teacher_prediction_from_sample(d, cfg)
    x = torch.from_numpy(sample_to_feature(d, bundle.cfg)).float().unsqueeze(0).to(bundle.device)
    fixed = fix_sample_geometry(
        d,
        num_roots=bundle.model.num_roots,
        num_options=bundle.model.num_options,
        d_signature=int(getattr(bundle.model, "d_signature", 0)),
        d_future_signature=int(getattr(bundle.model, "d_future_signature", 0)),
    )
    option_features = torch.from_numpy(fixed["option_features"]).float().unsqueeze(0).to(bundle.device)
    root_valid = torch.from_numpy(fixed["root_valid"]).bool().unsqueeze(0).to(bundle.device)
    option_valid = torch.from_numpy(fixed["option_valid"]).bool().unsqueeze(0).to(bundle.device)
    runtime_cfg = cfg or bundle.cfg
    bucket_id = torch.tensor([regime_id_from_cfg(runtime_cfg)], dtype=torch.long, device=bundle.device)
    singleton_group = torch.zeros((1, 1), dtype=torch.long, device=bundle.device)
    singleton_nominal = torch.tensor([
        1.0 if float(np.asarray(d.get("is_nominal", 0)).reshape(-1)[0]) > 0.5 else 0.0
    ], dtype=torch.float32, device=bundle.device)
    cphr_features = None
    if bool(getattr(bundle.model, "direct_recovery_absolute_physical_headroom_correction", False)):
        cphr_features = torch.from_numpy(
            direct_absolute_physical_headroom_features_from_sample(d, bundle.cfg)
        ).float().unsqueeze(0).to(bundle.device)
    erwf_features = None
    if bool(getattr(bundle.model, "direct_recovery_absolute_executable_witness_correction", False)):
        erwf_features = torch.from_numpy(
            direct_executable_recovery_witness_features_from_sample(
                d, bundle.cfg, num_options=bundle.model.num_options
            )
        ).float().unsqueeze(0).to(bundle.device)
    common_witness_features = None
    if bool(
        getattr(bundle.model, "direct_recovery_absolute_common_witness_correction", False)
        or getattr(bundle.model, "direct_recovery_absolute_quantifier_witness_correction", False)
    ):
        common_witness_features = torch.from_numpy(
            direct_common_recovery_witness_features_from_sample(
                d, bundle.cfg, num_options=bundle.model.num_options
            )
        ).float().unsqueeze(0).to(bundle.device)
    semantic_witness_features = None
    if bool(getattr(bundle.model, "direct_recovery_absolute_semantic_witness_correction", False)):
        semantic_witness_features = torch.from_numpy(
            direct_semantic_recovery_witness_features_from_sample(
                d, bundle.cfg, num_options=bundle.model.num_options
            )
        ).float().unsqueeze(0).to(bundle.device)
    out = bundle.model(
        x, option_features, bucket_id=bucket_id, group_index=singleton_group, is_nominal=singleton_nominal,
        absolute_physical_headroom_features=cphr_features,
        absolute_executable_witness_features=erwf_features,
        absolute_common_witness_features=common_witness_features,
        absolute_semantic_witness_features=semantic_witness_features,
        root_valid=root_valid, option_valid=option_valid,
    )
    p = torch.softmax(out["root_logits"].masked_fill(~root_valid, -1.0e4), dim=-1)
    r_dep, r_orc, gap, q = torch_oc_mero(
        out["margins"],
        p,
        out["c_star"],
        alpha=float((bundle.cfg.get("ocmero", {}) or {}).get("alpha", 0.2)),
        beta=float((bundle.cfg.get("ocmero", {}) or {}).get("beta", 0.2)),
        option_valid=option_valid,
        root_valid=root_valid,
        use_lcvar=not bool((bundle.cfg.get("ablation", {}) or {}).get("without_lower_tail", False)),
        use_obs_kernel=not bool((bundle.cfg.get("ablation", {}) or {}).get("without_observation_kernel", False)),
        top_m=int((bundle.cfg.get("ocmero", {}) or {}).get("top_m", 8)),
    )
    return Prediction(
        r_dep=float(r_dep.squeeze(0).detach().cpu().item()),
        r_orc=float(r_orc.squeeze(0).detach().cpu().item()),
        gap=float(gap.squeeze(0).detach().cpu().item()),
        q=q.squeeze(0).detach().cpu().numpy().astype(np.float32),
        root_probs=p.squeeze(0).detach().cpu().numpy().astype(np.float32),
        c_star=out["c_star"].squeeze(0).detach().cpu().numpy().astype(np.float32),
        margins=out["margins"].squeeze(0).detach().cpu().numpy().astype(np.float32),
        direct_recovery_value=(None if "direct_recovery_value_logit" not in out else float((out["direct_recovery_value_logit"] if str(getattr(bundle.model, "direct_recovery_value_output", "probability")) == "score" else torch.sigmoid(out["direct_recovery_value_logit"])).squeeze(0).detach().cpu().item())),
        direct_recovery_std=(None if "direct_recovery_value_logvar" not in out else float(torch.exp(0.5 * out["direct_recovery_value_logvar"]).squeeze(0).detach().cpu().item())),
        direct_recovery_opportunity=(None if "direct_recovery_opportunity_logit" not in out else float(torch.sigmoid(out["direct_recovery_opportunity_logit"]).squeeze(0).detach().cpu().item())),
        direct_recovery_opportunity_logit=(None if "direct_recovery_opportunity_logit" not in out else float(out["direct_recovery_opportunity_logit"].squeeze(0).detach().cpu().item())),
        direct_recovery_harm=(None if "direct_recovery_harm_logit" not in out else float(torch.sigmoid(out["direct_recovery_harm_logit"]).squeeze(0).detach().cpu().item())),
        direct_recovery_harm_logit=(None if "direct_recovery_harm_logit" not in out else float(out["direct_recovery_harm_logit"].squeeze(0).detach().cpu().item())),
        direct_recovery_absolute_feasibility=(None if "direct_recovery_absolute_feasibility_probability" not in out else float(out["direct_recovery_absolute_feasibility_probability"].squeeze(0).detach().cpu().item())),
        direct_recovery_absolute_feasibility_logit=(None if "direct_recovery_absolute_feasibility_logit" not in out else float(out["direct_recovery_absolute_feasibility_logit"].squeeze(0).detach().cpu().item())),
        direct_recovery_quantifier_best_common_viability=(None if "direct_recovery_absolute_quantifier_best_common_viability" not in out else float(out["direct_recovery_absolute_quantifier_best_common_viability"].squeeze(0).detach().cpu().item())),
        direct_recovery_quantifier_universal_failure=(None if "direct_recovery_absolute_quantifier_universal_failure" not in out else float(out["direct_recovery_absolute_quantifier_universal_failure"].squeeze(0).detach().cpu().item())),
        direct_recovery_quantifier_positive_option_count=(None if "direct_recovery_absolute_quantifier_positive_option_count" not in out else float(out["direct_recovery_absolute_quantifier_positive_option_count"].squeeze(0).detach().cpu().item())),
        direct_recovery_quantifier_max_common_support=(None if "direct_recovery_absolute_quantifier_max_common_support" not in out else float(out["direct_recovery_absolute_quantifier_max_common_support"].squeeze(0).detach().cpu().item())),
        direct_recovery_semantic_best_common_viability=(None if "direct_recovery_absolute_semantic_best_common_viability" not in out else float(out["direct_recovery_absolute_semantic_best_common_viability"].squeeze(0).detach().cpu().item())),
        direct_recovery_semantic_universal_failure=(None if "direct_recovery_absolute_semantic_universal_failure" not in out else float(out["direct_recovery_absolute_semantic_universal_failure"].squeeze(0).detach().cpu().item())),
        direct_recovery_semantic_positive_option_count=(None if "direct_recovery_absolute_semantic_positive_option_count" not in out else float(out["direct_recovery_absolute_semantic_positive_option_count"].squeeze(0).detach().cpu().item())),
        direct_recovery_semantic_max_common_support=(None if "direct_recovery_absolute_semantic_max_common_support" not in out else float(out["direct_recovery_absolute_semantic_max_common_support"].squeeze(0).detach().cpu().item())),
        direct_recovery_semantic_best_barriers=(None if "direct_recovery_absolute_semantic_best_barriers" not in out else out["direct_recovery_absolute_semantic_best_barriers"].squeeze(0).detach().cpu().numpy().astype(np.float32)),
        direct_recovery_semantic_limiting_constraint=(None if "direct_recovery_absolute_semantic_limiting_constraint" not in out else int(out["direct_recovery_absolute_semantic_limiting_constraint"].squeeze(0).detach().cpu().item())),
        direct_recovery_semantic_classlocal_lcvar_viability=(None if "direct_recovery_absolute_semantic_classlocal_lcvar_viability" not in out else float(out["direct_recovery_absolute_semantic_classlocal_lcvar_viability"].squeeze(0).detach().cpu().item())),
        direct_recovery_semantic_classlocal_viable_root_mass=(None if "direct_recovery_absolute_semantic_classlocal_viable_root_mass" not in out else float(out["direct_recovery_absolute_semantic_classlocal_viable_root_mass"].squeeze(0).detach().cpu().item())),
        direct_recovery_semantic_classlocal_selected_support_mean=(None if "direct_recovery_absolute_semantic_classlocal_selected_support_mean" not in out else float(out["direct_recovery_absolute_semantic_classlocal_selected_support_mean"].squeeze(0).detach().cpu().item())),
        direct_recovery_rank=(None if "direct_recovery_rank_logit" not in out else float(out["direct_recovery_rank_logit"].squeeze(0).detach().cpu().item())),
        direct_recovery_delta=(None if "direct_recovery_delta_mean" not in out else float(out["direct_recovery_delta_mean"].squeeze(0).detach().cpu().item())),
        direct_recovery_delta_std=(None if "direct_recovery_delta_logvar" not in out else float(torch.exp(0.5 * out["direct_recovery_delta_logvar"]).squeeze(0).detach().cpu().item())),
        direct_recovery_component_harm=(
            None
            if "direct_recovery_evidence_component_harm_probabilities" not in out
            else out["direct_recovery_evidence_component_harm_probabilities"]
            .squeeze(0).detach().cpu().numpy().astype(np.float32)
        ),
        direct_recovery_component_margins=(
            None
            if "direct_recovery_evidence_predicted_component_margins" not in out
            else out["direct_recovery_evidence_predicted_component_margins"]
            .squeeze(0).detach().cpu().numpy().astype(np.float32)
        ),
        direct_recovery_native_certificate=(
            None
            if "direct_recovery_evidence_native_certificate" not in out
            else out["direct_recovery_evidence_native_certificate"]
            .squeeze(0).detach().cpu().numpy().astype(np.float32)
        ),
        recovery_root_probs=p.squeeze(0).detach().cpu().numpy().astype(np.float32),
    )
