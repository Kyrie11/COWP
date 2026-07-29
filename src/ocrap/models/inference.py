from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ocrap.algorithms.ocmero import oc_mero, torch_oc_mero
from ocrap.models.data import OPTION_FEATURE_DIM, fix_sample_geometry, sample_to_feature, samples_to_feature_matrix
from ocrap.models.ocrap import OCRAPModel


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
    direct_recovery_rank: float | None = None
    direct_recovery_delta: float | None = None
    direct_recovery_delta_std: float | None = None


@dataclass
class ModelBundle:
    model: OCRAPModel
    cfg: dict[str, Any]
    device: torch.device


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


@torch.no_grad()
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
    runtime_cfg = cfg or bundle.cfg
    bucket_ids = torch.full((len(ds),), regime_id_from_cfg(runtime_cfg), dtype=torch.long, device=bundle.device)
    # predict_samples is normally called on one complete scene-time candidate set
    # (closed-loop replan or calibration group), so all candidates share a group id.
    group_index = torch.zeros((len(ds), 1), dtype=torch.long, device=bundle.device)
    is_nominal = torch.tensor([
        1.0 if float(np.asarray(d.get("is_nominal", 0)).reshape(-1)[0]) > 0.5 else 0.0 for d in ds
    ], dtype=torch.float32, device=bundle.device)
    out = bundle.model(
        xs, option_features, bucket_id=bucket_ids, group_index=group_index, is_nominal=is_nominal
    )
    root_valid = torch.from_numpy(np.stack([f["root_valid"] for f in fixed], axis=0)).bool().to(bundle.device)
    p = torch.softmax(out["root_logits"].masked_fill(~root_valid, -1.0e4), dim=-1)
    option_valid = torch.from_numpy(np.stack([f["option_valid"] for f in fixed], axis=0)).bool().to(bundle.device)
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
    c_np = out["c_star"].detach().cpu().numpy().astype(np.float32)
    m_np = out["margins"].detach().cpu().numpy().astype(np.float32)
    direct_mean_np = None
    direct_std_np = None
    direct_opp_np = None
    direct_opp_logit_np = None
    direct_harm_np = None
    direct_harm_logit_np = None
    direct_rank_np = None
    direct_delta_np = None
    direct_delta_std_np = None
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
        if "direct_recovery_rank_logit" in out:
            direct_rank_np = out["direct_recovery_rank_logit"].detach().cpu().numpy().astype(np.float32)
        if "direct_recovery_delta_mean" in out:
            direct_delta_np = out["direct_recovery_delta_mean"].detach().cpu().numpy().astype(np.float32)
            direct_delta_std_np = torch.exp(0.5 * out["direct_recovery_delta_logvar"]).detach().cpu().numpy().astype(np.float32)
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
                direct_recovery_rank=(None if direct_rank_np is None else float(direct_rank_np[i])),
                direct_recovery_delta=(None if direct_delta_np is None else float(direct_delta_np[i])),
                direct_recovery_delta_std=(None if direct_delta_std_np is None else float(direct_delta_std_np[i])),
            )
        )
    return preds


@torch.no_grad()
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
    runtime_cfg = cfg or bundle.cfg
    bucket_id = torch.tensor([regime_id_from_cfg(runtime_cfg)], dtype=torch.long, device=bundle.device)
    singleton_group = torch.zeros((1, 1), dtype=torch.long, device=bundle.device)
    singleton_nominal = torch.tensor([
        1.0 if float(np.asarray(d.get("is_nominal", 0)).reshape(-1)[0]) > 0.5 else 0.0
    ], dtype=torch.float32, device=bundle.device)
    out = bundle.model(
        x, option_features, bucket_id=bucket_id, group_index=singleton_group, is_nominal=singleton_nominal
    )
    root_valid = torch.from_numpy(fixed["root_valid"]).bool().unsqueeze(0).to(bundle.device)
    p = torch.softmax(out["root_logits"].masked_fill(~root_valid, -1.0e4), dim=-1)
    option_valid = torch.from_numpy(fixed["option_valid"]).bool().unsqueeze(0).to(bundle.device)
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
        direct_recovery_rank=(None if "direct_recovery_rank_logit" not in out else float(out["direct_recovery_rank_logit"].squeeze(0).detach().cpu().item())),
        direct_recovery_delta=(None if "direct_recovery_delta_mean" not in out else float(out["direct_recovery_delta_mean"].squeeze(0).detach().cpu().item())),
        direct_recovery_delta_std=(None if "direct_recovery_delta_logvar" not in out else float(torch.exp(0.5 * out["direct_recovery_delta_logvar"]).squeeze(0).detach().cpu().item())),
    )
