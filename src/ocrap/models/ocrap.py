from __future__ import annotations

import torch
from torch import nn

from ocrap.algorithms.ocmero import torch_oc_mero
from ocrap.v48_74_signed_viability import enabled as _v48_74_signed_viability_enabled
from ocrap.algorithms.lcv import torch_normalize_weights, torch_weighted_lcvar, torch_weighted_lcvar_influence
from .encoders import FlatFeatureLayout, MLPEncoder, StructuredTokenEncoder


class RecoverySetTournament(nn.Module):
    """Permutation-equivariant recovery-only set ranker.

    The candidate-level value score is deliberately excluded.  Each recovery
    token is projected from nominal-relative features, contextualised against
    the other recovery candidates in the same scene-time group, and scored by
    a shared head.  Nominal is pinned to zero and is not part of the tournament.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        hidden_dim = max(int(num_heads), int(hidden_dim))
        if hidden_dim % int(num_heads) != 0:
            hidden_dim = int(num_heads) * ((hidden_dim + int(num_heads) - 1) // int(num_heads))
        self.hidden_dim = int(hidden_dim)
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, int(num_heads), dropout=float(dropout), batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.score = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.score.weight)
        nn.init.zeros_(self.score.bias)

    def encode(
        self,
        relative_features: torch.Tensor,
        group_index: torch.Tensor | None,
        is_nominal: torch.Tensor | None,
    ) -> torch.Tensor:
        """Return candidate-set context without changing proposal semantics.

        Nominal rows remain zero.  Recovery rows contain the frozen tournament
        representation used by the proposal head, which is substantially lower
        dimensional and more data-efficient than the raw relative feature vector.
        """
        context = relative_features.new_zeros((relative_features.shape[0], self.hidden_dim))
        if group_index is None or is_nominal is None or relative_features.shape[0] <= 1:
            return context
        groups = group_index.to(device=relative_features.device)
        groups = groups.reshape(-1, 1) if groups.dim() == 1 else groups.reshape(groups.shape[0], -1)
        nominal_mask = is_nominal.to(device=relative_features.device).reshape(-1) > 0.5
        if groups.shape[0] != relative_features.shape[0] or nominal_mask.shape[0] != relative_features.shape[0]:
            return context
        for key in torch.unique(groups, dim=0):
            idx = torch.where((groups == key.unsqueeze(0)).all(dim=1))[0]
            recs = idx[~nominal_mask[idx]]
            if recs.numel() == 0:
                continue
            recovery_features = relative_features.index_select(0, recs)
            token = self.input_proj(self.input_norm(recovery_features)).unsqueeze(0)
            attended, _ = self.attn(token, token, token, need_weights=False)
            token = self.norm1(token + attended)
            token = self.norm2(token + self.ffn(token)).squeeze(0)
            context.index_copy_(0, recs, token.to(dtype=context.dtype))
        return context

    def score_from_context(
        self,
        context: torch.Tensor,
        group_index: torch.Tensor | None,
        is_nominal: torch.Tensor | None,
    ) -> torch.Tensor:
        scores = context.new_zeros((context.shape[0],))
        if group_index is None or is_nominal is None or context.shape[0] <= 1:
            return scores
        groups = group_index.to(device=context.device)
        groups = groups.reshape(-1, 1) if groups.dim() == 1 else groups.reshape(groups.shape[0], -1)
        nominal_mask = is_nominal.to(device=context.device).reshape(-1) > 0.5
        if groups.shape[0] != context.shape[0] or nominal_mask.shape[0] != context.shape[0]:
            return scores
        for key in torch.unique(groups, dim=0):
            idx = torch.where((groups == key.unsqueeze(0)).all(dim=1))[0]
            recs = idx[~nominal_mask[idx]]
            if recs.numel() == 0:
                continue
            group_scores = self.score(context.index_select(0, recs)).squeeze(-1)
            if group_scores.numel() > 1:
                group_scores = group_scores - group_scores.mean()
            scores.index_copy_(0, recs, group_scores.to(dtype=scores.dtype))
        return scores

    def forward(
        self,
        relative_features: torch.Tensor,
        group_index: torch.Tensor | None,
        is_nominal: torch.Tensor | None,
    ) -> torch.Tensor:
        context = self.encode(relative_features, group_index, is_nominal)
        return self.score_from_context(context, group_index, is_nominal)


class ObservationConditionedActionFrontierBridge(nn.Module):
    """Low-rank, regime-agnostic action-by-observation interaction.

    The scene path cannot produce a context by itself: every output term is
    multiplicatively gated by a candidate-minus-nominal executable action.  The
    raw signed action path preserves magnitude, while a magnitude-gated direction
    path improves conditioning.  With a zero action difference the output is
    exactly zero, including in finite precision.
    """

    def __init__(self, action_dim: int, observation_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.action_dim = int(action_dim)
        self.observation_dim = int(observation_dim)
        self.hidden_dim = max(8, int(hidden_dim))
        self.action_raw = nn.Linear(self.action_dim, self.hidden_dim, bias=False)
        self.action_direction_norm = nn.LayerNorm(self.action_dim, elementwise_affine=False)
        self.action_direction = nn.Linear(self.action_dim, self.hidden_dim, bias=False)
        self.observation = nn.Sequential(
            nn.LayerNorm(self.observation_dim),
            nn.Linear(self.observation_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.output = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.hidden_dim, bias=False),
            nn.GELU(),
            nn.Dropout(float(max(0.0, dropout))),
            nn.Linear(self.hidden_dim, self.hidden_dim, bias=False),
        )

    def forward(self, action_relative: torch.Tensor, nominal_observation: torch.Tensor) -> torch.Tensor:
        if action_relative.shape[-1] != self.action_dim:
            raise ValueError(
                f"OCAF action dimension mismatch: got {action_relative.shape[-1]}, expected {self.action_dim}"
            )
        if nominal_observation.shape[-1] != self.observation_dim:
            raise ValueError(
                "OCAF observation dimension mismatch: "
                f"got {nominal_observation.shape[-1]}, expected {self.observation_dim}"
            )
        # Compute the magnitude gate in float32 and clamp before sqrt.  Exact
        # nominal rows have a zero action difference; sqrt(x) has an undefined
        # derivative at x=0 and can inject NaNs when the raw context is not
        # detached (for example in diagnostic or future joint-training runs).
        # The clamp does not violate the zero-action contract because both
        # bias-free action projections are exactly zero at a zero input.
        scale_sq = action_relative.float().square().mean(dim=-1, keepdim=True)
        scale = scale_sq.clamp_min(1.0e-12).sqrt().to(dtype=action_relative.dtype)
        raw = self.action_raw(action_relative)
        direction = self.action_direction(self.action_direction_norm(action_relative)) * scale
        action = raw + direction
        scene = torch.tanh(self.observation(nominal_observation))
        return self.output(torch.cat([action, action * scene], dim=-1))


class DualObservationConditionedActionFrontierBridge(nn.Module):
    """Task-decoupled OCAF contexts with identical continuous inputs.

    Benefit and harm see the same regime-free candidate-minus-nominal action and
    nominal observation, but they do not share trainable interaction parameters.
    This prevents dense harm gradients from rotating the sparse benefit context
    (and vice versa) while preserving the single continuous physical policy.
    """

    def __init__(self, action_dim: int, observation_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.benefit = ObservationConditionedActionFrontierBridge(
            action_dim, observation_dim, hidden_dim, dropout
        )
        self.harm = ObservationConditionedActionFrontierBridge(
            action_dim, observation_dim, hidden_dim, dropout
        )

    def forward(
        self, action_relative: torch.Tensor, nominal_observation: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.benefit(action_relative, nominal_observation),
            self.harm(action_relative, nominal_observation),
        )


class _ZeroObservationConditionedActionFrontierBranch(nn.Module):
    """Parameter-free placeholder for a globally unsupported harm coordinate."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = int(hidden_dim)

    def forward(self, action_relative: torch.Tensor, nominal_observation: torch.Tensor) -> torch.Tensor:
        del nominal_observation
        return action_relative.new_zeros((action_relative.shape[0], self.hidden_dim))


class FactorizedObservationConditionedActionFrontierBridge(nn.Module):
    """Task- and component-decoupled continuous OCAF contexts.

    v48.40 established that benefit/harm interaction decoupling improves rare
    safety-frontier discrimination.  The remaining harm head still multiplexes
    physically different veto factors (DRS, deployability, gap, ...), whose
    prevalences and observation dependencies differ substantially.  FCFR gives
    every harm component its own observation-conditioned action interaction
    bridge while preserving *identical* regime-free inputs and the same exact
    non-compensatory max-veto downstream.

    No regime id, bucket router, regime-specific threshold, or case policy is
    introduced.  A zero candidate-minus-nominal action produces exact zeros in
    every branch.
    """

    def __init__(
        self,
        action_dim: int,
        observation_dim: int,
        hidden_dim: int,
        dropout: float,
        component_count: int,
        component_reliability: tuple[float, ...] | None = None,
    ):
        super().__init__()
        self.component_count = int(component_count)
        reliability = list(component_reliability or ())
        if len(reliability) < self.component_count:
            reliability.extend([1.0] * (self.component_count - len(reliability)))
        self.component_reliability = tuple(float(x) for x in reliability[: self.component_count])
        self.benefit = ObservationConditionedActionFrontierBridge(
            action_dim, observation_dim, hidden_dim, dropout
        )
        self.harm_components = nn.ModuleList(
            [
                ObservationConditionedActionFrontierBridge(
                    action_dim, observation_dim, hidden_dim, dropout
                )
                if self.component_reliability[i] > 0.0
                else _ZeroObservationConditionedActionFrontierBranch(hidden_dim)
                for i in range(self.component_count)
            ]
        )

    def forward(
        self, action_relative: torch.Tensor, nominal_observation: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        benefit = self.benefit(action_relative, nominal_observation)
        harm = torch.stack(
            [branch(action_relative, nominal_observation) for branch in self.harm_components],
            dim=1,
        )
        return benefit, harm


class OCRAPModel(nn.Module):
    """Neural OC-RAP model with a learned root-query decoder.

    The previous implementation predicted all roots from one global scene
    embedding.  That was useful as a compact smoke model but did not match the
    paper's pipeline, where learned root queries cross-attend to scene-prefix
    tokens and each root obtains its own observation embedding and recovery
    margins.  This module keeps the MLP fallback, but the default structured
    path now implements the paper-aligned decoder:

    scene/prefix tokens -> K learned root queries -> root probabilities,
    post-prefix observation embeddings, recovery signatures, and per-option
    margins conditioned on recovery-option embeddings.
    """

    def __init__(
        self,
        input_dim: int,
        num_roots: int = 8,
        num_options: int = 24,
        d_model: int = 128,
        d_obs: int = 64,
        tau_obs: float = 1.0,
        encoder_type: str = "mlp",
        feature_layout: dict | None = None,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        d_signature: int = 0,
        d_future_signature: int = 0,
        option_feature_dim: int = 0,
        direct_recovery_value_head: bool = False,
        direct_recovery_value_pooling: str = "scene",
        direct_recovery_value_output: str = "probability",
        direct_recovery_value_regime_conditioning: bool = False,
        direct_recovery_value_num_regimes: int = 4,
        direct_recovery_value_regime_dim: int = 16,
        direct_recovery_opportunity_head: bool = False,
        direct_recovery_harm_head: bool = False,
        direct_recovery_value_experts: bool = False,
        direct_recovery_value_num_experts: int = 2,
        direct_recovery_value_expert_routing: str = "bucket",
        direct_recovery_value_router_temperature: float = 1.0,
        direct_recovery_value_router_pooling: str = "candidate",
        direct_recovery_expert_disagreement_penalty: float = 0.5,
        direct_recovery_set_context: bool = False,
        direct_recovery_set_context_hidden: int = 192,
        direct_recovery_set_context_dropout: float = 0.1,
        direct_recovery_preference_head: bool = False,
        direct_recovery_preference_hidden: int = 96,
        direct_recovery_preference_dropout: float = 0.05,
        direct_recovery_preference_context: bool = False,
        direct_recovery_preference_context_hidden: int = 128,
        direct_recovery_relative_features_include_absolute: bool = True,
        direct_recovery_set_tournament: bool = False,
        direct_recovery_set_tournament_hidden: int = 48,
        direct_recovery_set_tournament_heads: int = 4,
        direct_recovery_set_tournament_dropout: float = 0.05,
        direct_recovery_set_tournament_replace_base: bool = True,
        direct_recovery_delta_head: bool = False,
        direct_recovery_delta_regime_experts: bool = False,
        direct_recovery_delta_policy_features: bool = False,
        direct_recovery_delta_hidden: int = 128,
        direct_recovery_delta_dropout: float = 0.05,
        direct_recovery_delta_initial_logvar: float = -4.605170186,
        direct_recovery_delta_mode: str = "gaussian",
        direct_recovery_evidence_calibrator: bool = False,
        direct_recovery_evidence_calibrator_hidden: int = 8,
        direct_recovery_evidence_calibrator_scale: float = 0.25,
        direct_recovery_evidence_calibrator_mode: str = "center_width",
        direct_recovery_evidence_calibrator_context: bool = False,
        direct_recovery_evidence_calibrator_context_detach: bool = True,
        direct_recovery_evidence_calibrator_context_source: str = "relative",
        direct_recovery_evidence_interaction_hidden: int = 64,
        direct_recovery_evidence_interaction_dropout: float = 0.05,
        direct_recovery_evidence_dual_interaction_bridge: bool = False,
        direct_recovery_evidence_factorized_harm_interaction: bool = False,
        direct_recovery_evidence_partial_pool_harm_residual: bool = False,
        direct_recovery_evidence_partial_pool_harm_residual_scale: float = 0.50,
        direct_recovery_evidence_rank_benefit_skip: bool = False,
        direct_recovery_evidence_rank_benefit_gain_init: float = 1.0,
        direct_recovery_evidence_postprefix_obs_transport_benefit: bool = False,
        direct_recovery_evidence_postprefix_obs_transport_harm: bool = False,
        direct_recovery_evidence_postprefix_obs_transport_scale: float = 1.0,
        direct_recovery_evidence_roct_benefit: bool = False,
        direct_recovery_evidence_roct_deployability: bool = False,
        direct_recovery_evidence_roct_scale: float = 1.0,
        direct_recovery_evidence_roct_alpha: float = 0.2,
        direct_recovery_evidence_roct_beta: float = 0.2,
        direct_recovery_evidence_roct_top_m: int = 8,
        direct_recovery_evidence_roct_option_temperature: float = 0.35,
        direct_recovery_evidence_common_measure_root_mass: bool = False,
        direct_recovery_absolute_feasibility_head: bool = False,
        direct_recovery_absolute_option_margin_correction: bool = False,
        direct_recovery_absolute_physical_headroom_correction: bool = False,
        direct_recovery_absolute_executable_witness_correction: bool = False,
        direct_recovery_absolute_common_witness_correction: bool = False,
        direct_recovery_absolute_quantifier_witness_correction: bool = False,
        direct_recovery_absolute_semantic_witness_correction: bool = False,
        direct_recovery_semantic_witness_active_set_alignment: bool = True,
        direct_recovery_semantic_witness_path_stop_alignment: bool = True,
        direct_recovery_semantic_witness_classlocal_transport: bool = False,
        direct_recovery_semantic_witness_route_alignment: bool = False,
        direct_recovery_semantic_witness_reentry_alignment: bool = False,
        direct_recovery_semantic_witness_control_projection: bool = False,
        direct_recovery_semantic_witness_boundary_transport: bool = False,
        direct_recovery_semantic_witness_projection_fidelity_weighting: bool = False,
        direct_recovery_semantic_witness_active_constraint_typed_source: bool = False,
        direct_recovery_semantic_witness_root_tail_source: bool = False,
        direct_recovery_semantic_witness_tail_localization: bool = False,
        direct_recovery_semantic_witness_structured_tail_field: bool = False,
        direct_recovery_semantic_witness_signed_tail_channels: bool = False,
        direct_recovery_semantic_witness_counterfactual_tail_response: bool = False,
        direct_recovery_semantic_witness_demand_normalized_fidelity: bool = False,
        direct_recovery_semantic_witness_robust_occupancy: bool = False,
        direct_recovery_semantic_witness_soft_occupancy_disagreement: bool = False,
        direct_recovery_semantic_witness_boundary_localized_occupancy_trust: bool = False,
        direct_recovery_semantic_witness_history_occupancy_reachability: bool = False,
        direct_recovery_semantic_witness_interaction_box_support: bool = False,
        direct_recovery_semantic_witness_interaction_hull_support: bool = False,
        direct_recovery_semantic_witness_interaction_anchor_support: bool = False,
        direct_recovery_semantic_witness_interaction_response_support: bool = False,
        direct_recovery_evidence_native_certificate_preservation: bool = False,
        direct_recovery_evidence_native_margin_complete_preservation: bool = False,
        direct_recovery_evidence_native_advantage_preservation: bool = False,
        direct_recovery_evidence_native_exact_advantage_preservation: bool = False,
        direct_recovery_evidence_native_boundary_complete_advantage_preservation: bool = False,
        direct_recovery_evidence_physical_student_drs: bool = False,
        direct_recovery_evidence_native_drs_tolerance: float = 0.05,
        direct_recovery_evidence_native_deployability_tolerance: float = 0.05,
        direct_recovery_evidence_native_dep_boundary_aligned: bool = False,
        direct_recovery_evidence_native_gap_tolerance: float = 0.05,
        direct_recovery_evidence_native_positive_gain: float = 0.015,
        direct_recovery_evidence_calibrator_shared: bool = False,
        direct_recovery_evidence_calibrator_regime_scale: float = 0.25,
        direct_recovery_evidence_unified_experts: bool = False,
        direct_recovery_evidence_component_heads: bool = False,
        direct_recovery_evidence_component_count: int = 3,
        direct_recovery_evidence_component_scale: float = 6.0,
        direct_recovery_evidence_benefit_residual_scale: float = 1.0,
        direct_recovery_evidence_unbounded_benefit_factor: bool = False,
        direct_recovery_evidence_unbounded_harm_factors: bool = False,
        direct_recovery_evidence_component_reliability: str | tuple[float, ...] | None = "",
        direct_recovery_evidence_concord: bool = False,
        direct_recovery_evidence_consensus_disagreement_penalty: float = 0.15,
        direct_recovery_evidence_consensus_prior_scale: float = 1.0,
        direct_recovery_evidence_admission_head: bool = False,
        direct_recovery_evidence_admission_scale: float = 2.0,
        direct_recovery_evidence_admission_bounded: bool = True,
        direct_recovery_evidence_admission_prior_detach: bool = True,
        direct_recovery_evidence_admission_prior_mode: str = "risk_centered",
        direct_recovery_evidence_slack_temperature: float = 0.025,
        direct_recovery_evidence_slack_penalty: float = 1.0,
        direct_recovery_evidence_frontier_cap_temperature: float = 0.10,
        direct_recovery_evidence_benefit_margin_temperature: float = 0.025,
        direct_recovery_evidence_joint_reserve_temperature: float = 0.025,
        direct_recovery_evidence_reserve_factor_alignment: bool = False,
        direct_recovery_evidence_frontier: bool = False,
        direct_recovery_evidence_component_prior_logit: float = -2.0,
    ):
        super().__init__()
        self.num_roots = int(num_roots)
        self.num_options = int(num_options)
        self.d_obs = int(d_obs)
        self.tau_obs = float(max(tau_obs, 1e-6))
        self.encoder_type = str(encoder_type)
        self.feature_layout = feature_layout or {}
        self.d_model = int(d_model)
        self.d_signature = int(d_signature)
        self.d_future_signature = int(d_future_signature)
        self.option_feature_dim = int(option_feature_dim)
        self.direct_recovery_value_head = bool(direct_recovery_value_head)
        self.direct_recovery_value_pooling = str(direct_recovery_value_pooling or "scene").strip().lower()
        self.direct_recovery_value_output = str(direct_recovery_value_output or "probability").strip().lower()
        self.direct_recovery_value_regime_conditioning = bool(direct_recovery_value_regime_conditioning)
        self.direct_recovery_value_num_regimes = max(1, int(direct_recovery_value_num_regimes))
        self.direct_recovery_value_regime_dim = max(1, int(direct_recovery_value_regime_dim))
        self.direct_recovery_opportunity_head = bool(direct_recovery_opportunity_head)
        self.direct_recovery_harm_head = bool(direct_recovery_harm_head)
        self.direct_recovery_value_experts = bool(direct_recovery_value_experts)
        self.direct_recovery_value_num_experts = max(1, int(direct_recovery_value_num_experts))
        self.direct_recovery_value_expert_routing = str(
            direct_recovery_value_expert_routing or "bucket"
        ).strip().lower()
        self.direct_recovery_value_router_temperature = float(
            max(direct_recovery_value_router_temperature, 1.0e-3)
        )
        self.direct_recovery_value_router_pooling = str(
            direct_recovery_value_router_pooling or "candidate"
        ).strip().lower()
        self.direct_recovery_expert_disagreement_penalty = float(
            max(direct_recovery_expert_disagreement_penalty, 0.0)
        )
        self.direct_recovery_set_context = bool(direct_recovery_set_context)
        self.direct_recovery_set_context_hidden = max(16, int(direct_recovery_set_context_hidden))
        self.direct_recovery_set_context_dropout = float(max(0.0, direct_recovery_set_context_dropout))
        self.direct_recovery_preference_head = bool(direct_recovery_preference_head)
        self.direct_recovery_preference_hidden = max(16, int(direct_recovery_preference_hidden))
        self.direct_recovery_preference_dropout = float(max(0.0, direct_recovery_preference_dropout))
        self.direct_recovery_preference_context = bool(direct_recovery_preference_context)
        self.direct_recovery_preference_context_hidden = max(16, int(direct_recovery_preference_context_hidden))
        self.direct_recovery_relative_features_include_absolute = bool(direct_recovery_relative_features_include_absolute)
        self.direct_recovery_set_tournament = bool(direct_recovery_set_tournament)
        self.direct_recovery_set_tournament_hidden = max(16, int(direct_recovery_set_tournament_hidden))
        self.direct_recovery_set_tournament_heads = max(1, int(direct_recovery_set_tournament_heads))
        self.direct_recovery_set_tournament_dropout = float(max(0.0, direct_recovery_set_tournament_dropout))
        self.direct_recovery_set_tournament_replace_base = bool(direct_recovery_set_tournament_replace_base)
        self.direct_recovery_delta_head = bool(direct_recovery_delta_head)
        self.direct_recovery_delta_regime_experts = bool(direct_recovery_delta_regime_experts)
        self.direct_recovery_delta_policy_features = bool(direct_recovery_delta_policy_features)
        self.direct_recovery_delta_hidden = max(16, int(direct_recovery_delta_hidden))
        self.direct_recovery_delta_dropout = float(max(0.0, direct_recovery_delta_dropout))
        self.direct_recovery_delta_initial_logvar = float(direct_recovery_delta_initial_logvar)
        self.direct_recovery_delta_mode = str(direct_recovery_delta_mode or "gaussian").strip().lower()
        self.direct_recovery_evidence_calibrator = bool(direct_recovery_evidence_calibrator)
        self.direct_recovery_evidence_calibrator_hidden = max(4, int(direct_recovery_evidence_calibrator_hidden))
        self.direct_recovery_evidence_calibrator_scale = float(max(0.0, direct_recovery_evidence_calibrator_scale))
        self.direct_recovery_evidence_calibrator_mode = str(
            direct_recovery_evidence_calibrator_mode or "center_width"
        ).strip().lower()
        self.direct_recovery_evidence_calibrator_context = bool(direct_recovery_evidence_calibrator_context)
        self.direct_recovery_evidence_calibrator_context_detach = bool(
            direct_recovery_evidence_calibrator_context_detach
        )
        self.direct_recovery_evidence_calibrator_context_source = str(
            direct_recovery_evidence_calibrator_context_source or "relative"
        ).strip().lower()
        self.direct_recovery_evidence_interaction_hidden = max(
            8, int(direct_recovery_evidence_interaction_hidden)
        )
        self.direct_recovery_evidence_interaction_dropout = float(
            max(0.0, direct_recovery_evidence_interaction_dropout)
        )
        # v48.40 DCFR: decouple observation-conditioned interaction parameters
        # between sparse benefit and dense harm tasks without any regime input.
        self.direct_recovery_evidence_dual_interaction_bridge = bool(
            direct_recovery_evidence_dual_interaction_bridge
        )
        # v48.41 FCFR.  Harm factors have different physical semantics and
        # label prevalences; sharing the same interaction representation can
        # recreate the negative transfer that v48.40 removed between benefit
        # and aggregate harm.  This option is meaningful only with a physical
        # interaction context and component heads, but keeping it as a model
        # flag makes the checkpoint contract explicit and auditable.
        self.direct_recovery_evidence_factorized_harm_interaction = bool(
            direct_recovery_evidence_factorized_harm_interaction
        )
        # v48.42 HPFR: retain the empirically stronger shared harm OCAF/trunk,
        # then allow only a small zero-initialised component-specific correction
        # from a detached copy of the same continuous evidence.  This is partial
        # pooling: physical factors may refine their frontier without rotating
        # the shared interaction representation that v48.41 full factorisation
        # showed was valuable.  No regime id or regime-specific policy exists.
        self.direct_recovery_evidence_partial_pool_harm_residual = bool(
            direct_recovery_evidence_partial_pool_harm_residual
        )
        self.direct_recovery_evidence_partial_pool_harm_residual_scale = float(
            max(0.0, direct_recovery_evidence_partial_pool_harm_residual_scale)
        )
        self.direct_recovery_evidence_rank_benefit_skip = bool(
            direct_recovery_evidence_rank_benefit_skip
        )
        self.direct_recovery_evidence_rank_benefit_gain_init = float(
            max(1.0e-4, direct_recovery_evidence_rank_benefit_gain_init)
        )
        # v48.43 POET: expose the candidate-induced post-prefix observation
        # equivalence geometry to the evidence bridge.  This is not a regime
        # router: every candidate in every evaluation slice is mapped through
        # the same frozen root/observation model and then expressed relative to
        # the unique nominal action in its scene-time group.
        self.direct_recovery_evidence_postprefix_obs_transport_benefit = bool(
            direct_recovery_evidence_postprefix_obs_transport_benefit
        )
        self.direct_recovery_evidence_postprefix_obs_transport_harm = bool(
            direct_recovery_evidence_postprefix_obs_transport_harm
        )
        self.direct_recovery_evidence_postprefix_obs_transport_scale = float(
            max(0.0, direct_recovery_evidence_postprefix_obs_transport_scale)
        )
        self.direct_recovery_evidence_postprefix_obs_signature_dim = 4
        # v48.44 ROCT (Recovery-Option Compatibility Transport): unlike POET,
        # which summarizes only post-prefix observation ambiguity, ROCT exposes
        # whether observation-equivalent latent roots actually share a recovery
        # option.  The structural teacher is frozen and candidate-relative; the
        # learned correction is zero-init and never consumes a regime label.
        self.direct_recovery_evidence_roct_benefit = bool(
            direct_recovery_evidence_roct_benefit
        )
        self.direct_recovery_evidence_roct_deployability = bool(
            direct_recovery_evidence_roct_deployability
        )
        self.direct_recovery_evidence_roct_scale = float(
            max(0.0, direct_recovery_evidence_roct_scale)
        )
        self.direct_recovery_evidence_roct_alpha = float(direct_recovery_evidence_roct_alpha)
        self.direct_recovery_evidence_roct_beta = float(direct_recovery_evidence_roct_beta)
        self.direct_recovery_evidence_roct_top_m = int(direct_recovery_evidence_roct_top_m)
        self.direct_recovery_evidence_roct_option_temperature = float(
            max(1.0e-4, direct_recovery_evidence_roct_option_temperature)
        )
        # v48.57 CMRI (Common-Measure Root Invariance): a scene-time candidate
        # set is a collection of counterfactual actions under one observed latent
        # world distribution.  OC-MERO comparisons therefore use the unique
        # nominal candidate's predicted root logits as a shared integration
        # measure while keeping every candidate's observation kernel and recovery
        # margins action-specific.  The raw root logits are never retrained or
        # recalibrated by this mechanism; it is a zero-parameter projection at the
        # recovery aggregation boundary.
        self.direct_recovery_evidence_common_measure_root_mass = bool(
            direct_recovery_evidence_common_measure_root_mass
        )
        # v48.58 RIFA: absolute feasibility is a logically separate deployment
        # predicate from candidate-vs-nominal recovery improvement.  The readout
        # consumes only frozen absolute OC-MERO/ROCT coordinates; no regime id,
        # relative score, threshold search, or proposal-set statistic enters it.
        self.direct_recovery_absolute_feasibility_head = bool(
            direct_recovery_absolute_feasibility_head
        )
        # v48.59 ORFC (Option-Resolved Feasibility Correction): instead of
        # reclassifying an 8-D compressed absolute summary, learn only one
        # global correction per recovery option and re-run the unchanged
        # OC-MERO absolute source.  The correction is regime-agnostic, never
        # sees candidate-vs-nominal evidence, and is zero-initialized so the
        # initial predicate is exactly the v48.58-B native R_dep=0 boundary.
        self.direct_recovery_absolute_option_margin_correction = bool(
            direct_recovery_absolute_option_margin_correction
        )
        # v48.60 CPHR (Contextual Physical Headroom Reserve): source-only,
        # regime-agnostic correction of the native absolute deployability logit
        # using signed continuous headroom computed from the executable prefix
        # and currently observed agents.  The six weights are zero-initialized,
        # non-negative/bounded at use time, and there is no free bias, so epoch 0
        # is exactly the v48.58-B native R_dep=0 source rather than a threshold shift.
        self.direct_recovery_absolute_physical_headroom_correction = bool(
            direct_recovery_absolute_physical_headroom_correction
        )
        # v48.61 ERWF (Executable Recovery Witness Field): a candidate x recovery-
        # option continuation field, computed only from the executable prefix,
        # deterministic recovery controller and current observation.  The field
        # corrects frozen root-option margins before the unchanged OC-MERO source.
        self.direct_recovery_absolute_executable_witness_correction = bool(
            direct_recovery_absolute_executable_witness_correction
        )
        # v48.62 OC-CWRF: finite-time physical recovery barrier + observation-
        # consistent common-option support.  This is one shared source mechanism,
        # not a regime router or an option-specific free bias.
        self.direct_recovery_absolute_common_witness_correction = bool(
            direct_recovery_absolute_common_witness_correction
        )
        # v48.63 OC-QARW: quantifier-aligned common witness.  Feasibility is
        # existential over common recovery options, whereas a negative veto is
        # admissible only when *all* common options fail.  This corrects the
        # per-option negative-veto asymmetry exposed by v48.62 without changing
        # the executable witness field, Stage-I, threshold, or regime policy.
        self.direct_recovery_absolute_quantifier_witness_correction = bool(
            direct_recovery_absolute_quantifier_witness_correction
        )
        # v48.64 OC-SARW: semantics-aligned recovery witness.  The candidate x
        # option continuation, common-option support and quantifier logic remain
        # frozen from v48.63; only the *observable constraint semantics* are
        # repaired.  The two factor flags are preregistered ablations, not
        # regime-conditioned behavior.
        self.direct_recovery_absolute_semantic_witness_correction = bool(
            direct_recovery_absolute_semantic_witness_correction
        )
        self.direct_recovery_semantic_witness_active_set_alignment = bool(
            direct_recovery_semantic_witness_active_set_alignment
        )
        self.direct_recovery_semantic_witness_path_stop_alignment = bool(
            direct_recovery_semantic_witness_path_stop_alignment
        )
        # v48.65 OC-CLRW: keep the v48.64 observable physical barrier but
        # transport its bounded correction at the OC-MERO observation-class
        # q[i,l] interface. Distinguishable observation classes may therefore
        # support different recovery options while compatible roots remain
        # coupled inside q. This is a factor flag, never a regime input.
        self.direct_recovery_semantic_witness_classlocal_transport = bool(
            direct_recovery_semantic_witness_classlocal_transport
        )
        # v48.66 OC-ACRW: after v48.65 falsified class-local transport as the
        # dominant source bottleneck, extend the *candidate-global* executable
        # witness with observation-certifiable active constraints that were
        # absent from the v48.64 physical certificate.  These are factor flags,
        # not regime routers or learned per-option parameters.
        self.direct_recovery_semantic_witness_route_alignment = bool(
            direct_recovery_semantic_witness_route_alignment
        )
        self.direct_recovery_semantic_witness_reentry_alignment = bool(
            direct_recovery_semantic_witness_reentry_alignment
        )
        # v48.67 OC-PBRW separates two failure layers exposed by v48.66:
        # (1) realize the recovery controller inside the observable actuator
        # envelope by construction; (2) transport a trusted positive witness
        # toward the absolute zero boundary with a bounded residual rather than
        # an arbitrary additive offset. Neither flag is regime-conditioned.
        self.direct_recovery_semantic_witness_control_projection = bool(
            direct_recovery_semantic_witness_control_projection
        )
        self.direct_recovery_semantic_witness_boundary_transport = bool(
            direct_recovery_semantic_witness_boundary_transport
        )
        # v48.68 OC-RTRW: Q_CTRLPROJ removed the post-hoc control veto but
        # admitted many low-trust witnesses.  Preserve the magnitude of the raw
        # desired-command control violation as a *soft support fidelity* rather
        # than a hard feasibility veto, and optionally robustify observable
        # occupancy with CV/observed-acceleration hypotheses.  Both are shared
        # observation-only semantics, never regime inputs.
        self.direct_recovery_semantic_witness_projection_fidelity_weighting = bool(
            direct_recovery_semantic_witness_projection_fidelity_weighting
        )
        # v48.77 OC-ACTSI: a signed min-certificate is piecewise by its active
        # constraint.  The historical two-gain transport collapses that mode
        # identity before learning.  Preserve the same non-compensatory physical
        # certificate, but allow the signed-margin source to use one shared
        # positive/negative slope per active constraint.  This is not a regime
        # router, option-ID bias, feature-weighted sum, or final admission head.
        self.direct_recovery_semantic_witness_active_constraint_typed_source = bool(
            direct_recovery_semantic_witness_active_constraint_typed_source
        )
        # v48.78 OC-RTSI closes the option-wise gain-transport family after
        # V48.77 STOP.  The new source changes *within-option root-tail shape*
        # with a single observation-coordinate vector shared by every option,
        # candidate and regime.  A weighted zero-mean projection removes the
        # option-translation degree exactly.  The optional nested-tail flag uses
        # only the frozen OC-MERO LCVAR influence (no teacher future or labels).
        self.direct_recovery_semantic_witness_root_tail_source = bool(
            direct_recovery_semantic_witness_root_tail_source
        )
        self.direct_recovery_semantic_witness_tail_localization = bool(
            direct_recovery_semantic_witness_tail_localization
        )
        self.direct_recovery_semantic_witness_structured_tail_field = bool(
            direct_recovery_semantic_witness_structured_tail_field
        )
        self.direct_recovery_semantic_witness_signed_tail_channels = bool(
            direct_recovery_semantic_witness_signed_tail_channels
        )
        self.direct_recovery_semantic_witness_counterfactual_tail_response = bool(
            direct_recovery_semantic_witness_counterfactual_tail_response
        )
        if self.direct_recovery_semantic_witness_structured_tail_field and not self.direct_recovery_semantic_witness_root_tail_source:
            raise ValueError("v48.82 structured tail field requires root-tail source")
        if self.direct_recovery_semantic_witness_signed_tail_channels and not self.direct_recovery_semantic_witness_structured_tail_field:
            raise ValueError("v48.82 signed tail channels require structured tail field")
        if self.direct_recovery_semantic_witness_counterfactual_tail_response and not (
            self.direct_recovery_semantic_witness_structured_tail_field
            and self.direct_recovery_semantic_witness_signed_tail_channels
        ):
            raise ValueError("v48.83 counterfactual tail response requires the signed structured tail field")
        if self.direct_recovery_semantic_witness_tail_localization and not self.direct_recovery_semantic_witness_root_tail_source:
            raise ValueError("tail localization requires the v48.78 root-tail source")
        if self.direct_recovery_semantic_witness_root_tail_source and self.direct_recovery_semantic_witness_active_constraint_typed_source:
            raise ValueError("v48.78 root-tail source replaces the v48.77 active-constraint gain table")
        if self.direct_recovery_semantic_witness_root_tail_source and self.direct_recovery_semantic_witness_classlocal_transport:
            raise ValueError("v48.78 root-tail source cannot be combined with learned class-local transport")
        if self.direct_recovery_semantic_witness_root_tail_source and self.direct_recovery_semantic_witness_boundary_transport:
            raise ValueError("v48.78 root-tail source does not reopen boundary transport")
        # v48.69 OC-DTRW keeps the validated v48.68 projection-fidelity signal
        # but tempers it by observation-derived recovery demand.  Urgent Near/
        # Contact recoveries often require a large actuator projection even when
        # they are the correct safe action; a Safe-like low-demand state should
        # retain the exact v48.68 penalty.  This is a shared observation-only
        # trust semantic, never a regime flag.
        self.direct_recovery_semantic_witness_demand_normalized_fidelity = bool(
            direct_recovery_semantic_witness_demand_normalized_fidelity
        )
        if self.direct_recovery_semantic_witness_demand_normalized_fidelity and not (
            self.direct_recovery_semantic_witness_projection_fidelity_weighting
            and self.direct_recovery_semantic_witness_control_projection
        ):
            raise ValueError(
                "demand-normalized projection fidelity requires projection-fidelity weighting and control projection"
            )
        self.direct_recovery_semantic_witness_robust_occupancy = bool(
            direct_recovery_semantic_witness_robust_occupancy
        )
        # v48.70 OC-DOTW: retain the validated single-CV physical certificate
        # but use disagreement with the already-observable bounded-acceleration
        # counterfactual as strictly-positive epistemic support trust.
        self.direct_recovery_semantic_witness_soft_occupancy_disagreement = bool(
            direct_recovery_semantic_witness_soft_occupancy_disagreement
        )
        if self.direct_recovery_semantic_witness_soft_occupancy_disagreement and not (
            self.direct_recovery_semantic_witness_projection_fidelity_weighting
            and self.direct_recovery_semantic_witness_control_projection
        ):
            raise ValueError(
                "soft occupancy disagreement requires projection-fidelity weighting and control projection"
            )
        # v48.71 OC-BORW: raw CV-vs-CA displacement disagreement is not itself
        # a safety quantity.  Separate (a) localization at the physical
        # clearance boundary and (b) a set-valued acceleration reachability tube
        # derived from observed history.  Both are strictly-positive support
        # trust only; CV remains the signed certificate and boundary transport
        # remains a separate downstream mechanism.
        self.direct_recovery_semantic_witness_boundary_localized_occupancy_trust = bool(
            direct_recovery_semantic_witness_boundary_localized_occupancy_trust
        )
        self.direct_recovery_semantic_witness_history_occupancy_reachability = bool(
            direct_recovery_semantic_witness_history_occupancy_reachability
        )
        if (
            self.direct_recovery_semantic_witness_boundary_localized_occupancy_trust
            or self.direct_recovery_semantic_witness_history_occupancy_reachability
        ) and not (
            self.direct_recovery_semantic_witness_projection_fidelity_weighting
            and self.direct_recovery_semantic_witness_control_projection
        ):
            raise ValueError(
                "boundary-localized/history occupancy trust requires projection-fidelity weighting and control projection"
            )
        if (
            self.direct_recovery_semantic_witness_boundary_localized_occupancy_trust
            or self.direct_recovery_semantic_witness_history_occupancy_reachability
        ) and self.direct_recovery_semantic_witness_soft_occupancy_disagreement:
            raise ValueError(
                "v48.71 occupancy reachability trust replaces, rather than stacks, v48.70 soft occupancy disagreement"
            )
        if (
            self.direct_recovery_semantic_witness_boundary_localized_occupancy_trust
            or self.direct_recovery_semantic_witness_history_occupancy_reachability
        ) and self.direct_recovery_semantic_witness_robust_occupancy:
            raise ValueError(
                "v48.71 occupancy reachability trust cannot be combined with the rejected hard robust-occupancy min"
            )

        # v48.72 OC-IORW: use the same observation-history acceleration evidence
        # as v48.71, but resolve reachable occupancy along the candidate-specific
        # ego-agent interaction normal rather than via an isotropic circumscribed
        # ball.  The hull arm further removes unobserved Cartesian corner
        # combinations by using the support function of conv({0,a_tau}).
        self.direct_recovery_semantic_witness_interaction_box_support = bool(
            direct_recovery_semantic_witness_interaction_box_support
        )
        self.direct_recovery_semantic_witness_interaction_hull_support = bool(
            direct_recovery_semantic_witness_interaction_hull_support
        )
        interaction_oriented = (
            self.direct_recovery_semantic_witness_interaction_box_support
            or self.direct_recovery_semantic_witness_interaction_hull_support
        )
        if interaction_oriented and not (
            self.direct_recovery_semantic_witness_projection_fidelity_weighting
            and self.direct_recovery_semantic_witness_control_projection
        ):
            raise ValueError(
                "v48.72 interaction-oriented reachability requires projection-fidelity weighting and control projection"
            )
        if interaction_oriented and (
            self.direct_recovery_semantic_witness_soft_occupancy_disagreement
            or self.direct_recovery_semantic_witness_boundary_localized_occupancy_trust
            or self.direct_recovery_semantic_witness_history_occupancy_reachability
            or self.direct_recovery_semantic_witness_robust_occupancy
        ):
            raise ValueError(
                "v48.72 interaction-oriented reachability replaces prior occupancy trust mechanisms rather than stacking them"
            )
        if self.direct_recovery_semantic_witness_interaction_hull_support and not self.direct_recovery_semantic_witness_interaction_box_support:
            # Hull support is the nested Main arm.  Requiring the box flag makes
            # its metadata encode the causal chain explicitly while the model
            # consumes only the hull coordinate when enabled.
            raise ValueError("v48.72 hull support requires interaction box support flag")

        # v48.73 OC-IRRW: retain v48.72 joint/directional acceleration geometry,
        # but constrain its temporal evolution from the current observed state.
        self.direct_recovery_semantic_witness_interaction_anchor_support = bool(
            direct_recovery_semantic_witness_interaction_anchor_support
        )
        self.direct_recovery_semantic_witness_interaction_response_support = bool(
            direct_recovery_semantic_witness_interaction_response_support
        )
        interaction_response = (
            self.direct_recovery_semantic_witness_interaction_anchor_support
            or self.direct_recovery_semantic_witness_interaction_response_support
        )
        if interaction_response and not (
            self.direct_recovery_semantic_witness_interaction_box_support
            and self.direct_recovery_semantic_witness_interaction_hull_support
            and self.direct_recovery_semantic_witness_projection_fidelity_weighting
            and self.direct_recovery_semantic_witness_control_projection
        ):
            raise ValueError(
                "v48.73 interaction-response reachability requires the v48.72 empirical-hull projected-recovery chain"
            )
        if self.direct_recovery_semantic_witness_interaction_response_support and not self.direct_recovery_semantic_witness_interaction_anchor_support:
            raise ValueError("v48.73 response support requires interaction anchor support flag")
        absolute_source_count = sum([
            self.direct_recovery_absolute_feasibility_head,
            self.direct_recovery_absolute_option_margin_correction,
            self.direct_recovery_absolute_physical_headroom_correction,
            self.direct_recovery_absolute_executable_witness_correction,
            self.direct_recovery_absolute_common_witness_correction,
            self.direct_recovery_absolute_quantifier_witness_correction,
            self.direct_recovery_absolute_semantic_witness_correction,
        ])
        if absolute_source_count > 1:
            raise ValueError(
                "AFE, ORFC, CPHR, ERWF, OC-CWRF, OC-QARW, and OC-SARW absolute-source corrections are mutually exclusive"
            )
        # v48.48 NCP: preserve paper-native OC-MERO DRS/deployability coordinates
        # at the final non-compensatory admission interface instead of asking a
        # downstream proxy head to learn their sign/scale again.
        self.direct_recovery_evidence_native_certificate_preservation = bool(
            direct_recovery_evidence_native_certificate_preservation
        )
        # v48.49 DCP: extend v48.48 NCP from two hard risk coordinates to a
        # decision-complete, monotone native transport.  The margin-complete arm
        # preserves local zero-boundary geometry and native gap quality; the
        # advantage arm transports the same native recovery value to the benefit
        # side of the non-compensatory reserve.  Both are regime-agnostic and add
        # no learned parameters.
        self.direct_recovery_evidence_native_margin_complete_preservation = bool(
            direct_recovery_evidence_native_margin_complete_preservation
        )
        self.direct_recovery_evidence_native_advantage_preservation = bool(
            direct_recovery_evidence_native_advantage_preservation
        )
        # v48.50: exact-coordinate NAP uses the hard predicted DRS that the
        # teacher/evaluator use, rather than the v48.49 smooth boundary mass.
        self.direct_recovery_evidence_native_exact_advantage_preservation = bool(
            direct_recovery_evidence_native_exact_advantage_preservation
        )
        # v48.51 BC-NAP: the hard certificate owns material decision sign, while
        # the boundary-resolved coordinate owns ordering inside the existing
        # positive-gain equivalence band. This is parameter-free and uses the
        # same global materiality boundary as calibration.
        self.direct_recovery_evidence_native_boundary_complete_advantage_preservation = bool(
            direct_recovery_evidence_native_boundary_complete_advantage_preservation
        )
        # v48.53 CSE: q chooses the observation-consistent recovery option,
        # while the selected predicted physical margin owns root success.  This
        # mirrors the teacher/evaluator certificate composition and is enabled
        # only as an explicit experimental factor.
        self.direct_recovery_evidence_physical_student_drs = bool(
            direct_recovery_evidence_physical_student_drs
        )
        if self.direct_recovery_evidence_physical_student_drs and not self.direct_recovery_evidence_native_certificate_preservation:
            raise ValueError("physical student DRS requires native certificate preservation")
        if self.direct_recovery_evidence_native_exact_advantage_preservation and not self.direct_recovery_evidence_native_advantage_preservation:
            raise ValueError("exact native advantage preservation requires native advantage preservation")
        if self.direct_recovery_evidence_native_boundary_complete_advantage_preservation and not self.direct_recovery_evidence_native_advantage_preservation:
            raise ValueError("boundary-complete native advantage preservation requires native advantage preservation")
        if self.direct_recovery_evidence_native_exact_advantage_preservation and self.direct_recovery_evidence_native_boundary_complete_advantage_preservation:
            raise ValueError("exact and boundary-complete native advantage preservation are mutually exclusive")
        self.direct_recovery_evidence_native_drs_tolerance = float(
            max(0.0, direct_recovery_evidence_native_drs_tolerance)
        )
        self.direct_recovery_evidence_native_deployability_tolerance = float(
            max(0.0, direct_recovery_evidence_native_deployability_tolerance)
        )
        self.direct_recovery_evidence_native_dep_boundary_aligned = bool(
            direct_recovery_evidence_native_dep_boundary_aligned
        )
        self.direct_recovery_evidence_native_gap_tolerance = float(
            max(0.0, direct_recovery_evidence_native_gap_tolerance)
        )
        self.direct_recovery_evidence_native_positive_gain = float(
            max(0.0, direct_recovery_evidence_native_positive_gain)
        )
        if (
            self.direct_recovery_evidence_native_margin_complete_preservation
            or self.direct_recovery_evidence_native_advantage_preservation
        ) and not self.direct_recovery_evidence_native_certificate_preservation:
            raise ValueError(
                "v48.49 native decision-complete transport requires native certificate preservation"
            )
        if self.direct_recovery_evidence_native_certificate_preservation and int(direct_recovery_evidence_component_count) < 2:
            raise ValueError(
                "native certificate preservation requires DRS and deployability component coordinates"
            )
        if self.direct_recovery_evidence_native_margin_complete_preservation and int(direct_recovery_evidence_component_count) < 3:
            raise ValueError(
                "native margin-complete preservation requires DRS, deployability, and gap component coordinates"
            )
        self.direct_recovery_evidence_roct_signature_dim = 4
        if not (0.0 < self.direct_recovery_evidence_roct_alpha <= 1.0):
            raise ValueError("direct_recovery_evidence_roct_alpha must be in (0,1]")
        if not (0.0 < self.direct_recovery_evidence_roct_beta <= 1.0):
            raise ValueError("direct_recovery_evidence_roct_beta must be in (0,1]")
        if (self.direct_recovery_evidence_roct_benefit or self.direct_recovery_evidence_roct_deployability):
            if self.direct_recovery_evidence_roct_scale <= 0.0:
                raise ValueError("ROCT requires direct_recovery_evidence_roct_scale > 0")
            if not bool(direct_recovery_evidence_component_heads):
                raise ValueError("ROCT requires component-head physical evidence")
            if self.direct_recovery_evidence_roct_deployability and int(direct_recovery_evidence_component_count) < 2:
                raise ValueError("ROCT deployability transport requires component index 1")
        if (
            self.direct_recovery_evidence_postprefix_obs_transport_benefit
            or self.direct_recovery_evidence_postprefix_obs_transport_harm
        ):
            if not self.direct_recovery_evidence_dual_interaction_bridge:
                raise ValueError(
                    "post-prefix observation transport requires the dual OCAF bridge"
                )
            if self.direct_recovery_evidence_calibrator_context_source != "physical_interaction":
                raise ValueError(
                    "post-prefix observation transport requires physical_interaction context"
                )
            if self.direct_recovery_evidence_postprefix_obs_transport_scale <= 0.0:
                raise ValueError(
                    "post-prefix observation transport scale must be positive when enabled"
                )
        self.direct_recovery_evidence_calibrator_shared = bool(
            direct_recovery_evidence_calibrator_shared
        )
        self.direct_recovery_evidence_calibrator_regime_scale = float(
            max(0.0, direct_recovery_evidence_calibrator_regime_scale)
        )
        self.direct_recovery_evidence_unified_experts = bool(
            direct_recovery_evidence_unified_experts
        )
        self.direct_recovery_evidence_component_heads = bool(
            direct_recovery_evidence_component_heads
        )
        self.direct_recovery_evidence_component_count = max(3, min(5, int(
            direct_recovery_evidence_component_count
        )))
        self.direct_recovery_evidence_component_scale = float(
            max(0.0, direct_recovery_evidence_component_scale)
        )
        self.direct_recovery_evidence_benefit_residual_scale = float(
            max(0.0, direct_recovery_evidence_benefit_residual_scale)
        )
        # v48.39 DRFR: benefit and harm are signed physical regression
        # coordinates. Keep the legacy bounded parameterisation as the default
        # so older checkpoints retain their inference semantics; v48.39 opts in
        # independently for benefit and harm to support a clean 2x2 ablation.
        self.direct_recovery_evidence_unbounded_benefit_factor = bool(
            direct_recovery_evidence_unbounded_benefit_factor
        )
        self.direct_recovery_evidence_unbounded_harm_factors = bool(
            direct_recovery_evidence_unbounded_harm_factors
        )
        raw_component_reliability = direct_recovery_evidence_component_reliability
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
        if not reliability_values:
            reliability_values = [1.0] * self.direct_recovery_evidence_component_count
        if len(reliability_values) < self.direct_recovery_evidence_component_count:
            reliability_values.extend(
                [1.0] * (self.direct_recovery_evidence_component_count - len(reliability_values))
            )
        reliability_values = [
            min(1.0, max(0.0, float(x)))
            for x in reliability_values[: self.direct_recovery_evidence_component_count]
        ]
        self.direct_recovery_evidence_component_reliability = tuple(reliability_values)
        self.register_buffer(
            "_direct_recovery_evidence_component_reliability",
            torch.tensor(reliability_values, dtype=torch.float32),
            persistent=False,
        )
        self.direct_recovery_evidence_concord = bool(direct_recovery_evidence_concord)
        if self.direct_recovery_evidence_partial_pool_harm_residual:
            if self.direct_recovery_evidence_factorized_harm_interaction:
                raise ValueError(
                    "partial-pool harm residual and full factorized-harm interaction are mutually exclusive"
                )
            if not self.direct_recovery_evidence_component_heads or not self.direct_recovery_evidence_concord:
                raise ValueError(
                    "partial-pool harm residual requires concord component heads"
                )
            if self.direct_recovery_evidence_partial_pool_harm_residual_scale <= 0.0:
                raise ValueError("partial-pool harm residual scale must be positive when enabled")
        self.direct_recovery_evidence_consensus_disagreement_penalty = float(
            max(0.0, direct_recovery_evidence_consensus_disagreement_penalty)
        )
        self.direct_recovery_evidence_consensus_prior_scale = float(
            max(0.0, direct_recovery_evidence_consensus_prior_scale)
        )
        self.direct_recovery_evidence_admission_head = bool(
            direct_recovery_evidence_admission_head
        )
        self.direct_recovery_evidence_admission_scale = float(
            max(0.0, direct_recovery_evidence_admission_scale)
        )
        self.direct_recovery_evidence_admission_bounded = bool(
            direct_recovery_evidence_admission_bounded
        )
        self.direct_recovery_evidence_admission_prior_detach = bool(
            direct_recovery_evidence_admission_prior_detach
        )
        self.direct_recovery_evidence_admission_prior_mode = str(
            direct_recovery_evidence_admission_prior_mode or "risk_centered"
        ).strip().lower()
        if self.direct_recovery_evidence_admission_prior_mode not in {
            "risk_centered", "benefit_only", "safety_slack", "barrier_gated_slack",
            "frontier_capped_slack", "joint_reserve",
        }:
            raise ValueError(
                "Unsupported direct_recovery_evidence_admission_prior_mode="
                f"{direct_recovery_evidence_admission_prior_mode!r}"
            )
        self.direct_recovery_evidence_slack_temperature = float(
            max(1.0e-6, direct_recovery_evidence_slack_temperature)
        )
        self.direct_recovery_evidence_slack_penalty = float(
            max(0.0, direct_recovery_evidence_slack_penalty)
        )
        self.direct_recovery_evidence_frontier_cap_temperature = float(
            max(1.0e-4, direct_recovery_evidence_frontier_cap_temperature)
        )
        # v48.38 RFR: benefit and component logits are converted back into the
        # same signed physical-margin units used by factor supervision before
        # being composed into a deterministic noncompensatory recovery reserve.
        self.direct_recovery_evidence_benefit_margin_temperature = float(
            max(1.0e-6, direct_recovery_evidence_benefit_margin_temperature)
        )
        self.direct_recovery_evidence_joint_reserve_temperature = float(
            max(1.0e-6, direct_recovery_evidence_joint_reserve_temperature)
        )
        self.direct_recovery_evidence_reserve_factor_alignment = bool(
            direct_recovery_evidence_reserve_factor_alignment
        )
        self.direct_recovery_evidence_frontier = bool(direct_recovery_evidence_frontier)
        self.direct_recovery_evidence_component_prior_logit = float(
            direct_recovery_evidence_component_prior_logit
        )
        if self.direct_recovery_evidence_calibrator_mode not in {
            "center_width", "simplex_context", "dual_tail_context"
        }:
            raise ValueError(
                "Unsupported direct_recovery_evidence_calibrator_mode="
                f"{direct_recovery_evidence_calibrator_mode!r}"
            )
        if self.direct_recovery_evidence_calibrator_context_source not in {
            "relative", "tournament", "physical_relative", "physical_interaction"
        }:
            raise ValueError(
                "Unsupported direct_recovery_evidence_calibrator_context_source="
                f"{direct_recovery_evidence_calibrator_context_source!r}"
            )
        if (
            self.direct_recovery_evidence_calibrator_context
            and self.direct_recovery_evidence_calibrator_context_source == "tournament"
            and not self.direct_recovery_set_tournament
        ):
            raise ValueError("tournament evidence context requires direct_recovery_set_tournament=true")
        if (
            self.direct_recovery_evidence_calibrator_context
            and self.direct_recovery_evidence_calibrator_context_source in {
                "physical_relative", "physical_interaction"
            }
            and self.encoder_type != "structured_transformer"
        ):
            raise ValueError("physical evidence context requires structured_transformer")
        if self.direct_recovery_evidence_unified_experts and not self.direct_recovery_delta_regime_experts:
            raise ValueError(
                "unified expert evidence requires direct_recovery_delta_regime_experts=true"
            )
        if self.direct_recovery_evidence_component_heads and not self.direct_recovery_evidence_unified_experts:
            raise ValueError(
                "component evidence heads currently require unified expert evidence"
            )
        if self.direct_recovery_evidence_concord and not self.direct_recovery_evidence_unified_experts:
            raise ValueError("CONCORD evidence requires unified frozen source experts")
        if self.direct_recovery_delta_mode not in {"gaussian", "ordinal_evidence"}:
            raise ValueError(f"Unsupported direct_recovery_delta_mode={direct_recovery_delta_mode!r}")
        if self.direct_recovery_value_expert_routing not in {
            "bucket",
            "hard_bucket",
            "soft_observation",
            "soft",
            "moe",
            "uniform",
            "uniform_robust",
            "robust_observation",
            "robust_ensemble",
            "risk_ensemble",
        }:
            raise ValueError(
                "Unsupported direct_recovery_value_expert_routing="
                f"{direct_recovery_value_expert_routing!r}"
            )
        if self.direct_recovery_value_output not in {"probability", "score"}:
            raise ValueError(f"Unsupported direct_recovery_value_output={direct_recovery_value_output!r}")
        if self.direct_recovery_value_router_pooling not in {"candidate", "scene", "shared_raw", "ego_shared_raw"}:
            raise ValueError(
                "Unsupported direct_recovery_value_router_pooling="
                f"{direct_recovery_value_router_pooling!r}"
            )

        if self.encoder_type == "structured_transformer":
            layout = FlatFeatureLayout(**self.feature_layout)
            self.encoder = StructuredTokenEncoder(
                layout=layout,
                d_model=d_model,
                num_layers=num_layers,
                num_heads=num_heads,
                dropout=dropout,
            )
        else:
            self.encoder = MLPEncoder(input_dim, d_model)

        # Root-query decoder: K learned latent root slots attend to the encoded
        # scene-prefix tokens.  For the MLP fallback, the single global embedding
        # is treated as a one-token memory.
        self.root_queries = nn.Parameter(torch.randn(1, self.num_roots, d_model) * 0.02)
        self.root_cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.root_self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.root_norm1 = nn.LayerNorm(d_model)
        self.root_norm2 = nn.LayerNorm(d_model)
        self.root_ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.root_norm3 = nn.LayerNorm(d_model)

        self.option_embeddings = nn.Parameter(torch.randn(1, 1, self.num_options, d_model) * 0.02)
        self.option_feature_proj = (
            nn.Sequential(nn.Linear(self.option_feature_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model))
            if self.option_feature_dim > 0
            else None
        )
        self.root_logit_head = nn.Linear(d_model, 1)
        self.margin_head = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self.obs_embed_head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, self.d_obs))
        self.utility_head = nn.Linear(d_model, 1)
        # v40 OC-UVRA: decouple the calibrated OC-MERO certificate from the
        # candidate preference signal.  The head predicts a bounded deployable
        # recovery value and aleatoric log-variance for each executable prefix.
        # This avoids forcing R_dep, DRS, and the oracle gap to absorb a separate
        # counterfactual ranking objective.
        # v41 OC-CAVA: the v40 head consumed only the frozen CLS token.  The
        # measured validation loss stayed close to a uniform candidate ranking,
        # which means the frozen v39 CLS representation discarded much of the
        # prefix-specific signal needed for counterfactual action comparison.
        # ``candidate_concat`` exposes the encoded ego/prefix/macro/state/control
        # tokens to a small trainable adapter without changing any OC-MERO head.
        direct_in_dim = d_model
        self.direct_candidate_raw_dim = 0
        self.direct_candidate_feature_dim = 0
        self.direct_candidate_physical_feature_dim = 0
        self.direct_candidate_physical_slices: tuple[tuple[int, int], ...] = ()
        self.direct_observation_feature_dim = 0
        self.direct_observation_slices: tuple[tuple[int, int], ...] = ()
        self.direct_ego_feature_dim = 0
        if self.encoder_type == "structured_transformer":
            layout = FlatFeatureLayout(**self.feature_layout)
            self.direct_ego_feature_dim = int(layout.ego_dim)
            prefix_semantic_start = int(layout.ego_dim)
            prefix_semantic_end = prefix_semantic_start + int(layout.prefix_param_dim + layout.num_macros)
            trajectory_start = prefix_semantic_end + int(layout.scalar_dim)
            trajectory_end = trajectory_start + int(layout.prefix_flat_dim + layout.control_flat_dim)
            self.direct_candidate_feature_dim = trajectory_end
            # The compact v48.35 evidence bridge receives only executable action
            # geometry/control.  It deliberately excludes ego state and the scalar
            # block (utility, hard_violation, harm_proxy, feasibility, nominal flag,
            # and time index), preventing direct target/selector shortcut leakage.
            self.direct_candidate_physical_slices = (
                (prefix_semantic_start, prefix_semantic_end),
                (trajectory_start, trajectory_end),
            )
            self.direct_candidate_physical_feature_dim = int(
                layout.prefix_param_dim + layout.num_macros
                + layout.prefix_flat_dim + layout.control_flat_dim
            )
            # OCAF scene pressure is anchored on the nominal row.  It contains
            # ego and shared observation context, while excluding every candidate
            # action field and the utility/harm/feasibility audit scalar block.
            self.direct_observation_slices = (
                (0, int(layout.ego_dim)),
                (trajectory_end, int(input_dim)),
            )
            self.direct_observation_feature_dim = int(
                layout.ego_dim + int(input_dim) - trajectory_end
            )
        if self.direct_recovery_value_pooling in {"candidate_concat", "prefix_concat", "action_concat"}:
            direct_in_dim = 6 * d_model
        elif self.direct_recovery_value_pooling in {"candidate_concat_raw", "action_concat_raw", "certificate_action_adapter"}:
            # Certificate-preserving dual pathway: keep the frozen contextual
            # OC-MERO tokens, but expose raw ego/prefix/macro/state/control
            # blocks to the trainable action-value adapter.  This prevents a
            # frozen certificate encoder from discarding counterfactual action
            # differences while leaving all calibrated certificate heads intact.
            if self.encoder_type != "structured_transformer":
                raise ValueError("candidate_concat_raw requires structured_transformer")
            self.direct_candidate_raw_dim = self.direct_candidate_feature_dim
            direct_in_dim = 6 * d_model + self.direct_candidate_raw_dim
        # Optional regime embedding retained only for controlled ablations. The
        # v44 OC-RAVA default disables it: deployment uses one observation-only
        # value/opportunity model and never receives the evaluation bucket as a
        # neural input. Bucket ids remain available solely to separate training
        # groups and to support an explicit leakage ablation if needed.
        self.direct_regime_embedding = (
            nn.Embedding(self.direct_recovery_value_num_regimes, self.direct_recovery_value_regime_dim)
            if self.direct_recovery_value_head and self.direct_recovery_value_regime_conditioning
            else None
        )
        if self.direct_recovery_value_router_pooling == "scene":
            direct_router_in_dim = d_model
        elif self.direct_recovery_value_router_pooling in {"shared_raw", "ego_shared_raw"}:
            if self.encoder_type != "structured_transformer" or self.direct_candidate_feature_dim <= 0:
                raise ValueError("shared raw router pooling requires structured_transformer")
            direct_router_in_dim = int(input_dim) - self.direct_candidate_feature_dim
            if self.direct_recovery_value_router_pooling == "ego_shared_raw":
                direct_router_in_dim += self.direct_ego_feature_dim
            if direct_router_in_dim <= 0:
                raise ValueError("shared raw router pooling produced an empty observation feature")
        else:
            direct_router_in_dim = direct_in_dim
        # v48.3 OC-TRAC-NASC: candidate value is intrinsically set-relative.
        # The adapter is permutation-equivariant within a scene-time candidate set,
        # explicitly anchored to nominal, and leaves the base pointwise path intact
        # when a complete candidate set is unavailable.
        self.direct_set_context_adapter = (
            nn.Sequential(
                nn.LayerNorm(4 * direct_in_dim),
                nn.Linear(4 * direct_in_dim, self.direct_recovery_set_context_hidden),
                nn.GELU(),
                nn.Dropout(self.direct_recovery_set_context_dropout),
                nn.Linear(self.direct_recovery_set_context_hidden, direct_in_dim),
                nn.LayerNorm(direct_in_dim),
            )
            if self.direct_recovery_value_head and self.direct_recovery_set_context
            else None
        )
        # Start conservatively so v47/v48 pointwise knowledge remains the initial
        # solution and the set residual is learned only when it improves ranking.
        self.direct_set_context_gate = (
            nn.Parameter(torch.tensor(-2.5)) if self.direct_set_context_adapter is not None else None
        )
        # v48.4 ZI-NASC: a warm-started checkpoint must initially reproduce the
        # inherited pointwise policy exactly.  A merely small sigmoid gate still
        # injects a random LayerNorm residual and can destroy candidate AUC before
        # the set adapter learns anything.  Zero-initialize the residual projection;
        # gradients still flow into it on the first update and into earlier adapter
        # layers thereafter.
        if self.direct_set_context_adapter is not None:
            residual_projection = self.direct_set_context_adapter[4]
            if isinstance(residual_projection, nn.Linear):
                nn.init.zeros_(residual_projection.weight)
                nn.init.zeros_(residual_projection.bias)
        if self.direct_regime_embedding is not None:
            direct_in_dim += self.direct_recovery_value_regime_dim
        direct_out_dim = 2 + int(self.direct_recovery_opportunity_head) + int(self.direct_recovery_harm_head)

        # v48.5 ECPR: ranking and admission are separate tasks.  The value head
        # estimates candidate-vs-nominal gain and uncertainty, while this
        # zero-initialized residual preference head learns only the within-set
        # ordering.  At warm start rank == value exactly, preserving v48.1/v48.4
        # checkpoint behaviour until ranking evidence is observed.
        self.direct_preference_adapter = (
            nn.Sequential(
                nn.LayerNorm(direct_in_dim),
                nn.Linear(direct_in_dim, self.direct_recovery_preference_hidden),
                nn.GELU(),
                nn.Dropout(self.direct_recovery_preference_dropout),
                nn.Linear(self.direct_recovery_preference_hidden, 1),
            )
            if self.direct_recovery_value_head and self.direct_recovery_preference_head
            else None
        )
        if self.direct_preference_adapter is not None:
            pref_projection = self.direct_preference_adapter[-1]
            if isinstance(pref_projection, nn.Linear):
                nn.init.zeros_(pref_projection.weight)
                nn.init.zeros_(pref_projection.bias)

        # v48.6 RPGC: do not let a set adapter rewrite the value representation.
        # A separate, zero-initialized relative-context residual augments only the
        # ranking score.  This preserves the v48.5 pointwise preference solution
        # while exposing candidate-minus-nominal and group summaries to ranking.
        relative_in_dim = (4 if self.direct_recovery_relative_features_include_absolute else 3) * direct_in_dim
        self.direct_preference_context_adapter = (
            nn.Sequential(
                nn.LayerNorm(relative_in_dim),
                nn.Linear(relative_in_dim, self.direct_recovery_preference_context_hidden),
                nn.GELU(),
                nn.Dropout(self.direct_recovery_preference_dropout),
                nn.Linear(self.direct_recovery_preference_context_hidden, 1),
            )
            if self.direct_recovery_value_head and self.direct_recovery_preference_context
            else None
        )
        if self.direct_preference_context_adapter is not None:
            ctx_projection = self.direct_preference_context_adapter[-1]
            if isinstance(ctx_projection, nn.Linear):
                nn.init.zeros_(ctx_projection.weight)
                nn.init.zeros_(ctx_projection.bias)

        # v48.11 CASTER: a standalone recovery-only set tournament replaces the
        # inherited candidate-level value ranking.  The latter has repeatedly
        # shown high candidate AUC but near-zero or negative groupwise top-1.
        self.direct_preference_set_ranker = (
            RecoverySetTournament(
                relative_in_dim,
                self.direct_recovery_set_tournament_hidden,
                self.direct_recovery_set_tournament_heads,
                self.direct_recovery_set_tournament_dropout,
            )
            if self.direct_recovery_value_head and self.direct_recovery_set_tournament
            else None
        )

        # Directly estimate candidate-minus-nominal PCD gain and uncertainty.
        # Subtracting two absolute predictions and adding their variances assumes
        # independent errors, even though both candidates share a scene encoder;
        # that assumption made v48.5 admission probabilities excessively diffuse.
        delta_input_dim = relative_in_dim + (2 if self.direct_recovery_delta_policy_features else 0)

        def _make_delta_adapter() -> nn.Sequential:
            adapter = nn.Sequential(
                nn.LayerNorm(delta_input_dim),
                nn.Linear(delta_input_dim, self.direct_recovery_delta_hidden),
                nn.GELU(),
                nn.Dropout(self.direct_recovery_delta_dropout),
                nn.Linear(self.direct_recovery_delta_hidden, 2),
            )
            projection = adapter[-1]
            if isinstance(projection, nn.Linear):
                nn.init.zeros_(projection.weight)
                nn.init.zeros_(projection.bias)
                with torch.no_grad():
                    projection.bias[1] = self.direct_recovery_delta_initial_logvar
            return adapter

        self.direct_delta_adapters = (
            nn.ModuleList([_make_delta_adapter(), _make_delta_adapter()])
            if self.direct_recovery_value_head and self.direct_recovery_delta_head
            and self.direct_recovery_delta_regime_experts
            else None
        )
        self.direct_delta_adapter = (
            _make_delta_adapter()
            if self.direct_recovery_value_head and self.direct_recovery_delta_head
            and not self.direct_recovery_delta_regime_experts
            else None
        )

        # v48.17 BRIDGE: retain the zero-initialised identity correction, but
        # optionally expose the frozen candidate-vs-nominal context representation.
        # v48.16 used only four scalars (source center/width and two rank margins);
        # certificate results showed that this input was insufficient to distinguish
        # beneficial, dead-zone, and harmful candidates with nearly identical source
        # scores.  The low-rank bottleneck keeps target adaptation small while adding
        # the observables needed for conditional correction.
        evidence_context_dim = 0
        if self.direct_recovery_evidence_calibrator_context:
            if (
                self.direct_recovery_evidence_calibrator_context_source == "tournament"
                and self.direct_preference_set_ranker is not None
            ):
                evidence_context_dim = self.direct_preference_set_ranker.hidden_dim
            elif self.direct_recovery_evidence_calibrator_context_source == "physical_relative":
                # v48.35 CONTINUOUS-FRONTIER ablation: action-only executable-prefix
                # difference to nominal.
                evidence_context_dim = self.direct_candidate_physical_feature_dim
            elif self.direct_recovery_evidence_calibrator_context_source == "physical_interaction":
                # v48.36 OCAF: continuous observation pressure modulates an
                # executable candidate-minus-nominal action without a regime switch.
                evidence_context_dim = self.direct_recovery_evidence_interaction_hidden
            else:
                evidence_context_dim = relative_in_dim
        self.direct_evidence_interaction_bridge = (
            (
                FactorizedObservationConditionedActionFrontierBridge(
                    self.direct_candidate_physical_feature_dim,
                    self.direct_observation_feature_dim,
                    self.direct_recovery_evidence_interaction_hidden,
                    self.direct_recovery_evidence_interaction_dropout,
                    self.direct_recovery_evidence_component_count,
                    self.direct_recovery_evidence_component_reliability,
                )
                if self.direct_recovery_evidence_factorized_harm_interaction
                else DualObservationConditionedActionFrontierBridge(
                    self.direct_candidate_physical_feature_dim,
                    self.direct_observation_feature_dim,
                    self.direct_recovery_evidence_interaction_hidden,
                    self.direct_recovery_evidence_interaction_dropout,
                )
                if self.direct_recovery_evidence_dual_interaction_bridge
                else ObservationConditionedActionFrontierBridge(
                    self.direct_candidate_physical_feature_dim,
                    self.direct_observation_feature_dim,
                    self.direct_recovery_evidence_interaction_hidden,
                    self.direct_recovery_evidence_interaction_dropout,
                )
            )
            if self.direct_recovery_evidence_calibrator_context
            and self.direct_recovery_evidence_calibrator_context_source == "physical_interaction"
            else None
        )
        # v48.43 POET (Post-prefix Observation-Equivalence Transport).
        # Four bounded, model-predicted structural coordinates summarize how the
        # candidate changes root uncertainty and observation aliasing after the
        # executable prefix.  Separate zero-init adapters let the 2x2 ablation
        # test benefit-side and harm-side identifiability without changing the
        # shared deployment rule.  Bias-free projections keep the nominal row
        # exactly unchanged because the signature is candidate-minus-nominal.
        def _make_postprefix_obs_transport_adapter() -> nn.Linear:
            adapter = nn.Linear(
                self.direct_recovery_evidence_postprefix_obs_signature_dim,
                self.direct_recovery_evidence_interaction_hidden,
                bias=False,
            )
            nn.init.zeros_(adapter.weight)
            return adapter

        self.direct_evidence_postprefix_obs_transport_benefit = (
            _make_postprefix_obs_transport_adapter()
            if self.direct_recovery_evidence_postprefix_obs_transport_benefit
            else None
        )
        self.direct_evidence_postprefix_obs_transport_harm = (
            _make_postprefix_obs_transport_adapter()
            if self.direct_recovery_evidence_postprefix_obs_transport_harm
            else None
        )

        # v48.44 ROCT uses the frozen OC-MERO geometry as a compact structural
        # teacher.  Benefit receives one scalar correction; only the deployability
        # physical veto coordinate receives the safety-side correction.  This
        # deliberately avoids another generic shared-harm residual or full
        # component factorization, both of which have already failed empirically.
        def _make_roct_adapter() -> nn.Linear:
            adapter = nn.Linear(
                self.direct_recovery_evidence_roct_signature_dim, 1, bias=False
            )
            nn.init.zeros_(adapter.weight)
            return adapter

        self.direct_evidence_roct_benefit = (
            _make_roct_adapter() if self.direct_recovery_evidence_roct_benefit else None
        )
        self.direct_evidence_roct_deployability = (
            _make_roct_adapter() if self.direct_recovery_evidence_roct_deployability else None
        )

        # v48.58 RIFA / Absolute Feasibility Evidence (AFE).  This is deliberately
        # a nine-parameter linear readout over [ROCT_abs(4), native_cert_abs(4)].
        # It is initialized to the raw native DEP zero boundary exactly at
        # dep_score=0.5; head-only adaptation can then correct source error without
        # rotating any Stage-I relative ranking/evidence representation.
        self.direct_absolute_feasibility_head = None
        if self.direct_recovery_absolute_feasibility_head:
            self.direct_absolute_feasibility_head = nn.Linear(
                self.direct_recovery_evidence_roct_signature_dim + 4, 1
            )
            with torch.no_grad():
                self.direct_absolute_feasibility_head.weight.zero_()
                self.direct_absolute_feasibility_head.bias.fill_(-2.0)
                # Native certificate coordinate 1 is sigmoid(R_dep).
                dep_index = self.direct_recovery_evidence_roct_signature_dim + 1
                self.direct_absolute_feasibility_head.weight[0, dep_index] = 4.0

        # v48.59 ORFC source correction.  A single option-wise bias vector is
        # the only trainable state.  It changes the absolute recovery margins
        # before OC-MERO aggregation, not the Stage-I rank/relative evidence.
        self.direct_absolute_option_margin_bias = None
        if self.direct_recovery_absolute_option_margin_correction:
            self.direct_absolute_option_margin_bias = nn.Parameter(
                torch.zeros(self.num_options, dtype=torch.float32)
            )

        # v48.60 CPHR.  Feature order:
        # [minimum clearance reserve, terminal clearance reserve, clearance gain,
        #  stopping reserve, control-envelope reserve, stability reserve].
        # Every coordinate is signed and bounded by tanh.  Positive means more
        # physical recovery headroom.  No regime indicator or teacher component
        # is an input; all six weights start at zero and are used through [0,2]
        # projection, preserving a bounded monotone physical correction.
        self.direct_absolute_physical_headroom_weight = None
        if self.direct_recovery_absolute_physical_headroom_correction:
            if self.encoder_type != "structured_transformer":
                raise ValueError("CPHR requires structured_transformer flat feature layout")
            self.direct_absolute_physical_headroom_weight = nn.Parameter(
                torch.zeros(6, dtype=torch.float32)
            )

        # v48.61 ERWF.  The six shared weights operate on option-resolved
        # executable continuation witness coordinates.  They are deliberately
        # shared across recovery modes and roots: all structure comes from the
        # physical candidate x option field, not an option ID bias or regime
        # router.  Zero initialization is execution-exact v48.58-B.
        self.direct_absolute_executable_witness_weight = None
        if self.direct_recovery_absolute_executable_witness_correction:
            if self.encoder_type != "structured_transformer":
                raise ValueError("ERWF requires structured_transformer flat feature layout")
            self.direct_absolute_executable_witness_weight = nn.Parameter(
                torch.zeros(6, dtype=torch.float32)
            )

        # v48.62 OC-CWRF learns only two global calibration gains.  All
        # selectivity comes from a deterministic non-compensatory physical
        # recovery barrier and frozen observation-consistent root commonality.
        # gain[0] rescues a positive common witness; gain[1] vetoes a negative
        # physical witness.  Both start at zero => execution-exact native B.
        self.direct_absolute_common_witness_gain = None
        if self.direct_recovery_absolute_common_witness_correction:
            if self.encoder_type != "structured_transformer":
                raise ValueError("OC-CWRF requires structured_transformer flat feature layout")
            self.direct_absolute_common_witness_gain = nn.Parameter(
                torch.zeros(2, dtype=torch.float32)
            )

        # v48.63 OC-QARW keeps the same deterministic 10-D candidate x option
        # continuation field and common-option support as OC-CWRF, but aligns
        # the negative evidence with the existential structure of recoverability.
        # gain[0] is an option-local positive rescue; gain[1] is a candidate-level
        # universal-failure veto.  Both start at zero => exact native B.
        self.direct_absolute_quantifier_witness_gain = None
        if self.direct_recovery_absolute_quantifier_witness_correction:
            if self.encoder_type != "structured_transformer":
                raise ValueError("OC-QARW requires structured_transformer flat feature layout")
            self.direct_absolute_quantifier_witness_gain = nn.Parameter(
                torch.zeros(2, dtype=torch.float32)
            )

        # v48.64 OC-SARW keeps the v48.63 two-gain quantifier interface.
        # v48.77 OC-ACTSI conditionally replaces only that scalar interface by
        # a 6 x 2 active-constraint typed table.  Rows follow the exact barrier
        # stack [clearance, stopping, control, stability, route, re-entry]; the
        # table is shared across roots/options/regimes and zero-init remains
        # execution-exact native B.
        self.direct_absolute_semantic_witness_gain = None
        if self.direct_recovery_absolute_semantic_witness_correction and not self.direct_recovery_semantic_witness_root_tail_source:
            if self.encoder_type != "structured_transformer":
                raise ValueError("OC-SARW requires structured_transformer flat feature layout")
            gain_shape = (6, 2) if self.direct_recovery_semantic_witness_active_constraint_typed_source else (2,)
            self.direct_absolute_semantic_witness_gain = nn.Parameter(
                torch.zeros(gain_shape, dtype=torch.float32)
            )

        # v48.78 OC-RTSI source state.  The only learned degree of freedom is
        # a single non-negative global scale on a *deterministic OC-MERO tail
        # basis*.  No observation-class/root embedding is learned or consumed
        # by this source.  The root x option deformation is p-centered for every
        # option, so it is algebraically outside the v48.64--77 option-translation
        # family while avoiding the v48.65 class-local learned-transport branch.
        self.direct_absolute_root_tail_source_scale = None
        self.direct_absolute_structured_tail_field_weight = None
        if self.direct_recovery_semantic_witness_root_tail_source:
            if not self.direct_recovery_absolute_semantic_witness_correction:
                raise ValueError("v48.78 root-tail source requires the semantic executable-witness source")
            if self.encoder_type != "structured_transformer":
                raise ValueError("v48.78 root-tail source requires structured_transformer")
            if self.direct_recovery_semantic_witness_structured_tail_field:
                # V48.82 OC-SNTF: a shared diagonal bilinear root-option field.
                # Zero initialization is execution-exact native; there are no
                # root/option/regime IDs and no generic MLP.  The signed arm
                # uses separate reserve/debt channels selected by the native
                # root-option margin sign, not by a regime label.
                channels = 2 if self.direct_recovery_semantic_witness_signed_tail_channels else 1
                self.direct_absolute_structured_tail_field_weight = nn.Parameter(
                    torch.zeros((channels, d_model), dtype=torch.float32)
                )
            else:
                self.direct_absolute_root_tail_source_scale = nn.Parameter(
                    torch.zeros(1, dtype=torch.float32)
                )

        # v48.20 UNISON-BRIDGE: a single shared evidence model consumes both
        # frozen source experts, their consensus/disagreement, and the frozen
        # proposal context.  No bucket/regime id is exposed to this model.
        evidence_scalar_dim = 10 if self.direct_recovery_evidence_unified_experts else 4
        evidence_calibrator_input_dim = evidence_scalar_dim + evidence_context_dim
        if self.direct_recovery_evidence_component_heads:
            # benefit + all configured non-compensatory harm components.
            # v48.27 uses DRS/deployability/gap/hard-rule/harm-proxy (5).
            evidence_calibrator_output_dim = 1 + self.direct_recovery_evidence_component_count
        else:
            evidence_calibrator_output_dim = (
                3 if self.direct_recovery_evidence_calibrator_mode == "simplex_context" else 2
            )

        def _make_evidence_calibrator(output_dim: int | None = None) -> nn.Sequential:
            final_dim = evidence_calibrator_output_dim if output_dim is None else int(output_dim)
            adapter = nn.Sequential(
                nn.LayerNorm(evidence_calibrator_input_dim),
                nn.Linear(evidence_calibrator_input_dim, self.direct_recovery_evidence_calibrator_hidden),
                nn.GELU(),
                nn.Linear(self.direct_recovery_evidence_calibrator_hidden, final_dim),
            )
            projection = adapter[-1]
            if isinstance(projection, nn.Linear):
                nn.init.zeros_(projection.weight)
                nn.init.zeros_(projection.bias)
            return adapter

        self.direct_evidence_unified_calibrator = (
            _make_evidence_calibrator()
            if self.direct_recovery_value_head
            and self.direct_recovery_delta_head
            and self.direct_recovery_evidence_calibrator
            and self.direct_recovery_evidence_unified_experts
            and not self.direct_recovery_evidence_concord
            else None
        )
        # v48.21 CONCORD-BRIDGE decouples the sparse safe-benefit task from the
        # much denser component-risk task.  A single shared trunk in v48.20 let
        # the 45--54% harmful labels dominate the roughly 3% benefit labels.
        # Both adapters remain bucket-invariant and consume the same symmetric
        # frozen-expert/context representation; only their parameters are
        # decoupled to prevent negative transfer.
        self.direct_evidence_concord_benefit_calibrator = (
            _make_evidence_calibrator(1)
            if self.direct_recovery_value_head
            and self.direct_recovery_delta_head
            and self.direct_recovery_evidence_calibrator
            and self.direct_recovery_evidence_unified_experts
            and self.direct_recovery_evidence_concord
            else None
        )
        # v48.41 FCFR: a deliberately low-capacity monotone skip from the
        # frozen recovery preference advantage into the *bounded* benefit
        # residual.  v48.40 rows show that rank_adv carries strong Contact
        # safe-positive ordering information while the learned opportunity
        # frontier underuses it.  The positive softplus gain cannot invert that
        # ordering, and the downstream tanh keeps the HAF benefit correction
        # bounded (unlike the falsified v48.39 unbounded-benefit ablation).
        if self.direct_recovery_evidence_rank_benefit_skip:
            init_gain = torch.tensor(
                self.direct_recovery_evidence_rank_benefit_gain_init,
                dtype=torch.float32,
            )
            self.direct_evidence_rank_benefit_log_gain = nn.Parameter(
                torch.log(torch.expm1(init_gain))
            )
        else:
            self.register_parameter("direct_evidence_rank_benefit_log_gain", None)
        self.direct_evidence_concord_harm_calibrator = (
            (
                nn.ModuleList(
                    [
                        _make_evidence_calibrator(1)
                        for _ in range(self.direct_recovery_evidence_component_count)
                    ]
                )
                if self.direct_recovery_evidence_factorized_harm_interaction
                and self.direct_recovery_evidence_component_heads
                else _make_evidence_calibrator(
                    self.direct_recovery_evidence_component_count
                    if self.direct_recovery_evidence_component_heads else 1
                )
            )
            if self.direct_recovery_value_head
            and self.direct_recovery_delta_head
            and self.direct_recovery_evidence_calibrator
            and self.direct_recovery_evidence_unified_experts
            and self.direct_recovery_evidence_concord
            else None
        )
        # v48.42 HPFR component residuals operate on a detached copy of the
        # shared harm evidence.  Each supported physical component gets one
        # zero-initialised scalar readout; unsupported coordinates are kept as
        # parameter-free zero placeholders.  The residual is bounded in raw-logit
        # space before the legacy bounded component factor, preserving the exact
        # v48.40-A/B semantics at initialisation and preventing another unbounded
        # factor experiment.
        if (
            self.direct_recovery_evidence_partial_pool_harm_residual
            and self.direct_recovery_evidence_component_heads
            and self.direct_evidence_concord_harm_calibrator is not None
            and not self.direct_recovery_evidence_factorized_harm_interaction
        ):
            residual_heads: list[nn.Module] = []
            for component_index in range(self.direct_recovery_evidence_component_count):
                if self.direct_recovery_evidence_component_reliability[component_index] <= 0.0:
                    residual_heads.append(nn.Identity())
                    continue
                head = nn.Sequential(
                    nn.LayerNorm(evidence_calibrator_input_dim),
                    nn.Linear(evidence_calibrator_input_dim, 1, bias=False),
                )
                nn.init.zeros_(head[-1].weight)
                residual_heads.append(head)
            self.direct_evidence_concord_harm_component_residuals = nn.ModuleList(residual_heads)
        else:
            self.direct_evidence_concord_harm_component_residuals = None
        # v48.22 COVENANT-BRIDGE factorises three distinct hypotheses:
        # raw recovery benefit, componentwise harm, and final safe admissibility.
        # The admission adapter never receives a regime id.  It starts from a
        # detached benefit/risk prior and learns only the residual needed to
        # identify candidates that are simultaneously useful and safe.
        self.direct_evidence_concord_admission_calibrator = (
            _make_evidence_calibrator(1)
            if self.direct_recovery_value_head
            and self.direct_recovery_delta_head
            and self.direct_recovery_evidence_calibrator
            and self.direct_recovery_evidence_unified_experts
            and self.direct_recovery_evidence_concord
            and self.direct_recovery_evidence_admission_head
            else None
        )
        self.direct_evidence_calibrators = (
            nn.ModuleList([_make_evidence_calibrator(), _make_evidence_calibrator()])
            if self.direct_recovery_value_head and self.direct_recovery_delta_head
            and self.direct_recovery_evidence_calibrator
            and not self.direct_recovery_evidence_unified_experts
            else None
        )
        # v48.19 legacy partial pooling.  v48.20 disables this branch and uses
        # ``direct_evidence_unified_calibrator`` so inference never first chooses
        # a regime expert and then applies a regime-specific strategy.
        self.direct_evidence_shared_calibrator = (
            _make_evidence_calibrator()
            if self.direct_evidence_calibrators is not None
            and self.direct_recovery_evidence_calibrator_shared
            else None
        )

        def _make_direct_head() -> nn.Sequential:
            return nn.Sequential(
                nn.LayerNorm(direct_in_dim),
                nn.Linear(direct_in_dim, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, direct_out_dim),
            )

        # v45 OC-RAVE: Near-contact and Contact share the observation encoder but
        # use lightweight task experts.  The two datasets may contain the same
        # current observation with different pressure-future teachers, so one
        # unconditional head can receive contradictory labels even after groups
        # are separated.  In soft-observation mode every expert is evaluated and
        # continuously fused from observable scene/prefix features; dataset
        # buckets remain available only for loss stratification/calibration and
        # for the explicit legacy hard-routing ablation.  Safe does not use the
        # direct branch.
        self.direct_value_heads = (
            nn.ModuleList([_make_direct_head() for _ in range(self.direct_recovery_value_num_experts)])
            if self.direct_recovery_value_head and self.direct_recovery_value_experts
            else None
        )
        self.direct_value_head = (
            _make_direct_head()
            if self.direct_recovery_value_head and not self.direct_recovery_value_experts
            else None
        )
        # Observation-conditioned soft routing avoids treating a hand-authored
        # regime id as a policy input.  The router consumes the same observable
        # scene/prefix representation as the experts and produces continuous
        # mixture weights.  Legacy hard bucket routing remains available as an
        # explicit ablation and for loading older checkpoints.
        self.direct_value_router = (
            nn.Sequential(
                nn.LayerNorm(direct_router_in_dim),
                nn.Linear(direct_router_in_dim, max(16, d_model // 2)),
                nn.GELU(),
                nn.Linear(max(16, d_model // 2), self.direct_recovery_value_num_experts),
            )
            if self.direct_value_heads is not None
            and self.direct_recovery_value_expert_routing in {"soft_observation", "soft", "moe", "robust_observation"}
            else None
        )
        self.root_signature_head = nn.Linear(d_model, self.d_signature) if self.d_signature > 0 else None
        self.root_future_signature_head = nn.Linear(d_model, self.d_future_signature) if self.d_future_signature > 0 else None

    def _scene_tokens(self, x: torch.Tensor) -> torch.Tensor:
        if self.encoder_type == "structured_transformer" and hasattr(self.encoder, "forward_tokens"):
            return self.encoder.forward_tokens(x)  # type: ignore[attr-defined]
        h = self.encoder(x)
        return h.unsqueeze(1)

    def _decode_roots(self, memory: torch.Tensor) -> torch.Tensor:
        B = memory.shape[0]
        q0 = self.root_queries.expand(B, -1, -1)
        q, _ = self.root_cross_attn(q0, memory, memory, need_weights=False)
        q = self.root_norm1(q0 + q)
        qs, _ = self.root_self_attn(q, q, q, need_weights=False)
        q = self.root_norm2(q + qs)
        q = self.root_norm3(q + self.root_ffn(q))
        return q

    def _option_tokens(self, x: torch.Tensor, option_features: torch.Tensor | None) -> torch.Tensor:
        learned = self.option_embeddings.expand(x.shape[0], self.num_roots, -1, -1)
        if option_features is None or self.option_feature_proj is None:
            return learned
        opt_feat = option_features.to(dtype=x.dtype, device=x.device)
        if opt_feat.dim() == 2:
            opt_feat = opt_feat.unsqueeze(0).expand(x.shape[0], -1, -1)
        if opt_feat.shape[1] != self.num_options:
            # Keep inference robust when an older checkpoint/sample has a shorter
            # option list: pad or truncate to the checkpoint geometry.
            fixed = torch.zeros((opt_feat.shape[0], self.num_options, opt_feat.shape[-1]), dtype=opt_feat.dtype, device=opt_feat.device)
            n = min(self.num_options, opt_feat.shape[1])
            fixed[:, :n] = opt_feat[:, :n]
            opt_feat = fixed
        semantic = self.option_feature_proj(opt_feat).unsqueeze(1).expand(-1, self.num_roots, -1, -1)
        return learned + semantic

    def _candidate_raw_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw candidate-conditioned blocks used only by the value adapter."""
        if self.direct_candidate_raw_dim <= 0:
            return x[:, :0]
        # FlatFeatureLayout places ego, prefix parameters, macro/scalars, prefix
        # states and controls first, so the candidate-conditioned slice is
        # contiguous and does not include future/audit labels.
        return x[:, : self.direct_candidate_raw_dim]

    @staticmethod
    def _noncompensatory_smooth_cap(
        free_logit: torch.Tensor, safety_cap_logit: torch.Tensor, temperature: float
    ) -> torch.Tensor:
        """Differentiable upper cap that is never above either input."""
        tau = max(float(temperature), 1.0e-4)
        stacked = torch.stack([free_logit, safety_cap_logit], dim=-1)
        return -tau * torch.logsumexp(-stacked / tau, dim=-1)

    @staticmethod
    def _candidate_minus_nominal(
        values: torch.Tensor,
        group_index: torch.Tensor | None,
        is_nominal: torch.Tensor | None,
    ) -> torch.Tensor:
        """Return candidate-minus-nominal values with exact group provenance.

        v48.38 RFR composes benefit and component-safety factors only after
        converting each learned factor into a nominal-relative physical margin.
        The operation is regime agnostic and fails closed to zeros for malformed
        groups instead of borrowing information across scenes.
        """
        out = torch.zeros_like(values)
        if group_index is None or is_nominal is None or values.shape[0] <= 0:
            return out
        groups = group_index.to(device=values.device)
        groups = groups.reshape(-1, 1) if groups.dim() == 1 else groups.reshape(groups.shape[0], -1)
        nominal_mask = is_nominal.to(device=values.device).reshape(-1) > 0.5
        if groups.shape[0] != values.shape[0] or nominal_mask.shape[0] != values.shape[0]:
            return out
        for key in torch.unique(groups, dim=0):
            idx = torch.where((groups == key.unsqueeze(0)).all(dim=1))[0]
            noms = idx[nominal_mask[idx]]
            if noms.numel() != 1:
                continue
            group_rows = values.index_select(0, idx)
            nominal_row = values.index_select(0, noms[:1])
            out.index_copy_(0, idx, group_rows - nominal_row)
        return out

    @classmethod
    def _counterfactual_tail_response(
        cls,
        interaction: torch.Tensor,
        group_index: torch.Tensor | None,
        is_nominal: torch.Tensor | None,
    ) -> torch.Tensor:
        """Candidate-minus-nominal latent root-option response for v48.83.

        The absolute v48.82 root-option field can explain scene context even when
        a candidate action did not *cause* a recoverability change.  V48.83 removes
        that scene-common component by differencing the frozen root-option
        interaction against the unique nominal action in the same scene-time
        counterfactual set.  The operation uses observation/candidate-side frozen
        states only: no teacher metadata, future label, regime ID or relative-ranker
        output enters it.  Malformed groups fail closed to exact zeros through
        :meth:`_candidate_minus_nominal`, and the nominal row is exactly zero.
        """
        return cls._candidate_minus_nominal(interaction, group_index, is_nominal)

    @staticmethod
    def _common_measure_root_logits(
        root_logits: torch.Tensor,
        group_index: torch.Tensor | None,
        is_nominal: torch.Tensor | None,
        root_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Broadcast one nominal root posterior across each counterfactual set.

        v48.57 CMRI separates the *latent-world measure* from action-specific
        recovery consequences.  The learned root decoder is left untouched; only
        the logits consumed by OC-MERO/native-certificate aggregation are
        projected.  Malformed groups (no unique nominal, singleton, mismatched
        metadata) fail closed to their original logits instead of borrowing a
        root posterior from another scene-time group.

        Using the nominal anchor rather than a candidate-set average is important:
        the common measure must not change when proposal candidates are added,
        removed, or reordered.  A second fail-closed condition requires the
        root-valid support mask to be identical across the group when masks are
        supplied.  Otherwise a nominal mass could be projected onto a root slot
        whose candidate margin was never supervised, which would no longer be a
        mathematically common measure.
        """
        if group_index is None or is_nominal is None or root_logits.shape[0] <= 1:
            return root_logits
        groups = group_index.to(device=root_logits.device)
        groups = groups.reshape(-1, 1) if groups.dim() == 1 else groups.reshape(groups.shape[0], -1)
        nominal_mask = is_nominal.to(device=root_logits.device).reshape(-1) > 0.5
        if groups.shape[0] != root_logits.shape[0] or nominal_mask.shape[0] != root_logits.shape[0]:
            return root_logits
        valid_mask = None
        if root_valid is not None:
            candidate_valid = root_valid.to(device=root_logits.device, dtype=torch.bool)
            candidate_valid = (
                candidate_valid.reshape(-1, 1)
                if candidate_valid.dim() == 1
                else candidate_valid.reshape(candidate_valid.shape[0], -1)
            )
            if candidate_valid.shape != root_logits.shape:
                return root_logits
            valid_mask = candidate_valid
        projected = root_logits.clone()
        for key in torch.unique(groups, dim=0):
            idx = torch.where((groups == key.unsqueeze(0)).all(dim=1))[0]
            if idx.numel() <= 1:
                continue
            noms = idx[nominal_mask[idx]]
            if noms.numel() != 1:
                continue
            if valid_mask is not None:
                group_valid = valid_mask.index_select(0, idx)
                nominal_valid = valid_mask.index_select(0, noms[:1])
                if not bool(torch.all(group_valid == nominal_valid)):
                    continue
            nominal_logits = root_logits.index_select(0, noms[:1])
            shared_logits = nominal_logits.expand(idx.numel(), -1).contiguous()
            projected.index_copy_(0, idx, shared_logits)
        return projected

    def _direct_candidate_raw_relative_features(
        self,
        x: torch.Tensor,
        group_index: torch.Tensor | None,
        is_nominal: torch.Tensor | None,
    ) -> torch.Tensor:
        """Return executable-prefix raw features relative to the nominal action.

        The slice contains prefix parameters, macro identity, prefix states, and
        controls only.  It excludes ego state plus the scalar utility/hard/harm/
        feasibility/nominal/time block before candidate-minus-nominal subtraction.
        This prevents target shortcuts and absolute scene identity from entering
        the compact evidence calibrators.  The operation is permutation-equivariant
        and uses no regime label.
        """
        if not self.direct_candidate_physical_slices:
            return x[:, :0]
        raw = torch.cat(
            [x[:, start:end] for start, end in self.direct_candidate_physical_slices],
            dim=-1,
        )
        out = torch.zeros_like(raw)
        if raw.shape[-1] == 0 or group_index is None or is_nominal is None:
            return out
        groups = group_index.to(device=raw.device)
        groups = groups.reshape(-1, 1) if groups.dim() == 1 else groups.reshape(groups.shape[0], -1)
        nominal_mask = is_nominal.to(device=raw.device).reshape(-1) > 0.5
        if groups.shape[0] != raw.shape[0] or nominal_mask.shape[0] != raw.shape[0]:
            return out
        for key in torch.unique(groups, dim=0):
            idx = torch.where((groups == key.unsqueeze(0)).all(dim=1))[0]
            noms = idx[nominal_mask[idx]]
            if noms.numel() != 1:
                continue
            # Avoid CUDA advanced-index assignment with a tensor-valued slice
            # boundary.  PyTorch 2.5/CUDA can route the broadcast through
            # index_put_ and mis-compute the expanded RHS geometry.  Explicit
            # row selection plus index_copy_ has identical semantics, preserves
            # gradients, and is stable on both CPU and CUDA.
            group_rows = raw.index_select(0, idx)
            nominal_row = raw.index_select(0, noms[:1])
            out.index_copy_(0, idx, group_rows - nominal_row)
        return out

    def _direct_nominal_observation_features(
        self,
        x: torch.Tensor,
        group_index: torch.Tensor | None,
        is_nominal: torch.Tensor | None,
    ) -> torch.Tensor:
        """Broadcast observation-only features from the unique nominal row.

        Candidate rows cannot replace the scene anchor.  Audit scalars and every
        candidate action field are excluded by construction, and no regime label
        is consumed.
        """
        if not self.direct_observation_slices:
            return x[:, :0]
        raw = torch.cat([x[:, start:end] for start, end in self.direct_observation_slices], dim=-1)
        out = torch.zeros_like(raw)
        if raw.shape[-1] == 0 or group_index is None or is_nominal is None:
            return out
        groups = group_index.to(device=raw.device)
        groups = groups.reshape(-1, 1) if groups.dim() == 1 else groups.reshape(groups.shape[0], -1)
        nominal_mask = is_nominal.to(device=raw.device).reshape(-1) > 0.5
        if groups.shape[0] != raw.shape[0] or nominal_mask.shape[0] != raw.shape[0]:
            return out
        for key in torch.unique(groups, dim=0):
            idx = torch.where((groups == key.unsqueeze(0)).all(dim=1))[0]
            noms = idx[nominal_mask[idx]]
            if noms.numel() != 1:
                continue
            # Do not rely on implicit [1, D] -> [N, D] broadcasting inside a
            # CUDA advanced-index assignment.  The v48.36 A30 run failed here
            # with an internal Indexing.cu assertion (N*D=4232 versus D*D=279841
            # for N=8, D=529).  Materialise the intended source geometry and use
            # index_copy_ so the row count is explicit to the kernel.
            nominal_row = raw.index_select(0, noms[:1])
            broadcast_rows = nominal_row.expand(idx.numel(), -1).contiguous()
            out.index_copy_(0, idx, broadcast_rows)
        return out

    @staticmethod
    def _postprefix_observation_equivalence_signature(
        root_logits: torch.Tensor,
        obs_embeddings: torch.Tensor,
        tau_obs: float,
    ) -> torch.Tensor:
        """Return a bounded structural summary of predicted post-prefix observability.

        Coordinates are deliberately low-dimensional and regime agnostic:
        normalized root entropy, probability-weighted observation-alias mass,
        probability-weighted peak alias pressure, and maximum root probability.
        All are functions only of the model's predicted latent-root distribution
        and post-prefix observation kernel for the candidate action.
        """
        p = torch.softmax(root_logits.float(), dim=-1)
        k = int(p.shape[-1])
        if k <= 1:
            entropy = p.new_zeros((p.shape[0],))
        else:
            entropy = -(p * p.clamp_min(1.0e-8).log()).sum(dim=-1)
            entropy = entropy / float(torch.log(torch.tensor(float(k))).item())
        obs = obs_embeddings.float()
        diff = obs.unsqueeze(2) - obs.unsqueeze(1)
        dist2 = diff.square().mean(dim=-1)
        compatibility = torch.exp(-dist2 / max(float(tau_obs), 1.0e-6)).clamp(0.0, 1.0)
        eye = torch.eye(k, dtype=compatibility.dtype, device=compatibility.device).unsqueeze(0)
        offdiag = compatibility * (1.0 - eye)
        alias_mass = torch.einsum('bi,bij,bj->b', p, offdiag, p)
        peak_alias = (offdiag.amax(dim=-1) * p).sum(dim=-1)
        root_peak = p.amax(dim=-1)
        signature = torch.stack([entropy, alias_mass, peak_alias, root_peak], dim=-1)
        return signature.clamp(0.0, 1.0)

    def _direct_postprefix_observation_signature(
        self, memory: torch.Tensor
    ) -> torch.Tensor:
        """Compute frozen candidate-specific post-prefix observability evidence.

        Evidence adaptation must not rotate the root decoder or observation
        kernel.  The structural signal is therefore detached by construction;
        only the tiny transport projection is trainable.
        """
        with torch.no_grad():
            root_tokens = self._decode_roots(memory.detach())
            root_logits = self.root_logit_head(root_tokens).squeeze(-1)
            obs_embeddings = self.obs_embed_head(root_tokens)
            return self._postprefix_observation_equivalence_signature(
                root_logits, obs_embeddings, self.tau_obs
            ).to(dtype=memory.dtype).detach()

    @staticmethod
    def _recovery_option_compatibility_signature(
        root_logits: torch.Tensor,
        obs_embeddings: torch.Tensor,
        margins: torch.Tensor,
        tau_obs: float,
        alpha: float,
        beta: float,
        top_m: int,
        option_temperature: float,
        root_valid: torch.Tensor | None = None,
        option_valid: torch.Tensor | None = None,
        return_native_certificate: bool = False,
        physical_student_drs: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Return the frozen observation-consistent recovery compatibility signature.

        v48.43 POET only measured whether roots remain observation-aliased.  The
        paper's deployability definition is stricter: aliased roots must admit a
        *shared recovery option*.  ROCT therefore combines the exact predicted
        OC-MERO deployable score/gap with a soft pairwise shared-option conflict
        pressure and the probability mass with a feasible shared option.  No
        regime label or audit stratum is consumed.
        """
        logits = root_logits.float()
        B, K = logits.shape
        if root_valid is not None:
            rv = root_valid.to(device=logits.device, dtype=torch.bool)
            if rv.dim() == 1:
                rv = rv.unsqueeze(0).expand(B, -1)
            if rv.shape != logits.shape:
                rv = None
        else:
            rv = None
        if rv is not None:
            logits = logits.masked_fill(~rv, -1.0e4)
        p = torch.softmax(logits, dim=-1)
        if rv is not None:
            p = torch.where(rv, p, torch.zeros_like(p))
            p = p / p.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)

        obs = obs_embeddings.float()
        diff = obs.unsqueeze(2) - obs.unsqueeze(1)
        dist2 = diff.square().mean(dim=-1)
        compatibility = torch.exp(-dist2 / max(float(tau_obs), 1.0e-6)).clamp(0.0, 1.0)
        eye_bool = torch.eye(K, dtype=torch.bool, device=compatibility.device).unsqueeze(0)
        compatibility = torch.where(
            eye_bool, torch.ones_like(compatibility), compatibility
        )

        ov = option_valid
        if ov is not None:
            ov = ov.to(device=margins.device, dtype=torch.bool)
            if ov.dim() == 1:
                ov = ov.unsqueeze(0).expand(B, -1)
            if ov.shape[0] != B or ov.shape[1] != margins.shape[-1]:
                ov = None

        r_dep, _r_orc, gap, q = torch_oc_mero(
            margins.float(),
            p,
            compatibility,
            alpha=float(alpha),
            beta=float(beta),
            option_valid=ov,
            root_valid=rv,
            use_lcvar=True,
            use_obs_kernel=True,
            top_m=int(top_m),
        )

        # Pairwise option compatibility: for two observation-aliased roots, the
        # common recovery support is the best option's minimum success across the
        # pair.  Weight only off-diagonal alias mass.
        tau = max(float(option_temperature), 1.0e-4)
        success = torch.sigmoid(margins.float() / tau)
        if ov is not None:
            success = torch.where(ov.unsqueeze(1), success, torch.zeros_like(success))
        pair_common = torch.minimum(
            success.unsqueeze(2), success.unsqueeze(1)
        ).amax(dim=-1)
        offdiag = compatibility * (~eye_bool).to(dtype=compatibility.dtype)
        pair_weight = p.unsqueeze(2) * p.unsqueeze(1) * offdiag
        alias_mass = pair_weight.sum(dim=(1, 2)).clamp(0.0, 1.0)
        common_num = (pair_weight * pair_common).sum(dim=(1, 2))
        common_support = torch.where(
            alias_mass > 1.0e-8,
            common_num / alias_mass.clamp_min(1.0e-8),
            torch.ones_like(alias_mass),
        ).clamp(0.0, 1.0)
        conflict_pressure = (alias_mass * (1.0 - common_support)).clamp(0.0, 1.0)

        q_best = q.amax(dim=-1)
        shared_feasible = torch.sigmoid(q_best / tau)
        if rv is not None:
            shared_feasible = torch.where(rv, shared_feasible, torch.zeros_like(shared_feasible))
        shared_feasible_mass = (p * shared_feasible).sum(dim=-1).clamp(0.0, 1.0)
        # v48.53 CSE tests whether the *student/deployment* certificate must
        # share the same composition as the physical teacher: robust q selects
        # the observation-consistent option, then the selected predicted margin
        # (not q itself) owns the physical zero crossing.  The legacy q-hard
        # coordinate is retained byte-for-byte when the factor is disabled.
        if bool(physical_student_drs):
            valid_q = torch.ones_like(q, dtype=torch.bool)
            if rv is not None:
                valid_q = valid_q & rv.unsqueeze(-1)
            if ov is not None:
                valid_q = valid_q & ov.unsqueeze(1)
            valid_q = valid_q & torch.isfinite(q)
            option_score = q.clamp(-5.0, 5.0) + 0.01 * ((q >= 0.0) & valid_q).to(dtype=q.dtype)
            option_score = torch.where(valid_q, option_score, torch.full_like(option_score, -1.0e9))
            opt = option_score.argmax(dim=-1)
            selected_margin = torch.gather(margins.float(), 2, opt.unsqueeze(-1)).squeeze(-1)
            selected_valid = torch.ones_like(selected_margin, dtype=torch.bool)
            if rv is not None:
                selected_valid = selected_valid & rv
            if ov is not None:
                selected_valid = selected_valid & torch.gather(
                    ov.unsqueeze(1).expand(-1, K, -1), 2, opt.unsqueeze(-1)
                ).squeeze(-1)
            hard_shared_feasible = (selected_margin >= 0.0).to(dtype=p.dtype)
            hard_shared_feasible = torch.where(
                selected_valid, hard_shared_feasible, torch.zeros_like(hard_shared_feasible)
            )
        else:
            hard_shared_feasible = (q_best >= 0.0).to(dtype=p.dtype)
            if rv is not None:
                hard_shared_feasible = torch.where(
                    rv, hard_shared_feasible, torch.zeros_like(hard_shared_feasible)
                )
        native_drs = (p * hard_shared_feasible).sum(dim=-1).clamp(0.0, 1.0)

        dep_unit = (0.5 * (torch.tanh(r_dep) + 1.0)).clamp(0.0, 1.0)
        gap_unit = torch.tanh(torch.relu(gap)).clamp(0.0, 1.0)
        signature = torch.stack(
            [dep_unit, gap_unit, conflict_pressure, shared_feasible_mass], dim=-1
        )
        if not return_native_certificate:
            return signature
        # v48.49 keeps the first two v48.48 coordinates byte-for-byte compatible
        # and appends two monotone native coordinates.  shared_feasible_mass is a
        # strictly monotone boundary-resolution of q_best around the *same* zero
        # frontier (tau is the already-frozen global ROCT option temperature).
        # gap_quality exactly matches the teacher convention exp(-max(gap,0)).
        dep_score = torch.sigmoid(r_dep).clamp(0.0, 1.0)
        gap_quality = torch.exp(-torch.relu(gap).clamp(max=20.0)).clamp(0.0, 1.0)
        native_certificate = torch.stack(
            [native_drs, dep_score, shared_feasible_mass, gap_quality], dim=-1
        )
        return signature, native_certificate

    def _direct_recovery_option_compatibility_evidence(
        self,
        memory: torch.Tensor,
        x: torch.Tensor,
        option_features: torch.Tensor | None,
        group_index: torch.Tensor | None = None,
        is_nominal: torch.Tensor | None = None,
        root_valid: torch.Tensor | None = None,
        option_valid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute frozen ROCT signature and native OC-MERO certificate once."""
        with torch.no_grad():
            root_tokens = self._decode_roots(memory.detach())
            root_logits = self.root_logit_head(root_tokens).squeeze(-1)
            if self.direct_recovery_evidence_common_measure_root_mass:
                root_logits = self._common_measure_root_logits(
                    root_logits, group_index, is_nominal, root_valid
                )
            obs_embeddings = self.obs_embed_head(root_tokens)
            root_expand = root_tokens.unsqueeze(2).expand(-1, -1, self.num_options, -1)
            opt_expand = self._option_tokens(x, option_features)
            margins = self.margin_head(torch.cat([root_expand, opt_expand], dim=-1)).squeeze(-1)
            signature, native = self._recovery_option_compatibility_signature(
                root_logits, obs_embeddings, margins, self.tau_obs,
                self.direct_recovery_evidence_roct_alpha,
                self.direct_recovery_evidence_roct_beta,
                self.direct_recovery_evidence_roct_top_m,
                self.direct_recovery_evidence_roct_option_temperature,
                root_valid=root_valid, option_valid=option_valid,
                return_native_certificate=True,
                physical_student_drs=self.direct_recovery_evidence_physical_student_drs,
            )
            return (
                signature.to(dtype=memory.dtype).detach(),
                native.to(dtype=memory.dtype).detach(),
            )

    def _direct_absolute_physical_headroom_features(
        self, supplied_features: torch.Tensor | None, *, batch_size: int, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        """Validate the v48.60.1 full-prefix CPHR side-channel.

        v48.60.0 reconstructed physical coordinates from the encoder's 80-D
        prefix block.  That block truncates a 10x9 executable prefix and cannot
        represent the true terminal state.  CPHR therefore fails closed unless
        the data/inference layer supplies the six coordinates computed from the
        complete raw executable prefix.  Stage-I still receives the unchanged
        historical flat feature tensor.
        """
        if supplied_features is None:
            raise RuntimeError(
                "CPHR full-prefix features missing: v48.60.1 requires "
                "direct_absolute_physical_headroom_features side-channel; "
                "legacy v48.60.0 truncated-prefix reconstruction is disabled"
            )
        feat = supplied_features.to(device=device, dtype=dtype)
        if feat.ndim != 2 or feat.shape[0] != int(batch_size) or feat.shape[1] != 6:
            raise RuntimeError(
                f"invalid CPHR full-prefix feature shape {tuple(feat.shape)}; expected ({batch_size}, 6)"
            )
        if not bool(torch.isfinite(feat).all()):
            raise RuntimeError("non-finite CPHR full-prefix feature value")
        return feat.detach()

    def _direct_physical_headroom_absolute_feasibility(
        self, x: torch.Tensor, native_recovery_certificate: torch.Tensor | None,
        physical_headroom_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """v48.60 CPHR source-only correction of native absolute feasibility."""
        if self.direct_absolute_physical_headroom_weight is None:
            return None
        if native_recovery_certificate is None:
            raise RuntimeError("CPHR requires frozen native OC-MERO certificate")
        native_probability = native_recovery_certificate[:, 1].detach().to(dtype=x.dtype).clamp(1.0e-6, 1.0 - 1.0e-6)
        native_logit = torch.logit(native_probability)
        features = self._direct_absolute_physical_headroom_features(
            physical_headroom_features, batch_size=x.shape[0], dtype=x.dtype, device=x.device
        )
        # Projected bounded weights encode the physical monotonicity assumption:
        # more signed headroom cannot reduce feasibility.  The no-bias source is
        # exact-native at initialization and cannot learn a global threshold shift.
        weights = self.direct_absolute_physical_headroom_weight.clamp(0.0, 2.0)
        correction = (features * weights.view(1, -1)).sum(dim=-1)
        logit = native_logit + correction
        probability = torch.sigmoid(logit)
        return logit, probability, features, weights


    def _direct_absolute_executable_witness_features(
        self,
        supplied_features: torch.Tensor | None,
        *,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Validate the v48.61 option-resolved ERWF side channel [B,L,6]."""
        if supplied_features is None:
            raise RuntimeError(
                "ERWF features missing: v48.61 requires the full executable "
                "candidate x recovery-option continuation witness side-channel"
            )
        feat = supplied_features.to(device=device, dtype=dtype)
        expected = (int(batch_size), int(self.num_options), 6)
        if feat.ndim != 3 or tuple(feat.shape) != expected:
            raise RuntimeError(
                f"invalid ERWF feature shape {tuple(feat.shape)}; expected {expected}"
            )
        if not bool(torch.isfinite(feat).all()):
            raise RuntimeError("non-finite ERWF feature value")
        return feat.detach()

    def _direct_executable_witness_absolute_feasibility(
        self,
        memory: torch.Tensor,
        x: torch.Tensor,
        option_features: torch.Tensor | None,
        executable_witness_features: torch.Tensor | None,
        group_index: torch.Tensor | None = None,
        is_nominal: torch.Tensor | None = None,
        root_valid: torch.Tensor | None = None,
        option_valid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """v48.61 ERWF source: correct native margins with executable witnesses.

        Frozen Stage-I predicts root probabilities, observation equivalence and
        root x option margins exactly as before.  ERWF adds one shared signed
        physical correction to each option's margin, then re-runs the unchanged
        observation-consistent OC-MERO source.  No candidate-level threshold
        shift, free option bias, regime ID or privileged teacher component enters
        the source.  At zero weights this is execution-identical to native B.
        """
        if self.direct_absolute_executable_witness_weight is None:
            return None
        with torch.no_grad():
            root_tokens = self._decode_roots(memory.detach())
            root_logits = self.root_logit_head(root_tokens).squeeze(-1)
            if self.direct_recovery_evidence_common_measure_root_mass:
                root_logits = self._common_measure_root_logits(
                    root_logits, group_index, is_nominal, root_valid
                )
            obs_embeddings = self.obs_embed_head(root_tokens)
            root_expand = root_tokens.unsqueeze(2).expand(-1, -1, self.num_options, -1)
            opt_expand = self._option_tokens(x, option_features)
            base_margins = self.margin_head(
                torch.cat([root_expand, opt_expand], dim=-1)
            ).squeeze(-1)
            root_logits = root_logits.detach()
            obs_embeddings = obs_embeddings.detach()
            base_margins = base_margins.detach()
        features = self._direct_absolute_executable_witness_features(
            executable_witness_features,
            batch_size=x.shape[0],
            dtype=memory.dtype,
            device=memory.device,
        )
        # A shared non-negative physical readout preserves coordinate semantics;
        # the signed features themselves determine whether a particular
        # continuation raises or lowers that option's margin.
        weights = self.direct_absolute_executable_witness_weight.clamp(0.0, 2.0)
        option_correction = torch.einsum("blf,f->bl", features, weights)
        corrected_margins = base_margins + option_correction.unsqueeze(1)
        _signature, native = self._recovery_option_compatibility_signature(
            root_logits,
            obs_embeddings,
            corrected_margins,
            self.tau_obs,
            self.direct_recovery_evidence_roct_alpha,
            self.direct_recovery_evidence_roct_beta,
            self.direct_recovery_evidence_roct_top_m,
            self.direct_recovery_evidence_roct_option_temperature,
            root_valid=root_valid,
            option_valid=option_valid,
            return_native_certificate=True,
            physical_student_drs=self.direct_recovery_evidence_physical_student_drs,
        )
        probability = native[:, 1].to(dtype=memory.dtype).clamp(1.0e-6, 1.0 - 1.0e-6)
        logit = torch.logit(probability)
        return logit, probability, features, weights

    def _direct_absolute_common_witness_features(
        self,
        supplied_features: torch.Tensor | None,
        *,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Validate the v48.62 OC-CWRF side channel [B,L,10]."""
        if supplied_features is None:
            raise RuntimeError(
                "OC-CWRF features missing: v48.62 requires the option-resolved "
                "finite-time recovery witness side-channel"
            )
        feat = supplied_features.to(device=device, dtype=dtype)
        expected = (int(batch_size), int(self.num_options), 10)
        if feat.ndim != 3 or tuple(feat.shape) != expected:
            raise RuntimeError(
                f"invalid OC-CWRF feature shape {tuple(feat.shape)}; expected {expected}"
            )
        if not bool(torch.isfinite(feat).all()):
            raise RuntimeError("non-finite OC-CWRF feature value")
        return feat.detach()

    def _direct_absolute_semantic_witness_features(
        self, supplied_features: torch.Tensor | None, *, batch_size: int,
        dtype: torch.dtype, device: torch.device,
    ) -> torch.Tensor:
        """Validate the v48.64/v48.66 semantic witness side channel.

        v48.64/v48.65 use schema-1 [B,L,12].  v48.66 keeps those coordinates
        byte-semantically identical and appends route/re-entry in schema-2
        [B,L,14] whenever either active-constraint factor is enabled.
        """
        if supplied_features is None:
            raise RuntimeError(
                "OC-SARW features missing: v48.64 requires the semantics-aligned "
                "option-resolved recovery witness side-channel"
            )
        feat = supplied_features.to(device=device, dtype=dtype)
        feature_dim = (
            22 if (
                self.direct_recovery_semantic_witness_interaction_anchor_support
                or self.direct_recovery_semantic_witness_interaction_response_support
            ) else
            (20 if (
                self.direct_recovery_semantic_witness_interaction_box_support
                or self.direct_recovery_semantic_witness_interaction_hull_support
            ) else
            (18 if (
                self.direct_recovery_semantic_witness_boundary_localized_occupancy_trust
                or self.direct_recovery_semantic_witness_history_occupancy_reachability
            ) else
            (15 if self.direct_recovery_semantic_witness_soft_occupancy_disagreement else
            (14 if (
                self.direct_recovery_semantic_witness_route_alignment
                or self.direct_recovery_semantic_witness_reentry_alignment
                or self.direct_recovery_semantic_witness_control_projection
                or self.direct_recovery_semantic_witness_boundary_transport
                or self.direct_recovery_semantic_witness_projection_fidelity_weighting
                or self.direct_recovery_semantic_witness_demand_normalized_fidelity
                or self.direct_recovery_semantic_witness_robust_occupancy
            ) else 12))))
        )
        expected = (int(batch_size), int(self.num_options), feature_dim)
        if feat.ndim != 3 or tuple(feat.shape) != expected:
            raise RuntimeError(
                f"invalid OC-SARW feature shape {tuple(feat.shape)}; expected {expected}"
            )
        if not bool(torch.isfinite(feat).all()):
            raise RuntimeError("non-finite OC-SARW feature value")
        return feat.detach()

    def _direct_common_witness_absolute_feasibility(
        self,
        memory: torch.Tensor,
        x: torch.Tensor,
        option_features: torch.Tensor | None,
        common_witness_features: torch.Tensor | None,
        group_index: torch.Tensor | None = None,
        is_nominal: torch.Tensor | None = None,
        root_valid: torch.Tensor | None = None,
        option_valid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """v48.62 OC-CWRF source: non-compensatory common recovery witness.

        Two facts are required before an option receives positive rescue:
        (1) its executable continuation establishes a finite-time physical
            recovery barrier, and
        (2) the *same option* remains a plausible recovery across observation-
            equivalent latent roots.

        Physical hard constraints are composed by min, not a compensatory sum.
        Clearance/stability use ``max(invariant, finite-time recovery)`` so a
        Contact state is not rejected merely because the initial point already
        violates a barrier.  The frozen root model provides only relative
        option-commonality; no latent root ID, teacher future, regime ID or free
        per-option parameter is introduced.  Zero gains exactly recover native B.
        """
        if self.direct_absolute_common_witness_gain is None:
            return None
        with torch.no_grad():
            root_tokens = self._decode_roots(memory.detach())
            root_logits = self.root_logit_head(root_tokens).squeeze(-1)
            if self.direct_recovery_evidence_common_measure_root_mass:
                root_logits = self._common_measure_root_logits(
                    root_logits, group_index, is_nominal, root_valid
                )
            obs_embeddings = self.obs_embed_head(root_tokens)
            root_expand = root_tokens.unsqueeze(2).expand(-1, -1, self.num_options, -1)
            opt_expand = self._option_tokens(x, option_features)
            base_margins = self.margin_head(
                torch.cat([root_expand, opt_expand], dim=-1)
            ).squeeze(-1)
            root_logits = root_logits.detach()
            obs_embeddings = obs_embeddings.detach()
            base_margins = base_margins.detach()

        features = self._direct_absolute_common_witness_features(
            common_witness_features,
            batch_size=x.shape[0], dtype=memory.dtype, device=memory.device,
        )
        (h_min, h_terminal, h_gain, h_stop, h_control, h_stab_min,
         h_stab_terminal, h_stab_gain, h_clear_floor_gain, h_stab_floor_gain) = [
            features[..., i] for i in range(10)
        ]
        # Invariance OR finite-time recovery.  A recovery branch is accepted only
        # if it reaches a positive terminal barrier, improves from the initial
        # violated state, and never becomes worse than that initial state.  This
        # prevents terminal-only recovery from hiding secondary contact/instability.
        clear_recovery = torch.minimum(h_terminal, h_gain)
        clear_recovery_ok = (clear_recovery > 0.0) & (h_clear_floor_gain >= 0.0)
        clearance_barrier = torch.where(clear_recovery_ok, clear_recovery, h_min)
        stab_recovery = torch.minimum(h_stab_terminal, h_stab_gain)
        stab_recovery_ok = (stab_recovery > 0.0) & (h_stab_floor_gain >= 0.0)
        stability_barrier = torch.where(stab_recovery_ok, stab_recovery, h_stab_min)

        if option_features is None:
            raise RuntimeError("OC-CWRF requires recovery option semantic features")
        of = option_features.to(device=memory.device, dtype=memory.dtype)
        if of.ndim != 3 or of.shape[0] != x.shape[0] or of.shape[1] != self.num_options or of.shape[2] < 8:
            raise RuntimeError(f"invalid option feature shape for OC-CWRF: {tuple(of.shape)}")
        # Public option semantics, not learned per-option bias.  Stopping reserve
        # is a hard barrier only for modes whose controller semantics require it.
        stop_active = (of[..., 0] > 0.5) | (of[..., 1] > 0.5) | (of[..., 3] > 0.5) | (of[..., 4] > 0.5)
        stop_barrier = torch.where(stop_active, h_stop, torch.ones_like(h_stop))
        physical_viability = torch.minimum(
            torch.minimum(clearance_barrier, stop_barrier),
            torch.minimum(h_control, stability_barrier),
        )

        # Frozen observation-consistent common-option support.  Use relative
        # option preference within each root, avoiding the native source's global
        # negative offset from turning commonality into another false veto.
        logits = root_logits.float()
        B, K = logits.shape
        rv = root_valid.to(device=logits.device, dtype=torch.bool) if root_valid is not None else None
        if rv is not None and rv.dim() == 1:
            rv = rv.unsqueeze(0).expand(B, -1)
        if rv is not None and rv.shape != logits.shape:
            rv = None
        if rv is not None:
            logits = logits.masked_fill(~rv, -1.0e4)
        p = torch.softmax(logits, dim=-1)
        if rv is not None:
            p = torch.where(rv, p, torch.zeros_like(p))
            p = p / p.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)

        ov = option_valid.to(device=base_margins.device, dtype=torch.bool) if option_valid is not None else None
        if ov is not None and ov.dim() == 1:
            ov = ov.unsqueeze(0).expand(B, -1)
        if ov is not None and (ov.shape[0] != B or ov.shape[1] != self.num_options):
            ov = None
        margin_for_best = base_margins.float()
        if ov is not None:
            margin_for_best = margin_for_best.masked_fill(~ov.unsqueeze(1), -1.0e4)
        best = margin_for_best.amax(dim=-1, keepdim=True)
        tau = max(float(self.direct_recovery_evidence_roct_option_temperature), 1.0e-4)
        relative_support = torch.exp(((margin_for_best - best) / tau).clamp(-20.0, 0.0))
        if ov is not None:
            relative_support = torch.where(ov.unsqueeze(1), relative_support, torch.zeros_like(relative_support))
        if rv is not None:
            relative_support = torch.where(rv.unsqueeze(-1), relative_support, torch.zeros_like(relative_support))

        obs = obs_embeddings.float()
        dist2 = (obs.unsqueeze(2) - obs.unsqueeze(1)).square().mean(dim=-1)
        compatibility = torch.exp(-dist2 / max(float(self.tau_obs), 1.0e-6)).clamp(0.0, 1.0)
        eye = torch.eye(K, dtype=torch.bool, device=compatibility.device).unsqueeze(0)
        offdiag = compatibility * (~eye).to(dtype=compatibility.dtype)
        if rv is not None:
            offdiag = offdiag * (rv.unsqueeze(2) & rv.unsqueeze(1)).to(dtype=offdiag.dtype)
        pair_weight = p.unsqueeze(2) * p.unsqueeze(1) * offdiag
        alias_mass = pair_weight.sum(dim=(1, 2))
        pair_common = torch.minimum(relative_support.unsqueeze(2), relative_support.unsqueeze(1))
        common_num = (pair_weight.unsqueeze(-1) * pair_common).sum(dim=(1, 2))
        pair_common_support = common_num / alias_mass.unsqueeze(-1).clamp_min(1.0e-8)
        root_weighted_support = (p.unsqueeze(-1) * relative_support).sum(dim=1)
        common_support = torch.where(
            (alias_mass > 1.0e-8).unsqueeze(-1), pair_common_support, root_weighted_support
        ).clamp(0.0, 1.0)
        if ov is not None:
            common_support = torch.where(ov, common_support, torch.zeros_like(common_support))
        common_support = common_support.detach().to(dtype=memory.dtype)

        gains = self.direct_absolute_common_witness_gain.clamp(0.0, 2.0)
        option_correction = (
            gains[0] * common_support * torch.relu(physical_viability)
            - gains[1] * torch.relu(-physical_viability)
        )
        corrected_margins = base_margins + option_correction.unsqueeze(1)
        _signature, native = self._recovery_option_compatibility_signature(
            root_logits, obs_embeddings, corrected_margins, self.tau_obs,
            self.direct_recovery_evidence_roct_alpha,
            self.direct_recovery_evidence_roct_beta,
            self.direct_recovery_evidence_roct_top_m,
            self.direct_recovery_evidence_roct_option_temperature,
            root_valid=root_valid, option_valid=option_valid,
            return_native_certificate=True,
            physical_student_drs=self.direct_recovery_evidence_physical_student_drs,
        )
        probability = native[:, 1].to(dtype=memory.dtype).clamp(1.0e-6, 1.0 - 1.0e-6)
        logit = torch.logit(probability)
        return logit, probability, features, gains, physical_viability.detach(), common_support

    def _direct_quantifier_witness_absolute_feasibility(
        self,
        memory: torch.Tensor,
        x: torch.Tensor,
        option_features: torch.Tensor | None,
        common_witness_features: torch.Tensor | None,
        group_index: torch.Tensor | None = None,
        is_nominal: torch.Tensor | None = None,
        root_valid: torch.Tensor | None = None,
        option_valid: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    ] | None:
        """v48.63 OC-QARW: quantifier-aligned common recovery witness.

        Recoverability is existential over a *common* recovery option.  Positive
        evidence therefore remains option-local: one observation-consistent
        executable option may rescue the native source.  Infeasibility has the
        dual universal semantics: a negative correction is allowed only when
        every valid common option has negative physical viability.  This avoids
        the v48.62 error mode where failure of many irrelevant options produced a
        broad candidate-level downward shift even when one option could recover.

        The physical field and common-option support are identical to v48.62;
        only the logical composition changes.  No regime id, teacher future,
        option-specific parameter, threshold search, or Stage-I update is used.
        Zero gains are execution-exact native B.
        """
        if self.direct_absolute_quantifier_witness_gain is None:
            return None
        with torch.no_grad():
            root_tokens = self._decode_roots(memory.detach())
            root_logits = self.root_logit_head(root_tokens).squeeze(-1)
            if self.direct_recovery_evidence_common_measure_root_mass:
                root_logits = self._common_measure_root_logits(
                    root_logits, group_index, is_nominal, root_valid
                )
            obs_embeddings = self.obs_embed_head(root_tokens)
            root_expand = root_tokens.unsqueeze(2).expand(-1, -1, self.num_options, -1)
            opt_expand = self._option_tokens(x, option_features)
            base_margins = self.margin_head(
                torch.cat([root_expand, opt_expand], dim=-1)
            ).squeeze(-1)
            root_logits = root_logits.detach()
            obs_embeddings = obs_embeddings.detach()
            base_margins = base_margins.detach()

        features = self._direct_absolute_common_witness_features(
            common_witness_features,
            batch_size=x.shape[0], dtype=memory.dtype, device=memory.device,
        )
        (h_min, h_terminal, h_gain, h_stop, h_control, h_stab_min,
         h_stab_terminal, h_stab_gain, h_clear_floor_gain, h_stab_floor_gain) = [
            features[..., i] for i in range(10)
        ]
        clear_recovery = torch.minimum(h_terminal, h_gain)
        clear_recovery_ok = (clear_recovery > 0.0) & (h_clear_floor_gain >= 0.0)
        clearance_barrier = torch.where(clear_recovery_ok, clear_recovery, h_min)
        stab_recovery = torch.minimum(h_stab_terminal, h_stab_gain)
        stab_recovery_ok = (stab_recovery > 0.0) & (h_stab_floor_gain >= 0.0)
        stability_barrier = torch.where(stab_recovery_ok, stab_recovery, h_stab_min)

        if option_features is None:
            raise RuntimeError("OC-QARW requires recovery option semantic features")
        of = option_features.to(device=memory.device, dtype=memory.dtype)
        if of.ndim != 3 or of.shape[0] != x.shape[0] or of.shape[1] != self.num_options or of.shape[2] < 8:
            raise RuntimeError(f"invalid option feature shape for OC-QARW: {tuple(of.shape)}")
        stop_active = (of[..., 0] > 0.5) | (of[..., 1] > 0.5) | (of[..., 3] > 0.5) | (of[..., 4] > 0.5)
        stop_barrier = torch.where(stop_active, h_stop, torch.ones_like(h_stop))
        physical_viability = torch.minimum(
            torch.minimum(clearance_barrier, stop_barrier),
            torch.minimum(h_control, stability_barrier),
        )

        # Same frozen observation-consistent common-option support as v48.62.
        logits = root_logits.float()
        B, K = logits.shape
        rv = root_valid.to(device=logits.device, dtype=torch.bool) if root_valid is not None else None
        if rv is not None and rv.dim() == 1:
            rv = rv.unsqueeze(0).expand(B, -1)
        if rv is not None and rv.shape != logits.shape:
            rv = None
        if rv is not None:
            logits = logits.masked_fill(~rv, -1.0e4)
        p = torch.softmax(logits, dim=-1)
        if rv is not None:
            p = torch.where(rv, p, torch.zeros_like(p))
            p = p / p.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)

        ov = option_valid.to(device=base_margins.device, dtype=torch.bool) if option_valid is not None else None
        if ov is not None and ov.dim() == 1:
            ov = ov.unsqueeze(0).expand(B, -1)
        if ov is not None and (ov.shape[0] != B or ov.shape[1] != self.num_options):
            ov = None
        margin_for_best = base_margins.float()
        if ov is not None:
            margin_for_best = margin_for_best.masked_fill(~ov.unsqueeze(1), -1.0e4)
        best = margin_for_best.amax(dim=-1, keepdim=True)
        tau = max(float(self.direct_recovery_evidence_roct_option_temperature), 1.0e-4)
        relative_support = torch.exp(((margin_for_best - best) / tau).clamp(-20.0, 0.0))
        if ov is not None:
            relative_support = torch.where(ov.unsqueeze(1), relative_support, torch.zeros_like(relative_support))
        if rv is not None:
            relative_support = torch.where(rv.unsqueeze(-1), relative_support, torch.zeros_like(relative_support))

        obs = obs_embeddings.float()
        dist2 = (obs.unsqueeze(2) - obs.unsqueeze(1)).square().mean(dim=-1)
        compatibility = torch.exp(-dist2 / max(float(self.tau_obs), 1.0e-6)).clamp(0.0, 1.0)
        eye = torch.eye(K, dtype=torch.bool, device=compatibility.device).unsqueeze(0)
        offdiag = compatibility * (~eye).to(dtype=compatibility.dtype)
        if rv is not None:
            offdiag = offdiag * (rv.unsqueeze(2) & rv.unsqueeze(1)).to(dtype=offdiag.dtype)
        pair_weight = p.unsqueeze(2) * p.unsqueeze(1) * offdiag
        alias_mass = pair_weight.sum(dim=(1, 2))
        pair_common = torch.minimum(relative_support.unsqueeze(2), relative_support.unsqueeze(1))
        common_num = (pair_weight.unsqueeze(-1) * pair_common).sum(dim=(1, 2))
        pair_common_support = common_num / alias_mass.unsqueeze(-1).clamp_min(1.0e-8)
        root_weighted_support = (p.unsqueeze(-1) * relative_support).sum(dim=1)
        common_support = torch.where(
            (alias_mass > 1.0e-8).unsqueeze(-1), pair_common_support, root_weighted_support
        ).clamp(0.0, 1.0)
        if ov is not None:
            common_support = torch.where(ov, common_support, torch.zeros_like(common_support))
        common_support = common_support.detach().to(dtype=memory.dtype)

        gains = self.direct_absolute_quantifier_witness_gain.clamp(0.0, 2.0)
        positive_rescue = gains[0] * common_support * torch.relu(physical_viability)

        # Quantifier alignment: feasibility is exists(option), so infeasibility
        # is certified only by failure of the best supported valid option.  A
        # single negative option cannot veto a candidate if another common option
        # remains viable.  Low common support also weakens, rather than strengthens,
        # a negative claim because lack of support is epistemic absence, not proof.
        supported_viability = common_support * physical_viability
        if ov is not None:
            masked_supported = supported_viability.masked_fill(~ov, -1.0e4)
            valid_any = ov.any(dim=-1)
            best_common_viability = masked_supported.amax(dim=-1)
            best_common_viability = torch.where(
                valid_any, best_common_viability, torch.zeros_like(best_common_viability)
            )
            positive_option_count = ((supported_viability > 0.0) & ov).sum(dim=-1).to(dtype=memory.dtype)
            max_common_support = torch.where(
                ov, common_support, torch.zeros_like(common_support)
            ).amax(dim=-1)
        else:
            best_common_viability = supported_viability.amax(dim=-1)
            positive_option_count = (supported_viability > 0.0).sum(dim=-1).to(dtype=memory.dtype)
            max_common_support = common_support.amax(dim=-1)
        universal_failure = torch.relu(-best_common_viability)
        if self.direct_recovery_semantic_witness_boundary_transport:
            # v48.67 bounded boundary transport.  The v48.66 additive lift had
            # no relation to a root-option's native deficit: even an all-positive
            # trusted witness could remain below R_dep=0.  Convert the tanh
            # certificate back to a normalized signed reserve and interpolate
            # monotonically toward the common-support-scaled positive target.
            # gain=0 is exact native B; gain=2 reaches the trusted target without
            # lowering an already safer native margin.
            eps = torch.finfo(physical_viability.dtype).eps * 16.0
            cert_margin = torch.atanh(physical_viability.clamp(-1.0 + eps, 1.0 - eps))
            # Absolute admission needs the sign/boundary, not arbitrarily large
            # slack magnitude.  Cap at one normalized reserve unit so the
            # transport cannot explode when every active barrier is far safe.
            cert_margin = torch.relu(cert_margin.float()).clamp(max=1.0)
            certified_target = common_support.float() * cert_margin
            target = certified_target.unsqueeze(1).to(dtype=base_margins.dtype)
            rho = (gains[0] / 2.0).clamp(0.0, 1.0)
            positive_delta = torch.relu(target - base_margins)
            positive_delta = positive_delta * (certified_target > 0.0).unsqueeze(1).to(dtype=positive_delta.dtype)
            corrected_margins = (
                base_margins + rho * positive_delta
                - gains[1] * universal_failure.unsqueeze(-1).unsqueeze(1)
            )
        else:
            positive_rescue = gains[0] * common_support * torch.relu(physical_viability)
            option_correction = positive_rescue - gains[1] * universal_failure.unsqueeze(-1)
            corrected_margins = base_margins + option_correction.unsqueeze(1)
        _signature, native = self._recovery_option_compatibility_signature(
            root_logits, obs_embeddings, corrected_margins, self.tau_obs,
            self.direct_recovery_evidence_roct_alpha,
            self.direct_recovery_evidence_roct_beta,
            self.direct_recovery_evidence_roct_top_m,
            self.direct_recovery_evidence_roct_option_temperature,
            root_valid=root_valid, option_valid=option_valid,
            return_native_certificate=True,
            physical_student_drs=self.direct_recovery_evidence_physical_student_drs,
        )
        probability = native[:, 1].to(dtype=memory.dtype).clamp(1.0e-6, 1.0 - 1.0e-6)
        logit = torch.logit(probability)
        return (
            logit, probability, features, gains, physical_viability.detach(), common_support,
            best_common_viability.detach(), universal_failure.detach(),
            positive_option_count.detach(), max_common_support.detach(),
        )

    def _direct_semantic_witness_absolute_feasibility(
        self,
        memory: torch.Tensor,
        x: torch.Tensor,
        option_features: torch.Tensor | None,
        semantic_witness_features: torch.Tensor | None,
        group_index: torch.Tensor | None = None,
        is_nominal: torch.Tensor | None = None,
        root_valid: torch.Tensor | None = None,
        option_valid: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None,
        torch.Tensor | None,
    ] | None:
        """v48.64 OC-SARW: semantics-aligned common executable witness.

        v48.63 showed that quantifier alignment alone is inert because most
        teacher-safe-positive candidates have no *positive observable physical
        witness* despite high common-option support.  OC-SARW therefore holds
        the candidate x option rollout, common support, exists/forall logic,
        Stage-I and threshold fixed, and changes only two constraint semantics:

        1. stopping reserve is measured along executable free path capacity,
           instead of terminal radial clearance;
        2. stability is active only when the observable prefix is already in
           contact/unstable or the option explicitly stabilizes post-contact.

        Both switches are global factor-ablation flags, never regime inputs.
        Zero gains remain execution-exact native B.
        """
        if self.direct_absolute_semantic_witness_gain is None and self.direct_absolute_root_tail_source_scale is None and self.direct_absolute_structured_tail_field_weight is None:
            return None
        with torch.no_grad():
            root_tokens = self._decode_roots(memory.detach())
            root_logits = self.root_logit_head(root_tokens).squeeze(-1)
            if self.direct_recovery_evidence_common_measure_root_mass:
                root_logits = self._common_measure_root_logits(
                    root_logits, group_index, is_nominal, root_valid
                )
            obs_embeddings = self.obs_embed_head(root_tokens)
            root_expand = root_tokens.unsqueeze(2).expand(-1, -1, self.num_options, -1)
            opt_expand = self._option_tokens(x, option_features)
            base_margins = self.margin_head(
                torch.cat([root_expand, opt_expand], dim=-1)
            ).squeeze(-1)
            root_logits = root_logits.detach()
            obs_embeddings = obs_embeddings.detach()
            base_margins = base_margins.detach()

        features = self._direct_absolute_semantic_witness_features(
            semantic_witness_features,
            batch_size=x.shape[0], dtype=memory.dtype, device=memory.device,
        )
        (h_min, h_terminal, h_gain, h_stop_legacy, h_control, h_stab_min,
         h_stab_terminal, h_stab_gain, h_clear_floor_gain, h_stab_floor_gain,
         h_path_stop, stability_active_obs) = [features[..., i] for i in range(12)]
        if features.shape[-1] >= 14:
            h_route = features[..., 12]
            h_reentry = features[..., 13]
        else:
            h_route = torch.ones_like(h_min)
            h_reentry = torch.ones_like(h_min)
        h_occupancy_optimism = (
            features[..., 14] if features.shape[-1] >= 15 else torch.zeros_like(h_min)
        )
        h_current_boundary_deficit = (
            features[..., 15] if features.shape[-1] >= 18 else torch.zeros_like(h_min)
        )
        h_history_occupancy_optimism = (
            features[..., 16] if features.shape[-1] >= 18 else torch.zeros_like(h_min)
        )
        h_history_boundary_deficit = (
            features[..., 17] if features.shape[-1] >= 18 else torch.zeros_like(h_min)
        )
        h_interaction_box_optimism = (
            features[..., 18] if features.shape[-1] >= 20 else torch.zeros_like(h_min)
        )
        h_interaction_hull_optimism = (
            features[..., 19] if features.shape[-1] >= 20 else torch.zeros_like(h_min)
        )
        h_interaction_anchor_optimism = (
            features[..., 20] if features.shape[-1] >= 22 else torch.zeros_like(h_min)
        )
        h_interaction_response_optimism = (
            features[..., 21] if features.shape[-1] >= 22 else torch.zeros_like(h_min)
        )

        clear_recovery = torch.minimum(h_terminal, h_gain)
        clear_recovery_ok = (clear_recovery > 0.0) & (h_clear_floor_gain >= 0.0)
        clearance_barrier = torch.where(clear_recovery_ok, clear_recovery, h_min)
        stab_recovery = torch.minimum(h_stab_terminal, h_stab_gain)
        stab_recovery_ok = (stab_recovery > 0.0) & (h_stab_floor_gain >= 0.0)
        raw_stability_barrier = torch.where(stab_recovery_ok, stab_recovery, h_stab_min)

        if option_features is None:
            raise RuntimeError("OC-SARW requires recovery option semantic features")
        of = option_features.to(device=memory.device, dtype=memory.dtype)
        if of.ndim != 3 or of.shape[0] != x.shape[0] or of.shape[1] != self.num_options or of.shape[2] < 8:
            raise RuntimeError(f"invalid option feature shape for OC-SARW: {tuple(of.shape)}")
        stop_active = (of[..., 0] > 0.5) | (of[..., 1] > 0.5) | (of[..., 3] > 0.5) | (of[..., 4] > 0.5)
        chosen_stop = h_path_stop if self.direct_recovery_semantic_witness_path_stop_alignment else h_stop_legacy
        stop_barrier = torch.where(stop_active, chosen_stop, torch.ones_like(chosen_stop))
        if self.direct_recovery_semantic_witness_active_set_alignment:
            stability_barrier = torch.where(
                stability_active_obs > 0.5, raw_stability_barrier, torch.ones_like(raw_stability_barrier)
            )
        else:
            stability_barrier = raw_stability_barrier

        # In the projected-control factor the actual executable recovery trace
        # is already magnitude/rate/jerk feasible by construction.  Do not
        # re-veto the same policy using the historical desired-command barrier;
        # certify only the remaining environment/state-dependent constraints.
        effective_control_barrier = (
            torch.ones_like(h_control)
            if self.direct_recovery_semantic_witness_control_projection
            else h_control
        )
        barriers = [clearance_barrier, stop_barrier, effective_control_barrier, stability_barrier]
        if self.direct_recovery_semantic_witness_route_alignment:
            barriers.append(h_route)
        if self.direct_recovery_semantic_witness_reentry_alignment:
            barriers.append(h_reentry)
        barrier_stack = torch.stack(barriers, dim=-1)
        physical_viability, limiting_constraint = barrier_stack.min(dim=-1)

        # Exact frozen common-option support from v48.62/v48.63.
        logits = root_logits.float()
        B, K = logits.shape
        rv = root_valid.to(device=logits.device, dtype=torch.bool) if root_valid is not None else None
        if rv is not None and rv.dim() == 1:
            rv = rv.unsqueeze(0).expand(B, -1)
        if rv is not None and rv.shape != logits.shape:
            rv = None
        if rv is not None:
            logits = logits.masked_fill(~rv, -1.0e4)
        p = torch.softmax(logits, dim=-1)
        if rv is not None:
            p = torch.where(rv, p, torch.zeros_like(p))
            p = p / p.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)

        ov = option_valid.to(device=base_margins.device, dtype=torch.bool) if option_valid is not None else None
        if ov is not None and ov.dim() == 1:
            ov = ov.unsqueeze(0).expand(B, -1)
        if ov is not None and (ov.shape[0] != B or ov.shape[1] != self.num_options):
            ov = None
        margin_for_best = base_margins.float()
        if ov is not None:
            margin_for_best = margin_for_best.masked_fill(~ov.unsqueeze(1), -1.0e4)
        best = margin_for_best.amax(dim=-1, keepdim=True)
        tau = max(float(self.direct_recovery_evidence_roct_option_temperature), 1.0e-4)
        relative_support = torch.exp(((margin_for_best - best) / tau).clamp(-20.0, 0.0))
        if ov is not None:
            relative_support = torch.where(ov.unsqueeze(1), relative_support, torch.zeros_like(relative_support))
        if rv is not None:
            relative_support = torch.where(rv.unsqueeze(-1), relative_support, torch.zeros_like(relative_support))

        obs = obs_embeddings.float()
        dist2 = (obs.unsqueeze(2) - obs.unsqueeze(1)).square().mean(dim=-1)
        compatibility = torch.exp(-dist2 / max(float(self.tau_obs), 1.0e-6)).clamp(0.0, 1.0)
        eye = torch.eye(K, dtype=torch.bool, device=compatibility.device).unsqueeze(0)
        offdiag = compatibility * (~eye).to(dtype=compatibility.dtype)
        if rv is not None:
            offdiag = offdiag * (rv.unsqueeze(2) & rv.unsqueeze(1)).to(dtype=offdiag.dtype)
        pair_weight = p.unsqueeze(2) * p.unsqueeze(1) * offdiag
        alias_mass = pair_weight.sum(dim=(1, 2))
        pair_common = torch.minimum(relative_support.unsqueeze(2), relative_support.unsqueeze(1))
        common_num = (pair_weight.unsqueeze(-1) * pair_common).sum(dim=(1, 2))
        pair_common_support = common_num / alias_mass.unsqueeze(-1).clamp_min(1.0e-8)
        root_weighted_support = (p.unsqueeze(-1) * relative_support).sum(dim=1)
        common_support = torch.where(
            (alias_mass > 1.0e-8).unsqueeze(-1), pair_common_support, root_weighted_support
        ).clamp(0.0, 1.0)
        if ov is not None:
            common_support = torch.where(ov, common_support, torch.zeros_like(common_support))
        common_support = common_support.detach().to(dtype=memory.dtype)

        # v48.68 projection-fidelity factor.  Under Q_CTRLPROJ the actual
        # applied controls are feasible by construction, so h_control must not
        # become a hard veto.  However v48.67 showed that erasing the raw
        # desired-command violation entirely admits many false certificates.
        # Convert the raw normalized control violation into a strictly-positive
        # confidence multiplier: zero/positive reserve -> 1, one violated
        # normalized unit -> 1/2, larger violations decay smoothly.  Because the
        # multiplier never changes sign, witness *existence* is unchanged; only
        # the amount of positive rescue is discounted when realizing the mode
        # requires a large actuator projection.
        if self.direct_recovery_semantic_witness_projection_fidelity_weighting:
            eps_fid = torch.finfo(h_control.dtype).eps * 16.0
            raw_control_margin = torch.atanh(
                h_control.float().clamp(-1.0 + eps_fid, 1.0 - eps_fid)
            )
            control_violation = torch.relu(-raw_control_margin)
            if self.direct_recovery_semantic_witness_demand_normalized_fidelity:
                # v48.69 OC-DTRW: v48.68 T proved raw projection severity is a
                # useful trust signal, but it disproportionately suppressed the
                # scarce safe-positive candidates that are already in an
                # observed recovery-demanding state.  Reconstruct the normalized
                # candidate-terminal clearance deficit directly from the frozen
                # signed witness coordinates:
                #   atanh(h_gain)-atanh(h_terminal)
                #     = (d_safe-prefix_terminal_clearance)/distance_scale.
                # The stability analogue is identical in normalized yaw units.
                # A zero-demand state is execution-exact v48.68 T.  Increasing
                # observed demand only *tempers* the soft projection penalty; it
                # never changes physical witness sign or creates a free rescue.
                clear_terminal_margin = torch.atanh(
                    h_terminal.float().clamp(-1.0 + eps_fid, 1.0 - eps_fid)
                )
                clear_gain_margin = torch.atanh(
                    h_gain.float().clamp(-1.0 + eps_fid, 1.0 - eps_fid)
                )
                clearance_demand = torch.relu(clear_gain_margin - clear_terminal_margin)

                stab_terminal_margin = torch.atanh(
                    h_stab_terminal.float().clamp(-1.0 + eps_fid, 1.0 - eps_fid)
                )
                stab_gain_margin = torch.atanh(
                    h_stab_gain.float().clamp(-1.0 + eps_fid, 1.0 - eps_fid)
                )
                stability_demand = torch.relu(stab_gain_margin - stab_terminal_margin)
                if self.direct_recovery_semantic_witness_active_set_alignment:
                    stability_demand = stability_demand * (stability_active_obs > 0.5).to(
                        dtype=stability_demand.dtype
                    )
                recovery_demand = torch.maximum(clearance_demand, stability_demand)
                projection_fidelity = (
                    (1.0 + recovery_demand)
                    / (1.0 + recovery_demand + control_violation)
                ).to(dtype=memory.dtype)
            else:
                projection_fidelity = (1.0 / (1.0 + control_violation)).to(dtype=memory.dtype)
            common_support = common_support * projection_fidelity

        if self.direct_recovery_semantic_witness_soft_occupancy_disagreement:
            # v48.70 raw point-disagreement trust (historical ablation only).
            eps_occ = torch.finfo(h_occupancy_optimism.dtype).eps * 16.0
            occupancy_optimism_gap = torch.relu(torch.atanh(
                h_occupancy_optimism.float().clamp(-1.0 + eps_occ, 1.0 - eps_occ)
            ))
            occupancy_trust = (1.0 / (1.0 + occupancy_optimism_gap)).to(dtype=memory.dtype)
            common_support = common_support * occupancy_trust

        if (
            self.direct_recovery_semantic_witness_boundary_localized_occupancy_trust
            or self.direct_recovery_semantic_witness_history_occupancy_reachability
        ):
            # v48.71 OC-BORW factorial semantics:
            #   H: current-CA boundary deficit (boundary localization only)
            #   J: history-tube raw CV optimism (history reachability only)
            #   K: history-tube boundary deficit (both factors / Main)
            # The trust variable is a normalized non-negative reserve deficit,
            # and w=1/(1+deficit) is strictly positive.  Thus the intervention
            # cannot alter witness existence/sign and only discounts the amount
            # of positive rescue when an observation-supported occupancy model
            # actually threatens the physical clearance boundary.
            if self.direct_recovery_semantic_witness_history_occupancy_reachability:
                h_occ_risk = (
                    h_history_boundary_deficit
                    if self.direct_recovery_semantic_witness_boundary_localized_occupancy_trust
                    else h_history_occupancy_optimism
                )
            else:
                h_occ_risk = h_current_boundary_deficit
            eps_occ71 = torch.finfo(h_occ_risk.dtype).eps * 16.0
            occupancy_risk = torch.relu(torch.atanh(
                h_occ_risk.float().clamp(-1.0 + eps_occ71, 1.0 - eps_occ71)
            ))
            occupancy_reachability_trust = (1.0 / (1.0 + occupancy_risk)).to(
                dtype=memory.dtype
            )
            common_support = common_support * occupancy_reachability_trust

        if (
            self.direct_recovery_semantic_witness_interaction_box_support
            or self.direct_recovery_semantic_witness_interaction_hull_support
        ):
            # v48.72 uses static interaction-oriented ambiguity geometry.  v48.73
            # keeps those diagnostics execution-exact but *replaces* their trust
            # multiplier with a current-state-anchored temporal response support
            # when the new nested flags are enabled.  No occupancy trust factors
            # are stacked.  Every selected risk is non-negative and w>0, so the
            # historical CV physical-certificate sign/set remains frozen.
            if self.direct_recovery_semantic_witness_interaction_response_support:
                h_occ = h_interaction_response_optimism
            elif self.direct_recovery_semantic_witness_interaction_anchor_support:
                h_occ = h_interaction_anchor_optimism
            else:
                h_occ = (
                    h_interaction_hull_optimism
                    if self.direct_recovery_semantic_witness_interaction_hull_support
                    else h_interaction_box_optimism
                )
            if _v48_74_signed_viability_enabled() and (
                self.direct_recovery_semantic_witness_interaction_anchor_support
                or self.direct_recovery_semantic_witness_interaction_response_support
            ):
                # V48.74 coordinates 20/21 are already raw non-negative
                # normalized viability debts, so consume them directly.  The
                # historical v48.72/v48.73 coordinates are tanh-encoded and keep
                # the exact atanh decoder below when the V48.74 switch is off.
                interaction_risk = torch.relu(h_occ.float())
            else:
                eps_occ = torch.finfo(h_occ.dtype).eps * 16.0
                interaction_risk = torch.relu(torch.atanh(
                    h_occ.float().clamp(-1.0 + eps_occ, 1.0 - eps_occ)
                ))
            interaction_trust = (1.0 / (1.0 + interaction_risk)).to(dtype=memory.dtype)
            common_support = common_support * interaction_trust

        gains = (
            self.direct_absolute_semantic_witness_gain.clamp(0.0, 2.0)
            if self.direct_absolute_semantic_witness_gain is not None
            else torch.zeros(2, dtype=memory.dtype, device=memory.device)
        )
        root_tail_source = self.direct_recovery_semantic_witness_root_tail_source
        typed_source = self.direct_recovery_semantic_witness_active_constraint_typed_source
        if typed_source:
            if gains.shape != (6, 2):
                raise RuntimeError(f"OC-ACTSI requires semantic gain shape (6,2), got {tuple(gains.shape)}")
            if self.direct_recovery_semantic_witness_classlocal_transport:
                raise RuntimeError("OC-ACTSI is candidate-global and cannot be combined with class-local transport")
            if self.direct_recovery_semantic_witness_boundary_transport:
                raise RuntimeError("OC-ACTSI does not reopen boundary transport")
            if barrier_stack.shape[-1] != 6:
                raise RuntimeError("OC-ACTSI requires the fixed six-slot active-constraint stack including route and re-entry")

        # v48.65 OC-CLRW factor: the paper's OC-MERO information pattern is
        # observation-class local.  q[i,l] has already aggregated all roots
        # that are compatible with anchor observation i, so it is the correct
        # locus for a deployable correction.  v48.64 instead formed one
        # candidate-level support c_l and broadcast the same correction to all
        # roots.  The class-local branch leaves v48.64 numerical behavior
        # execution-exact when disabled (covered by regression tests).
        if self.direct_recovery_semantic_witness_classlocal_transport:
            obs = obs_embeddings.float()
            dist2_q = (obs.unsqueeze(2) - obs.unsqueeze(1)).square().mean(dim=-1)
            compatibility_q = torch.exp(
                -dist2_q / max(float(self.tau_obs), 1.0e-6)
            ).clamp(0.0, 1.0)
            eye_q = torch.eye(K, dtype=torch.bool, device=compatibility_q.device).unsqueeze(0)
            compatibility_q = torch.where(
                eye_q, torch.ones_like(compatibility_q), compatibility_q
            )
            native_r_dep, _native_r_orc, _native_gap, q_base = torch_oc_mero(
                base_margins.float(), p, compatibility_q,
                alpha=float(self.direct_recovery_evidence_roct_alpha),
                beta=float(self.direct_recovery_evidence_roct_beta),
                option_valid=ov, root_valid=rv, use_lcvar=True, use_obs_kernel=True,
                top_m=int(self.direct_recovery_evidence_roct_top_m),
            )
            q_for_support = q_base
            if ov is not None:
                q_for_support = q_for_support.masked_fill(~ov.unsqueeze(1), -1.0e9)
            q_best = q_for_support.amax(dim=-1, keepdim=True)
            class_support = torch.exp(
                ((q_for_support - q_best) / tau).clamp(-20.0, 0.0)
            )
            if ov is not None:
                class_support = torch.where(
                    ov.unsqueeze(1), class_support, torch.zeros_like(class_support)
                )
            if rv is not None:
                class_support = torch.where(
                    rv.unsqueeze(-1), class_support, torch.zeros_like(class_support)
                )

            class_supported_viability = (
                class_support * physical_viability.float().unsqueeze(1)
            )
            if ov is not None:
                class_supported_for_best = class_supported_viability.masked_fill(
                    ~ov.unsqueeze(1), -1.0e9
                )
            else:
                class_supported_for_best = class_supported_viability
            per_class_best, per_class_best_option = class_supported_for_best.max(dim=-1)
            if rv is not None:
                per_class_best = torch.where(rv, per_class_best, torch.zeros_like(per_class_best))

            # Rescue only the option that is locally compatible with each
            # observation class; a class receives a negative correction only
            # when all locally supported observable continuations fail.
            positive_rescue_q = (
                gains[0] * class_support.to(dtype=memory.dtype)
                * torch.relu(physical_viability).unsqueeze(1)
            )
            class_failure = torch.relu(-per_class_best).to(dtype=memory.dtype)
            q_corrected = (
                q_base.to(dtype=memory.dtype) + positive_rescue_q
                - gains[1] * class_failure.unsqueeze(-1)
            )
            if ov is not None:
                q_corrected = torch.where(
                    ov.unsqueeze(1), q_corrected, torch.full_like(q_corrected, -1.0e9)
                )
            r_per_class = q_corrected.amax(dim=-1)
            corrected_r_dep = torch_weighted_lcvar(
                r_per_class.float(), p.float(),
                float(self.direct_recovery_evidence_roct_alpha),
            )
            probability = torch.sigmoid(corrected_r_dep).to(dtype=memory.dtype).clamp(
                1.0e-6, 1.0 - 1.0e-6
            )
            logit = torch.logit(probability)

            classlocal_lcvar_viability = torch_weighted_lcvar(
                per_class_best.float(), p.float(),
                float(self.direct_recovery_evidence_roct_alpha),
            ).to(dtype=memory.dtype)
            viable_root_mass = (
                p * (per_class_best > 0.0).to(dtype=p.dtype)
            ).sum(dim=-1).to(dtype=memory.dtype)
            selected_support = torch.gather(
                class_support, 2, per_class_best_option.unsqueeze(-1)
            ).squeeze(-1)
            selected_support_mean = (p * selected_support).sum(dim=-1).to(dtype=memory.dtype)

            # For limiting-constraint diagnosis choose the weakest valid
            # observation class, then that class's best supported option.
            weak_score = per_class_best.clone()
            if rv is not None:
                weak_score = weak_score.masked_fill(~rv, float('inf'))
            weak_class = weak_score.argmin(dim=-1)
            batch_index = torch.arange(B, device=memory.device)
            best_option = per_class_best_option[batch_index, weak_class]
            best_barriers = barrier_stack[batch_index, best_option]
            best_limiting_constraint = limiting_constraint[batch_index, best_option].to(dtype=memory.dtype)

            # Compatibility field retained for the existing audit schema: in
            # class-local mode this counts observation classes that have at
            # least one positive supported option (not root-option pairs).
            positive_class = per_class_best > 0.0
            if rv is not None:
                positive_class = positive_class & rv
            positive_option_count = positive_class.sum(dim=1).to(dtype=memory.dtype)
            max_common_support = class_support.amax(dim=(1, 2)).to(dtype=memory.dtype)
            universal_failure = torch.relu(-classlocal_lcvar_viability)
            return (
                logit, probability, features, gains, physical_viability.detach(),
                class_support.detach().to(dtype=memory.dtype),
                classlocal_lcvar_viability.detach(), universal_failure.detach(),
                positive_option_count.detach(), max_common_support.detach(),
                best_barriers.detach(), best_limiting_constraint.detach(),
                classlocal_lcvar_viability.detach(), viable_root_mass.detach(),
                selected_support_mean.detach(),
            )

        supported_viability = common_support * physical_viability
        if ov is not None:
            masked_supported = supported_viability.masked_fill(~ov, -1.0e4)
            valid_any = ov.any(dim=-1)
            best_option = masked_supported.argmax(dim=-1)
            best_common_viability = masked_supported.amax(dim=-1)
            best_common_viability = torch.where(valid_any, best_common_viability, torch.zeros_like(best_common_viability))
            positive_option_count = ((supported_viability > 0.0) & ov).sum(dim=-1).to(dtype=memory.dtype)
            max_common_support = torch.where(ov, common_support, torch.zeros_like(common_support)).amax(dim=-1)
        else:
            best_option = supported_viability.argmax(dim=-1)
            best_common_viability = supported_viability.amax(dim=-1)
            positive_option_count = (supported_viability > 0.0).sum(dim=-1).to(dtype=memory.dtype)
            max_common_support = common_support.amax(dim=-1)
        universal_failure = torch.relu(-best_common_viability)
        if root_tail_source:
            # V48.78 OC-RTSI / V48.82 OC-SNTF: reshape the deployable lower tail
            # without restoring an option-wise translation degree.
            # not another option-wise translation.  The basis is derived only
            # from the frozen nested OC-MERO operator; the sole trainable state
            # is a shared scalar scale.  This is deliberately not the v48.65
            # learned class-local transport: no class/root embedding, ID, or
            # learned mixer enters the source.
            structured_tail_field = self.direct_recovery_semantic_witness_structured_tail_field
            if structured_tail_field:
                if self.direct_absolute_structured_tail_field_weight is None:
                    raise RuntimeError("v48.82 structured tail field enabled without its weights")
            elif self.direct_absolute_root_tail_source_scale is None:
                raise RuntimeError("v48.78 root-tail source is enabled without its source scale")
            if self.direct_recovery_semantic_witness_projection_fidelity_weighting:
                raise RuntimeError("v48.78 preregistered root-tail source keeps projection fidelity OFF")

            p_rt = p.float()
            scale_rt = None
            if not structured_tail_field:
                scale_rt = self.direct_absolute_root_tail_source_scale.to(
                    device=memory.device, dtype=torch.float32
                ).clamp(0.0, 2.0).reshape(1, 1, 1)

            # Keep the historical physical witness and common-option support
            # as a signed option amplitude.  Unlike v48.64--77, this amplitude
            # is never broadcast as a constant option translation.
            option_amplitude = (common_support.float() * physical_viability.float()).detach()

            # Build the exact observation-compatible inner LCVAR influence for
            # every anchor and option.  I78 averages these lower-tail exposures
            # under the frozen root measure; J78 additionally composes them with
            # the outer LCVAR influence and the native deployable best option.
            with torch.no_grad():
                C_eff = compatibility.float()
                top_m = int(self.direct_recovery_evidence_roct_top_m)
                if top_m > 0 and top_m < K:
                    vals_rt, idx_rt = torch.topk(C_eff, k=top_m, dim=-1)
                    C_sparse = torch.zeros_like(C_eff).scatter(-1, idx_rt, vals_rt)
                    eye_rt = torch.eye(K, dtype=torch.bool, device=memory.device).unsqueeze(0)
                    C_eff = torch.where(
                        eye_rt, torch.maximum(C_sparse, torch.ones_like(C_sparse)), C_sparse
                    )

                scores_rt = base_margins.float().transpose(1, 2)  # [B,L,Kroot]
                inner_exposure = torch.zeros(
                    (B, K, self.num_options), dtype=torch.float32, device=memory.device
                )
                inner_by_anchor: list[torch.Tensor] = []
                for anchor_i in range(K):
                    w_i = torch_normalize_weights(C_eff[:, anchor_i, :] * p_rt)
                    w_i = w_i.unsqueeze(1).expand(-1, self.num_options, -1)
                    inner_inf = torch_weighted_lcvar_influence(
                        scores_rt, w_i, float(self.direct_recovery_evidence_roct_beta)
                    )  # [B,L,Kroot]
                    inner_by_anchor.append(inner_inf)
                    inner_exposure = inner_exposure + (
                        p_rt[:, anchor_i].view(B, 1, 1) * inner_inf.transpose(1, 2)
                    )

                tail_basis = inner_exposure
                if self.direct_recovery_semantic_witness_tail_localization:
                    _rdep_rt, _rorc_rt, _gap_rt, q_rt = torch_oc_mero(
                        base_margins.float(), p_rt, compatibility.float(),
                        alpha=float(self.direct_recovery_evidence_roct_alpha),
                        beta=float(self.direct_recovery_evidence_roct_beta),
                        option_valid=ov, root_valid=rv, use_lcvar=True, use_obs_kernel=True,
                        top_m=top_m,
                    )
                    q_for_best_rt = q_rt
                    if ov is not None:
                        q_for_best_rt = q_for_best_rt.masked_fill(~ov.unsqueeze(1), -1.0e9)
                    best_l_rt = q_for_best_rt.argmax(dim=-1)
                    r_per_anchor_rt = q_for_best_rt.amax(dim=-1)
                    outer_inf = torch_weighted_lcvar_influence(
                        r_per_anchor_rt, p_rt, float(self.direct_recovery_evidence_roct_alpha)
                    )
                    nested_exposure = torch.zeros_like(inner_exposure)
                    for anchor_i, inner_inf in enumerate(inner_by_anchor):
                        chosen = torch.nn.functional.one_hot(
                            best_l_rt[:, anchor_i], num_classes=self.num_options
                        ).to(dtype=inner_inf.dtype)
                        nested_exposure = nested_exposure + (
                            outer_inf[:, anchor_i].view(B, 1, 1)
                            * inner_inf.transpose(1, 2)
                            * chosen.unsqueeze(1)
                        )
                    tail_basis = nested_exposure

                if rv is not None:
                    tail_basis = torch.where(
                        rv.unsqueeze(-1), tail_basis, torch.zeros_like(tail_basis)
                    )
                if ov is not None:
                    tail_basis = torch.where(
                        ov.unsqueeze(1), tail_basis, torch.zeros_like(tail_basis)
                    )

                # Remove the option-wise p-weighted translation exactly, then
                # normalize only the deterministic basis magnitude.  A constant
                # tail exposure therefore becomes the zero intervention rather
                # than another hidden gain transport.
                basis_mean = (p_rt.unsqueeze(-1) * tail_basis).sum(dim=1, keepdim=True)
                centered_basis = tail_basis - basis_mean
                basis_norm = centered_basis.abs().amax(dim=1, keepdim=True)
                tail_basis = torch.where(
                    basis_norm > 1.0e-8,
                    centered_basis / basis_norm.clamp_min(1.0e-8),
                    torch.zeros_like(centered_basis),
                ).detach()

            if structured_tail_field:
                # Shared observation-derived root-option interaction.  The
                # elementwise product is a diagonal bilinear potential over the
                # frozen root and option tokens.  It is multiplied by the exact
                # nested-tail exposure and then p-centered per option, so the
                # source can redistribute reserve/debt across latent roots but
                # cannot translate an option wholesale.
                interaction = root_expand.float() * opt_expand.float()
                interaction = torch.nn.functional.layer_norm(interaction, (interaction.shape[-1],))
                if self.direct_recovery_semantic_witness_counterfactual_tail_response:
                    # V48.83 OC-CRTF: remove scene-common absolute-field leakage by
                    # conditioning the frozen root-option interaction on the
                    # candidate's *own* frozen root-option representation change relative
                    # to the unique nominal action in the same scene-time group.  This
                    # is a counterfactual latent response, not an absolute scene field.
                    # Nominal rows and malformed groups receive exactly zero correction,
                    # while no teacher label/future, regime ID, option ID or relative-
                    # ranker output enters this source.
                    interaction = self._counterfactual_tail_response(
                        interaction, group_index, is_nominal
                    ).detach()
                w_field = self.direct_absolute_structured_tail_field_weight.to(
                    device=memory.device, dtype=torch.float32
                )
                field_all = torch.einsum('bkld,cd->bklc', interaction, w_field) / (float(self.d_model) ** 0.5)
                if self.direct_recovery_semantic_witness_signed_tail_channels:
                    channel = (base_margins.float() < 0.0).to(dtype=torch.long).unsqueeze(-1)
                    field = field_all.gather(-1, channel).squeeze(-1)
                else:
                    field = field_all[..., 0]
                # The old normalized tail basis supplies the exact nested-tail
                # localization; abs(option_amplitude) is confidence only, so the
                # field itself learns the signed reserve/debt direction.
                raw_delta = tail_basis * torch.tanh(field) * option_amplitude.abs().unsqueeze(1)
                mean_delta = (p_rt.unsqueeze(-1) * raw_delta).sum(dim=1, keepdim=True)
                root_tail_delta = raw_delta - mean_delta
            else:
                root_tail_delta = (
                    scale_rt
                    * option_amplitude.unsqueeze(1)
                    * tail_basis
                )
            if rv is not None:
                root_tail_delta = torch.where(
                    rv.unsqueeze(-1), root_tail_delta, torch.zeros_like(root_tail_delta)
                )
            if ov is not None:
                root_tail_delta = torch.where(
                    ov.unsqueeze(1), root_tail_delta, torch.zeros_like(root_tail_delta)
                )
            corrected_margins = base_margins + root_tail_delta.to(dtype=base_margins.dtype)
        elif typed_source:
            # V48.77 OC-ACTSI structured source interface.  The physical sign
            # remains the exact non-compensatory minimum.  Learning sees only
            # the *identity of the binding constraint* of each option, which is
            # the natural piecewise mode of a min-defined signed viability
            # function.  Positive rescue is therefore typed per option.  When
            # every option is infeasible, the negative correction is typed by
            # the least-infeasible common option (the same best_option used by
            # the universal-failure diagnostic) and remains candidate-global.
            type_index = limiting_constraint.to(dtype=torch.long).clamp(0, 5)
            pos_gain = gains[:, 0][type_index]
            positive_rescue = pos_gain.to(dtype=memory.dtype) * common_support * torch.relu(physical_viability)
            batch_index_typed = torch.arange(B, device=memory.device)
            failure_type = type_index[batch_index_typed, best_option]
            neg_gain = gains[:, 1][failure_type].to(dtype=memory.dtype)
            option_correction = positive_rescue - (neg_gain * universal_failure).unsqueeze(-1)
            corrected_margins = base_margins + option_correction.unsqueeze(1)
        elif self.direct_recovery_semantic_witness_boundary_transport:
            # v48.67 bounded boundary transport.  The v48.66 additive lift had
            # no relation to a root-option's native deficit: even an all-positive
            # trusted witness could remain below R_dep=0.  Convert the tanh
            # certificate back to a normalized signed reserve and interpolate
            # monotonically toward the common-support-scaled positive target.
            # gain=0 is exact native B; gain=2 reaches the trusted target without
            # lowering an already safer native margin.
            eps = torch.finfo(physical_viability.dtype).eps * 16.0
            cert_margin = torch.atanh(physical_viability.clamp(-1.0 + eps, 1.0 - eps))
            # Absolute admission needs the sign/boundary, not arbitrarily large
            # slack magnitude.  Cap at one normalized reserve unit so the
            # transport cannot explode when every active barrier is far safe.
            cert_margin = torch.relu(cert_margin.float()).clamp(max=1.0)
            certified_target = common_support.float() * cert_margin
            target = certified_target.unsqueeze(1).to(dtype=base_margins.dtype)
            rho = (gains[0] / 2.0).clamp(0.0, 1.0)
            positive_delta = torch.relu(target - base_margins)
            positive_delta = positive_delta * (certified_target > 0.0).unsqueeze(1).to(dtype=positive_delta.dtype)
            corrected_margins = (
                base_margins + rho * positive_delta
                - gains[1] * universal_failure.unsqueeze(-1).unsqueeze(1)
            )
        else:
            positive_rescue = gains[0] * common_support * torch.relu(physical_viability)
            option_correction = positive_rescue - gains[1] * universal_failure.unsqueeze(-1)
            corrected_margins = base_margins + option_correction.unsqueeze(1)
        _signature, native = self._recovery_option_compatibility_signature(
            root_logits, obs_embeddings, corrected_margins, self.tau_obs,
            self.direct_recovery_evidence_roct_alpha,
            self.direct_recovery_evidence_roct_beta,
            self.direct_recovery_evidence_roct_top_m,
            self.direct_recovery_evidence_roct_option_temperature,
            root_valid=root_valid, option_valid=option_valid,
            return_native_certificate=True,
            physical_student_drs=self.direct_recovery_evidence_physical_student_drs,
        )
        probability = native[:, 1].to(dtype=memory.dtype).clamp(1.0e-6, 1.0 - 1.0e-6)
        logit = torch.logit(probability)

        batch_index = torch.arange(B, device=memory.device)
        best_barriers = barrier_stack[batch_index, best_option]
        best_limiting_constraint = limiting_constraint[batch_index, best_option].to(dtype=memory.dtype)
        return (
            logit, probability, features, gains, physical_viability.detach(), common_support,
            best_common_viability.detach(), universal_failure.detach(),
            positive_option_count.detach(), max_common_support.detach(),
            best_barriers.detach(), best_limiting_constraint.detach(),
            None, None, None,
        )

    def _direct_option_corrected_absolute_feasibility(
        self,
        memory: torch.Tensor,
        x: torch.Tensor,
        option_features: torch.Tensor | None,
        group_index: torch.Tensor | None = None,
        is_nominal: torch.Tensor | None = None,
        root_valid: torch.Tensor | None = None,
        option_valid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """v48.59 ORFC absolute source with option-resolved margin correction.

        All Stage-I witness tensors are computed under no-grad and detached.
        Gradients can flow only to ``direct_absolute_option_margin_bias``.
        The corrected margins are passed through the exact same observation-
        consistent OC-MERO operator as the native source, so zero bias is
        execution-identical to v48.58-B and the 0.5 probability threshold is
        still exactly the physical R_dep=0 boundary.
        """
        if self.direct_absolute_option_margin_bias is None:
            return None
        with torch.no_grad():
            root_tokens = self._decode_roots(memory.detach())
            root_logits = self.root_logit_head(root_tokens).squeeze(-1)
            if self.direct_recovery_evidence_common_measure_root_mass:
                root_logits = self._common_measure_root_logits(
                    root_logits, group_index, is_nominal, root_valid
                )
            obs_embeddings = self.obs_embed_head(root_tokens)
            root_expand = root_tokens.unsqueeze(2).expand(-1, -1, self.num_options, -1)
            opt_expand = self._option_tokens(x, option_features)
            base_margins = self.margin_head(
                torch.cat([root_expand, opt_expand], dim=-1)
            ).squeeze(-1)
            root_logits = root_logits.detach()
            obs_embeddings = obs_embeddings.detach()
            base_margins = base_margins.detach()
        corrected_margins = base_margins + self.direct_absolute_option_margin_bias.view(1, 1, -1)
        _signature, native = self._recovery_option_compatibility_signature(
            root_logits,
            obs_embeddings,
            corrected_margins,
            self.tau_obs,
            self.direct_recovery_evidence_roct_alpha,
            self.direct_recovery_evidence_roct_beta,
            self.direct_recovery_evidence_roct_top_m,
            self.direct_recovery_evidence_roct_option_temperature,
            root_valid=root_valid,
            option_valid=option_valid,
            return_native_certificate=True,
            physical_student_drs=self.direct_recovery_evidence_physical_student_drs,
        )
        probability = native[:, 1].to(dtype=memory.dtype).clamp(1.0e-6, 1.0 - 1.0e-6)
        logit = torch.logit(probability)
        return logit, probability

    def _direct_recovery_option_compatibility_signature(
        self,
        memory: torch.Tensor,
        x: torch.Tensor,
        option_features: torch.Tensor | None,
        group_index: torch.Tensor | None = None,
        is_nominal: torch.Tensor | None = None,
        root_valid: torch.Tensor | None = None,
        option_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        signature, _native = self._direct_recovery_option_compatibility_evidence(
            memory, x, option_features,
            group_index=group_index, is_nominal=is_nominal,
            root_valid=root_valid, option_valid=option_valid
        )
        return signature

    def _apply_direct_set_context(
        self,
        direct_features: torch.Tensor,
        group_index: torch.Tensor | None,
        is_nominal: torch.Tensor | None,
    ) -> torch.Tensor:
        """Nominal-anchored, permutation-equivariant candidate-set adapter.

        Each candidate is represented relative to the nominal candidate and to
        exchangeable mean/max summaries of recovery alternatives.  Singleton or
        malformed groups deliberately fall back to the original pointwise feature.
        """
        if self.direct_set_context_adapter is None or group_index is None or is_nominal is None:
            return direct_features
        if direct_features.shape[0] <= 1:
            return direct_features
        groups = group_index.to(device=direct_features.device)
        if groups.dim() == 1:
            groups = groups.reshape(-1, 1)
        else:
            groups = groups.reshape(groups.shape[0], -1)
        nominal_mask = is_nominal.to(device=direct_features.device).reshape(-1) > 0.5
        if groups.shape[0] != direct_features.shape[0] or nominal_mask.shape[0] != direct_features.shape[0]:
            return direct_features
        adapted = direct_features.clone()
        unique_groups = torch.unique(groups, dim=0)
        gate = torch.sigmoid(self.direct_set_context_gate) if self.direct_set_context_gate is not None else 1.0
        for key in unique_groups:
            idx = torch.where((groups == key.unsqueeze(0)).all(dim=1))[0]
            if idx.numel() <= 1:
                continue
            noms = idx[nominal_mask[idx]]
            recs = idx[~nominal_mask[idx]]
            if noms.numel() == 0 or recs.numel() == 0:
                continue
            nom_feat = direct_features.index_select(0, noms[:1])
            group_features = direct_features.index_select(0, idx)
            recovery_features = direct_features.index_select(0, recs)
            rel = group_features - nom_feat
            rec_rel = recovery_features - nom_feat
            mean_rel = rec_rel.mean(dim=0, keepdim=True).expand(idx.numel(), -1)
            max_rel = rec_rel.max(dim=0, keepdim=True).values.expand(idx.numel(), -1)
            context_input = torch.cat([group_features, rel, mean_rel, max_rel], dim=-1)
            residual = self.direct_set_context_adapter(context_input)
            adapted.index_copy_(0, idx, group_features + gate * residual)
        return adapted

    def _direct_group_relative_features(
        self,
        direct_features: torch.Tensor,
        group_index: torch.Tensor | None,
        is_nominal: torch.Tensor | None,
    ) -> torch.Tensor:
        """Return permutation-equivariant candidate-relative features.

        The output concatenates the pointwise feature, candidate-minus-nominal,
        mean recovery-relative feature and max recovery-relative feature.  It is
        used only by the v48.6 preference-context and delta heads, so the value
        branch remains pointwise and warm-start compatible.
        """
        zeros = torch.zeros_like(direct_features)
        if self.direct_recovery_relative_features_include_absolute:
            relative = torch.cat([direct_features, zeros, zeros, zeros], dim=-1)
        else:
            relative = torch.cat([zeros, zeros, zeros], dim=-1)
        if group_index is None or is_nominal is None or direct_features.shape[0] <= 1:
            return relative
        groups = group_index.to(device=direct_features.device)
        groups = groups.reshape(-1, 1) if groups.dim() == 1 else groups.reshape(groups.shape[0], -1)
        nominal_mask = is_nominal.to(device=direct_features.device).reshape(-1) > 0.5
        if groups.shape[0] != direct_features.shape[0] or nominal_mask.shape[0] != direct_features.shape[0]:
            return relative
        out = relative.clone()
        for key in torch.unique(groups, dim=0):
            idx = torch.where((groups == key.unsqueeze(0)).all(dim=1))[0]
            if idx.numel() <= 1:
                continue
            noms = idx[nominal_mask[idx]]
            recs = idx[~nominal_mask[idx]]
            if noms.numel() == 0 or recs.numel() == 0:
                continue
            nominal = direct_features.index_select(0, noms[:1])
            group_features = direct_features.index_select(0, idx)
            recovery_features = direct_features.index_select(0, recs)
            rel = group_features - nominal
            rec_rel = recovery_features - nominal
            mean_rel = rec_rel.mean(dim=0, keepdim=True).expand(idx.numel(), -1)
            max_rel = rec_rel.max(dim=0, keepdim=True).values.expand(idx.numel(), -1)
            if self.direct_recovery_relative_features_include_absolute:
                rows = torch.cat([group_features, rel, mean_rel, max_rel], dim=-1)
            else:
                rows = torch.cat([rel, mean_rel, max_rel], dim=-1)
            out.index_copy_(0, idx, rows)
        return out

    def _native_certificate_component_logits(
        self,
        native_certificate: torch.Tensor | None,
        group_index: torch.Tensor | None,
        is_nominal: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Transport native recovery coordinates to non-compensatory harm logits.

        v48.48 NCP preserves [hard DRS, sigmoid(R_dep)].  v48.49 DCP optionally
        upgrades this to [boundary-resolved DRS, sigmoid(R_dep), gap-quality].
        Every coordinate is higher-is-safer and only a fixed monotone transform
        of OC-MERO output; no learned proxy or regime-specific routing rule is
        introduced.  Legacy coordinates use ``nominal-candidate-tolerance``.
        Under v48.56 DRAC, DEP may instead use the exact absolute
        ``0.5-sigmoid(R_dep_candidate)`` boundary margin; positive still means
        harmful in either convention.
        """
        if not self.direct_recovery_evidence_native_certificate_preservation:
            return None, None
        if native_certificate is None:
            raise RuntimeError(
                "native certificate preservation enabled but OC-MERO native certificate is missing"
            )
        native = native_certificate.to(dtype=torch.float32)
        min_width = 4 if self.direct_recovery_evidence_native_margin_complete_preservation else 2
        if native.dim() != 2 or native.shape[-1] < min_width:
            raise RuntimeError(
                f"native OC-MERO certificate must have shape [batch, >={min_width}]"
            )
        if self.direct_recovery_evidence_native_margin_complete_preservation:
            # Coordinate order presented to the downstream component contract is
            # still DRS, deployability, gap.  Only the DRS transport changes from
            # the lossy hard indicator to its zero-centred monotone boundary mass.
            safer = torch.stack([native[:, 2], native[:, 1], native[:, 3]], dim=-1)
            tolerances = [
                self.direct_recovery_evidence_native_drs_tolerance,
                self.direct_recovery_evidence_native_deployability_tolerance,
                self.direct_recovery_evidence_native_gap_tolerance,
            ]
        else:
            safer = native[:, :2]
            tolerances = [
                self.direct_recovery_evidence_native_drs_tolerance,
                self.direct_recovery_evidence_native_deployability_tolerance,
            ]
        rel = self._candidate_minus_nominal(safer, group_index, is_nominal)
        tol = rel.new_tensor(tolerances)
        harmful_margins = -rel - tol.unsqueeze(0)
        if self.direct_recovery_evidence_native_dep_boundary_aligned:
            # v48.56 DRAC: deployability is boundary-bearing.  The teacher
            # non-deployable label is R_dep<0, equivalent to sigmoid(R_dep)<0.5.
            # Use that absolute zero boundary directly instead of nominal-relative
            # quality degradation.  DRS remains nominal-relative; GAP (if present
            # in historical margin-complete modes) is unchanged.
            harmful_margins = harmful_margins.clone()
            harmful_margins[:, 1] = 0.5 - safer[:, 1]
        logits = harmful_margins / max(self.direct_recovery_evidence_slack_temperature, 1.0e-6)
        return logits, harmful_margins

    def _native_certificate_benefit_logit(
        self,
        native_certificate: torch.Tensor | None,
        group_index: torch.Tensor | None,
        is_nominal: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Return native signed recovery-advantage evidence for v48.49 DCP.

        The training/calibration benefit target is the candidate-minus-nominal
        deployability-score advantage. v48.49 used a boundary-smoothed DRS;
        v48.50 tested a full exact hard-DRS replacement. v48.51 adds a
        boundary-complete mode: exact hard PCD owns material sign outside the
        existing positive-gain band, while smooth PCD resolves ordering inside
        the hard certificate's equivalence region.

        The benefit factor remains centred at the preregistered positive-gain
        boundary, so logit zero has an explicit cross-scene physical meaning.
        No learned proxy, regime route, or new threshold is introduced.
        """
        if not self.direct_recovery_evidence_native_advantage_preservation:
            return None, None, None
        if native_certificate is None:
            raise RuntimeError(
                "native advantage preservation enabled but OC-MERO native certificate is missing"
            )
        native = native_certificate.to(dtype=torch.float32)
        if native.dim() != 2 or native.shape[-1] < 4:
            raise RuntimeError("native advantage preservation requires [batch, >=4] certificate")
        exact_value = (native[:, 0] * native[:, 1] * native[:, 3]).clamp(0.0, 1.0)
        smooth_value = (native[:, 2] * native[:, 1] * native[:, 3]).clamp(0.0, 1.0)
        if self.direct_recovery_evidence_native_boundary_complete_advantage_preservation:
            # Boundary-complete decision equivalence: outside the already
            # preregistered materiality band, the deployed hard certificate is
            # authoritative for sign. Inside that equivalence band the hard DRS
            # is intentionally under-resolved, so retain v48.49's smooth local
            # ordering rather than collapsing all ties. No new threshold is
            # introduced: positive_gain is the same physical benefit boundary
            # used by the Natural gate and the frontier loss.
            exact_rel = self._candidate_minus_nominal(
                exact_value.unsqueeze(-1), group_index, is_nominal
            ).squeeze(-1)
            smooth_rel = self._candidate_minus_nominal(
                smooth_value.unsqueeze(-1), group_index, is_nominal
            ).squeeze(-1)
            gain = float(self.direct_recovery_evidence_native_positive_gain)
            rel_adv = torch.where(
                exact_rel >= gain,
                torch.maximum(exact_rel, smooth_rel),
                torch.where(
                    exact_rel <= -gain,
                    torch.minimum(exact_rel, smooth_rel),
                    smooth_rel,
                ),
            )
            # The returned per-row value is diagnostic only for BC-NAP because
            # the effective transport is candidate-relative by construction.
            # Keep the smooth physical PCD here so existing diagnostics remain
            # comparable to v48.49-A; the authoritative decision quantity is
            # direct_recovery_evidence_native_benefit_margin below.
            native_value = smooth_value
        else:
            native_value = exact_value if self.direct_recovery_evidence_native_exact_advantage_preservation else smooth_value
            rel_adv = self._candidate_minus_nominal(
                native_value.unsqueeze(-1), group_index, is_nominal
            ).squeeze(-1)
        benefit_margin = rel_adv - self.direct_recovery_evidence_native_positive_gain
        benefit_logit = benefit_margin / max(
            self.direct_recovery_evidence_benefit_margin_temperature, 1.0e-6
        )
        return benefit_logit, benefit_margin, native_value

    def _direct_outputs(
        self,
        memory: torch.Tensor,
        x: torch.Tensor,
        bucket_id: torch.Tensor | None,
        group_index: torch.Tensor | None = None,
        is_nominal: torch.Tensor | None = None,
        postprefix_observation_signature: torch.Tensor | None = None,
        recovery_option_compatibility_signature: torch.Tensor | None = None,
        native_recovery_certificate: torch.Tensor | None = None,
        absolute_physical_headroom_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Evaluate the policy-level recovery branch.

        v48 keeps experts observation-only and uses their disagreement as a
        conservative certificate: recovery gain and opportunity receive a lower
        confidence adjustment, while harmful-switch risk receives an upper
        confidence adjustment.  No hidden regime label is needed.
        """
        if self.direct_value_head is None and self.direct_value_heads is None:
            return {}
        scene_token = memory[:, 0]
        direct_features = scene_token
        if self.direct_recovery_value_pooling in {
            "candidate_concat", "prefix_concat", "action_concat",
            "candidate_concat_raw", "action_concat_raw", "certificate_action_adapter",
        }:
            if memory.shape[1] >= 6:
                direct_features = torch.cat([memory[:, i] for i in range(6)], dim=-1)
            else:
                direct_features = scene_token.repeat(1, 6)
            if self.direct_recovery_value_pooling in {
                "candidate_concat_raw", "action_concat_raw", "certificate_action_adapter"
            }:
                direct_features = torch.cat([direct_features, self._candidate_raw_features(x)], dim=-1)
        # Preserve an untouched pointwise path for the ranking/delta heads.  The
        # legacy NASC adapter may still be enabled as an ablation for the value
        # experts, but cannot silently contaminate the new relative certificate.
        pointwise_direct_features = direct_features
        direct_features = self._apply_direct_set_context(pointwise_direct_features, group_index, is_nominal)
        if self.direct_recovery_value_router_pooling == "shared_raw":
            router_features = x[:, self.direct_candidate_feature_dim:]
        elif self.direct_recovery_value_router_pooling == "ego_shared_raw":
            router_features = torch.cat(
                [x[:, : self.direct_ego_feature_dim], x[:, self.direct_candidate_feature_dim:]],
                dim=-1,
            )
        elif self.direct_recovery_value_router_pooling == "scene":
            router_features = scene_token
        else:
            router_features = direct_features
        if self.direct_regime_embedding is not None:
            if bucket_id is None:
                bucket_id = torch.full(
                    (x.shape[0],), min(3, self.direct_recovery_value_num_regimes - 1),
                    dtype=torch.long, device=x.device,
                )
            regime_id = bucket_id.to(device=x.device, dtype=torch.long).reshape(-1)
            regime_id = regime_id.clamp(0, self.direct_recovery_value_num_regimes - 1)
            regime_features = self.direct_regime_embedding(regime_id)
            direct_features = torch.cat([direct_features, regime_features], dim=-1)
            pointwise_direct_features = torch.cat([pointwise_direct_features, regime_features], dim=-1)

        relative_features = self._direct_group_relative_features(
            pointwise_direct_features, group_index, is_nominal
        )

        out: dict[str, torch.Tensor] = {}
        roct_signature_rel: torch.Tensor | None = None
        if recovery_option_compatibility_signature is not None:
            roct_signature_abs = recovery_option_compatibility_signature.to(dtype=memory.dtype).detach()
            roct_signature_rel = self._candidate_minus_nominal(
                roct_signature_abs, group_index, is_nominal
            )
            out["direct_recovery_evidence_roct_signature"] = roct_signature_abs
            out["direct_recovery_evidence_roct_signature_relative"] = roct_signature_rel
        native_component_logits, native_component_margins = self._native_certificate_component_logits(
            native_recovery_certificate, group_index, is_nominal
        )
        native_benefit_logit, native_benefit_margin, native_recovery_value = (
            self._native_certificate_benefit_logit(
                native_recovery_certificate, group_index, is_nominal
            )
        )
        if native_recovery_certificate is not None:
            out["direct_recovery_evidence_native_certificate"] = native_recovery_certificate.detach()
        if self.direct_absolute_feasibility_head is not None:
            if recovery_option_compatibility_signature is None or native_recovery_certificate is None:
                raise RuntimeError(
                    "absolute feasibility head requires absolute ROCT signature and native certificate"
                )
            absolute_feasibility_features = torch.cat(
                [
                    recovery_option_compatibility_signature.to(dtype=memory.dtype).detach(),
                    native_recovery_certificate.to(dtype=memory.dtype).detach(),
                ],
                dim=-1,
            )
            absolute_feasibility_logit = self.direct_absolute_feasibility_head(
                absolute_feasibility_features
            ).squeeze(-1)
            out["direct_recovery_absolute_feasibility_features"] = absolute_feasibility_features
            out["direct_recovery_absolute_feasibility_logit"] = absolute_feasibility_logit
            out["direct_recovery_absolute_feasibility_probability"] = torch.sigmoid(
                absolute_feasibility_logit
            )
        physical_abs = self._direct_physical_headroom_absolute_feasibility(
            x, native_recovery_certificate, physical_headroom_features=absolute_physical_headroom_features
        )
        if physical_abs is not None:
            absolute_feasibility_logit, absolute_feasibility_probability, physical_features, physical_weights = physical_abs
            out["direct_recovery_absolute_feasibility_logit"] = absolute_feasibility_logit
            out["direct_recovery_absolute_feasibility_probability"] = absolute_feasibility_probability
            out["direct_recovery_absolute_physical_headroom_features"] = physical_features
            out["direct_recovery_absolute_physical_headroom_weight"] = physical_weights
        if native_component_logits is not None:
            native_component_logits = native_component_logits.to(
                device=memory.device, dtype=memory.dtype
            )
            native_component_margins = native_component_margins.to(
                device=memory.device, dtype=memory.dtype
            )
            out["direct_recovery_evidence_native_component_logits"] = native_component_logits
            out["direct_recovery_evidence_native_component_margins"] = native_component_margins
        if native_benefit_logit is not None:
            native_benefit_logit = native_benefit_logit.to(device=memory.device, dtype=memory.dtype)
            native_benefit_margin = native_benefit_margin.to(device=memory.device, dtype=memory.dtype)
            native_recovery_value = native_recovery_value.to(device=memory.device, dtype=memory.dtype)
            out["direct_recovery_evidence_native_benefit_logit"] = native_benefit_logit
            out["direct_recovery_evidence_native_benefit_margin"] = native_benefit_margin
            out["direct_recovery_evidence_native_recovery_value"] = native_recovery_value
        rank_base: torch.Tensor | None = None
        if self.direct_value_heads is not None:
            all_direct = torch.stack([head(direct_features) for head in self.direct_value_heads], dim=1)
            routing = self.direct_recovery_value_expert_routing
            if routing in {"bucket", "hard_bucket"}:
                if bucket_id is None:
                    bucket_id = torch.ones((x.shape[0],), dtype=torch.long, device=x.device)
                expert_idx = (bucket_id.to(device=x.device, dtype=torch.long).reshape(-1) - 1)
                expert_idx = expert_idx.clamp(0, self.direct_recovery_value_num_experts - 1)
                weights = torch.nn.functional.one_hot(
                    expert_idx, num_classes=self.direct_recovery_value_num_experts
                ).to(dtype=all_direct.dtype)
                direct = (all_direct * weights.unsqueeze(-1)).sum(dim=1)
                rank_base = direct[:, 0]
            elif routing in {"uniform", "uniform_robust", "robust_ensemble", "risk_ensemble"}:
                weights = torch.full(
                    (x.shape[0], self.direct_recovery_value_num_experts),
                    1.0 / float(self.direct_recovery_value_num_experts),
                    dtype=all_direct.dtype, device=x.device,
                )
                mean = all_direct.mean(dim=1)
                std = all_direct.std(dim=1, unbiased=False)
                rank_base = mean[:, 0]
                if routing in {"uniform_robust", "robust_ensemble", "risk_ensemble"}:
                    direct = mean.clone()
                    lam = self.direct_recovery_expert_disagreement_penalty
                    direct[:, 0] = mean[:, 0] - lam * std[:, 0]
                    # Keep aleatoric log-variance conservative as well.
                    direct[:, 1] = mean[:, 1] + lam * std[:, 1]
                    cursor = 2
                    if self.direct_recovery_opportunity_head:
                        direct[:, cursor] = mean[:, cursor] - lam * std[:, cursor]
                        cursor += 1
                    if self.direct_recovery_harm_head:
                        direct[:, cursor] = mean[:, cursor] + lam * std[:, cursor]
                    out["direct_expert_disagreement"] = std
                    out["direct_expert_output_std"] = std
                else:
                    direct = mean
            else:
                if self.direct_value_router is None:
                    raise RuntimeError("soft expert routing configured without a router")
                router_logits = self.direct_value_router(router_features)
                weights = torch.softmax(
                    router_logits / self.direct_recovery_value_router_temperature, dim=-1
                )
                mean = (all_direct * weights.unsqueeze(-1)).sum(dim=1)
                rank_base = mean[:, 0]
                if routing == "robust_observation":
                    centered = all_direct - mean.unsqueeze(1)
                    std = torch.sqrt((weights.unsqueeze(-1) * centered.square()).sum(dim=1).clamp_min(1.0e-8))
                    direct = mean.clone()
                    lam = self.direct_recovery_expert_disagreement_penalty
                    direct[:, 0] = mean[:, 0] - lam * std[:, 0]
                    direct[:, 1] = mean[:, 1] + lam * std[:, 1]
                    cursor = 2
                    if self.direct_recovery_opportunity_head:
                        direct[:, cursor] = mean[:, cursor] - lam * std[:, cursor]
                        cursor += 1
                    if self.direct_recovery_harm_head:
                        direct[:, cursor] = mean[:, cursor] + lam * std[:, cursor]
                    out["direct_expert_disagreement"] = std
                    out["direct_expert_output_std"] = std
                else:
                    direct = mean
                out["direct_expert_logits"] = router_logits
            out["direct_expert_weights"] = weights
            out["direct_expert_outputs"] = all_direct
        elif self.direct_value_head is not None:
            direct = self.direct_value_head(direct_features)
            rank_base = direct[:, 0]
        else:
            raise RuntimeError("direct recovery branch configured without a head")

        out["direct_recovery_value_logit"] = direct[:, 0]
        out["direct_recovery_value_logvar"] = direct[:, 1].clamp(-7.0, 2.0)
        cursor = 2
        if self.direct_recovery_opportunity_head:
            out["direct_recovery_opportunity_logit"] = direct[:, cursor]
            cursor += 1
        if self.direct_recovery_harm_head:
            out["direct_recovery_harm_logit"] = direct[:, cursor]
        if rank_base is None:
            rank_base = direct[:, 0]
        rank_residual = torch.zeros_like(rank_base)
        if self.direct_preference_adapter is not None:
            pointwise_rank_residual = self.direct_preference_adapter(pointwise_direct_features).squeeze(-1)
            out["direct_recovery_rank_pointwise_residual"] = pointwise_rank_residual
            rank_residual = rank_residual + pointwise_rank_residual
        if self.direct_preference_context_adapter is not None:
            context_rank_residual = self.direct_preference_context_adapter(relative_features).squeeze(-1)
            out["direct_recovery_rank_context_residual"] = context_rank_residual
            rank_residual = rank_residual + context_rank_residual
        out["direct_recovery_rank_residual"] = rank_residual
        inherited_rank = rank_base + rank_residual
        if self.direct_preference_set_ranker is not None:
            tournament_context = self.direct_preference_set_ranker.encode(
                relative_features, group_index, is_nominal
            )
            tournament_rank = self.direct_preference_set_ranker.score_from_context(
                tournament_context, group_index, is_nominal
            )
            out["direct_recovery_tournament_context"] = tournament_context
            out["direct_recovery_rank_tournament"] = tournament_rank
            rank_logit = tournament_rank if self.direct_recovery_set_tournament_replace_base else inherited_rank + tournament_rank
        else:
            rank_logit = inherited_rank
        out["direct_recovery_rank_logit"] = rank_logit

        delta_features = relative_features
        if self.direct_recovery_delta_policy_features:
            policy_features = relative_features.new_zeros((relative_features.shape[0], 2))
            if group_index is not None and is_nominal is not None:
                groups = group_index.to(device=relative_features.device)
                groups = groups.reshape(-1, 1) if groups.dim() == 1 else groups.reshape(groups.shape[0], -1)
                nominal_mask = is_nominal.to(device=relative_features.device).reshape(-1) > 0.5
                if groups.shape[0] == rank_logit.shape[0] and nominal_mask.shape[0] == rank_logit.shape[0]:
                    for key in torch.unique(groups, dim=0):
                        idx = torch.where((groups == key.unsqueeze(0)).all(dim=1))[0]
                        noms = idx[nominal_mask[idx]]
                        recs = idx[~nominal_mask[idx]]
                        if noms.numel() == 0 or recs.numel() == 0:
                            continue
                        nom_rank = rank_logit[noms[0]]
                        policy_features[recs, 0] = rank_logit[recs] - nom_rank
                        for rec in recs:
                            others = recs[recs != rec]
                            second = rank_logit[others].max() if others.numel() else nom_rank
                            policy_features[rec, 1] = rank_logit[rec] - second
            delta_features = torch.cat([relative_features, policy_features], dim=-1)
            out["direct_recovery_policy_features"] = policy_features

        delta_expert_idx = None
        if bucket_id is None:
            delta_expert_idx = torch.zeros((delta_features.shape[0],), dtype=torch.long, device=delta_features.device)
        else:
            delta_expert_idx = (bucket_id.to(device=delta_features.device, dtype=torch.long).reshape(-1) - 1).clamp(0, 1)
        if self.direct_delta_adapters is not None:
            all_delta = torch.stack([adapter(delta_features) for adapter in self.direct_delta_adapters], dim=1)
            delta = all_delta[torch.arange(all_delta.shape[0], device=all_delta.device), delta_expert_idx]
            out["direct_recovery_delta_expert_outputs"] = all_delta
        elif self.direct_delta_adapter is not None:
            delta = self.direct_delta_adapter(delta_features)
        else:
            delta = None

        evidence_calibrator_residual = None
        unified_benefit_logit = None
        unified_harm_logit = None
        unified_component_harm_logits = None
        effective_component_harm_logits = None
        unified_admission_logit = None
        calibrator_enabled = (
            self.direct_evidence_calibrators is not None
            or self.direct_evidence_unified_calibrator is not None
            or self.direct_evidence_concord_benefit_calibrator is not None
        )
        if delta is not None and calibrator_enabled:
            if self.direct_recovery_delta_policy_features and "direct_recovery_policy_features" in out:
                calibrator_policy = out["direct_recovery_policy_features"].to(dtype=delta.dtype)
            else:
                calibrator_policy = delta.new_zeros((delta.shape[0], 2))

            unified_evidence = (
                self.direct_evidence_unified_calibrator is not None
                or self.direct_evidence_concord_benefit_calibrator is not None
            )
            if unified_evidence:
                if self.direct_delta_adapters is None or "direct_recovery_delta_expert_outputs" not in out:
                    raise RuntimeError("unified evidence configured without frozen delta expert outputs")
                expert_delta = out["direct_recovery_delta_expert_outputs"].to(dtype=delta.dtype).detach()
                expert_mean = expert_delta.mean(dim=1)
                expert_disagreement = (expert_delta[:, 0] - expert_delta[:, 1]).abs()
                if self.direct_evidence_concord_benefit_calibrator is not None:
                    # Permutation-invariant expert statistics.  The evidence model
                    # cannot infer a regime from expert ordering and never receives
                    # bucket ids.  Consensus and disagreement are observables, not
                    # a hidden hard router.
                    center_e = expert_delta[:, :, 0]
                    half_e = 0.5 * torch.nn.functional.softplus(expert_delta[:, :, 1])
                    benefit_e = center_e - half_e
                    harm_e = -(center_e + half_e)
                    benefit_stats = torch.stack(
                        [benefit_e.mean(dim=1), benefit_e.amin(dim=1),
                         benefit_e.amax(dim=1), benefit_e.amax(dim=1)-benefit_e.amin(dim=1)],
                        dim=-1,
                    )
                    calibrator_parts = [
                        expert_mean, expert_disagreement, benefit_stats,
                        calibrator_policy.detach(),
                    ]
                else:
                    calibrator_parts = [
                        expert_delta.reshape(expert_delta.shape[0], -1),
                        expert_mean,
                        expert_disagreement,
                        calibrator_policy.detach(),
                    ]
            else:
                calibrator_parts = [delta[:, :2], calibrator_policy]

            if self.direct_recovery_evidence_calibrator_context:
                if (
                    self.direct_recovery_evidence_calibrator_context_source == "tournament"
                    and "direct_recovery_tournament_context" in out
                ):
                    calibrator_context = out["direct_recovery_tournament_context"].to(dtype=delta.dtype)
                elif self.direct_recovery_evidence_calibrator_context_source == "physical_relative":
                    calibrator_context = self._direct_candidate_raw_relative_features(
                        x, group_index, is_nominal
                    ).to(dtype=delta.dtype)
                    if self.direct_recovery_evidence_calibrator_context_detach:
                        calibrator_context = calibrator_context.detach()
                    out["direct_recovery_evidence_physical_relative_context"] = calibrator_context
                elif self.direct_recovery_evidence_calibrator_context_source == "physical_interaction":
                    if self.direct_evidence_interaction_bridge is None:
                        raise RuntimeError("physical_interaction configured without OCAF bridge")
                    action_context = self._direct_candidate_raw_relative_features(
                        x, group_index, is_nominal
                    ).to(dtype=delta.dtype)
                    observation_context = self._direct_nominal_observation_features(
                        x, group_index, is_nominal
                    ).to(dtype=delta.dtype)
                    if self.direct_recovery_evidence_calibrator_context_detach:
                        # Detach raw/upstream representations, not the trainable OCAF bridge.
                        action_context = action_context.detach()
                        observation_context = observation_context.detach()
                    if self.direct_recovery_evidence_factorized_harm_interaction:
                        benefit_context, harm_component_contexts = self.direct_evidence_interaction_bridge(
                            action_context, observation_context
                        )
                        harm_context = harm_component_contexts.mean(dim=1)
                        calibrator_context = 0.5 * (benefit_context + harm_context)
                        out["direct_recovery_evidence_benefit_interaction_context"] = benefit_context
                        out["direct_recovery_evidence_harm_interaction_context"] = harm_context
                        out["direct_recovery_evidence_component_harm_interaction_contexts"] = (
                            harm_component_contexts
                        )
                    elif self.direct_recovery_evidence_dual_interaction_bridge:
                        benefit_context, harm_context = self.direct_evidence_interaction_bridge(
                            action_context, observation_context
                        )
                        # Generic/legacy consumers receive a symmetric diagnostic
                        # context only; concord benefit/harm heads below receive
                        # their task-specific contexts. No regime id is introduced.
                        calibrator_context = 0.5 * (benefit_context + harm_context)
                        out["direct_recovery_evidence_benefit_interaction_context"] = benefit_context
                        out["direct_recovery_evidence_harm_interaction_context"] = harm_context
                    else:
                        calibrator_context = self.direct_evidence_interaction_bridge(
                            action_context, observation_context
                        )
                        benefit_context = calibrator_context
                        harm_context = calibrator_context

                    if (
                        self.direct_evidence_postprefix_obs_transport_benefit is not None
                        or self.direct_evidence_postprefix_obs_transport_harm is not None
                    ):
                        signature_abs = postprefix_observation_signature
                        if signature_abs is None:
                            signature_abs = self._direct_postprefix_observation_signature(memory)
                        signature_abs = signature_abs.to(device=delta.device, dtype=delta.dtype).detach()
                        signature_rel = self._candidate_minus_nominal(
                            signature_abs, group_index, is_nominal
                        ).detach()
                        scale = self.direct_recovery_evidence_postprefix_obs_transport_scale
                        if self.direct_evidence_postprefix_obs_transport_benefit is not None:
                            benefit_transport = (
                                self.direct_evidence_postprefix_obs_transport_benefit(signature_rel) * scale
                            )
                            benefit_context = benefit_context + benefit_transport
                            out["direct_recovery_evidence_postprefix_obs_benefit_transport"] = benefit_transport
                        if self.direct_evidence_postprefix_obs_transport_harm is not None:
                            harm_transport = (
                                self.direct_evidence_postprefix_obs_transport_harm(signature_rel) * scale
                            )
                            harm_context = harm_context + harm_transport
                            out["direct_recovery_evidence_postprefix_obs_harm_transport"] = harm_transport
                        # The generic diagnostic context stays symmetric; task
                        # calibrators below receive their branch-specific contexts.
                        calibrator_context = 0.5 * (benefit_context + harm_context)
                        out["direct_recovery_evidence_benefit_interaction_context"] = benefit_context
                        out["direct_recovery_evidence_harm_interaction_context"] = harm_context
                        out["direct_recovery_evidence_postprefix_obs_signature"] = signature_abs
                        out["direct_recovery_evidence_postprefix_obs_signature_relative"] = signature_rel

                    out["direct_recovery_evidence_physical_relative_context"] = action_context
                    out["direct_recovery_evidence_nominal_observation_context"] = observation_context
                    out["direct_recovery_evidence_interaction_context"] = calibrator_context
                else:
                    calibrator_context = relative_features.to(dtype=delta.dtype)
                    if self.direct_recovery_evidence_calibrator_context_detach:
                        calibrator_context = calibrator_context.detach()
                    benefit_context = calibrator_context
                    harm_context = calibrator_context
                calibrator_parts.append(calibrator_context)
            calibrator_input = torch.cat(calibrator_parts, dim=-1)
            benefit_calibrator_input = calibrator_input
            harm_calibrator_input = calibrator_input
            if (
                self.direct_recovery_evidence_calibrator_context
                and self.direct_recovery_evidence_calibrator_context_source == "physical_interaction"
                and (
                    self.direct_recovery_evidence_dual_interaction_bridge
                    or self.direct_recovery_evidence_factorized_harm_interaction
                )
            ):
                # Replace only the final context block; scalar evidence and raw
                # physical inputs remain identical between tasks.
                scalar_width = calibrator_input.shape[-1] - benefit_context.shape[-1]
                scalar_input = calibrator_input[:, :scalar_width]
                benefit_calibrator_input = torch.cat([scalar_input, benefit_context], dim=-1)
                harm_calibrator_input = torch.cat([scalar_input, harm_context], dim=-1)

            if self.direct_evidence_concord_benefit_calibrator is not None:
                benefit_raw = self.direct_evidence_concord_benefit_calibrator(
                    benefit_calibrator_input
                ).squeeze(-1)
                if self.direct_recovery_evidence_rank_benefit_skip:
                    if calibrator_policy.shape[-1] < 1 or self.direct_evidence_rank_benefit_log_gain is None:
                        raise RuntimeError("rank-benefit skip configured without rank policy evidence")
                    rank_gain = torch.nn.functional.softplus(
                        self.direct_evidence_rank_benefit_log_gain
                    )
                    benefit_raw = benefit_raw + rank_gain * calibrator_policy[:, 0]
                    out["direct_recovery_evidence_rank_benefit_gain"] = rank_gain.expand_as(
                        benefit_raw
                    )
                if self.direct_recovery_evidence_factorized_harm_interaction:
                    if not isinstance(self.direct_evidence_concord_harm_calibrator, nn.ModuleList):
                        raise RuntimeError("factorized harm interaction requires component calibrators")
                    component_contexts = out.get(
                        "direct_recovery_evidence_component_harm_interaction_contexts"
                    )
                    if component_contexts is None:
                        raise RuntimeError("factorized harm interaction context missing")
                    harm_parts = []
                    for component_index, head in enumerate(
                        self.direct_evidence_concord_harm_calibrator
                    ):
                        if self.direct_recovery_evidence_component_reliability[component_index] <= 0.0:
                            # Exact compute-only pruning for globally unsupported
                            # coordinates. Their effective logit remains the
                            # semantic non-harm prior and they are masked from
                            # the learned reserve exactly as before.
                            harm_parts.append(component_contexts.new_zeros((component_contexts.shape[0], 1)))
                            continue
                        component_input = torch.cat(
                            [scalar_input, component_contexts[:, component_index, :]], dim=-1
                        )
                        harm_parts.append(head(component_input))
                    harm_raw = torch.cat(harm_parts, dim=-1)
                else:
                    harm_raw = self.direct_evidence_concord_harm_calibrator(harm_calibrator_input)
                    if self.direct_evidence_concord_harm_component_residuals is not None:
                        detached_harm_input = harm_calibrator_input.detach()
                        residual_parts = []
                        for component_index, head in enumerate(
                            self.direct_evidence_concord_harm_component_residuals
                        ):
                            if self.direct_recovery_evidence_component_reliability[component_index] <= 0.0:
                                residual_parts.append(
                                    harm_raw.new_zeros((harm_raw.shape[0], 1))
                                )
                            else:
                                residual_parts.append(head(detached_harm_input))
                        component_residual_raw = torch.cat(residual_parts, dim=-1)
                        component_residual_raw = (
                            torch.tanh(component_residual_raw)
                            * self.direct_recovery_evidence_partial_pool_harm_residual_scale
                        )
                        harm_raw = harm_raw + component_residual_raw
                        out[
                            "direct_recovery_evidence_partial_pool_harm_component_residuals"
                        ] = component_residual_raw
                admission_raw = (
                    self.direct_evidence_concord_admission_calibrator(benefit_calibrator_input).squeeze(-1)
                    if self.direct_evidence_concord_admission_calibrator is not None
                    else None
                )
                if self.direct_recovery_evidence_unbounded_benefit_factor:
                    # v48.39 DRFR.  v48.38 development rows expose a hard
                    # dynamic-range mismatch: safe-positive benefit headroom is
                    # often O(0.4) while the legacy bounded residual can change
                    # the physical margin by at most tau_b*0.75=0.0375.  The
                    # final layer is zero-initialised, so a linear residual keeps
                    # the exact source identity at initialisation while allowing
                    # continuous signed-margin regression to represent the data.
                    benefit_residual = (
                        benefit_raw * self.direct_recovery_evidence_benefit_residual_scale
                    )
                else:
                    benefit_residual = (
                        torch.tanh(benefit_raw) * self.direct_recovery_evidence_calibrator_scale
                    )
                # Consensus transfer replaces v48.20's exact min envelope.  The
                # exact min let one mismatched frozen expert destroy otherwise
                # useful Near benefit evidence.  Mean consensus preserves shared
                # source information while an explicit disagreement penalty keeps
                # transfer conservative without selecting a regime expert.
                base_benefit = self.direct_recovery_evidence_consensus_prior_scale * (
                    benefit_e.mean(dim=1) - (
                        self.direct_recovery_evidence_consensus_disagreement_penalty
                        * (benefit_e.amax(dim=1) - benefit_e.amin(dim=1))
                    )
                )
                unified_benefit_logit = base_benefit + benefit_residual
                if self.direct_evidence_roct_benefit is not None and roct_signature_rel is not None:
                    roct_benefit_correction = (
                        torch.tanh(self.direct_evidence_roct_benefit(roct_signature_rel).squeeze(-1))
                        * self.direct_recovery_evidence_roct_scale
                    )
                    unified_benefit_logit = unified_benefit_logit + roct_benefit_correction
                    out["direct_recovery_evidence_roct_benefit_correction"] = roct_benefit_correction
                if self.direct_recovery_evidence_component_heads:
                    # v48.28: the semantic prior is -2.  A scale of 2 capped
                    # candidate component logits at zero, so the model could
                    # never represent p(harm)>0.5 after a veto tolerance was
                    # exceeded.  The wider bounded range retains stable logits
                    # while allowing strong harmful evidence.
                    raw_component_residual = harm_raw[
                        :, : self.direct_recovery_evidence_component_count
                    ]
                    if self.direct_recovery_evidence_unbounded_harm_factors:
                        # v48.39 DRFR.  With prior=-2, scale=6 and tanh, the
                        # largest representable component margin at tau_h=.025
                        # is only 0.10, yet v48.38 teacher veto margins reach
                        # roughly 0.95.  A zero-initialised unbounded residual
                        # preserves the semantic prior at step zero and removes
                        # this representational ceiling without a regime branch.
                        component_residual = (
                            raw_component_residual
                            * self.direct_recovery_evidence_component_scale
                        )
                    else:
                        component_residual = (
                            torch.tanh(raw_component_residual)
                            * self.direct_recovery_evidence_component_scale
                        )
                    # v48.23 FRONTIER: a zero network output must mean the
                    # nominal-relative component is inside the configured safety
                    # deadband, not p(harm)=0.5.  With tolerance=0.05 and
                    # temperature=0.025 the teacher target at equality is
                    # sigmoid(-2), hence the principled default prior -2.0.
                    if self.direct_recovery_evidence_frontier:
                        unified_component_harm_logits = (
                            component_residual
                            + self.direct_recovery_evidence_component_prior_logit
                        )
                    else:
                        unified_component_harm_logits = component_residual
                    # v48.31 CONTRACT-SLACK-RANK: some physical coordinates can
                    # be degenerate or nearly unsupported in a fixed dataset (for
                    # example a constant harm_proxy).  A max-veto over an
                    # unsupported learned coordinate can dominate every regime even
                    # though no data identify its sign.  Reliability is global and
                    # regime-agnostic: it shrinks only unsupported coordinates toward
                    # the semantic non-harm prior, while the independent measured
                    # hard veto remains unchanged at deployment.
                    if (
                        self.direct_evidence_roct_deployability is not None
                        and roct_signature_rel is not None
                    ):
                        roct_deployability_correction = (
                            torch.tanh(self.direct_evidence_roct_deployability(roct_signature_rel).squeeze(-1))
                            * self.direct_recovery_evidence_roct_scale
                        )
                        deployability_basis = unified_component_harm_logits.new_zeros(
                            (self.direct_recovery_evidence_component_count,)
                        )
                        deployability_basis[1] = 1.0
                        unified_component_harm_logits = (
                            unified_component_harm_logits
                            + roct_deployability_correction.unsqueeze(-1) * deployability_basis.unsqueeze(0)
                        )
                        out["direct_recovery_evidence_roct_deployability_correction"] = (
                            roct_deployability_correction
                        )
                    if native_component_logits is not None:
                        unified_component_harm_logits = unified_component_harm_logits.clone()
                        native_width = int(native_component_logits.shape[-1])
                        unified_component_harm_logits[:, :native_width] = native_component_logits
                        out["direct_recovery_evidence_native_certificate_preserved"] = (
                            unified_component_harm_logits.new_ones(())
                        )
                    reliability = self._direct_recovery_evidence_component_reliability.to(
                        device=unified_component_harm_logits.device,
                        dtype=unified_component_harm_logits.dtype,
                    )
                    neutral_component_logit = (
                        unified_component_harm_logits.new_tensor(
                            self.direct_recovery_evidence_component_prior_logit
                        )
                        if self.direct_recovery_evidence_frontier
                        else unified_component_harm_logits.new_zeros(())
                    )
                    effective_component_harm_logits = neutral_component_logit + reliability * (
                        unified_component_harm_logits - neutral_component_logit
                    )
                    unified_harm_logit = effective_component_harm_logits.amax(dim=-1)
                    out["direct_recovery_evidence_effective_component_harm_logits"] = (
                        effective_component_harm_logits
                    )
                    # v48.41 diagnostic contract: publish the exact effective
                    # component probabilities used by the global harm max.  This
                    # is diagnostic/provenance only; deployment selection still
                    # uses the signed component margins and the same max-veto.
                    out["direct_recovery_evidence_component_harm_probabilities"] = (
                        torch.sigmoid(effective_component_harm_logits)
                    )
                    out["direct_recovery_evidence_component_reliability"] = reliability.expand_as(
                        effective_component_harm_logits
                    )
                    anchor_harm_residual = (
                        component_residual
                        if self.direct_recovery_evidence_frontier
                        else effective_component_harm_logits
                    )
                    evidence_calibrator_residual = torch.cat(
                        [benefit_residual.unsqueeze(-1), anchor_harm_residual], dim=-1
                    )
                else:
                    harm_residual = (
                        torch.tanh(harm_raw.reshape(-1))
                        * self.direct_recovery_evidence_component_scale
                    )
                    unified_harm_logit = harm_residual
                    evidence_calibrator_residual = torch.stack(
                        [benefit_residual, harm_residual], dim=-1
                    )
                if admission_raw is not None:
                    # Admission has a separate semantic target: raw benefit AND
                    # no component veto. Historical stages detached the prior so
                    # sparse admission gradients could not distort the factor
                    # heads. v48.32 optionally couples the deployment-exact safe
                    # utility back into the compact benefit/component calibrators.
                    # This remains one regime-agnostic candidate-vs-nominal model;
                    # the flag controls gradient flow only, never inference logic.
                    prior_benefit = (
                        unified_benefit_logit.detach()
                        if self.direct_recovery_evidence_admission_prior_detach
                        else unified_benefit_logit
                    )
                    prior_components = (
                        effective_component_harm_logits.detach()
                        if self.direct_recovery_evidence_admission_prior_detach
                        else effective_component_harm_logits
                    )
                    prior_harm = (
                        unified_harm_logit.detach()
                        if self.direct_recovery_evidence_admission_prior_detach
                        else unified_harm_logit
                    )
                    residual_safety_gate = None
                    admission_safety_cap_logit = None
                    if self.direct_recovery_evidence_admission_prior_mode in {
                        "safety_slack", "barrier_gated_slack", "frontier_capped_slack"
                    }:
                        # Unified candidate-vs-nominal signed physical margins.
                        predicted_component_margins = (
                            self.direct_recovery_evidence_slack_temperature
                            * prior_components
                        )
                        max_predicted_veto_margin = predicted_component_margins.amax(dim=-1)
                        if self.direct_recovery_evidence_admission_prior_mode == "barrier_gated_slack":
                            # v48.34 legacy soft barrier.  Retained for ablation only.
                            tau = max(self.direct_recovery_evidence_slack_temperature, 1.0e-6)
                            residual_safety_gate = torch.sigmoid(
                                -max_predicted_veto_margin / tau
                            )
                            slack_barrier = tau * torch.nn.functional.softplus(
                                max_predicted_veto_margin / tau
                            )
                            admission_prior = (
                                residual_safety_gate * prior_benefit
                                - self.direct_recovery_evidence_slack_penalty * slack_barrier
                            )
                            out["direct_recovery_evidence_barrier_safety_gate"] = residual_safety_gate
                        elif self.direct_recovery_evidence_admission_prior_mode == "frontier_capped_slack":
                            # v48.35 non-compensatory frontier.  A component logit of
                            # zero is the shared signed-margin boundary.  The final
                            # admission is capped by its worst component; benefit or
                            # a learned residual can never cross an unsafe cap.
                            tau = max(self.direct_recovery_evidence_slack_temperature, 1.0e-6)
                            admission_safety_cap_logit = -max_predicted_veto_margin / tau
                            slack_barrier = torch.relu(max_predicted_veto_margin)
                            admission_prior = prior_benefit
                            out["direct_recovery_evidence_frontier_safety_cap_logit"] = (
                                admission_safety_cap_logit
                            )
                        else:
                            slack_barrier = torch.relu(max_predicted_veto_margin)
                            admission_prior = (
                                prior_benefit
                                - self.direct_recovery_evidence_slack_penalty * slack_barrier
                            )
                        out["direct_recovery_evidence_predicted_component_margins"] = predicted_component_margins
                        out["direct_recovery_evidence_max_predicted_veto_margin"] = max_predicted_veto_margin
                        out["direct_recovery_evidence_slack_barrier"] = slack_barrier
                    elif self.direct_recovery_evidence_admission_prior_mode == "benefit_only":
                        # v48.29: non-compensatory factors are calibrated as a
                        # separate veto. Penalising the same max-risk logit again
                        # inside the admission score double-counted one noisy
                        # factor and suppressed safe-positive actions. The
                        # admission residual is still trained with harmful
                        # candidates mapped to negative safe utility.
                        admission_prior = prior_benefit
                    elif self.direct_recovery_evidence_frontier:
                        # Center the risk penalty at the semantic non-harm prior.
                        # Zero residual therefore reproduces the transferred
                        # benefit evidence instead of forcing all-abstain.
                        prior_penalty = torch.nn.functional.softplus(
                            unified_harm_logit.new_tensor(
                                self.direct_recovery_evidence_component_prior_logit
                            )
                        )
                        admission_prior = (
                            prior_benefit
                            - (
                                torch.nn.functional.softplus(prior_harm)
                                - prior_penalty
                            )
                        )
                    else:
                        admission_prior = (
                            prior_benefit
                            - torch.nn.functional.softplus(prior_harm)
                        )
                    # v48.25 INTEGRITY-BRIDGE optionally removes the tanh ceiling.
                    # The zero-initialised head still preserves the transferred prior
                    # exactly, while an unbounded linear residual can cross the actual
                    # nominal-vs-recovery decision boundary when the source prior is
                    # conservatively negative. Global gradient clipping retains numerical control.
                    admission_basis = (
                        torch.tanh(admission_raw)
                        if self.direct_recovery_evidence_admission_bounded
                        else admission_raw
                    )
                    admission_residual = (
                        admission_basis * self.direct_recovery_evidence_admission_scale
                    )
                    if residual_safety_gate is not None:
                        admission_residual = residual_safety_gate * admission_residual
                    free_admission_logit = admission_prior + admission_residual
                    if admission_safety_cap_logit is not None:
                        # Smooth minimum is always <= the exact minimum, so the
                        # differentiable continuation preserves the hard safety cap.
                        unified_admission_logit = self._noncompensatory_smooth_cap(
                            free_admission_logit, admission_safety_cap_logit,
                            self.direct_recovery_evidence_frontier_cap_temperature,
                        )
                        out["direct_recovery_evidence_free_admission_logit"] = free_admission_logit
                    else:
                        unified_admission_logit = free_admission_logit
                    evidence_calibrator_residual = torch.cat(
                        [evidence_calibrator_residual, admission_residual.unsqueeze(-1)], dim=-1
                    )
                    out["direct_recovery_evidence_concord_admission_raw"] = admission_raw
                    out["direct_recovery_evidence_admission_prior"] = admission_prior
                out["direct_recovery_evidence_expert_benefit_logits"] = benefit_e
                out["direct_recovery_evidence_expert_harm_logits"] = harm_e
                out["direct_recovery_evidence_expert_base"] = torch.stack(
                    [base_benefit, harm_e.amax(dim=1)], dim=-1
                )
                out["direct_recovery_evidence_concord_benefit_raw"] = benefit_raw
                out["direct_recovery_evidence_concord_harm_raw"] = harm_raw
            elif self.direct_evidence_unified_calibrator is not None:
                combined_residual = self.direct_evidence_unified_calibrator(calibrator_input)
                benefit_residual = (
                    torch.tanh(combined_residual[:, 0])
                    * self.direct_recovery_evidence_calibrator_scale
                )
                # Conservative continuous envelope over the two frozen source
                # experts.  Equal experts reproduce the source exactly; expert
                # disagreement lowers benefit confidence and raises harm.
                center_e = expert_delta[:, :, 0]
                half_e = 0.5 * torch.nn.functional.softplus(expert_delta[:, :, 1])
                benefit_e = center_e - half_e
                harm_e = -(center_e + half_e)
                # Exact lower/upper envelopes are used instead of normalized
                # soft-min/soft-max.  The normalized smooth forms lie *inside*
                # the expert/component range and can therefore overstate benefit
                # or let two low-risk components compensate one high-risk veto.
                # Exact amin/amax are bucket-invariant, preserve the zero-residual
                # source identity when experts/components agree, and implement the
                # intended non-compensatory safety semantics.
                base_benefit = benefit_e.amin(dim=1)
                base_harm = harm_e.amax(dim=1)
                unified_benefit_logit = base_benefit + benefit_residual
                if self.direct_evidence_roct_benefit is not None and roct_signature_rel is not None:
                    roct_benefit_correction = (
                        torch.tanh(self.direct_evidence_roct_benefit(roct_signature_rel).squeeze(-1))
                        * self.direct_recovery_evidence_roct_scale
                    )
                    unified_benefit_logit = unified_benefit_logit + roct_benefit_correction
                    out["direct_recovery_evidence_roct_benefit_correction"] = roct_benefit_correction
                if self.direct_recovery_evidence_component_heads:
                    # FACET changed the harm target semantics from signed total
                    # PCD to componentwise non-compensatory vetoes.  Reusing the
                    # old total-PCD harm logit as an additive base anchors every
                    # component to a source signal that the v48.19 certificate
                    # showed to be near random.  Benefit therefore transfers from
                    # the source experts, while component harm heads are a semantic
                    # reset: zero-initialized, bounded, absolute candidate-vs-
                    # nominal logits learned jointly across all regimes.
                    unified_component_harm_logits = (
                        torch.tanh(combined_residual[:, 1 : 1 + self.direct_recovery_evidence_component_count])
                        * self.direct_recovery_evidence_component_scale
                    )
                    if (
                        self.direct_evidence_roct_deployability is not None
                        and roct_signature_rel is not None
                    ):
                        roct_deployability_correction = (
                            torch.tanh(self.direct_evidence_roct_deployability(roct_signature_rel).squeeze(-1))
                            * self.direct_recovery_evidence_roct_scale
                        )
                        deployability_basis = unified_component_harm_logits.new_zeros(
                            (self.direct_recovery_evidence_component_count,)
                        )
                        deployability_basis[1] = 1.0
                        unified_component_harm_logits = (
                            unified_component_harm_logits
                            + roct_deployability_correction.unsqueeze(-1) * deployability_basis.unsqueeze(0)
                        )
                        out["direct_recovery_evidence_roct_deployability_correction"] = (
                            roct_deployability_correction
                        )
                    if native_component_logits is not None:
                        unified_component_harm_logits = unified_component_harm_logits.clone()
                        native_width = int(native_component_logits.shape[-1])
                        unified_component_harm_logits[:, :native_width] = native_component_logits
                        out["direct_recovery_evidence_native_certificate_preserved"] = (
                            unified_component_harm_logits.new_ones(())
                        )
                    unified_harm_logit = unified_component_harm_logits.amax(dim=-1)
                    evidence_calibrator_residual = torch.cat(
                        [benefit_residual.unsqueeze(-1), unified_component_harm_logits],
                        dim=-1,
                    )
                else:
                    harm_residual = (
                        torch.tanh(combined_residual[:, 1])
                        * self.direct_recovery_evidence_calibrator_scale
                    )
                    unified_harm_logit = base_harm + harm_residual
                    evidence_calibrator_residual = torch.stack(
                        [benefit_residual, harm_residual], dim=-1
                    )
                out["direct_recovery_evidence_expert_benefit_logits"] = benefit_e
                out["direct_recovery_evidence_expert_harm_logits"] = harm_e
                out["direct_recovery_evidence_expert_base"] = torch.stack(
                    [base_benefit, base_harm], dim=-1
                )
                out["direct_recovery_evidence_unified_residual_raw"] = combined_residual
            else:
                all_residuals = torch.stack(
                    [adapter(calibrator_input) for adapter in self.direct_evidence_calibrators], dim=1
                )
                regime_residual = all_residuals[
                    torch.arange(all_residuals.shape[0], device=all_residuals.device), delta_expert_idx
                ]
                if self.direct_evidence_shared_calibrator is not None:
                    shared_residual = self.direct_evidence_shared_calibrator(calibrator_input)
                    combined_residual = (
                        shared_residual
                        + self.direct_recovery_evidence_calibrator_regime_scale * regime_residual
                    )
                    out["direct_recovery_evidence_shared_residual_raw"] = shared_residual
                    out["direct_recovery_evidence_regime_residual_raw"] = regime_residual
                else:
                    combined_residual = regime_residual
                evidence_calibrator_residual = (
                    torch.tanh(combined_residual)
                    * self.direct_recovery_evidence_calibrator_scale
                )
                out["direct_recovery_evidence_calibrator_outputs"] = all_residuals
                if self.direct_recovery_evidence_calibrator_mode == "center_width":
                    delta = delta + evidence_calibrator_residual

            out["direct_recovery_evidence_calibrator_input"] = calibrator_input
            out["direct_recovery_evidence_calibrator_residual"] = evidence_calibrator_residual

        if native_benefit_logit is not None:
            # v48.49 NAP: overwrite, do not add.  The point is to remove the free
            # learned proxy on the benefit side exactly as v48.48 NCP removed it
            # for critical safety components.
            unified_benefit_logit = native_benefit_logit
            out["direct_recovery_evidence_native_advantage_preserved"] = (
                native_benefit_logit.new_ones(())
            )
            out["direct_recovery_evidence_native_boundary_complete_advantage_preserved"] = (
                native_benefit_logit.new_tensor(
                    1.0 if self.direct_recovery_evidence_native_boundary_complete_advantage_preservation else 0.0
                )
            )

        if delta is not None:
            nominal_mask = None
            if is_nominal is not None and is_nominal.numel() == delta.shape[0]:
                nominal_mask = is_nominal.to(device=delta.device).reshape(-1) > 0.5
            if self.direct_recovery_delta_mode == "ordinal_evidence":
                # v48.10 COPE: the exact PCD advantage is strongly tri-modal
                # (harm / dead-zone / benefit).  A continuous regressor collapsed
                # toward zero and made conformal radii span the whole target range.
                # Parameterise two ordered cumulative logits instead:
                #   P(benefit) <= P(non-harm), P(harm)=1-P(non-harm).
                center = delta[:, 0]
                half_width = 0.5 * torch.nn.functional.softplus(delta[:, 1])
                nonharm_logit = center + half_width
                benefit_logit = center - half_width
                if unified_benefit_logit is not None and unified_harm_logit is not None:
                    benefit_logit = unified_benefit_logit
                    harm_logit = unified_harm_logit
                    nonharm_logit = -harm_logit
                else:
                    # Preserve the legacy identity exactly: nominal non-harm is
                    # pinned before deriving the complementary harm logit.
                    if nominal_mask is not None:
                        nonharm_logit = torch.where(
                            nominal_mask, torch.zeros_like(nonharm_logit), nonharm_logit
                        )
                        benefit_logit = torch.where(
                            nominal_mask, torch.zeros_like(benefit_logit), benefit_logit
                        )
                    harm_logit = -nonharm_logit
                if (
                    evidence_calibrator_residual is not None
                    and self.direct_recovery_evidence_calibrator_mode == "dual_tail_context"
                    and unified_benefit_logit is None
                ):
                    # v48.18 DUET-BRIDGE: benefit and harm are independent tails.
                    # A candidate can be simultaneously uncertain/ambiguous in both
                    # tails; deployment then applies the harm veto rather than
                    # forcing one class probability down through a simplex softmax.
                    benefit_logit = benefit_logit + evidence_calibrator_residual[:, 0]
                    harm_logit = harm_logit + evidence_calibrator_residual[:, 1]
                    # The calibrator is a recovery-candidate correction only.
                    # Keep nominal evidence exactly at the source identity even
                    # after the residual output biases have been trained.
                    if nominal_mask is not None:
                        benefit_logit = torch.where(
                            nominal_mask, torch.zeros_like(benefit_logit), benefit_logit
                        )
                        harm_logit = torch.where(
                            nominal_mask, torch.zeros_like(harm_logit), harm_logit
                        )
                    nonharm_logit = -harm_logit
                if (
                    self.direct_recovery_evidence_admission_prior_mode == "joint_reserve"
                    and unified_benefit_logit is not None
                    and effective_component_harm_logits is not None
                ):
                    # v48.38 RFR / ROBUST-FRONTIER-RESERVE.
                    # v48.37 showed that the learned admission residual selected
                    # epoch zero while later updates worsened deployment-exact
                    # metrics. Admission is therefore no longer a separately
                    # learned sparse classifier in this mode. It is the exact
                    # piecewise-linear physical AND of benefit headroom and worst-component safety
                    # headroom, shared continuously across all audit strata.
                    if self.direct_recovery_evidence_reserve_factor_alignment:
                        # v48.39 DRFR: use exactly the signed factor coordinates
                        # seen by factor supervision.  The externally published
                        # nominal evidence is pinned to logit zero before the
                        # loss forms candidate-minus-nominal deltas, so the loss
                        # trains the recovery-candidate logits themselves.  The
                        # v48.38 reserve instead subtracted the *pre-pin* nominal
                        # here, cancelling the component prior (-2) and shifting
                        # every safety headroom by about one tolerance (0.05).
                        # Keep old behaviour behind the flag for checkpoint
                        # compatibility; all v48.39 arms require this alignment.
                        reserve_benefit_logit = unified_benefit_logit
                        reserve_component_logits = effective_component_harm_logits
                    else:
                        reserve_benefit_logit = self._candidate_minus_nominal(
                            unified_benefit_logit, group_index, is_nominal
                        )
                        reserve_component_logits = self._candidate_minus_nominal(
                            effective_component_harm_logits, group_index, is_nominal
                        )
                    predicted_benefit_margin = (
                        self.direct_recovery_evidence_benefit_margin_temperature
                        * reserve_benefit_logit
                    )
                    predicted_component_margins = (
                        self.direct_recovery_evidence_slack_temperature
                        * reserve_component_logits
                    )
                    # Only globally supported learned coordinates may define
                    # the learned reserve.  Reliability-zero coordinates are
                    # pinned to a constant neutral prior; after nominal
                    # subtraction their margin is exactly zero.  Including that
                    # zero in a max would force safety_headroom <= 0 and make
                    # positive reserve mathematically impossible whenever any
                    # coordinate is unsupported (the v48.37 support contract is
                    # [1,1,1,0,0]).  Mask them exactly as the RFR training loss
                    # does.  Independent measured hard vetoes remain downstream.
                    reserve_supported = (
                        self._direct_recovery_evidence_component_reliability
                        .to(device=predicted_component_margins.device) > 0.0
                    )
                    if not bool(reserve_supported.any()):
                        raise RuntimeError(
                            "joint_reserve requires at least one supported harm coordinate"
                        )
                    reserve_components = torch.where(
                        reserve_supported.unsqueeze(0),
                        predicted_component_margins,
                        predicted_component_margins.new_tensor(-1.0e6),
                    )
                    max_predicted_veto_margin = reserve_components.amax(dim=-1)
                    predicted_safety_headroom = -max_predicted_veto_margin
                    # Exact min preserves the physical zero boundary: reserve
                    # is positive iff both benefit headroom and safety headroom
                    # are positive.  It is piecewise differentiable and never
                    # allows one factor to compensate the other.
                    joint_reserve_margin = torch.minimum(
                        predicted_benefit_margin, predicted_safety_headroom
                    )
                    unified_admission_logit = (
                        joint_reserve_margin
                        / self.direct_recovery_evidence_joint_reserve_temperature
                    )
                    out["direct_recovery_evidence_predicted_benefit_margin"] = predicted_benefit_margin
                    out["direct_recovery_evidence_predicted_component_margins"] = predicted_component_margins
                    out["direct_recovery_evidence_max_predicted_veto_margin"] = max_predicted_veto_margin
                    out["direct_recovery_evidence_predicted_safety_headroom"] = predicted_safety_headroom
                    out["direct_recovery_evidence_joint_reserve_margin"] = joint_reserve_margin
                    out["direct_recovery_evidence_admission_prior"] = unified_admission_logit

                if nominal_mask is not None and unified_benefit_logit is not None:
                    benefit_logit = torch.where(
                        nominal_mask, torch.zeros_like(benefit_logit), benefit_logit
                    )
                    harm_logit = torch.where(
                        nominal_mask, torch.zeros_like(harm_logit), harm_logit
                    )
                    nonharm_logit = -harm_logit
                    if unified_component_harm_logits is not None:
                        unified_component_harm_logits = torch.where(
                            nominal_mask.unsqueeze(-1),
                            torch.zeros_like(unified_component_harm_logits),
                            unified_component_harm_logits,
                        )
                    if unified_admission_logit is not None:
                        unified_admission_logit = torch.where(
                            nominal_mask, torch.zeros_like(unified_admission_logit), unified_admission_logit
                        )
                benefit_prob = torch.sigmoid(benefit_logit)
                harm_prob = torch.sigmoid(harm_logit)
                if (
                    evidence_calibrator_residual is not None
                    and self.direct_recovery_evidence_calibrator_mode == "simplex_context"
                ):
                    # Identity-preserving tri-class correction.  Adding a bounded
                    # residual to source log-probabilities and renormalising with
                    # softmax preserves a valid harmful/dead/beneficial simplex,
                    # while allowing each tail to move independently.
                    eps = torch.finfo(benefit_prob.dtype).eps
                    dead_prob = (1.0 - benefit_prob - harm_prob).clamp_min(eps)
                    source_probs = torch.stack(
                        [harm_prob.clamp_min(eps), dead_prob, benefit_prob.clamp_min(eps)],
                        dim=-1,
                    )
                    calibrated_probs = torch.softmax(
                        torch.log(source_probs) + evidence_calibrator_residual, dim=-1
                    )
                    harm_prob = calibrated_probs[:, 0]
                    dead_prob = calibrated_probs[:, 1]
                    benefit_prob = calibrated_probs[:, 2]
                    if nominal_mask is not None:
                        harm_prob = torch.where(
                            nominal_mask, torch.full_like(harm_prob, 0.5), harm_prob
                        )
                        benefit_prob = torch.where(
                            nominal_mask, torch.full_like(benefit_prob, 0.5), benefit_prob
                        )
                        dead_prob = torch.where(
                            nominal_mask, torch.zeros_like(dead_prob), dead_prob
                        )
                    harm_logit = torch.logit(harm_prob.clamp(eps, 1.0 - eps))
                    benefit_logit = torch.logit(benefit_prob.clamp(eps, 1.0 - eps))
                    nonharm_logit = torch.logit((1.0 - harm_prob).clamp(eps, 1.0 - eps))
                    out["direct_recovery_evidence_class_probabilities"] = torch.stack(
                        [harm_prob, dead_prob, benefit_prob], dim=-1
                    )
                if unified_admission_logit is not None:
                    admission_prob = torch.sigmoid(unified_admission_logit)
                    # Nominal is pinned to probability 0.5, so subtracting 0.5
                    # yields an exact candidate-vs-nominal admission score.
                    evidence_score = admission_prob - 0.5
                    out["direct_recovery_admission_logit"] = unified_admission_logit
                    out["direct_recovery_admission_probability"] = admission_prob
                else:
                    evidence_score = benefit_prob - harm_prob
                out["direct_recovery_evidence_nonharm_logit"] = nonharm_logit
                out["direct_recovery_evidence_benefit_logit"] = benefit_logit
                out["direct_recovery_evidence_harm_logit"] = harm_logit
                out["direct_recovery_evidence_score"] = evidence_score
                if unified_component_harm_logits is not None:
                    out["direct_recovery_evidence_component_harm_logits"] = unified_component_harm_logits
                    out["direct_recovery_evidence_component_harm_probabilities"] = torch.sigmoid(
                        unified_component_harm_logits
                    )
                # Reuse the established admission plumbing.  These logits are
                # already candidate-vs-nominal evidence; nominal is pinned to 0.
                out["direct_recovery_opportunity_logit"] = benefit_logit
                out["direct_recovery_harm_logit"] = harm_logit
                out["direct_recovery_delta_mean"] = evidence_score
                out["direct_recovery_delta_logvar"] = torch.full_like(
                    evidence_score, self.direct_recovery_delta_initial_logvar
                )
            else:
                delta_mean = torch.tanh(delta[:, 0])
                delta_logvar = delta[:, 1].clamp(-7.0, 2.0)
                if nominal_mask is not None:
                    delta_mean = torch.where(nominal_mask, torch.zeros_like(delta_mean), delta_mean)
                out["direct_recovery_delta_mean"] = delta_mean
                out["direct_recovery_delta_logvar"] = delta_logvar
        return out

    def forward(
        self,
        x: torch.Tensor,
        option_features: torch.Tensor | None = None,
        bucket_id: torch.Tensor | None = None,
        group_index: torch.Tensor | None = None,
        is_nominal: torch.Tensor | None = None,
        direct_only: bool = False,
        witness_only: bool = False,
        witness_observation_only: bool = False,
        absolute_physical_headroom_features: torch.Tensor | None = None,
        absolute_executable_witness_features: torch.Tensor | None = None,
        absolute_common_witness_features: torch.Tensor | None = None,
        absolute_semantic_witness_features: torch.Tensor | None = None,
        root_valid: torch.Tensor | None = None,
        option_valid: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if witness_observation_only and not witness_only:
            raise ValueError("witness_observation_only requires witness_only=True")
        memory = self._scene_tokens(x)
        if direct_only:
            roct_signature = None
            native_certificate = None
            if (
                self.direct_evidence_roct_benefit is not None
                or self.direct_evidence_roct_deployability is not None
                or self.direct_recovery_evidence_native_certificate_preservation
                or self.direct_absolute_feasibility_head is not None
                or self.direct_absolute_physical_headroom_weight is not None
                or self.direct_absolute_executable_witness_weight is not None
                or self.direct_absolute_common_witness_gain is not None
                or self.direct_absolute_quantifier_witness_gain is not None
                or self.direct_absolute_semantic_witness_gain is not None
                or self.direct_absolute_root_tail_source_scale is not None
                or self.direct_absolute_structured_tail_field_weight is not None
            ):
                roct_signature, native_certificate = self._direct_recovery_option_compatibility_evidence(
                    memory, x, option_features,
                    group_index=group_index, is_nominal=is_nominal,
                    root_valid=root_valid, option_valid=option_valid
                )
            direct_out = self._direct_outputs(
                memory, x, bucket_id, group_index, is_nominal,
                recovery_option_compatibility_signature=roct_signature,
                native_recovery_certificate=native_certificate,
                absolute_physical_headroom_features=absolute_physical_headroom_features,
            )
            corrected_abs = self._direct_option_corrected_absolute_feasibility(
                memory, x, option_features, group_index=group_index, is_nominal=is_nominal,
                root_valid=root_valid, option_valid=option_valid,
            )
            if corrected_abs is not None:
                abs_logit, abs_probability = corrected_abs
                direct_out["direct_recovery_absolute_feasibility_logit"] = abs_logit
                direct_out["direct_recovery_absolute_feasibility_probability"] = abs_probability
                direct_out["direct_recovery_absolute_option_margin_bias"] = (
                    self.direct_absolute_option_margin_bias
                )
            erwf_abs = self._direct_executable_witness_absolute_feasibility(
                memory, x, option_features, absolute_executable_witness_features,
                group_index=group_index, is_nominal=is_nominal,
                root_valid=root_valid, option_valid=option_valid,
            )
            if erwf_abs is not None:
                abs_logit, abs_probability, erwf_features, erwf_weights = erwf_abs
                direct_out["direct_recovery_absolute_feasibility_logit"] = abs_logit
                direct_out["direct_recovery_absolute_feasibility_probability"] = abs_probability
                direct_out["direct_recovery_absolute_executable_witness_features"] = erwf_features
                direct_out["direct_recovery_absolute_executable_witness_weight"] = erwf_weights
            common_abs = self._direct_common_witness_absolute_feasibility(
                memory, x, option_features, absolute_common_witness_features,
                group_index=group_index, is_nominal=is_nominal,
                root_valid=root_valid, option_valid=option_valid,
            )
            if common_abs is not None:
                abs_logit, abs_probability, cw_features, cw_gains, cw_viability, cw_support = common_abs
                direct_out["direct_recovery_absolute_feasibility_logit"] = abs_logit
                direct_out["direct_recovery_absolute_feasibility_probability"] = abs_probability
                direct_out["direct_recovery_absolute_common_witness_features"] = cw_features
                direct_out["direct_recovery_absolute_common_witness_gain"] = cw_gains
                direct_out["direct_recovery_absolute_common_witness_viability"] = cw_viability
                direct_out["direct_recovery_absolute_common_option_support"] = cw_support
            quant_abs = self._direct_quantifier_witness_absolute_feasibility(
                memory, x, option_features, absolute_common_witness_features,
                group_index=group_index, is_nominal=is_nominal,
                root_valid=root_valid, option_valid=option_valid,
            )
            if quant_abs is not None:
                (abs_logit, abs_probability, qw_features, qw_gains, qw_viability, qw_support,
                 qw_best, qw_failure, qw_positive_count, qw_max_support) = quant_abs
                direct_out["direct_recovery_absolute_feasibility_logit"] = abs_logit
                direct_out["direct_recovery_absolute_feasibility_probability"] = abs_probability
                direct_out["direct_recovery_absolute_quantifier_witness_features"] = qw_features
                direct_out["direct_recovery_absolute_quantifier_witness_gain"] = qw_gains
                direct_out["direct_recovery_absolute_quantifier_witness_viability"] = qw_viability
                direct_out["direct_recovery_absolute_quantifier_common_option_support"] = qw_support
                direct_out["direct_recovery_absolute_quantifier_best_common_viability"] = qw_best
                direct_out["direct_recovery_absolute_quantifier_universal_failure"] = qw_failure
                direct_out["direct_recovery_absolute_quantifier_positive_option_count"] = qw_positive_count
                direct_out["direct_recovery_absolute_quantifier_max_common_support"] = qw_max_support
            semantic_abs = self._direct_semantic_witness_absolute_feasibility(
                memory, x, option_features, absolute_semantic_witness_features,
                group_index=group_index, is_nominal=is_nominal,
                root_valid=root_valid, option_valid=option_valid,
            )
            if semantic_abs is not None:
                (abs_logit, abs_probability, sw_features, sw_gains, sw_viability, sw_support,
                 sw_best, sw_failure, sw_positive_count, sw_max_support, sw_best_barriers,
                 sw_limiting_constraint, sw_classlocal_lcvar, sw_classlocal_viable_mass,
                 sw_classlocal_support_mean) = semantic_abs
                direct_out["direct_recovery_absolute_feasibility_logit"] = abs_logit
                direct_out["direct_recovery_absolute_feasibility_probability"] = abs_probability
                direct_out["direct_recovery_absolute_semantic_witness_features"] = sw_features
                direct_out["direct_recovery_absolute_semantic_witness_gain"] = sw_gains
                if self.direct_absolute_root_tail_source_scale is not None:
                    direct_out["direct_recovery_absolute_root_tail_source_scale"] = self.direct_absolute_root_tail_source_scale
                if self.direct_absolute_structured_tail_field_weight is not None:
                    direct_out["direct_recovery_absolute_structured_tail_field_weight"] = self.direct_absolute_structured_tail_field_weight
                direct_out["direct_recovery_absolute_semantic_witness_viability"] = sw_viability
                direct_out["direct_recovery_absolute_semantic_common_option_support"] = sw_support
                direct_out["direct_recovery_absolute_semantic_best_common_viability"] = sw_best
                direct_out["direct_recovery_absolute_semantic_universal_failure"] = sw_failure
                direct_out["direct_recovery_absolute_semantic_positive_option_count"] = sw_positive_count
                direct_out["direct_recovery_absolute_semantic_max_common_support"] = sw_max_support
                direct_out["direct_recovery_absolute_semantic_best_barriers"] = sw_best_barriers
                direct_out["direct_recovery_absolute_semantic_limiting_constraint"] = sw_limiting_constraint
                if sw_classlocal_lcvar is not None:
                    direct_out["direct_recovery_absolute_semantic_classlocal_lcvar_viability"] = sw_classlocal_lcvar
                    direct_out["direct_recovery_absolute_semantic_classlocal_viable_root_mass"] = sw_classlocal_viable_mass
                    direct_out["direct_recovery_absolute_semantic_classlocal_selected_support_mean"] = sw_classlocal_support_mean
            return direct_out
        scene_token = memory[:, 0]
        root_tokens = self._decode_roots(memory)

        root_logits = self.root_logit_head(root_tokens).squeeze(-1)
        recovery_root_logits = root_logits
        if self.direct_recovery_evidence_common_measure_root_mass:
            recovery_root_logits = self._common_measure_root_logits(
                root_logits, group_index, is_nominal, root_valid
            )
        obs_embeddings = self.obs_embed_head(root_tokens)

        margins = None
        if not witness_observation_only:
            root_expand = root_tokens.unsqueeze(2).expand(-1, -1, self.num_options, -1)
            opt_expand = self._option_tokens(x, option_features)
            margins = self.margin_head(torch.cat([root_expand, opt_expand], dim=-1)).squeeze(-1)

        diff = obs_embeddings.unsqueeze(2) - obs_embeddings.unsqueeze(1)
        dist2 = (diff * diff).mean(dim=-1)
        C = torch.exp(-dist2 / self.tau_obs).clamp(0.0, 1.0)
        eye = torch.eye(self.num_roots, dtype=C.dtype, device=C.device).unsqueeze(0)
        C = C * (1 - eye) + eye

        out: dict[str, torch.Tensor] = {
            "root_logits": root_logits,
            "obs_embeddings": obs_embeddings,
            "c_star": C,
        }
        if self.direct_recovery_evidence_common_measure_root_mass:
            # Diagnostic/deployment-side recovery measure.  ``root_logits`` stays
            # raw so legacy root losses, calibration diagnostics, and checkpoint
            # semantics remain byte-compatible when CMRI is enabled downstream.
            out["recovery_root_logits"] = recovery_root_logits
        if margins is not None:
            out["margins"] = margins
        # v48.47 DS-OFR witness stages update only the paper-native observation
        # or margin witness.  Skipping utility/direct-policy/diagnostic heads here
        # is execution-equivalent for those stages because every skipped parameter
        # is frozen and every corresponding loss weight is exactly zero.
        if witness_only:
            return out
        out["utility"] = self.utility_head(scene_token).squeeze(-1)
        postprefix_signature = None
        if (
            self.direct_evidence_postprefix_obs_transport_benefit is not None
            or self.direct_evidence_postprefix_obs_transport_harm is not None
        ):
            with torch.no_grad():
                postprefix_signature = self._postprefix_observation_equivalence_signature(
                    recovery_root_logits.detach(), obs_embeddings.detach(), self.tau_obs
                ).to(dtype=memory.dtype)
        roct_signature = None
        native_certificate = None
        if (
            self.direct_evidence_roct_benefit is not None
            or self.direct_evidence_roct_deployability is not None
            or self.direct_recovery_evidence_native_certificate_preservation
            or self.direct_absolute_feasibility_head is not None
            or self.direct_absolute_physical_headroom_weight is not None
            or self.direct_absolute_common_witness_gain is not None
            or self.direct_absolute_quantifier_witness_gain is not None
        ):
            with torch.no_grad():
                roct_signature, native_certificate = self._recovery_option_compatibility_signature(
                    recovery_root_logits.detach(), obs_embeddings.detach(), margins.detach(), self.tau_obs,
                    self.direct_recovery_evidence_roct_alpha,
                    self.direct_recovery_evidence_roct_beta,
                    self.direct_recovery_evidence_roct_top_m,
                    self.direct_recovery_evidence_roct_option_temperature,
                    root_valid=root_valid, option_valid=option_valid,
                    return_native_certificate=True,
                )
                roct_signature = roct_signature.to(dtype=memory.dtype).detach()
                native_certificate = native_certificate.to(dtype=memory.dtype).detach()
        out.update(
            self._direct_outputs(
                memory, x, bucket_id, group_index, is_nominal,
                postprefix_observation_signature=postprefix_signature,
                recovery_option_compatibility_signature=roct_signature,
                native_recovery_certificate=native_certificate,
                absolute_physical_headroom_features=absolute_physical_headroom_features,
            )
        )
        corrected_abs = self._direct_option_corrected_absolute_feasibility(
            memory, x, option_features, group_index=group_index, is_nominal=is_nominal,
            root_valid=root_valid, option_valid=option_valid,
        )
        if corrected_abs is not None:
            abs_logit, abs_probability = corrected_abs
            out["direct_recovery_absolute_feasibility_logit"] = abs_logit
            out["direct_recovery_absolute_feasibility_probability"] = abs_probability
            out["direct_recovery_absolute_option_margin_bias"] = (
                self.direct_absolute_option_margin_bias
            )
        erwf_abs = self._direct_executable_witness_absolute_feasibility(
            memory, x, option_features, absolute_executable_witness_features,
            group_index=group_index, is_nominal=is_nominal,
            root_valid=root_valid, option_valid=option_valid,
        )
        if erwf_abs is not None:
            abs_logit, abs_probability, erwf_features, erwf_weights = erwf_abs
            out["direct_recovery_absolute_feasibility_logit"] = abs_logit
            out["direct_recovery_absolute_feasibility_probability"] = abs_probability
            out["direct_recovery_absolute_executable_witness_features"] = erwf_features
            out["direct_recovery_absolute_executable_witness_weight"] = erwf_weights
        common_abs = self._direct_common_witness_absolute_feasibility(
            memory, x, option_features, absolute_common_witness_features,
            group_index=group_index, is_nominal=is_nominal,
            root_valid=root_valid, option_valid=option_valid,
        )
        if common_abs is not None:
            abs_logit, abs_probability, cw_features, cw_gains, cw_viability, cw_support = common_abs
            out["direct_recovery_absolute_feasibility_logit"] = abs_logit
            out["direct_recovery_absolute_feasibility_probability"] = abs_probability
            out["direct_recovery_absolute_common_witness_features"] = cw_features
            out["direct_recovery_absolute_common_witness_gain"] = cw_gains
            out["direct_recovery_absolute_common_witness_viability"] = cw_viability
            out["direct_recovery_absolute_common_option_support"] = cw_support
        quant_abs = self._direct_quantifier_witness_absolute_feasibility(
            memory, x, option_features, absolute_common_witness_features,
            group_index=group_index, is_nominal=is_nominal,
            root_valid=root_valid, option_valid=option_valid,
        )
        if quant_abs is not None:
            (abs_logit, abs_probability, qw_features, qw_gains, qw_viability, qw_support,
             qw_best, qw_failure, qw_positive_count, qw_max_support) = quant_abs
            out["direct_recovery_absolute_feasibility_logit"] = abs_logit
            out["direct_recovery_absolute_feasibility_probability"] = abs_probability
            out["direct_recovery_absolute_quantifier_witness_features"] = qw_features
            out["direct_recovery_absolute_quantifier_witness_gain"] = qw_gains
            out["direct_recovery_absolute_quantifier_witness_viability"] = qw_viability
            out["direct_recovery_absolute_quantifier_common_option_support"] = qw_support
            out["direct_recovery_absolute_quantifier_best_common_viability"] = qw_best
            out["direct_recovery_absolute_quantifier_universal_failure"] = qw_failure
            out["direct_recovery_absolute_quantifier_positive_option_count"] = qw_positive_count
            out["direct_recovery_absolute_quantifier_max_common_support"] = qw_max_support
        semantic_abs = self._direct_semantic_witness_absolute_feasibility(
            memory, x, option_features, absolute_semantic_witness_features,
            group_index=group_index, is_nominal=is_nominal,
            root_valid=root_valid, option_valid=option_valid,
        )
        if semantic_abs is not None:
            (abs_logit, abs_probability, sw_features, sw_gains, sw_viability, sw_support,
             sw_best, sw_failure, sw_positive_count, sw_max_support, sw_best_barriers,
             sw_limiting_constraint, sw_classlocal_lcvar, sw_classlocal_viable_mass,
             sw_classlocal_support_mean) = semantic_abs
            out["direct_recovery_absolute_feasibility_logit"] = abs_logit
            out["direct_recovery_absolute_feasibility_probability"] = abs_probability
            out["direct_recovery_absolute_semantic_witness_features"] = sw_features
            out["direct_recovery_absolute_semantic_witness_gain"] = sw_gains
            if self.direct_absolute_root_tail_source_scale is not None:
                out["direct_recovery_absolute_root_tail_source_scale"] = self.direct_absolute_root_tail_source_scale
            if self.direct_absolute_structured_tail_field_weight is not None:
                out["direct_recovery_absolute_structured_tail_field_weight"] = self.direct_absolute_structured_tail_field_weight
            out["direct_recovery_absolute_semantic_witness_viability"] = sw_viability
            out["direct_recovery_absolute_semantic_common_option_support"] = sw_support
            out["direct_recovery_absolute_semantic_best_common_viability"] = sw_best
            out["direct_recovery_absolute_semantic_universal_failure"] = sw_failure
            out["direct_recovery_absolute_semantic_positive_option_count"] = sw_positive_count
            out["direct_recovery_absolute_semantic_max_common_support"] = sw_max_support
            out["direct_recovery_absolute_semantic_best_barriers"] = sw_best_barriers
            out["direct_recovery_absolute_semantic_limiting_constraint"] = sw_limiting_constraint
            if sw_classlocal_lcvar is not None:
                out["direct_recovery_absolute_semantic_classlocal_lcvar_viability"] = sw_classlocal_lcvar
                out["direct_recovery_absolute_semantic_classlocal_viable_root_mass"] = sw_classlocal_viable_mass
                out["direct_recovery_absolute_semantic_classlocal_selected_support_mean"] = sw_classlocal_support_mean
        if self.root_signature_head is not None:
            out["root_signature"] = self.root_signature_head(root_tokens)
        if self.root_future_signature_head is not None:
            out["root_future_signature"] = self.root_future_signature_head(root_tokens)
        return out
