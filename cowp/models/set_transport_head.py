from __future__ import annotations

import math

import torch
from torch import nn


class SetTransportCertificateHead(nn.Module):
    """Candidate-conditioned transport of a *supported* natural option set.

    The denominator of option-preservation ratio (OPR) must contain only root
    options that are both valid and naturally low-burden.  v8 normalized over all
    decoder slots, including padded/high-burden modes, which makes the aggregate
    certificate non-identifiable and systematically biases OPR/conflict mass.

    This head predicts three separate objects:
      1. root support (valid + naturally low-burden) from the natural decoder;
      2. candidate-conditioned conflict/retention for each same-root mode;
      3. candidate-conditioned safe response existence and minimum burden.
    """

    def __init__(self, d_model: int = 128, hidden: int = 64, source_count: int = 4):
        super().__init__()
        h = int(hidden)
        self.source_count = int(source_count)
        self.cand = nn.Linear(d_model, h, bias=False)
        self.agent = nn.Linear(d_model, h, bias=False)
        self.graph = nn.Linear(d_model, h, bias=False)
        self.mode = nn.Linear(d_model, h, bias=False)
        self.norm = nn.LayerNorm(h)
        # conflict logit, intervention-conditioned burden parameter, epistemic uncertainty.
        # Keeping three channels preserves v8 checkpoint shape compatibility while
        # replacing the unconstrained "retain" classifier with an explicit burden field.
        self.mode_out = nn.Sequential(nn.GELU(), nn.Linear(h, 3))
        # A bounded *logit-space* residual preserves monotonic analytic structure
        # and avoids the flat gradients created by probability-space clamping.
        self.calibration = nn.Sequential(
            nn.Linear(d_model * 3, h), nn.GELU(), nn.LayerNorm(h), nn.Linear(h, 1), nn.Tanh()
        )

    @staticmethod
    def _soft_min(
        value: torch.Tensor,
        support: torch.Tensor,
        tau: float,
        *,
        empty_value: float = 2.0,
    ) -> torch.Tensor:
        tau = max(float(tau), 1.0e-3)
        mass = support.sum(dim=-1)
        normalized = support / mass.unsqueeze(-1).clamp_min(1.0e-8)
        out = -tau * torch.logsumexp(normalized.clamp_min(1.0e-8).log() - value / tau, dim=-1)
        return torch.where(mass > 1.0e-4, out.clamp_min(0.0), torch.full_like(out, float(empty_value)))

    def forward(
        self,
        *,
        z_agent: torch.Tensor,
        z_candidate: torch.Tensor,
        z_graph: torch.Tensor,
        critical_indices: torch.Tensor,
        natural: dict[str, torch.Tensor],
        response: dict[str, torch.Tensor],
        beta: torch.Tensor,
        alpha_opr: float = 0.35,
        gamma: float = 0.10,
        conflict_mass_floor: float = 0.10,
        burden_temperature: float = 0.08,
        gate_temperature: float = 0.06,
        calibration_scale: float = 0.10,
    ) -> dict[str, torch.Tensor]:
        B, K, D = z_candidate.shape
        A = critical_indices.shape[1]
        idx = critical_indices.clamp(0, max(z_agent.shape[1] - 1, 0)).long().unsqueeze(-1).expand(B, A, D)
        zcrit = torch.gather(z_agent, 1, idx)

        mode_latent = natural["mode_latent"]  # [B,A,M,D]
        M = mode_latent.shape[2]
        h = (
            self.cand(z_candidate)[:, :, None, None, :]
            + self.agent(zcrit)[:, None, :, None, :]
            + self.graph(z_graph)[:, None, None, None, :]
            + self.mode(mode_latent)[:, None, :, :, :]
        )
        raw = self.mode_out(self.norm(h))
        conflict_prob = torch.sigmoid(raw[..., 0])
        # A positive burden field gives the mode-level certificate a physically
        # interpretable transport variable.  v8 predicted retention directly, which
        # could disagree with its own burden threshold and was hard to audit.
        mode_burden_under = torch.nn.functional.softplus(raw[..., 1]).clamp(max=2.0)
        mode_uncertainty = torch.sigmoid(raw[..., 2])

        # Root support is a probability *mass*, not a softmax over every slot.
        # Missing fields preserve backward compatibility with v8 checkpoints/tests.
        root_prior = torch.softmax(natural["logits"].float(), dim=-1)
        valid_logits = natural.get("valid_logits")
        low_logits = natural.get("low_neutral_logits")
        root_valid = torch.sigmoid(valid_logits.float()) if valid_logits is not None else torch.ones_like(root_prior)
        root_low = torch.sigmoid(low_logits.float()) if low_logits is not None else torch.ones_like(root_prior)
        root_support = (root_prior * root_valid * root_low).clamp_min(0.0)  # [B,A,M]
        low_natural_mass_root = root_support.sum(dim=-1).clamp(0.0, 1.0)  # [B,A]
        normalized_support_root = root_support / low_natural_mass_root.unsqueeze(-1).clamp_min(1.0e-8)

        support = root_support[:, None, :, :].expand(B, K, A, M)
        normalized_support = normalized_support_root[:, None, :, :].expand(B, K, A, M)
        low_natural_mass = low_natural_mass_root[:, None, :].expand(B, K, A)
        source_prob = torch.softmax(natural["source_logits"].float(), dim=-1)[:, None, :, :, :]
        priority_prob = torch.sigmoid(natural["priority_logits"].float())[:, None, :, :]

        # A root option is retained only if it remains geometrically clear and its
        # intervention-conditioned burden stays below the scene-specific threshold.
        # This is the differentiable counterpart of A_nat ∩ R_low in the paper.
        beta_mode = beta.float()[:, None, :, None].expand(B, K, A, M)
        burden_low_prob = torch.sigmoid((beta_mode - mode_burden_under) / max(float(gate_temperature), 1.0e-3))
        low_safe_option_prob = (1.0 - conflict_prob) * burden_low_prob
        retained_mass = (support * low_safe_option_prob).sum(dim=-1)
        opr = torch.where(
            low_natural_mass > 1.0e-5,
            retained_mass / low_natural_mass.clamp_min(1.0e-6),
            torch.ones_like(retained_mass),
        ).clamp(0.0, 1.0)
        natural_conflict_mass = (support * conflict_prob).sum(dim=-1).clamp(0.0, 1.0)
        priority_conflict_mass = (support * priority_prob * conflict_prob).sum(dim=-1).clamp(0.0, 1.0)

        natural_mass_by_source = (support[..., None] * source_prob).sum(dim=-2)
        conflict_mass_by_source = (support[..., None] * source_prob * conflict_prob[..., None]).sum(dim=-2)
        low_safe_mass_by_source = (support[..., None] * source_prob * low_safe_option_prob[..., None]).sum(dim=-2)
        source_opr = torch.where(
            natural_mass_by_source > 1.0e-6,
            low_safe_mass_by_source / natural_mass_by_source.clamp_min(1.0e-6),
            torch.ones_like(low_safe_mass_by_source),
        ).clamp(0.0, 1.0)

        neutral_burden = natural.get("neutral_burden")
        if neutral_burden is None:
            neutral_burden = torch.zeros_like(root_prior)
        natural_min_burden_root = self._soft_min(
            neutral_burden.float(), root_support, burden_temperature, empty_value=0.0
        )
        natural_min_burden = natural_min_burden_root[:, None, :].expand(B, K, A)

        response_safe = torch.sigmoid(response["safe_logits"].float())
        response_low = torch.sigmoid(response["low_logits"].float())
        response_valid = torch.sigmoid(response.get("valid_logits", torch.zeros_like(response_safe)).float())
        response_weight = torch.softmax(response.get("mode_logits", torch.zeros_like(response_safe)).float(), dim=-1)
        response_low_safe = response_safe * response_low
        valid_weight = response_weight * response_valid
        response_low_safe_mass = (
            (valid_weight * response_low_safe).sum(dim=-1)
            / valid_weight.sum(dim=-1).clamp_min(1.0e-6)
        ).clamp(0.0, 1.0)
        # Differentiable OR over valid slots. Padded slots contribute near zero.
        log_no_low_safe = (
            response_valid * torch.log1p(-response_low_safe.clamp(max=1.0 - 1.0e-6))
        ).sum(dim=-1)
        response_exist_low_safe = (1.0 - torch.exp(log_no_low_safe)).clamp(0.0, 1.0)
        min_safe_burden = self._soft_min(
            response["burden_total"].float(),
            response_weight * response_safe * response_valid,
            burden_temperature,
            empty_value=2.0,
        )
        coercion_increment = (min_safe_burden - natural_min_burden).clamp(0.0, 2.0)

        beta_pair = beta.float()[:, None, :].expand(B, K, A)
        gt = max(float(gate_temperature), 1.0e-3)
        conflict_gate = torch.sigmoid((natural_conflict_mass - float(conflict_mass_floor)) / gt)
        burden_gate = torch.sigmoid((min_safe_burden - (beta_pair + float(gamma))) / gt)
        option_gate = torch.sigmoid((float(alpha_opr) - opr) / gt)
        response_absence_gate = 1.0 - response_exist_low_safe
        failure_union = 1.0 - (
            (1.0 - burden_gate) * (1.0 - option_gate) * (1.0 - response_absence_gate)
        )
        analytic_witness = (conflict_gate * failure_union).clamp(1.0e-5, 1.0 - 1.0e-5)

        calib_in = torch.cat([
            z_candidate[:, :, None, :].expand(B, K, A, D),
            zcrit[:, None, :, :].expand(B, K, A, D),
            z_graph[:, None, None, :].expand(B, K, A, D),
        ], dim=-1)
        residual_logit = self.calibration(calib_in).squeeze(-1) * float(calibration_scale)
        analytic_logit = torch.logit(analytic_witness)
        witness_logit = analytic_logit + residual_logit
        witness_prob = torch.sigmoid(witness_logit)

        entropy = -(normalized_support * normalized_support.clamp_min(1.0e-8).log()).sum(dim=-1)
        entropy = entropy / max(math.log(float(max(M, 2))), 1.0e-6)
        support_gap = (1.0 - low_natural_mass).clamp(0.0, 1.0)
        uncertainty = (
            (normalized_support * mode_uncertainty).sum(dim=-1)
            + entropy
            + support_gap
        ).div(3.0).clamp(0.0, 1.0)

        return {
            "exist_logits": witness_logit,
            "witness_prob": witness_prob,
            "analytic_witness_prob": analytic_witness,
            "opr": opr,
            "min_safe_burden": min_safe_burden,
            "natural_min_burden": natural_min_burden,
            "coercion_increment": coercion_increment,
            "low_natural_mass": low_natural_mass,
            "natural_conflict_mass": natural_conflict_mass,
            "priority_conflict_mass": priority_conflict_mass,
            "response_low_safe_mass": response_low_safe_mass,
            "response_exist_low_safe": response_exist_low_safe,
            "natural_mass_by_source": natural_mass_by_source,
            "conflict_mass_by_source": conflict_mass_by_source,
            "low_safe_mass_by_source": low_safe_mass_by_source,
            "source_opr": source_opr,
            "mode_conflict_logit": raw[..., 0],
            "mode_burden_under": mode_burden_under,
            "mode_conflict_prob": conflict_prob,
            "mode_low_burden_prob": burden_low_prob,
            "mode_low_safe_prob": low_safe_option_prob,
            "root_support_prob": (root_valid * root_low).clamp(0.0, 1.0),
            "root_valid_prob": root_valid,
            "root_low_neutral_prob": root_low,
            "uncertainty": uncertainty,
        }
