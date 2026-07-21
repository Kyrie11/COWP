from __future__ import annotations

import torch
from torch import nn


class SetTransportCertificateHead(nn.Module):
    """Geometry-conditioned primitive-indexed set transport certificate.

    v9 predicted per-mode conflict/retention from additive latent embeddings only.
    That representation did not expose the candidate--natural relative geometry
    that actually defines a conflict.  v10 augments every primitive with compact
    differentiable trajectory geometry and uses response mixture weights when
    estimating same-root recovery, preventing diffuse root probabilities from
    creating spurious recovery mass.
    """

    def __init__(
        self,
        d_model: int = 128,
        hidden: int = 64,
        source_count: int = 4,
        geometry_steps: int = 16,
        geometry_dim: int = 8,
    ):
        super().__init__()
        h = int(hidden)
        self.source_count = int(source_count)
        self.geometry_steps = max(int(geometry_steps), 4)
        self.cand = nn.Linear(d_model, h, bias=False)
        self.agent = nn.Linear(d_model, h, bias=False)
        self.graph = nn.Linear(d_model, h, bias=False)
        self.mode = nn.Linear(d_model, h, bias=False)
        self.geometry = nn.Sequential(
            nn.Linear(int(geometry_dim), h), nn.GELU(), nn.LayerNorm(h), nn.Linear(h, h, bias=False)
        )
        self.norm = nn.LayerNorm(h)
        self.mode_out = nn.Sequential(nn.GELU(), nn.Linear(h, 3))
        self.calibration = nn.Sequential(
            nn.Linear(d_model * 3, h), nn.GELU(), nn.LayerNorm(h), nn.Linear(h, 1), nn.Tanh()
        )

    @staticmethod
    def _soft_min_burden(
        burden: torch.Tensor,
        safe_prob: torch.Tensor,
        valid_prob: torch.Tensor,
        mode_weight: torch.Tensor,
        tau: float,
    ) -> torch.Tensor:
        tau = max(float(tau), 1.0e-3)
        support = (safe_prob * valid_prob * mode_weight).clamp_min(1.0e-8)
        value = -tau * torch.logsumexp(support.log() - burden / tau, dim=-1)
        existence = support.sum(dim=-1)
        return torch.where(existence > 1.0e-4, value.clamp_min(0.0), torch.full_like(value, 2.0))

    def _relative_geometry(
        self,
        candidate_traj: torch.Tensor | None,
        natural_traj: torch.Tensor | None,
        *,
        B: int,
        K: int,
        A: int,
        M: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return [B,K,A,M,8] compact candidate--primitive geometry.

        Features are normalized min/mean/final distance, time to closest point,
        approach distance, mean heading agreement, mean relative speed and minimum
        footprint clearance.  Sampling keeps memory bounded for 64x6x24 banks.
        """
        if candidate_traj is None or natural_traj is None:
            return torch.zeros(B, K, A, M, 8, device=device, dtype=dtype)
        if candidate_traj.ndim != 4 or natural_traj.ndim != 5:
            return torch.zeros(B, K, A, M, 8, device=device, dtype=dtype)
        T = min(int(candidate_traj.shape[-2]), int(natural_traj.shape[-2]))
        if T <= 0:
            return torch.zeros(B, K, A, M, 8, device=device, dtype=dtype)
        S = min(self.geometry_steps, T)
        sample = torch.linspace(0, T - 1, steps=S, device=device).round().long().unique()
        cand = candidate_traj[..., sample, :].float()
        nat = natural_traj[..., sample, :].float()
        cxy = cand[..., :2][:, :, None, None, :, :]
        nxy = nat[..., :2][:, None, :, :, :, :]
        delta = cxy - nxy
        dist = torch.linalg.vector_norm(delta, dim=-1)
        min_dist, min_idx = dist.min(dim=-1)
        mean_dist = dist.mean(dim=-1)
        final_dist = dist[..., -1]
        tmin = min_idx.float() / max(float(dist.shape[-1] - 1), 1.0)
        approach = dist[..., 0] - min_dist

        if cand.shape[-1] >= 3 and nat.shape[-1] >= 3:
            ch = cand[..., 2][:, :, None, None, :]
            nh = nat[..., 2][:, None, :, :, :]
            heading_agreement = torch.cos(ch - nh).mean(dim=-1)
        else:
            heading_agreement = torch.zeros_like(min_dist)
        if cand.shape[-1] >= 5 and nat.shape[-1] >= 5:
            cv = cand[..., 3:5][:, :, None, None, :, :]
            nv = nat[..., 3:5][:, None, :, :, :, :]
            rel_speed = torch.linalg.vector_norm(cv - nv, dim=-1).mean(dim=-1)
        else:
            rel_speed = torch.zeros_like(min_dist)
        if cand.shape[-1] >= 7 and nat.shape[-1] >= 7:
            cr = 0.5 * torch.linalg.vector_norm(cand[..., 5:7], dim=-1)[:, :, None, None, :]
            nr = 0.5 * torch.linalg.vector_norm(nat[..., 5:7], dim=-1)[:, None, :, :, :]
            clearance = (dist - cr - nr).min(dim=-1).values
        else:
            clearance = min_dist

        feat = torch.stack(
            [
                min_dist / 20.0,
                mean_dist / 30.0,
                final_dist / 30.0,
                tmin,
                approach / 20.0,
                heading_agreement,
                rel_speed / 10.0,
                clearance / 10.0,
            ],
            dim=-1,
        )
        return torch.nan_to_num(feat, nan=0.0, posinf=5.0, neginf=-5.0).clamp(-5.0, 5.0).to(dtype=dtype)

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
        candidate_traj: torch.Tensor | None = None,
        natural_traj: torch.Tensor | None = None,
        alpha_opr: float = 0.35,
        gamma: float = 0.10,
        conflict_mass_floor: float = 0.10,
        burden_temperature: float = 0.08,
        gate_temperature: float = 0.06,
        calibration_scale: float = 0.10,
        root_mass_scale: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        B, K, D = z_candidate.shape
        A = critical_indices.shape[1]
        idx = critical_indices.clamp(0, max(z_agent.shape[1] - 1, 0)).long().unsqueeze(-1).expand(B, A, D)
        zcrit = torch.gather(z_agent, 1, idx)

        mode_latent = natural["mode_latent"]
        M = mode_latent.shape[2]
        geometry_feat = self._relative_geometry(
            candidate_traj, natural_traj, B=B, K=K, A=A, M=M,
            device=z_candidate.device, dtype=z_candidate.dtype,
        )
        h = (
            self.cand(z_candidate)[:, :, None, None, :]
            + self.agent(zcrit)[:, None, :, None, :]
            + self.graph(z_graph)[:, None, None, None, :]
            + self.mode(mode_latent)[:, None, :, :, :]
            + self.geometry(geometry_feat)
        )
        raw = self.mode_out(self.norm(h))
        conflict_logit = raw[..., 0]
        retain_logit = raw[..., 1]
        conflict_prob = torch.sigmoid(conflict_logit)
        retain_prob = torch.sigmoid(retain_logit)
        mode_uncertainty = torch.sigmoid(raw[..., 2])

        natural_weight = torch.softmax(natural["logits"].float(), dim=-1)[:, None, :, :].expand(B, K, A, M)
        source_prob = torch.softmax(natural["source_logits"].float(), dim=-1)[:, None, :, :, :]
        priority_prob = torch.sigmoid(natural["priority_logits"].float())[:, None, :, :]

        low_safe_option_prob = retain_prob * (1.0 - conflict_prob)
        opr = (natural_weight * low_safe_option_prob).sum(dim=-1).clamp(0.0, 1.0)
        natural_conflict_mass = (natural_weight * conflict_prob).sum(dim=-1).clamp(0.0, 1.0)
        priority_conflict_mass = (natural_weight * priority_prob * conflict_prob).sum(dim=-1).clamp(0.0, 1.0)
        natural_mass_by_source = (natural_weight[..., None] * source_prob).sum(dim=-2)
        conflict_mass_by_source = (natural_weight[..., None] * source_prob * conflict_prob[..., None]).sum(dim=-2)
        low_safe_mass_by_source = (natural_weight[..., None] * source_prob * low_safe_option_prob[..., None]).sum(dim=-2)
        source_opr = low_safe_mass_by_source / natural_mass_by_source.clamp_min(1.0e-6)

        response_safe = torch.sigmoid(response["safe_logits"].float())
        response_low = torch.sigmoid(response["low_logits"].float())
        response_valid = torch.sigmoid(response.get("valid_logits", torch.zeros_like(response_safe)).float())
        response_weight = torch.softmax(response.get("mode_logits", torch.zeros_like(response_safe)).float(), dim=-1)
        response_low_safe = response_safe * response_low * response_valid
        response_low_safe_mass = (response_weight * response_low_safe).sum(dim=-1).clamp(0.0, 1.0)
        response_exist_low_safe = 1.0 - torch.prod((1.0 - response_low_safe).clamp(1.0e-5, 1.0), dim=-1)

        root_logits = response.get("root_logits")
        if root_logits is not None and root_logits.shape[-1] == M:
            root_prob = torch.softmax(root_logits.float(), dim=-1)
            # v9 omitted response mixture weights here.  With 32 diffuse slots,
            # the product-of-complements greatly overestimated root recovery.
            root_low_safe_mass = (
                root_prob * response_low_safe[..., None] * response_weight[..., None]
            ).sum(dim=-2)
            root_response_exist = (root_low_safe_mass * float(root_mass_scale)).clamp(0.0, 1.0)
        else:
            root_response_exist = low_safe_option_prob
        conflicted_mass = natural_weight * conflict_prob
        conflict_denom = conflicted_mass.sum(dim=-1)
        root_recovery_mass = (conflicted_mass * root_response_exist).sum(dim=-1) / conflict_denom.clamp_min(1.0e-6)
        root_recovery_mass = torch.where(
            conflict_denom > 1.0e-6, root_recovery_mass, torch.ones_like(root_recovery_mass)
        ).clamp(0.0, 1.0)
        min_safe_burden = self._soft_min_burden(
            response["burden_total"].float(), response_safe, response_valid, response_weight, burden_temperature
        )

        beta_pair = beta.float()[:, None, :].expand(B, K, A)
        gt = max(float(gate_temperature), 1.0e-3)
        floor = max(float(conflict_mass_floor), 1.0e-4)
        conflict_support = (natural_conflict_mass / floor).clamp(0.0, 1.0)
        conflict_gate = conflict_support * torch.sigmoid((natural_conflict_mass - floor) / gt)
        burden_gate = torch.sigmoid((min_safe_burden - (beta_pair + float(gamma))) / gt)
        option_gate = torch.sigmoid((float(alpha_opr) - opr) / gt)
        response_absence_gate = 1.0 - root_recovery_mass
        failure_union = 1.0 - (1.0 - burden_gate) * (1.0 - option_gate) * (1.0 - response_absence_gate)
        analytic_witness = (conflict_gate * failure_union).clamp(0.0, 1.0)

        calib_in = torch.cat([
            z_candidate[:, :, None, :].expand(B, K, A, D),
            zcrit[:, None, :, :].expand(B, K, A, D),
            z_graph[:, None, None, :].expand(B, K, A, D),
        ], dim=-1)
        residual = self.calibration(calib_in).squeeze(-1) * float(calibration_scale)
        witness_prob = (analytic_witness + residual).clamp(0.0, 1.0)
        eps = 1.0e-5
        witness_logit = torch.logit(witness_prob.clamp(eps, 1.0 - eps))
        uncertainty = (natural_weight * mode_uncertainty).sum(dim=-1).clamp(0.0, 1.0)

        return {
            "exist_logits": witness_logit,
            "witness_prob": witness_prob,
            "analytic_witness_prob": analytic_witness,
            "opr": opr,
            "min_safe_burden": min_safe_burden,
            "natural_conflict_mass": natural_conflict_mass,
            "priority_conflict_mass": priority_conflict_mass,
            "response_low_safe_mass": response_low_safe_mass,
            "response_exist_low_safe": response_exist_low_safe,
            "root_response_exist": root_response_exist,
            "root_recovery_mass": root_recovery_mass,
            "natural_mass_by_source": natural_mass_by_source,
            "conflict_mass_by_source": conflict_mass_by_source,
            "low_safe_mass_by_source": low_safe_mass_by_source,
            "source_opr": source_opr,
            "mode_conflict_logits": conflict_logit,
            "mode_retain_logits": retain_logit,
            "mode_conflict_prob": conflict_prob,
            "mode_retain_prob": retain_prob,
            "mode_uncertainty": mode_uncertainty,
            "uncertainty": uncertainty,
            "geometry_min_distance_norm": geometry_feat[..., 0],
            "geometry_clearance_norm": geometry_feat[..., 7],
        }
