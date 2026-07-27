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
        response_topk: int = 8,
    ):
        super().__init__()
        h = int(hidden)
        self.source_count = int(source_count)
        self.geometry_steps = max(int(geometry_steps), 4)
        self.response_topk = max(int(response_topk), 1)
        self.cand = nn.Linear(d_model, h, bias=False)
        self.agent = nn.Linear(d_model, h, bias=False)
        self.graph = nn.Linear(d_model, h, bias=False)
        self.mode = nn.Linear(d_model, h, bias=False)
        self.geometry = nn.Sequential(
            nn.Linear(int(geometry_dim), h), nn.GELU(), nn.LayerNorm(h), nn.Linear(h, h, bias=False)
        )
        self.norm = nn.LayerNorm(h)
        # Per-natural-root primitive outputs:
        #   0 conflict,
        #   1 retained-low-safe conditional on no conflict,
        #   2 epistemic/aleatoric error proxy,
        #   3 existence of a low-burden safe response transported from this root.
        #
        # v11 inferred item (3) indirectly from a generic unordered response bank
        # plus a 24-way root classifier.  That is unnecessarily hard and, more
        # importantly, does not condition response feasibility on the natural
        # option that the paper claims to transport.  The direct root-recovery
        # primitive is the minimal structural repair: it is still supervised by
        # the explicit response-set/root labels, while making the decision
        # certificate root indexed by construction.
        self.mode_out = nn.Sequential(nn.GELU(), nn.Linear(h, 4))
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
        # This quantity is existential: the least-burden safe response should not
        # disappear merely because its mixture probability is small.  v11 folded
        # ``mode_weight`` into support, turning an existential minimum into a
        # density-weighted expectation.  Normalize only the safe/valid support so
        # duplicate slots do not artificially reduce the soft minimum.
        support = (safe_prob * valid_prob).clamp_min(0.0)
        existence = support.amax(dim=-1)
        norm_support = support / support.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
        value = -tau * torch.logsumexp(
            norm_support.clamp_min(1.0e-8).log() - burden / tau, dim=-1
        )
        return torch.where(existence > 1.0e-4, value.clamp_min(0.0), torch.full_like(value, 2.0))

    @staticmethod
    def _soft_root_min_burden(
        burden: torch.Tensor,
        safe_prob: torch.Tensor,
        valid_prob: torch.Tensor,
        root_prob: torch.Tensor,
        tau: float,
    ) -> torch.Tensor:
        """Differentiable same-root minimum burden, Eq. b*_{ikm}.

        ``root_prob`` is [B,K,A,R,M].  The support is normalized independently
        for every root so response-slot duplication cannot lower the minimum.
        A root without a credible safe/valid response receives the finite
        emergency sentinel 2.0 used by the cache schema.
        """
        tau = max(float(tau), 1.0e-3)
        support = (
            safe_prob[..., None]
            * valid_prob[..., None]
            * root_prob.clamp_min(0.0)
        )
        existence = support.amax(dim=-2)
        norm = support / support.sum(dim=-2, keepdim=True).clamp_min(1.0e-8)
        value = -tau * torch.logsumexp(
            norm.clamp_min(1.0e-8).log() - burden[..., None] / tau,
            dim=-2,
        )
        return torch.where(
            existence > 1.0e-4,
            value.clamp(0.0, 2.0),
            torch.full_like(value, 2.0),
        )

    @staticmethod
    def _weighted_upper_cvar(
        values: torch.Tensor,
        weights: torch.Tensor,
        tail_mass: float,
    ) -> torch.Tensor:
        """Weighted upper-tail CVaR over the last/root dimension.

        ``tail_mass`` is the probability mass averaged from the worst end, e.g.
        0.25 averages the worst 25% of conflicted root mass.  Sorting is
        piecewise differentiable and exactly matches the finite-root definition,
        unlike a max or an unrelated global response minimum.
        """
        q = min(max(float(tail_mass), 1.0e-3), 1.0)
        w = weights.clamp_min(0.0)
        total = w.sum(dim=-1, keepdim=True)
        w = w / total.clamp_min(1.0e-8)
        v_sorted, order = values.sort(dim=-1, descending=True)
        w_sorted = torch.gather(w, -1, order)
        before = w_sorted.cumsum(dim=-1) - w_sorted
        remaining = torch.relu(torch.as_tensor(q, device=w.device, dtype=w.dtype) - before)
        take = torch.minimum(w_sorted, remaining)
        denom = take.sum(dim=-1).clamp_min(1.0e-8)
        cvar = (take * v_sorted).sum(dim=-1) / denom
        return torch.where(total.squeeze(-1) > 1.0e-8, cvar, torch.zeros_like(cvar))

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
        critical_mask: torch.Tensor | None = None,
        alpha_opr: float = 0.35,
        gamma: float = 0.10,
        conflict_mass_floor: float = 0.10,
        burden_temperature: float = 0.08,
        gate_temperature: float = 0.06,
        calibration_scale: float = 0.10,
        root_mass_scale: float = 1.0,
        candidate_tail_temperature: float = 0.12,
        root_probability_floor: float = 0.02,
        cvar_tail_mass: float = 0.25,
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
        retain_conditional_logit = raw[..., 1]
        conflict_prob = torch.sigmoid(conflict_logit)
        # Labels define retained-low-safe as an explicitly non-conflicting mode.
        # Factorize P(retained-low-safe) = P(no conflict) * P(low-safe | no
        # conflict), which removes impossible high-conflict/high-retention states
        # and prevents the downstream OPR computation from applying the same
        # no-conflict condition twice.
        retain_conditional_prob = torch.sigmoid(retain_conditional_logit)
        retain_prob = ((1.0 - conflict_prob) * retain_conditional_prob).clamp(1.0e-5, 1.0 - 1.0e-5)
        retain_logit = torch.logit(retain_prob)
        mode_uncertainty = torch.sigmoid(raw[..., 2])
        mode_recovery_logit = raw[..., 3]
        mode_recovery_prob = torch.sigmoid(mode_recovery_logit)

        natural_weight_raw = torch.softmax(natural["logits"].float(), dim=-1)
        eps_p = min(max(float(root_probability_floor), 0.0), 0.25)
        natural_weight_raw = (1.0 - eps_p) * natural_weight_raw + eps_p / max(M, 1)
        natural_weight = natural_weight_raw[:, None, :, :].expand(B, K, A, M)
        source_prob = torch.softmax(natural["source_logits"].float(), dim=-1)[:, None, :, :, :]
        priority_prob = torch.sigmoid(natural["priority_logits"].float())[:, None, :, :]

        # ``retain_prob`` already includes the no-conflict event by construction.
        low_safe_option_prob = retain_prob
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
        # The label is existential (there is at least one low-burden safe
        # response), not a response-mixture expectation.  v10 multiplied every
        # root contribution by the mixture weight and under-estimated sparse
        # recovery, while an all-slot noisy-OR lets duplicated/diffuse slots
        # manufacture near-one recovery.  Use the fuzzy existential operator
        # (max/supremum) over only the strongest slots.  It is bounded, invariant
        # to duplicate slots, and gives a uniform root assignment only 1/M mass.
        top_r = min(self.response_topk, int(response_low_safe.shape[-1]))
        response_strength, response_top_idx = torch.topk(response_low_safe, k=top_r, dim=-1)
        response_exist_low_safe = response_strength.amax(dim=-1)

        root_logits = response.get("root_logits")
        if root_logits is not None and root_logits.shape[-1] == M:
            root_prob = torch.softmax(root_logits.float(), dim=-1)
            gather_idx = response_top_idx[..., None].expand(*response_top_idx.shape, M)
            root_top = torch.gather(root_prob, dim=-2, index=gather_idx)
            root_contribution = (
                root_top * response_strength[..., None] * float(root_mass_scale)
            ).clamp(0.0, 1.0)
            response_root_exist_aux = root_contribution.amax(dim=-2)
        else:
            root_prob = torch.full(
                (*response_safe.shape, M),
                1.0 / max(M, 1),
                device=response_safe.device,
                dtype=response_safe.dtype,
            )
            response_root_exist_aux = low_safe_option_prob
        # Preserve the legacy response-bank reconstruction under its original
        # name for diagnostics/ablations.  The new root-indexed quantity is the
        # primary transport certificate used by the planner.
        root_response_exist = response_root_exist_aux
        root_transport_exist = mode_recovery_prob
        conflicted_mass = natural_weight * conflict_prob
        conflict_denom = conflicted_mass.sum(dim=-1)
        root_recovery_mass = (conflicted_mass * root_transport_exist).sum(dim=-1) / conflict_denom.clamp_min(1.0e-6)
        root_recovery_mass = torch.where(
            conflict_denom > 1.0e-6, root_recovery_mass, torch.ones_like(root_recovery_mass)
        ).clamp(0.0, 1.0)
        min_safe_burden = self._soft_min_burden(
            response["burden_total"].float(), response_safe, response_valid, response_weight, burden_temperature
        )
        root_min_safe_burden = self._soft_root_min_burden(
            response["burden_total"].float(),
            response_safe,
            response_valid,
            root_prob,
            burden_temperature,
        )

        beta_pair = beta.float()[:, None, :].expand(B, K, A)
        root_excess = torch.relu(root_min_safe_burden - beta_pair[..., None])
        conflicted_root_weight = natural_weight * conflict_prob
        tail_burden_excess = self._weighted_upper_cvar(
            root_excess,
            conflicted_root_weight,
            cvar_tail_mass,
        )
        gt = max(float(gate_temperature), 1.0e-3)
        floor = max(float(conflict_mass_floor), 1.0e-4)
        conflict_support = (natural_conflict_mass / floor).clamp(0.0, 1.0)
        conflict_gate = conflict_support * torch.sigmoid((natural_conflict_mass - floor) / gt)
        burden_gate = torch.sigmoid((tail_burden_excess - float(gamma)) / gt)
        option_gate = torch.sigmoid((float(alpha_opr) - opr) / gt)
        # No independent response-absence heuristic is needed: an absent same-root
        # response receives burden 2.0 and therefore enters the CVaR term.
        failure_union = 1.0 - (1.0 - burden_gate) * (1.0 - option_gate)
        analytic_witness = (conflict_gate * failure_union).clamp(0.0, 1.0)
        response_absence_gate = 1.0 - root_recovery_mass

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

        # ------------------------------------------------------------------
        # Budgeted Counterfactual Option Transport (BCOT) candidate certificate
        # ------------------------------------------------------------------
        # A max/any reduction over up to six pair certificates turns a moderate
        # pair-level false-positive rate into a very low candidate-level recall.
        # The paper's object is instead the amount of low-burden option mass that
        # ego removes.  Aggregate that mass with a smooth tail-risk term, while
        # preserving an explicit hard signal for a genuinely severe protected
        # pair.  This is monotone in conflict, option loss, burden excess and
        # failed same-root recovery, so a generic candidate classifier cannot
        # silently replace the proposed mechanism.
        unrecovered_mode_mass = natural_weight * conflict_prob * (1.0 - root_transport_exist)
        unrecovered_conflict_mass = unrecovered_mode_mass.sum(dim=-1).clamp(0.0, 1.0)
        option_shortfall = (
            torch.relu(torch.as_tensor(float(alpha_opr), device=opr.device, dtype=opr.dtype) - opr)
            / max(float(alpha_opr), 1.0e-6)
        ).clamp(0.0, 1.0)
        priority_support = (natural_weight * priority_prob).sum(dim=-1).clamp(0.0, 1.0)
        pair_transport_deficit = (
            0.55 * unrecovered_conflict_mass
            + 0.25 * natural_conflict_mass * burden_gate
            + 0.20 * option_shortfall
        ).clamp(0.0, 1.0)
        pair_severe_prob = (
            conflict_gate
            * response_absence_gate
            * burden_gate
            * priority_support
            * (1.0 - 0.5 * uncertainty)
        ).clamp(0.0, 1.0)

        if critical_mask is None:
            cmask = torch.ones(B, A, device=pair_transport_deficit.device, dtype=torch.bool)
        else:
            cmask = critical_mask.bool()
        cm = cmask[:, None, :].expand(B, K, A)
        priority_weight = torch.where(
            cm,
            0.25 + 0.75 * priority_support,
            torch.zeros_like(priority_support),
        )
        weight_denom = priority_weight.sum(dim=-1).clamp_min(1.0e-6)
        candidate_mean_deficit = (
            priority_weight * pair_transport_deficit
        ).sum(dim=-1) / weight_denom

        tail_tau = max(float(candidate_tail_temperature), 1.0e-3)
        tail_logits = pair_transport_deficit / tail_tau + torch.log(priority_weight.clamp_min(1.0e-8))
        tail_logits = torch.where(cm, tail_logits, torch.full_like(tail_logits, -1.0e4))
        tail_weight = torch.softmax(tail_logits, dim=-1) * cm.float()
        tail_weight = tail_weight / tail_weight.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)
        candidate_tail_deficit = (tail_weight * pair_transport_deficit).sum(dim=-1)
        candidate_severe_prob = torch.where(
            cm, pair_severe_prob, torch.zeros_like(pair_severe_prob)
        ).amax(dim=-1)
        candidate_transport_risk = (
            0.55 * candidate_mean_deficit
            + 0.30 * candidate_tail_deficit
            + 0.15 * candidate_severe_prob
        ).clamp(0.0, 1.0)
        candidate_mean_uncertainty = (
            priority_weight * uncertainty
        ).sum(dim=-1) / weight_denom
        candidate_tail_uncertainty = (tail_weight * uncertainty).sum(dim=-1)
        candidate_transport_uncertainty = (
            0.60 * candidate_mean_uncertainty + 0.40 * candidate_tail_uncertainty
        ).clamp(0.0, 1.0)

        return {
            "exist_logits": witness_logit,
            "witness_prob": witness_prob,
            "analytic_witness_prob": analytic_witness,
            "opr": opr,
            "min_safe_burden": min_safe_burden,
            "root_min_safe_burden": root_min_safe_burden,
            "tail_burden_excess": tail_burden_excess,
            "natural_conflict_mass": natural_conflict_mass,
            "priority_conflict_mass": priority_conflict_mass,
            "response_low_safe_mass": response_low_safe_mass,
            "response_exist_low_safe": response_exist_low_safe,
            "root_response_exist": root_response_exist,
            "root_transport_exist": root_transport_exist,
            "response_root_exist_aux": response_root_exist_aux,
            "root_recovery_mass": root_recovery_mass,
            "natural_mass_by_source": natural_mass_by_source,
            "conflict_mass_by_source": conflict_mass_by_source,
            "low_safe_mass_by_source": low_safe_mass_by_source,
            "source_opr": source_opr,
            "mode_conflict_logits": conflict_logit,
            "mode_retain_logits": retain_logit,
            "mode_retain_conditional_logits": retain_conditional_logit,
            "mode_conflict_prob": conflict_prob,
            "mode_retain_prob": retain_prob,
            "mode_retain_conditional_prob": retain_conditional_prob,
            "mode_uncertainty": mode_uncertainty,
            "mode_recovery_logits": mode_recovery_logit,
            "mode_recovery_prob": mode_recovery_prob,
            "uncertainty": uncertainty,
            "geometry_min_distance_norm": geometry_feat[..., 0],
            "geometry_clearance_norm": geometry_feat[..., 7],
            "unrecovered_conflict_mass": unrecovered_conflict_mass,
            "pair_transport_deficit": pair_transport_deficit,
            "pair_severe_prob": pair_severe_prob,
            "priority_support": priority_support,
            "candidate_mean_deficit": candidate_mean_deficit,
            "candidate_tail_deficit": candidate_tail_deficit,
            "candidate_severe_prob": candidate_severe_prob,
            "candidate_transport_risk": candidate_transport_risk,
            "candidate_transport_uncertainty": candidate_transport_uncertainty,
        }
