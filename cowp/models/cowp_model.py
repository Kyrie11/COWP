from __future__ import annotations

import torch
from torch import nn

from cowp.models.candidate_encoder import CandidateEncoder
from cowp.models.graph_encoder import GraphEncoder
from cowp.models.natural_decoder import NaturalDecoder
from cowp.models.planner_head import PlannerHead
from cowp.models.priority_head import PriorityClaimHead
from cowp.models.outcome_head import OutcomeRiskHead
from cowp.models.response_decoder import ResponseDecoder
from cowp.models.witness_decoder import WitnessDecoder
from cowp.models.set_transport_head import SetTransportCertificateHead
from cowp.data.womd_features import build_agent_history_from_womd, has_womd_state
from cowp.models.coordinate import ego_centric_inputs, infer_sdc_index


class COWPModel(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        # Keep the merged configuration on the module.  The planner forward pass
        # reads planning.evidential_probability_mix when combining witness-logit
        # probability with the evidential witness probability.  Earlier stages did
        # not touch this field, so the missing attribute only surfaced when
        # stage=planner was trained/evaluated.
        self.cfg = cfg
        m = cfg.get("model", cfg)
        d_model = int(m.get("d_model", 128))
        ab = cfg.get("ablation", {})
        self.graph = GraphEncoder(
            int(m.get("d_state", 11)),
            d_model,
            int(m.get("num_heads", 4)),
            int(m.get("num_layers", 3)),
            float(m.get("dropout", 0.1)),
            use_typed_edges=bool(ab.get("use_typed_edges", True)),
            use_dual_edge=bool(ab.get("use_dual_edge", True)),
            use_conflict_query=bool(ab.get("use_conflict_query", True)),
            edge_distance_scale_m=float(m.get("edge_distance_scale_m", 12.0)),
        )
        self.candidate_encoder = CandidateEncoder(d_model=d_model, dropout=float(m.get("dropout", 0.1)))
        self.natural_decoder = NaturalDecoder(
            d_model=d_model,
            modes=int(m.get("max_natural_alternatives", 24)),
            future_steps=int(m.get("future_steps", 80)),
            decoder_type=str(m.get("natural_decoder_type", "temporal_kinematic")),
            obs_capacity_scale=float(m.get("natural_obs_capacity_scale", 1.0)),
        )
        self.response_decoder = ResponseDecoder(
            d_model=d_model,
            responses=int(m.get("max_safe_responses", 32)),
            future_steps=int(m.get("future_steps", 80)),
            natural_modes=int(m.get("max_natural_alternatives", 24)),
        )
        self.witness_decoder = WitnessDecoder(d_model=d_model, token_count=int(m.get("token_count", 7)))
        self.set_transport = SetTransportCertificateHead(
            d_model=d_model,
            hidden=int(m.get("set_transport_hidden", 64)),
            source_count=4,
            geometry_steps=int(m.get("set_transport_geometry_steps", 16)),
            response_topk=int(m.get("set_transport_response_topk", 8)),
        )
        self.planner = PlannerHead(d_model=d_model)
        # Candidate-level calibrated non-coercive feasibility certificate.
        # The pairwise witness decoder remains the explanation/localization module,
        # but closed-loop selection should not depend solely on a max over noisy
        # pair certificates.  This head learns candidate-level NCF / false-safe
        # probabilities directly from the same candidate embedding and aggregate
        # witness features.
        self.candidate_certificate = nn.Sequential(
            nn.Linear(d_model + 10, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 3),
        )
        # v11-BCOT calibrates only monotone transport statistics.  The legacy
        # candidate head is retained for checkpoint compatibility and ablation,
        # but the main certificate no longer receives a generic candidate latent
        # that could solve false-safe classification while ignoring option
        # transport.
        self.transport_candidate_calibrator = nn.Sequential(
            nn.Linear(11, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Linear(64, 3),
            nn.Tanh(),
        )
        # Preserve the analytic BCOT certificate at v10 checkpoint load; the
        # residual starts exactly at zero and is learned only as calibration.
        nn.init.zeros_(self.transport_candidate_calibrator[3].weight)
        nn.init.zeros_(self.transport_candidate_calibrator[3].bias)
        self.priority_claim = PriorityClaimHead(d_model=d_model, dropout=float(m.get("dropout", 0.1)))
        self.outcome_risk = OutcomeRiskHead(d_model=d_model, dropout=float(m.get("dropout", 0.1)))
        self.max_agents = int(m.get("max_agents", 128))
        self.history_steps = int(m.get("history_steps", 11))
        self.d_state = int(m.get("d_state", 11))
        # Production must never synthesize encoder inputs from future trajectory
        # labels.  The legacy path is retained only for explicitly opted-in toy
        # tests and migration utilities.
        self.allow_label_only_state_fallback = bool(m.get("allow_label_only_state_fallback", False))
        self.require_explicit_sdc_index = bool(m.get("require_explicit_sdc_index", False))

    @staticmethod
    def _first_tensor(batch: dict[str, torch.Tensor], names: tuple[str, ...]) -> torch.Tensor | None:
        for name in names:
            value = batch.get(name)
            if value is not None and torch.is_tensor(value):
                return value
        return None

    def _critical_anchor7(self, agent_history: torch.Tensor, critical_idx: torch.Tensor) -> torch.Tensor:
        """Return [B,A,7] current-state anchor for critical-agent trajectory heads.

        Label generation stores absolute trajectories [x,y,heading,vx,vy,length,width].
        Predicting those absolute coordinates directly from an unconstrained linear
        head causes very large initial Stage-A losses in WOMD global coordinates.
        The model instead learns residual futures around each critical agent's
        current state, preserving the paper's absolute label/loss semantics.
        """
        if agent_history.ndim == 4:
            cur = agent_history[:, :, -1, :].float()
        elif agent_history.ndim == 3:
            cur = agent_history.float()
        else:
            raise ValueError(f"Cannot build critical anchors from agent_history shape {tuple(agent_history.shape)}")
        B, A = critical_idx.shape
        n_agent = cur.shape[1]
        idx = critical_idx.clamp(0, max(n_agent - 1, 0)).long().unsqueeze(-1).expand(B, A, cur.shape[-1])
        c = torch.gather(cur, 1, idx)
        anchor = torch.zeros(B, A, 7, device=cur.device, dtype=cur.dtype)
        if c.shape[-1] >= 2:
            anchor[..., 0:2] = c[..., 0:2]
        if c.shape[-1] >= 7:
            anchor[..., 2] = c[..., 6]
        if c.shape[-1] >= 9:
            anchor[..., 3:5] = c[..., 7:9]
        if c.shape[-1] >= 5:
            anchor[..., 5] = c[..., 3].clamp_min(0.1)
            anchor[..., 6] = c[..., 4].clamp_min(0.1)
        return anchor

    @staticmethod
    def _add_natural_anchor(pred: dict[str, torch.Tensor], anchor7: torch.Tensor) -> dict[str, torch.Tensor]:
        out = dict(pred)
        out["traj"] = pred["traj"] + anchor7[:, :, None, None, :]
        if "base_traj" in pred:
            out["base_traj"] = pred["base_traj"] + anchor7[:, :, None, None, :]
        return out

    @staticmethod
    def _add_response_anchor(pred: dict[str, torch.Tensor], anchor7: torch.Tensor) -> dict[str, torch.Tensor]:
        out = dict(pred)
        if "traj" in pred:
            out["traj"] = pred["traj"] + anchor7[:, None, :, None, None, :]
        return out

    def _agent_history_from_batch(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        # Prefer real WOMD tf.Example tensors from tensor_cache. Earlier versions
        # only checked state/history and state/all, so merged tensor caches silently
        # fell back to label-only natural trajectories.
        hist = self._first_tensor(batch, ("state/history", "womd/state/history"))
        if hist is not None:
            agent_history = hist.float()
            agent_valid = self._first_tensor(batch, ("state/agent_valid", "womd/state/agent_valid", "state/current/valid", "womd/state/current/valid"))
            agent_mask = agent_valid.bool() if agent_valid is not None and agent_valid.shape[:2] == agent_history.shape[:2] else torch.ones(agent_history.shape[:2], device=agent_history.device, dtype=torch.bool)
            return agent_history, agent_mask

        if has_womd_state(batch):
            return build_agent_history_from_womd(
                batch,
                max_agents=self.max_agents,
                history_steps=self.history_steps,
                d_state=self.d_state,
            )

        all_state = self._first_tensor(batch, ("state/all", "womd/state/all"))
        if all_state is not None:
            all_state = all_state.float()
            agent_history = all_state[:, :, : min(11, all_state.shape[2]), :11] if all_state.ndim == 4 else all_state[:, :, None, :11]
            agent_valid = self._first_tensor(batch, ("state/agent_valid", "womd/state/agent_valid"))
            agent_mask = agent_valid.bool() if agent_valid is not None and agent_valid.shape[:2] == agent_history.shape[:2] else torch.ones(agent_history.shape[:2], device=agent_history.device, dtype=torch.bool)
            return agent_history, agent_mask

        available = ", ".join(sorted(batch.keys())[:40])
        if not self.allow_label_only_state_fallback:
            raise RuntimeError(
                "Causal input violation: COWPModel received no real history/current "
                "state tensor. Refusing to reconstruct encoder state from future "
                "cowp/natural/traj labels. Rebuild/repair the tensor cache instead. "
                f"Available keys: {available}"
            )
        if "cowp/natural/traj" not in batch:
            raise KeyError(
                "Legacy label-only fallback was explicitly enabled, but "
                f"cowp/natural/traj is absent. Available keys: {available}"
            )

        # Explicitly opted-in legacy toy fallback. Never enable this in a train or
        # evaluation config used for reported results.
        nat = batch["cowp/natural/traj"].float()
        B, A = nat.shape[:2]
        max_idx = int(batch["cowp/critical/track_index"].max().item() + 1) if batch["cowp/critical/track_index"].numel() else A
        N = max(max_idx, A, 1)
        agent_history = torch.zeros(B, N, 1, self.d_state, device=nat.device)
        agent_mask = torch.zeros(B, N, device=nat.device, dtype=torch.bool)
        for a in range(A):
            idx = batch["cowp/critical/track_index"][:, a].clamp(0, N - 1).long()
            vals = nat[:, a, 0, 0]  # [x,y,heading,vx,vy,length,width]
            for b in range(B):
                agent_history[b, idx[b], 0, 0] = vals[b, 0]
                agent_history[b, idx[b], 0, 1] = vals[b, 1]
                agent_history[b, idx[b], 0, 3] = vals[b, 5].clamp_min(0.1)
                agent_history[b, idx[b], 0, 4] = vals[b, 6].clamp_min(0.1)
                agent_history[b, idx[b], 0, 5] = 1.5
                agent_history[b, idx[b], 0, 6] = vals[b, 2]
                agent_history[b, idx[b], 0, 7] = vals[b, 3]
                agent_history[b, idx[b], 0, 8] = vals[b, 4]
                agent_history[b, idx[b], 0, 9] = torch.linalg.norm(vals[b, 3:5])
                agent_history[b, idx[b], 0, 10] = 1.0
                agent_mask[b, idx[b]] = True
        agent_mask[:, 0] = True
        return agent_history, agent_mask

    @staticmethod
    def _safe_critical_indices(
        critical_idx: torch.Tensor,
        critical_mask: torch.Tensor,
        agent_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Clip gather indices and mask invisible critical-agent slots.

        Scenario proto labels store original track indices.  WOMD tf.Example model
        input is padded to a fixed agent dimension.  If a selected critical track
        lies outside the model-visible tensor, the slot must be ignored by all
        losses instead of being clamped and supervised as another agent.
        """
        n_agent = int(agent_mask.shape[1])
        in_range = (critical_idx >= 0) & (critical_idx < n_agent)
        safe_idx = critical_idx.clamp(0, max(n_agent - 1, 0)).long()
        visible = torch.gather(agent_mask.bool(), 1, safe_idx) if n_agent > 0 else torch.zeros_like(safe_idx, dtype=torch.bool)
        safe_mask = critical_mask.bool() & in_range & visible
        return safe_idx, safe_mask

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        stage: str | None = None,
        *,
        decode_response_traj: bool = True,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        stage = stage or "all"
        agent_history, agent_mask = self._agent_history_from_batch(batch)
        conflict = batch.get("map/conflict_regions")
        conflict_mask = batch.get("map/conflict_region_valid")
        sdc_index, ego_mask = infer_sdc_index(
            batch, agent_history, require_explicit=self.require_explicit_sdc_index
        )

        # Decode only the heads needed by the current stage.  Natural alternatives
        # should be conditioned on the root scene, not on a particular ego
        # candidate.  Response/witness/planner heads use the candidate-conditioned
        # graph.  This also avoids loading/encoding candidate tensors in Stage A.
        # Witness/planner stages need the root-scene natural latent.  Only the
        # dedicated natural/all stages decode the dense 24x80 trajectory bank.
        need_natural = stage in ("natural", "representation", "witness", "planner", "all")
        # Explicit mode-level transport targets are indexed by the unordered GT
        # natural set.  Decode the much smaller natural trajectory bank in
        # witness/planner stages so losses can perform permutation-invariant
        # nearest-mode alignment.  Dense response trajectories remain disabled.
        decode_natural_traj = stage in ("natural", "representation", "witness", "planner", "all")
        # Planner/witness stages need the compact response bank (classification and
        # burden heads) even when dense response trajectories are disabled.
        need_response = stage in ("response", "witness", "planner", "all")
        need_witness = stage in ("witness", "planner", "all")
        need_planner = stage in ("planner", "all")
        need_candidate_context = need_response or need_witness or need_planner

        raw_critical_idx = batch.get("cowp/critical/input_index", batch["cowp/critical/track_index"]).long()
        raw_critical_mask = batch.get("cowp/critical/valid")
        raw_critical_mask = raw_critical_mask.bool() if raw_critical_mask is not None else torch.ones_like(raw_critical_idx, dtype=torch.bool)
        critical_idx, critical_mask = self._safe_critical_indices(raw_critical_idx, raw_critical_mask, agent_mask)

        enc_scene = None
        enc_cond = None
        cand_traj = batch["cowp/candidates/trajectory"].float() if need_candidate_context else None
        cand_mask = batch["cowp/candidates/valid"].bool() if need_candidate_context else None
        z_cand = None

        enc_history, enc_candidates, enc_conflict = ego_centric_inputs(
            agent_history,
            cand_traj,
            conflict.float() if conflict is not None else None,
            sdc_index,
        )

        if need_natural:
            enc_scene = self.graph(
                enc_history,
                agent_mask,
                None,
                None,
                enc_conflict,
                conflict_mask.bool() if conflict_mask is not None else None,
                ego_mask=ego_mask,
            )

        if need_candidate_context:
            assert cand_traj is not None and cand_mask is not None and enc_candidates is not None
            enc_cond = self.graph(
                enc_history,
                agent_mask,
                enc_candidates,
                cand_mask,
                enc_conflict,
                conflict_mask.bool() if conflict_mask is not None else None,
                ego_mask=ego_mask,
            )
        enc = enc_cond if enc_cond is not None else enc_scene
        assert enc is not None

        out: dict[str, torch.Tensor | dict[str, torch.Tensor]] = {
            "enc": enc,
            "critical_idx": critical_idx,
            "critical_mask": critical_mask,
            "sdc_index": sdc_index,
        }
        if need_candidate_context:
            assert cand_traj is not None and cand_mask is not None and enc_cond is not None
            assert enc_candidates is not None
            z_cand = self.candidate_encoder(enc_candidates, batch["cowp/candidates/macro_type"].long())
            if "z_candidate_context" in enc_cond:
                z_cand = z_cand + enc_cond["z_candidate_context"]
            out["z_candidate"] = z_cand

        anchor7 = None
        if decode_natural_traj or need_response:
            anchor7 = self._critical_anchor7(agent_history, critical_idx)
        natural_out = None
        if need_natural:
            assert enc_scene is not None
            natural_out = self.natural_decoder(
                enc_scene["z_agent"],
                critical_idx,
                decode_traj=decode_natural_traj,
                anchor7=anchor7,
                dt=float(self.cfg.get("time", {}).get("dt", 0.1)),
            )
            if decode_natural_traj:
                assert anchor7 is not None
                natural_out = self._add_natural_anchor(natural_out, anchor7)
            out["natural"] = natural_out
        if need_response:
            assert z_cand is not None and enc_cond is not None and anchor7 is not None
            out["response"] = self._add_response_anchor(
                self.response_decoder(
                    enc_cond["z_agent"],
                    z_cand,
                    enc_cond["z_graph"],
                    critical_idx,
                    # Witness/planner stages also optimize response auxiliary
                    # losses.  v11 accepted --response-traj-weight for those
                    # stages but silently skipped the trajectory head, so enabling
                    # the loss would fail when response_loss accessed pred["traj"].
                    decode_traj=bool(
                        decode_response_traj
                        and stage in ("response", "witness", "planner", "all")
                    ),
                ),
                anchor7,
            )
        if need_witness:
            assert z_cand is not None and enc_cond is not None
            natural_latent = natural_out.get("latent") if isinstance(natural_out, dict) else None
            witness = self.witness_decoder(
                enc_cond["z_agent"], z_cand, enc_cond["z_graph"], critical_idx,
                natural_latent=natural_latent,
            )
            out["witness_proxy"] = witness
            response_out = out.get("response")
            if isinstance(natural_out, dict) and isinstance(response_out, dict) and "mode_latent" in natural_out:
                pcfg = self.cfg.get("planning", {})
                beta = batch.get("cowp/natural/beta")
                if beta is None:
                    beta = torch.full(
                        (z_cand.shape[0], critical_idx.shape[1]), 0.65,
                        device=z_cand.device, dtype=z_cand.dtype,
                    )
                set_certificate = self.set_transport(
                    z_agent=enc_cond["z_agent"],
                    z_candidate=z_cand,
                    z_graph=enc_cond["z_graph"],
                    critical_indices=critical_idx,
                    natural=natural_out,
                    response=response_out,
                    beta=beta,
                    candidate_traj=cand_traj,
                    natural_traj=natural_out.get("traj"),
                    critical_mask=critical_mask,
                    alpha_opr=float(pcfg.get("alpha_opr_infer", self.cfg.get("ncf", {}).get("alpha_opr", 0.35))),
                    gamma=float(pcfg.get("ncf_gamma_infer", self.cfg.get("ncf", {}).get("gamma", 0.10))),
                    conflict_mass_floor=float(pcfg.get("set_transport_conflict_mass_floor", self.cfg.get("ncf", {}).get("positive_min_natural_conflict_mass", 0.10))),
                    burden_temperature=float(pcfg.get("set_transport_burden_temperature", 0.08)),
                    gate_temperature=float(pcfg.get("set_transport_gate_temperature", 0.06)),
                    calibration_scale=float(pcfg.get("set_transport_calibration_scale", 0.10)),
                    root_mass_scale=float(pcfg.get("set_transport_root_mass_scale", 1.0)),
                    candidate_tail_temperature=float(pcfg.get("candidate_transport_tail_temperature", 0.12)),
                )
                out["set_certificate"] = set_certificate
                # The selector consumes the mechanistic certificate.  The proxy
                # decoder is retained for explanation-token supervision/ablation.
                witness = dict(witness)
                witness["proxy_exist_logits"] = witness["exist_logits"]
                witness["proxy_opr"] = witness["opr"]
                witness["proxy_burden_total"] = witness["burden_total"]
                witness["exist_logits"] = set_certificate["exist_logits"]
                witness["evidential_prob"] = set_certificate["witness_prob"]
                witness["epistemic_uncertainty"] = set_certificate["uncertainty"]
                witness["opr"] = set_certificate["opr"]
                witness["burden_total"] = set_certificate["min_safe_burden"]
                witness["c_i"] = set_certificate["natural_conflict_mass"]
                out["candidate_transport_risk"] = set_certificate["candidate_transport_risk"]
                out["candidate_transport_uncertainty"] = set_certificate["candidate_transport_uncertainty"]
                out["candidate_transport_severe_prob"] = set_certificate["candidate_severe_prob"]
            out["witness"] = witness
        if need_planner:
            assert z_cand is not None and cand_mask is not None
            witness = out.get("witness")
            assert isinstance(witness, dict)
            pcfg = self.cfg.get("planning", {})
            temp = max(float(pcfg.get("witness_temperature", 1.0)), 1e-3)
            bias = float(pcfg.get("witness_logit_bias", 0.0))
            logit_prob = torch.sigmoid((witness["exist_logits"] - bias) / temp)
            evidence_prob = witness.get("evidential_prob")
            source = str(pcfg.get("witness_probability_source", "mixed")).lower()
            if source == "logit" or evidence_prob is None:
                witness_prob = logit_prob
            elif source == "evidential":
                witness_prob = evidence_prob
            else:
                evidence_mix = float(pcfg.get("evidential_probability_mix", 0.5))
                witness_prob = (1.0 - evidence_mix) * logit_prob + evidence_mix * evidence_prob
            # Planner/certificate objectives must not backpropagate through the
            # staged witness decoder.  In v5 the very large candidate-certificate
            # loss reached witness_prob through this path and pushed nearly every
            # pair probability to one, despite planner_witness_scale being small.
            # The witness head still receives its own explicitly scaled witness loss.
            detach_witness = bool(pcfg.get("planner_detach_witness_features", True))
            detach_backbone = bool(pcfg.get("planner_detach_backbone_features", True))
            witness_for_planner = witness_prob.detach() if detach_witness else witness_prob
            opr_for_planner = witness["opr"].detach() if detach_witness else witness["opr"]
            # Strong gradient firewall: candidate certificate, physical outcome,
            # priority, and planner heads may calibrate pretrained representations,
            # but their losses must not rewrite the shared graph/candidate features
            # consumed by the pairwise witness decoder.  v6 detached witness output
            # tensors but still allowed all candidate/planner losses to reach the
            # witness indirectly through z_agent/z_graph/z_cand.
            z_cand_planner = z_cand.detach() if detach_backbone else z_cand
            z_agent_planner = enc_cond["z_agent"].detach() if detach_backbone else enc_cond["z_agent"]
            z_graph_planner = enc_cond["z_graph"].detach() if detach_backbone else enc_cond["z_graph"]
            out["priority_claim_logits"] = self.priority_claim(
                z_agent_planner,
                z_cand_planner,
                critical_idx,
                witness_for_planner,
                opr_for_planner,
            )
            out["outcome"] = self.outcome_risk(z_cand_planner)
            ego_utility = batch.get("cowp/candidates/ego_utility_prior", torch.zeros_like(cand_mask, dtype=torch.float32)).float()
            conventional_safe = batch.get("cowp/candidates/conventional_safe")
            if witness_prob.ndim == 3:
                if critical_mask is not None:
                    cm = critical_mask.bool()[:, None, :]
                else:
                    cm = torch.ones_like(witness_prob, dtype=torch.bool)
                wp_aux = torch.where(cm, witness_for_planner, torch.zeros_like(witness_for_planner))
                opr_aux = torch.where(cm, opr_for_planner, torch.ones_like(opr_for_planner))
                uncertainty = witness.get("epistemic_uncertainty")
                if uncertainty is None:
                    uncertainty = torch.zeros_like(witness_for_planner)
                uncertainty = uncertainty.detach() if detach_witness else uncertainty
                uncertainty = torch.where(cm, uncertainty, torch.zeros_like(uncertainty))
                burden = witness.get("burden_total", torch.zeros_like(witness_for_planner)).float()
                ci = witness.get("c_i", torch.zeros_like(witness_for_planner)).float()
                if detach_witness:
                    burden = burden.detach()
                    ci = ci.detach()
                beta = batch.get("cowp/natural/beta")
                if beta is None:
                    beta_pair = torch.full_like(witness_for_planner, 0.65)
                else:
                    beta_pair = beta.float()[:, None, :].expand_as(witness_prob)
                burden_excess = torch.where(cm, torch.relu(burden - beta_pair), torch.zeros_like(burden))
                ci_excess = torch.where(cm, torch.relu(ci), torch.zeros_like(ci))
                alpha = float(self.cfg.get("planning", {}).get("alpha_opr_infer", self.cfg.get("ncf", {}).get("alpha_opr", 0.35)))
                option_collapse = torch.where(
                    cm,
                    (torch.relu(torch.as_tensor(alpha, device=opr_aux.device, dtype=opr_aux.dtype) - opr_aux) / max(alpha, 1e-6)).clamp(0.0, 1.0),
                    torch.zeros_like(opr_aux),
                )
                denom = cm.float().sum(dim=-1).clamp_min(1.0)
                max_wit = wp_aux.max(dim=-1).values
                mean_wit = wp_aux.sum(dim=-1) / denom
                min_opr = opr_aux.min(dim=-1).values
                mean_opr = (torch.where(cm, opr_aux, torch.zeros_like(opr_aux)).sum(dim=-1) / denom).clamp(0.0, 1.0)
                max_burden_excess = burden_excess.max(dim=-1).values.clamp(0.0, 2.0) / 2.0
                max_ci_excess = ci_excess.max(dim=-1).values.clamp(0.0, 2.0) / 2.0
                collapse_fraction = option_collapse.sum(dim=-1) / denom
                max_uncertainty = uncertainty.max(dim=-1).values.clamp(0.0, 1.0)
                set_certificate = out.get("set_certificate")
                if isinstance(set_certificate, dict) and "candidate_transport_risk" in set_certificate:
                    structured_risk = set_certificate["candidate_transport_risk"].float()
                    mean_deficit = set_certificate["candidate_mean_deficit"].float()
                    tail_deficit = set_certificate["candidate_tail_deficit"].float()
                    severe_prob = set_certificate["candidate_severe_prob"].float()
                    transport_uncertainty = set_certificate["candidate_transport_uncertainty"].float()
                    unrecovered = set_certificate["unrecovered_conflict_mass"].float()
                    max_unrecovered = torch.where(cm, unrecovered, torch.zeros_like(unrecovered)).max(dim=-1).values
                else:
                    pair_set_risk = (
                        0.50 * wp_aux
                        + 0.25 * option_collapse
                        + 0.15 * burden_excess.clamp(0.0, 1.0)
                        + 0.10 * ci_excess.clamp(0.0, 1.0)
                    ).clamp(0.0, 1.0)
                    structured_risk = torch.where(cm, pair_set_risk, torch.zeros_like(pair_set_risk)).max(dim=-1).values
                    mean_deficit = structured_risk
                    tail_deficit = structured_risk
                    severe_prob = structured_risk
                    transport_uncertainty = max_uncertainty
                    max_unrecovered = max_ci_excess
            else:
                max_wit = witness_for_planner
                mean_wit = witness_for_planner
                min_opr = opr_for_planner
                mean_opr = opr_for_planner
                max_burden_excess = torch.zeros_like(max_wit)
                max_ci_excess = torch.zeros_like(max_wit)
                collapse_fraction = torch.relu(0.35 - min_opr).clamp(0.0, 1.0)
                max_uncertainty = torch.zeros_like(max_wit)
                structured_risk = (0.65 * max_wit + 0.35 * collapse_fraction).clamp(0.0, 1.0)
                mean_deficit = structured_risk
                tail_deficit = structured_risk
                severe_prob = structured_risk
                transport_uncertainty = max_uncertainty
                max_unrecovered = max_ci_excess
            safe_aux = torch.ones_like(max_wit) if conventional_safe is None else conventional_safe.float()
            cert_aux = torch.stack([
                ego_utility.float(), safe_aux,
                max_wit.float(), mean_wit.float(), min_opr.float(), mean_opr.float(),
                max_burden_excess.float(), max_ci_excess.float(),
                collapse_fraction.float(), max_uncertainty.float(),
            ], dim=-1)
            crit_fraction = (
                critical_mask.float().sum(dim=-1, keepdim=True) / max(float(critical_mask.shape[-1]), 1.0)
                if critical_mask is not None
                else torch.ones_like(structured_risk[:, :1])
            ).expand_as(structured_risk)
            transport_features = torch.stack([
                structured_risk.float(), mean_deficit.float(), tail_deficit.float(),
                severe_prob.float(), transport_uncertainty.float(),
                max_wit.float(), min_opr.float(), max_burden_excess.float(),
                max_ci_excess.float(), max_unrecovered.float(), crit_fraction.float(),
            ], dim=-1)
            cert_residual = self.transport_candidate_calibrator(
                transport_features.detach() if detach_backbone else transport_features
            )
            eps = 1.0e-4
            base_fs = (safe_aux * structured_risk).clamp(eps, 1.0 - eps)
            base_ncf = (safe_aux * (1.0 - structured_risk)).clamp(eps, 1.0 - eps)
            base_fs_logit = torch.logit(base_fs)
            base_ncf_logit = torch.logit(base_ncf)
            structured_weight = float(self.cfg.get("planning", {}).get("candidate_structured_logit_weight", 1.0))
            residual_scale = float(self.cfg.get("planning", {}).get("candidate_transport_residual_scale", 0.20))
            # Mechanism-only calibration is exposed separately.  The main COWP
            # selector consumes candidate_transport_risk directly, while these
            # logits support a transport-calibration ablation without a generic
            # candidate latent.
            out["candidate_transport_ncf_logit"] = structured_weight * base_ncf_logit + residual_scale * cert_residual[..., 0]
            out["candidate_transport_false_safe_logit"] = structured_weight * base_fs_logit + residual_scale * cert_residual[..., 1]
            out["candidate_transport_quality_logit"] = structured_weight * (base_ncf_logit - base_fs_logit) + residual_scale * cert_residual[..., 2]

            # Keep the generic candidate classifier as an explicit diagnostic /
            # candidate-only ablation.  v11 left this module instantiated but
            # never called it, so its parameters were dead and the reported
            # "candidate certificate" was actually another transport-calibrator
            # view.  Separating the outputs makes the causal claim auditable:
            # generic classification may be strong, but the default selector is
            # forbidden from using it.
            generic_logits = self.candidate_certificate(
                torch.cat([z_cand_planner, cert_aux], dim=-1)
            )
            out["candidate_ncf_logit"] = generic_logits[..., 0]
            out["candidate_false_safe_logit"] = generic_logits[..., 1]
            out["candidate_quality_logit"] = generic_logits[..., 2]
            out["candidate_structured_coercion_risk"] = structured_risk
            out["planner_score"] = self.planner(
                z_cand_planner,
                ego_utility,
                witness_for_planner,
                opr_for_planner,
                conventional_safe,
                critical_mask=critical_mask,
            )
        return out
