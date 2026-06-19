from __future__ import annotations

import torch
from torch import nn

from cowp.models.candidate_encoder import CandidateEncoder
from cowp.models.graph_encoder import GraphEncoder
from cowp.models.natural_decoder import NaturalDecoder
from cowp.models.planner_head import PlannerHead
from cowp.models.response_decoder import ResponseDecoder
from cowp.models.witness_decoder import WitnessDecoder
from cowp.data.womd_features import build_agent_history_from_womd, has_womd_state


class COWPModel(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
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
        self.natural_decoder = NaturalDecoder(d_model=d_model, modes=int(m.get("max_natural_alternatives", 24)), future_steps=int(m.get("future_steps", 80)))
        self.response_decoder = ResponseDecoder(d_model=d_model, responses=int(m.get("max_safe_responses", 32)), future_steps=int(m.get("future_steps", 80)))
        self.witness_decoder = WitnessDecoder(d_model=d_model, token_count=int(m.get("token_count", 7)))
        self.planner = PlannerHead(d_model=d_model)
        self.max_agents = int(m.get("max_agents", 128))
        self.history_steps = int(m.get("history_steps", 11))
        self.d_state = int(m.get("d_state", 11))

    def _agent_history_from_batch(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        # Prefer real WOMD tf.Example tensors from tensor_cache.  Earlier versions
        # only checked state/history and state/all, so merged tensor caches silently
        # fell back to label-only natural trajectories.
        if "state/history" in batch:
            agent_history = batch["state/history"].float()
            agent_mask = batch.get("state/agent_valid", torch.ones(agent_history.shape[:2], device=agent_history.device, dtype=torch.bool)).bool()
        elif has_womd_state(batch):
            agent_history, agent_mask = build_agent_history_from_womd(
                batch,
                max_agents=self.max_agents,
                history_steps=self.history_steps,
                d_state=self.d_state,
            )
        elif "state/all" in batch:
            all_state = batch["state/all"].float()
            agent_history = all_state[:, :, : min(11, all_state.shape[2]), :11] if all_state.ndim == 4 else all_state[:, :, None, :11]
            agent_mask = batch.get("state/agent_valid", torch.ones(agent_history.shape[:2], device=agent_history.device, dtype=torch.bool)).bool()
        else:
            # Last-resort toy/label-only fallback.  This path is intentionally kept
            # for unit tests and diagnostics, but production training should use
            # tensor_cache with WOMD state features.
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

    def forward(self, batch: dict[str, torch.Tensor], stage: str | None = None) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        stage = stage or "all"
        agent_history, agent_mask = self._agent_history_from_batch(batch)
        conflict = batch.get("map/conflict_regions")
        conflict_mask = batch.get("map/conflict_region_valid")

        # Decode only the heads needed by the current stage.  Natural alternatives
        # should be conditioned on the root scene, not on a particular ego
        # candidate.  Response/witness/planner heads use the candidate-conditioned
        # graph.  This also avoids loading/encoding candidate tensors in Stage A.
        need_natural = stage in ("natural", "representation", "all")
        need_response = stage in ("response", "all")
        need_witness = stage in ("witness", "planner", "all")
        need_planner = stage in ("planner", "all")
        need_candidate_context = need_response or need_witness or need_planner

        raw_critical_idx = batch.get("cowp/critical/input_index", batch["cowp/critical/track_index"]).long()
        raw_critical_mask = batch.get("cowp/critical/valid")
        raw_critical_mask = raw_critical_mask.bool() if raw_critical_mask is not None else torch.ones_like(raw_critical_idx, dtype=torch.bool)
        critical_idx, critical_mask = self._safe_critical_indices(raw_critical_idx, raw_critical_mask, agent_mask)

        enc_scene = None
        enc_cond = None
        cand_traj = None
        cand_mask = None
        z_cand = None

        if need_natural:
            enc_scene = self.graph(
                agent_history,
                agent_mask,
                None,
                None,
                conflict.float() if conflict is not None else None,
                conflict_mask.bool() if conflict_mask is not None else None,
            )

        if need_candidate_context:
            cand_traj = batch["cowp/candidates/trajectory"].float()
            cand_mask = batch["cowp/candidates/valid"].bool()
            enc_cond = self.graph(
                agent_history,
                agent_mask,
                cand_traj,
                cand_mask,
                conflict.float() if conflict is not None else None,
                conflict_mask.bool() if conflict_mask is not None else None,
            )
        enc = enc_cond if enc_cond is not None else enc_scene
        assert enc is not None

        out: dict[str, torch.Tensor | dict[str, torch.Tensor]] = {
            "enc": enc,
            "critical_idx": critical_idx,
            "critical_mask": critical_mask,
        }
        if need_candidate_context:
            assert cand_traj is not None and cand_mask is not None and enc_cond is not None
            z_cand = self.candidate_encoder(cand_traj, batch["cowp/candidates/macro_type"].long())
            if "z_candidate_context" in enc_cond:
                z_cand = z_cand + enc_cond["z_candidate_context"]
            out["z_candidate"] = z_cand

        if need_natural:
            assert enc_scene is not None
            out["natural"] = self.natural_decoder(enc_scene["z_agent"], critical_idx)
        if need_response:
            assert z_cand is not None and enc_cond is not None
            out["response"] = self.response_decoder(enc_cond["z_agent"], z_cand, enc_cond["z_graph"], critical_idx)
        if need_witness:
            assert z_cand is not None and enc_cond is not None
            witness = self.witness_decoder(enc_cond["z_agent"], z_cand, enc_cond["z_graph"], critical_idx)
            out["witness"] = witness
        if need_planner:
            assert z_cand is not None and cand_mask is not None
            witness = out.get("witness")
            assert isinstance(witness, dict)
            witness_prob = torch.sigmoid(witness["exist_logits"])
            out["planner_score"] = self.planner(
                z_cand,
                batch.get("cowp/candidates/ego_utility_prior", torch.zeros_like(cand_mask, dtype=torch.float32)).float(),
                witness_prob,
                witness["opr"],
                batch.get("cowp/candidates/conventional_safe"),
                critical_mask=critical_mask,
            )
        return out
