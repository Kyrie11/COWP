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
        self.natural_decoder = NaturalDecoder(d_model=d_model, modes=int(m.get("max_natural_alternatives", 24)), future_steps=int(m.get("future_steps", 80)))
        self.response_decoder = ResponseDecoder(d_model=d_model, responses=int(m.get("max_safe_responses", 32)), future_steps=int(m.get("future_steps", 80)))
        self.witness_decoder = WitnessDecoder(d_model=d_model, token_count=int(m.get("token_count", 7)))
        self.planner = PlannerHead(d_model=d_model)
        self.max_agents = int(m.get("max_agents", 128))
        self.history_steps = int(m.get("history_steps", 11))
        self.d_state = int(m.get("d_state", 11))

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

        if "cowp/natural/traj" not in batch:
            available = ", ".join(sorted(batch.keys())[:40])
            raise KeyError(
                "COWPModel requires encoder state tensors. Expected state/history, "
                "womd/state/history, state/all, womd/state/all, or WOMD "
                "state/{past,current}/x. The current batch also lacks "
                f"cowp/natural/traj for the legacy toy fallback. Available keys: {available}"
            )

        # Last-resort toy/label-only fallback. Production training should use
        # tensor_cache with WOMD state features; dataset validation normally
        # prevents this path.
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

        anchor7 = None
        if need_natural or need_response:
            anchor7 = self._critical_anchor7(agent_history, critical_idx)
        if need_natural:
            assert enc_scene is not None and anchor7 is not None
            out["natural"] = self._add_natural_anchor(self.natural_decoder(enc_scene["z_agent"], critical_idx), anchor7)
        if need_response:
            assert z_cand is not None and enc_cond is not None and anchor7 is not None
            out["response"] = self._add_response_anchor(
                self.response_decoder(
                    enc_cond["z_agent"],
                    z_cand,
                    enc_cond["z_graph"],
                    critical_idx,
                    decode_traj=decode_response_traj,
                ),
                anchor7,
            )
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
