from __future__ import annotations

import torch
from torch import nn


class NaturalDecoder(nn.Module):
    """Multi-branch natural alternative decoder.

    v13 adds a temporally structured, kinematic-residual decoder.  The former
    implementation used one linear layer to emit all ``M x T x 7`` values from a
    single agent token; mode embeddings were not connected to the trajectory
    head.  That made every time step compete through one very large projection
    and gave the model no useful motion prior.  The new decoder starts from a
    constant-velocity baseline and predicts bounded, cumulative residuals from
    explicit mode and time embeddings.  ``legacy_linear`` remains available for
    checkpoint ablations.
    """

    def __init__(
        self,
        d_model: int = 128,
        modes: int = 24,
        future_steps: int = 80,
        source_count: int = 4,
        decoder_type: str = "temporal_kinematic",
    ):
        super().__init__()
        self.modes = int(modes)
        self.future_steps = int(future_steps)
        self.source_count = int(source_count)
        self.decoder_type = str(decoder_type).lower()

        self.shared = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.LayerNorm(d_model))
        # Kept under the old name so legacy checkpoints can still be loaded and
        # explicitly evaluated with model.natural_decoder_type=legacy_linear.
        self.head = nn.Linear(d_model, self.modes * self.future_steps * 7)
        self.logit = nn.Linear(d_model, self.modes)
        self.source_logit = nn.Linear(d_model, self.modes * self.source_count)
        self.priority_logit = nn.Linear(d_model, self.modes)

        self.mode_embedding = nn.Parameter(torch.randn(self.modes, d_model) * 0.02)
        self.time_embedding = nn.Parameter(torch.randn(self.future_steps, d_model) * 0.02)
        self.mode_norm = nn.LayerNorm(d_model)
        self.temporal_norm = nn.LayerNorm(d_model)
        self.temporal_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 7),
        )
        # A newly initialized v13 decoder exactly reproduces the CV baseline.
        # This is much safer than starting tens of metres away in global space.
        nn.init.zeros_(self.temporal_head[-1].weight)
        nn.init.zeros_(self.temporal_head[-1].bias)

    def _temporal_kinematic_offsets(
        self,
        mode_latent: torch.Tensor,
        anchor7: torch.Tensor | None,
        dt: float,
    ) -> torch.Tensor:
        B, A, M, _ = mode_latent.shape
        time = self.time_embedding[None, None, None, :, :]
        h_mt = self.temporal_norm(mode_latent[:, :, :, None, :] + time)
        raw = self.temporal_head(h_mt)

        # Bounded per-step residuals give smooth trajectories while retaining a
        # wide 8 s reachable set.  Position and yaw are integrated; velocity is
        # predicted as a bounded offset; dimensions remain nearly constant.
        step_xy = torch.tanh(raw[..., 0:2]) * 0.45
        pos_res = torch.cumsum(step_xy, dim=3)
        yaw_res = torch.cumsum(torch.tanh(raw[..., 2:3]) * 0.035, dim=3)
        vel_res = torch.tanh(raw[..., 3:5]) * 8.0
        size_res = torch.tanh(raw[..., 5:7]) * 0.35
        residual = torch.cat([pos_res, yaw_res, vel_res, size_res], dim=-1)

        if anchor7 is None:
            return residual
        t = (
            torch.arange(1, self.future_steps + 1, device=mode_latent.device, dtype=mode_latent.dtype)
            * float(dt)
        )
        baseline = torch.zeros(B, A, 1, self.future_steps, 7, device=mode_latent.device, dtype=mode_latent.dtype)
        baseline[..., 0] = anchor7[:, :, None, None, 3] * t[None, None, None, :]
        baseline[..., 1] = anchor7[:, :, None, None, 4] * t[None, None, None, :]
        # Heading, velocity and box dimensions are offsets from the current-state
        # anchor and therefore remain zero in the CV baseline.
        return baseline + residual

    def forward(
        self,
        z_agent: torch.Tensor,
        critical_indices: torch.Tensor,
        *,
        decode_traj: bool = True,
        anchor7: torch.Tensor | None = None,
        dt: float = 0.1,
    ) -> dict[str, torch.Tensor]:
        B, A = critical_indices.shape
        idx = critical_indices.clamp(0, max(z_agent.shape[1] - 1, 0)).long().unsqueeze(-1).expand(B, A, z_agent.shape[-1])
        z = torch.gather(z_agent, 1, idx)
        h = self.shared(z)
        logits = self.logit(h)
        source_logits = self.source_logit(h).reshape(B, A, self.modes, self.source_count)
        priority_logits = self.priority_logit(h)
        mode_latent = self.mode_norm(h[:, :, None, :] + self.mode_embedding[None, None, :, :])
        out = {
            "latent": h,
            "mode_latent": mode_latent,
            "logits": logits,
            "source_logits": source_logits,
            "priority_logits": priority_logits,
        }
        if decode_traj:
            if self.decoder_type in {"legacy", "legacy_linear", "linear"}:
                out["traj"] = self.head(h).reshape(B, A, self.modes, self.future_steps, 7)
            else:
                out["traj"] = self._temporal_kinematic_offsets(mode_latent, anchor7, dt)
        return out
