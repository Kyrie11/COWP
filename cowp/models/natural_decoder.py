from __future__ import annotations

import math

import torch
from torch import nn

from cowp.core.constants import NaturalSource


class NaturalDecoder(nn.Module):
    """Counterfactual natural-option decoder with stable typed root identities.

    ``typed_kinematic_residual`` (v14) assigns every mode a permanent semantic
    source (OBS/NEU/PRIO), initializes it with a distinct kinematic prototype, and
    learns only a bounded residual around that prototype.  This prevents the v13
    failure mode where all 24 roots started as the same constant-velocity curve
    and global nearest-neighbour matching allowed many heterogeneous GT roots to
    collapse onto one predicted mode.

    The decoder emits *offsets* from ``anchor7``.  ``COWPModel`` adds the absolute
    current-state anchor after decoding, preserving existing label/loss semantics.
    ``temporal_kinematic`` and ``legacy_linear`` remain available for ablations.
    """

    def __init__(
        self,
        d_model: int = 128,
        modes: int = 24,
        future_steps: int = 80,
        source_count: int = 4,
        decoder_type: str = "typed_kinematic_residual",
    ):
        super().__init__()
        self.modes = int(modes)
        self.future_steps = int(future_steps)
        self.source_count = int(source_count)
        self.decoder_type = str(decoder_type).lower()

        self.shared = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.LayerNorm(d_model))
        # Kept under the old name for checkpoint-compatible legacy ablations.
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
        self.residual_gate = nn.Linear(d_model, 1)
        nn.init.zeros_(self.temporal_head[-1].weight)
        nn.init.zeros_(self.temporal_head[-1].bias)
        nn.init.zeros_(self.residual_gate.weight)
        nn.init.constant_(self.residual_gate.bias, -2.5)

        mode_source, accel, yaw_rate, speed_offset = self._make_typed_specs(self.modes)
        self.register_buffer("mode_source", mode_source, persistent=True)
        self.register_buffer("prototype_accel", accel, persistent=True)
        self.register_buffer("prototype_yaw_rate", yaw_rate, persistent=True)
        self.register_buffer("prototype_speed_offset", speed_offset, persistent=True)

        source_bias = torch.full((self.modes, self.source_count), -3.0, dtype=torch.float32)
        for m, src in enumerate(mode_source.tolist()):
            if 0 <= src < self.source_count:
                source_bias[m, src] = 3.0
        if self.source_count > int(NaturalSource.PAD):
            source_bias[:, int(NaturalSource.PAD)] = -6.0
        self.register_buffer("typed_source_bias", source_bias, persistent=True)

        priority_bias = torch.zeros(self.modes, dtype=torch.float32)
        priority_bias[mode_source == int(NaturalSource.PRIO)] = 1.5
        self.register_buffer("typed_priority_bias", priority_bias, persistent=True)

        # v15 uses source-specific residual capacity.  OBS must fit real curved
        # motion, while NEU/PRIO should stay close to their causal/rule priors.
        xy_scale = torch.full((self.modes,), 0.12, dtype=torch.float32)
        yaw_scale = torch.full((self.modes,), 0.012, dtype=torch.float32)
        vel_scale = torch.full((self.modes,), 3.0, dtype=torch.float32)
        gate_bias = torch.zeros(self.modes, dtype=torch.float32)
        obs_mask = mode_source == int(NaturalSource.OBS)
        neu_mask = mode_source == int(NaturalSource.NEU)
        prio_mask = mode_source == int(NaturalSource.PRIO)
        xy_scale[obs_mask] = 0.30
        yaw_scale[obs_mask] = 0.025
        vel_scale[obs_mask] = 5.0
        xy_scale[prio_mask] = 0.10
        yaw_scale[prio_mask] = 0.010
        gate_bias[obs_mask] = 1.5
        gate_bias[neu_mask] = 0.0
        gate_bias[prio_mask] = -0.25
        self.register_buffer("residual_xy_step_scale", xy_scale, persistent=True)
        self.register_buffer("residual_yaw_step_scale", yaw_scale, persistent=True)
        self.register_buffer("residual_velocity_scale", vel_scale, persistent=True)
        self.register_buffer("typed_residual_gate_bias", gate_bias, persistent=True)

    @staticmethod
    def _make_typed_specs(modes: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Eight diverse prototypes per semantic source for the default M=24.
        # OBS is broad and turning-capable; NEU is ego-neutral longitudinal
        # continuation; PRIO emphasizes maintained/progressive motion.
        obs = [
            (-2.0, -0.14, 0.0), (-2.0, 0.14, 0.0),
            (-0.5, -0.10, 0.0), (-0.5, 0.10, 0.0),
            (0.5, -0.06, 0.0), (0.5, 0.06, 0.0),
            (1.5, -0.12, 0.0), (1.5, 0.12, 0.0),
        ]
        neu = [
            (-3.0, 0.0, 0.0), (-2.0, 0.0, 0.0), (-1.0, 0.0, 0.0),
            (-0.4, 0.0, 0.0), (0.0, 0.0, 0.0), (0.5, 0.0, 0.0),
            (1.2, 0.0, 0.0), (2.0, 0.0, 0.0),
        ]
        prio = [
            (-1.0, 0.0, 0.0), (-0.3, 0.0, 0.0), (0.0, -0.04, 0.0),
            (0.0, 0.0, 0.8), (0.6, 0.0, 0.0), (1.2, 0.0, 0.0),
            (2.0, 0.0, 0.0), (3.0, 0.03, 0.0),
        ]
        groups = [
            (int(NaturalSource.OBS), obs),
            (int(NaturalSource.NEU), neu),
            (int(NaturalSource.PRIO), prio),
        ]
        specs: list[tuple[int, float, float, float]] = []
        # Interleave complete source groups, then cycle if a non-standard number
        # of modes is requested.  Default 24 preserves 8/8/8 exactly.
        flat = [(src, *p) for src, bank in groups for p in bank]
        while len(specs) < modes:
            specs.extend(flat[: min(len(flat), modes - len(specs))])
        specs = specs[:modes]
        return (
            torch.tensor([s[0] for s in specs], dtype=torch.long),
            torch.tensor([s[1] for s in specs], dtype=torch.float32),
            torch.tensor([s[2] for s in specs], dtype=torch.float32),
            torch.tensor([s[3] for s in specs], dtype=torch.float32),
        )

    def typed_kinematic_basis(self, anchor7: torch.Tensor, dt: float) -> torch.Tensor:
        """Return typed prototype offsets with shape ``[B,A,M,T,7]``."""
        if anchor7.ndim != 3 or anchor7.shape[-1] != 7:
            raise ValueError(f"anchor7 must be [B,A,7], got {tuple(anchor7.shape)}")
        dtype, device = anchor7.dtype, anchor7.device
        t = (torch.arange(1, self.future_steps + 1, device=device, dtype=dtype) * float(dt))
        accel = self.prototype_accel.to(device=device, dtype=dtype)[None, None, :, None]
        yaw_rate = self.prototype_yaw_rate.to(device=device, dtype=dtype)[None, None, :, None]
        speed_offset = self.prototype_speed_offset.to(device=device, dtype=dtype)[None, None, :, None]

        yaw0 = anchor7[..., 2, None, None]
        vx0 = anchor7[..., 3]
        vy0 = anchor7[..., 4]
        speed0 = torch.sqrt(vx0.square() + vy0.square()).clamp_min(0.0)[..., None, None]
        speed = (speed0 + speed_offset + accel * t[None, None, None, :]).clamp_min(0.0)
        yaw = yaw0 + yaw_rate * t[None, None, None, :]
        vx = speed * torch.cos(yaw)
        vy = speed * torch.sin(yaw)

        out = torch.zeros(
            anchor7.shape[0], anchor7.shape[1], self.modes, self.future_steps, 7,
            device=device, dtype=dtype,
        )
        out[..., 0] = torch.cumsum(vx * float(dt), dim=-1)
        out[..., 1] = torch.cumsum(vy * float(dt), dim=-1)
        out[..., 2] = yaw - yaw0
        out[..., 3] = vx - vx0[..., None, None]
        out[..., 4] = vy - vy0[..., None, None]
        # Length/width are absolute anchors outside this decoder, so zero offsets.
        return out

    def _learned_residual(self, mode_latent: torch.Tensor) -> torch.Tensor:
        time = self.time_embedding[None, None, None, :, :]
        h_mt = self.temporal_norm(mode_latent[:, :, :, None, :] + time)
        raw = self.temporal_head(h_mt)
        gate_bias = self.typed_residual_gate_bias.to(mode_latent)[None, None, :, None]
        gate = torch.sigmoid(self.residual_gate(mode_latent) + gate_bias)[..., None, :]

        xy_scale = self.residual_xy_step_scale.to(raw)[None, None, :, None, None]
        yaw_scale = self.residual_yaw_step_scale.to(raw)[None, None, :, None, None]
        vel_scale = self.residual_velocity_scale.to(raw)[None, None, :, None, None]
        step_xy = torch.tanh(raw[..., 0:2]) * xy_scale
        pos_res = torch.cumsum(step_xy, dim=3)
        yaw_res = torch.cumsum(torch.tanh(raw[..., 2:3]) * yaw_scale, dim=3)
        vel_res = torch.tanh(raw[..., 3:5]) * vel_scale
        size_res = torch.tanh(raw[..., 5:7]) * 0.10
        return torch.cat([pos_res, yaw_res, vel_res, size_res], dim=-1) * gate

    def _temporal_kinematic_offsets(
        self,
        mode_latent: torch.Tensor,
        anchor7: torch.Tensor | None,
        dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = self._learned_residual(mode_latent)
        if anchor7 is None:
            base = torch.zeros_like(residual)
        else:
            t = torch.arange(1, self.future_steps + 1, device=mode_latent.device, dtype=mode_latent.dtype) * float(dt)
            base = torch.zeros_like(residual)
            base[..., 0] = anchor7[:, :, None, None, 3] * t[None, None, None, :]
            base[..., 1] = anchor7[:, :, None, None, 4] * t[None, None, None, :]
        return base + residual, base, residual

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
        learned_source_logits = self.source_logit(h).reshape(B, A, self.modes, self.source_count)
        source_logits = 0.10 * learned_source_logits + self.typed_source_bias[None, None].to(learned_source_logits)
        priority_logits = self.priority_logit(h) + self.typed_priority_bias[None, None].to(h)
        mode_latent = self.mode_norm(h[:, :, None, :] + self.mode_embedding[None, None, :, :])
        out = {
            "latent": h,
            "mode_latent": mode_latent,
            "logits": logits,
            "source_logits": source_logits,
            "priority_logits": priority_logits,
            "mode_source": self.mode_source,
        }
        if decode_traj:
            if self.decoder_type in {"legacy", "legacy_linear", "linear"}:
                traj = self.head(h).reshape(B, A, self.modes, self.future_steps, 7)
                out.update({"traj": traj, "base_traj": torch.zeros_like(traj), "residual": traj})
            elif self.decoder_type in {"typed", "typed_kinematic", "typed_kinematic_residual", "typed_causal_residual", "tnob"}:
                if anchor7 is None:
                    base = torch.zeros(B, A, self.modes, self.future_steps, 7, device=h.device, dtype=h.dtype)
                else:
                    base = self.typed_kinematic_basis(anchor7, dt)
                residual = self._learned_residual(mode_latent)
                out.update({"traj": base + residual, "base_traj": base, "residual": residual})
            else:
                traj, base, residual = self._temporal_kinematic_offsets(mode_latent, anchor7, dt)
                out.update({"traj": traj, "base_traj": base, "residual": residual})
        return out
