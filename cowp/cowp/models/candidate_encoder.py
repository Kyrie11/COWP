from __future__ import annotations

from contextlib import nullcontext

import torch
from torch import nn


# Candidate trajectory layout is [x, y, heading, vx, vy, length, width].
# Fixed physical scales make the recurrent input dimensionless and keep every
# feature in a comparable range.  The buffer is non-persistent so checkpoints
# produced before this numeric fix remain strictly load-compatible.
_CANDIDATE_FEATURE_SCALE = (50.0, 50.0, 3.141592653589793, 20.0, 20.0, 5.0, 2.0)


class CandidateEncoder(nn.Module):
    """Encode an ego-candidate trajectory without letting padding corrupt AMP.

    The witness/transport stages backpropagate several certificate losses into
    this encoder.  Candidate banks are padded to ``K`` with all-zero global-frame
    trajectories.  After ego-centric translation those padded slots can become
    large ``-ego_origin`` sequences unless the validity mask is applied.  Feeding
    those values to a mixed-precision bidirectional GRU makes the first input
    matrices (``weight_ih_l0`` and its reverse direction) the first gradients to
    overflow.

    This implementation therefore enforces three invariants:

    1. invalid candidates are exactly zero before and after the encoder;
    2. heterogeneous physical features are converted to dimensionless scales;
    3. the recurrent/projection block is a small fp32 precision island.

    No parameter shape changes, so existing v16.x checkpoints can be loaded.
    """

    def __init__(
        self,
        d_model: int = 128,
        macro_count: int = 13,
        dropout: float = 0.1,
        *,
        force_fp32: bool = True,
        normalized_clip: float = 20.0,
    ):
        super().__init__()
        self.temporal = nn.GRU(
            input_size=7,
            hidden_size=d_model // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.macro_embed = nn.Embedding(macro_count, d_model)
        self.proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.force_fp32 = bool(force_fp32)
        self.normalized_clip = float(max(normalized_clip, 1.0))
        self.register_buffer(
            "candidate_feature_scale",
            torch.tensor(_CANDIDATE_FEATURE_SCALE, dtype=torch.float32),
            persistent=False,
        )

    @staticmethod
    def _validate_shapes(
        traj: torch.Tensor,
        macro_type: torch.Tensor,
        valid_mask: torch.Tensor | None,
    ) -> tuple[int, int, int, int]:
        if traj.ndim != 4:
            raise ValueError(f"candidate trajectory must be [B,K,T,7], got {tuple(traj.shape)}")
        B, K, T, D = traj.shape
        if D != 7:
            raise ValueError(f"candidate trajectory feature dimension must be 7, got {D}")
        if macro_type.shape != (B, K):
            raise ValueError(
                f"candidate macro_type must be [B,K]={B,K}, got {tuple(macro_type.shape)}"
            )
        if valid_mask is not None and valid_mask.shape != (B, K):
            raise ValueError(
                f"candidate valid_mask must be [B,K]={B,K}, got {tuple(valid_mask.shape)}"
            )
        return B, K, T, D

    def forward(
        self,
        traj: torch.Tensor,
        macro_type: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, K, T, D = self._validate_shapes(traj, macro_type, valid_mask)
        if valid_mask is None:
            valid = torch.ones(B, K, device=traj.device, dtype=torch.bool)
        else:
            valid = valid_mask.to(device=traj.device, dtype=torch.bool)

        # Work in fp32 before recurrent computation.  Invalid rows are removed
        # before normalization, so arbitrary padding values cannot enter cuDNN.
        # A non-finite value in a *valid* candidate is a corrupted cache item and
        # must be reported, not silently repaired into a different trajectory.
        raw = traj.float()
        bad_valid = valid[..., None, None] & ~torch.isfinite(raw)
        if bool(bad_valid.any().item()):
            first = torch.nonzero(bad_valid, as_tuple=False)[0].detach().cpu().tolist()
            raise FloatingPointError(
                "Non-finite feature in a valid candidate trajectory at "
                f"[batch,candidate,time,feature]={first}."
            )
        x = torch.where(valid[..., None, None], raw, torch.zeros_like(raw))
        scale = self.candidate_feature_scale.to(device=x.device, dtype=x.dtype)
        x = (x / scale.view(1, 1, 1, D)).clamp(
            min=-self.normalized_clip,
            max=self.normalized_clip,
        )

        device_type = x.device.type
        fp32_context = (
            torch.autocast(device_type=device_type, enabled=False)
            if self.force_fp32 and device_type in {"cuda", "cpu"}
            else nullcontext()
        )
        with fp32_context:
            h, _ = self.temporal(x.reshape(B * K, T, D).float())
            pooled = h.mean(dim=1).reshape(B, K, -1)
            macro_idx = macro_type.long().clamp(0, self.macro_embed.num_embeddings - 1)
            macro = self.macro_embed(macro_idx)
            encoded = self.proj(pooled + macro.float())

        # A padded token must stay a true padding token even though GRU biases and
        # macro embeddings are non-zero.  This also blocks all invalid-slot grads.
        return encoded * valid.unsqueeze(-1).to(encoded.dtype)
