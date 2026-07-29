from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


class MLPEncoder(nn.Module):
    def __init__(self, in_dim: int, d_model: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, d_model), nn.ReLU(), nn.Linear(d_model, d_model), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class FlatFeatureLayout:
    ego_dim: int = 9
    prefix_param_dim: int = 5
    num_macros: int = 16
    scalar_dim: int = 6
    prefix_flat_dim: int = 80
    control_flat_dim: int = 40
    agent_summary_dim: int = 8
    feature_max_agents: int = 32
    agent_token_dim: int = 10
    bev_channels: int = 7
    route_stats_dim: int = 6
    route_flat_dim: int = 64
    map_stats_dim: int = 6
    map_flat_dim: int = 64
    dyn_stats_dim: int = 6
    dyn_flat_dim: int = 32

    @property
    def bev_dim(self) -> int:
        return 2 * int(self.bev_channels)

    @property
    def total_dim(self) -> int:
        return (
            self.ego_dim + self.prefix_param_dim + self.num_macros + self.scalar_dim
            + self.prefix_flat_dim + self.control_flat_dim
            + self.agent_summary_dim + self.feature_max_agents * self.agent_token_dim
            + self.bev_dim + self.route_stats_dim + self.route_flat_dim
            + self.map_stats_dim + self.map_flat_dim + self.dyn_stats_dim + self.dyn_flat_dim
        )


class StructuredTokenEncoder(nn.Module):
    """Lightweight token transformer over grouped OC-RAP scene-prefix features.

    The full paper model uses agent/map transformers, ego-prefix encoding, BEV
    tokens, and learned root queries.  This encoder is a practical intermediate
    implementation: it keeps those semantic groups as separate tokens instead
    of collapsing everything through one flat MLP.
    """

    def __init__(
        self,
        layout: FlatFeatureLayout,
        d_model: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layout = layout
        self.d_model = int(d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.ego_proj = nn.Linear(layout.ego_dim, d_model)
        self.prefix_param_proj = nn.Linear(layout.prefix_param_dim, d_model)
        self.macro_scalar_proj = nn.Linear(layout.num_macros + layout.scalar_dim, d_model)
        self.prefix_state_proj = nn.Linear(layout.prefix_flat_dim, d_model)
        self.control_proj = nn.Linear(layout.control_flat_dim, d_model)
        self.agent_summary_proj = nn.Linear(layout.agent_summary_dim, d_model)
        self.agent_proj = nn.Linear(layout.agent_token_dim, d_model)
        self.bev_proj = nn.Linear(layout.bev_dim, d_model)
        self.route_proj = nn.Linear(layout.route_stats_dim + layout.route_flat_dim, d_model)
        self.map_proj = nn.Linear(layout.map_stats_dim + layout.map_flat_dim, d_model)
        self.dyn_proj = nn.Linear(layout.dyn_stats_dim + layout.dyn_flat_dim, d_model)
        self.max_tokens = 1 + 10 + int(layout.feature_max_agents)
        self.pos = nn.Parameter(torch.zeros(1, self.max_tokens, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def _split(self, x: torch.Tensor):
        L = self.layout
        idx = 0
        def take(n: int):
            nonlocal idx
            y = x[:, idx: idx + n]
            idx += n
            return y
        ego = take(L.ego_dim)
        prefix_param = take(L.prefix_param_dim)
        macro = take(L.num_macros)
        scalar = take(L.scalar_dim)
        prefix_state = take(L.prefix_flat_dim)
        control = take(L.control_flat_dim)
        agent_summary = take(L.agent_summary_dim)
        agents = take(L.feature_max_agents * L.agent_token_dim).reshape(x.shape[0], L.feature_max_agents, L.agent_token_dim)
        bev = take(L.bev_dim)
        route = take(L.route_stats_dim + L.route_flat_dim)
        maps = take(L.map_stats_dim + L.map_flat_dim)
        dyn = take(L.dyn_stats_dim + L.dyn_flat_dim)
        return ego, prefix_param, macro, scalar, prefix_state, control, agent_summary, agents, bev, route, maps, dyn

    def _tokens(self, x: torch.Tensor) -> torch.Tensor:
        ego, prefix_param, macro, scalar, prefix_state, control, agent_summary, agents, bev, route, maps, dyn = self._split(x)
        B = x.shape[0]
        tokens = [
            self.ego_proj(ego),
            self.prefix_param_proj(prefix_param),
            self.macro_scalar_proj(torch.cat([macro, scalar], dim=-1)),
            self.prefix_state_proj(prefix_state),
            self.control_proj(control),
            self.agent_summary_proj(agent_summary),
            self.bev_proj(bev),
            self.route_proj(route),
            self.map_proj(maps),
            self.dyn_proj(dyn),
        ]
        tok = torch.stack(tokens, dim=1)
        agent_tok = self.agent_proj(agents)
        cls = self.cls.expand(B, -1, -1)
        tok = torch.cat([cls, tok, agent_tok], dim=1)
        tok = tok + self.pos[:, :tok.shape[1], :]
        return self.encoder(tok)

    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Return encoded semantic tokens, including the leading scene token.

        The paper's encoder does not collapse the scene immediately: learned
        root queries attend over agent, map, ego-prefix, and BEV tokens.  Keeping
        this method separate preserves the previous ``forward`` API while giving
        the OC-RAP model access to the token set needed by the root-query decoder.
        """
        return self.norm(self._tokens(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.forward_tokens(x)
        return h[:, 0]
