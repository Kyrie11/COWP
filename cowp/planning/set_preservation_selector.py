from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class FrontierResult:
    frontier: torch.Tensor
    adjusted_scores: torch.Tensor
    guarded_base: torch.Tensor
    pareto_candidates: int


def _safe(x: torch.Tensor, *, nan: float = 1.0) -> torch.Tensor:
    return torch.nan_to_num(x.float(), nan=nan, posinf=1.0, neginf=0.0)


def _guard_base(
    base: torch.Tensor,
    score_risk: torch.Tensor,
    progress: torch.Tensor,
    action_risk: torch.Tensor,
    *,
    keep_min: int,
    cfg: dict[str, Any],
) -> torch.Tensor:
    base = base.bool()
    if not bool(base.any().item()):
        return base
    idx = torch.where(base)[0]
    need = min(max(int(keep_min), 1), int(idx.numel()))
    guarded = base.clone()

    progress = _safe(progress, nan=0.0)
    p_ref = progress[idx].max()
    min_abs = float(cfg.get("candidate_frontier_min_progress_m", 1.0))
    ratio = float(cfg.get("candidate_frontier_min_progress_ratio", 0.12))
    if float(p_ref.item()) > min_abs:
        pg = base & (progress >= torch.maximum(
            p_ref.new_tensor(min_abs), p_ref * ratio
        ))
        if int(pg.sum().item()) >= need:
            guarded = pg

    score_risk = _safe(score_risk)
    best = score_risk[idx].min()
    sg = base & (score_risk <= best + float(cfg.get("candidate_frontier_score_slack", 0.85)))
    joint = guarded & sg
    if int(joint.sum().item()) >= need:
        guarded = joint

    action_risk = _safe(action_risk)
    ag = guarded & (action_risk <= float(cfg.get("candidate_frontier_max_action_risk", 0.90)))
    if int(ag.sum().item()) >= need:
        guarded = ag
    return guarded


def _exact_topk(
    base: torch.Tensor,
    risk: torch.Tensor,
    tie: torch.Tensor,
    *,
    keep_fraction: float,
    keep_min: int,
    keep_max: int,
    eps: float,
) -> torch.Tensor:
    out = torch.zeros_like(base, dtype=torch.bool)
    idx = torch.where(base)[0]
    if idx.numel() == 0:
        return out
    n = int(idx.numel())
    k = max(int(keep_min), int(torch.ceil(risk.new_tensor(float(n) * float(keep_fraction))).item()))
    k = min(max(k, 1), n, max(int(keep_max), 1))
    r = _safe(risk[idx])
    tb = _safe(tie[idx], nan=0.0)
    if tb.numel() > 1:
        tb = (tb - tb.min()) / (tb.max() - tb.min()).clamp_min(1.0e-6)
    else:
        tb = torch.zeros_like(tb)
    order = torch.argsort(r + float(eps) * tb, stable=True)
    out[idx[order[:k]]] = True
    return out


def _epsilon_pareto(
    base: torch.Tensor,
    noncoercive_risk: torch.Tensor,
    physical_risk: torch.Tensor,
    utility_regret: torch.Tensor,
    tie: torch.Tensor,
    *,
    keep_min: int,
    keep_max: int,
    eps: float,
) -> tuple[torch.Tensor, int]:
    """Return an epsilon-Pareto semantic/physical/utility frontier.

    A candidate is removed only when another candidate is no worse in all three
    axes and meaningfully better in at least one.  This preserves the paper's
    lexicographic setting: response-set preservation is not collapsed into a
    single generic safety/utility scalar, while the final set remains bounded.
    """
    out = torch.zeros_like(base, dtype=torch.bool)
    idx = torch.where(base)[0]
    if idx.numel() == 0:
        return out, 0
    objectives = torch.stack([
        _safe(noncoercive_risk[idx]),
        _safe(physical_risk[idx]),
        _safe(utility_regret[idx]),
    ], dim=-1)
    # dom[i,j] means i dominates j.
    le = objectives[:, None, :] <= objectives[None, :, :] + float(eps)
    lt = objectives[:, None, :] < objectives[None, :, :] - float(eps)
    dominates = le.all(dim=-1) & lt.any(dim=-1)
    dominated = dominates.any(dim=0)
    pareto_local = torch.where(~dominated)[0]
    pareto_count = int(pareto_local.numel())
    chosen = idx[pareto_local]
    if chosen.numel() > max(int(keep_max), 1):
        composite = (
            _safe(noncoercive_risk[chosen])
            + 0.35 * _safe(physical_risk[chosen])
            + 0.15 * _safe(utility_regret[chosen])
            + float(eps) * _safe(tie[chosen], nan=0.0)
        )
        chosen = chosen[torch.argsort(composite, stable=True)[: max(int(keep_max), 1)]]
    if chosen.numel() < max(int(keep_min), 1):
        fill = _exact_topk(
            base, noncoercive_risk, tie,
            keep_fraction=1.0, keep_min=keep_min, keep_max=keep_min, eps=eps,
        )
        chosen = torch.where(fill)[0]
    out[chosen] = True
    return out, pareto_count


def select_set_preservation_frontier_1d(
    *,
    scores: torch.Tensor,
    base_mask: torch.Tensor,
    noncoercive_risk: torch.Tensor,
    score_risk: torch.Tensor,
    progress: torch.Tensor,
    progress_shortfall: torch.Tensor,
    action_risk: torch.Tensor,
    rule_risk: torch.Tensor,
    outcome_risk: torch.Tensor,
    ncf_probability: torch.Tensor | None,
    false_safe_probability: torch.Tensor | None,
    cfg: dict[str, Any],
) -> FrontierResult:
    keep_fraction = float(cfg.get("candidate_frontier_keep_fraction", 0.40))
    keep_min = int(cfg.get("candidate_frontier_min_keep", 1))
    keep_max = int(cfg.get("candidate_frontier_max_keep", 4))
    eps = float(cfg.get("candidate_frontier_tie_eps", 1.0e-3))
    guarded = _guard_base(
        base_mask, score_risk, progress, action_risk,
        keep_min=keep_min, cfg=cfg,
    )
    inf = torch.full_like(scores, float("inf"))
    if not bool(guarded.any().item()):
        return FrontierResult(torch.zeros_like(base_mask, dtype=torch.bool), inf, guarded, 0)

    tie = (
        0.70 * _safe(score_risk)
        + 0.20 * _safe(progress_shortfall)
        + 0.25 * _safe(outcome_risk)
        + 0.20 * _safe(action_risk)
    )
    mode = str(cfg.get("candidate_frontier_mode", "exact_topk")).lower()
    physical_risk = torch.maximum(_safe(action_risk), torch.maximum(_safe(rule_risk), _safe(outcome_risk)))
    if mode in {"epsilon_pareto", "pareto", "lexicographic_pareto"}:
        frontier, pareto_count = _epsilon_pareto(
            guarded,
            noncoercive_risk,
            physical_risk,
            progress_shortfall,
            tie,
            keep_min=keep_min,
            keep_max=keep_max,
            eps=float(cfg.get("candidate_pareto_epsilon", 0.025)),
        )
    else:
        frontier = _exact_topk(
            guarded,
            noncoercive_risk + float(cfg.get("candidate_frontier_shield_tie_mix", 0.08)) * physical_risk,
            tie,
            keep_fraction=keep_fraction,
            keep_min=keep_min,
            keep_max=keep_max,
            eps=eps,
        )
        pareto_count = 0

    if ncf_probability is not None and false_safe_probability is not None:
        screened = (
            frontier
            & (_safe(ncf_probability, nan=0.0) >= float(cfg.get("candidate_min_ncf_prob", 0.05)))
            & (_safe(false_safe_probability) <= float(cfg.get("candidate_max_false_safe_prob", 0.95)))
        )
        if bool(screened.any().item()):
            frontier = screened

    select = frontier.clone()
    nr = _safe(noncoercive_risk)
    vals = nr[select]
    if vals.numel():
        budget = float(cfg.get("candidate_selection_risk_budget", 0.18))
        budget_mask = select & (nr <= vals.min() + budget)
        if int(budget_mask.sum().item()) >= max(int(cfg.get("candidate_selection_min_keep", 1)), 1):
            select = budget_mask
    objective = (
        nr
        + float(cfg.get("candidate_selection_score_weight", 0.18)) * _safe(score_risk)
        + float(cfg.get("candidate_selection_progress_weight", 0.10)) * _safe(progress_shortfall)
        + float(cfg.get("candidate_selection_action_weight", 1.15)) * _safe(action_risk)
        + float(cfg.get("candidate_selection_rule_weight", 0.25)) * _safe(rule_risk)
        + float(cfg.get("candidate_selection_outcome_weight", 0.45)) * _safe(outcome_risk)
    )
    return FrontierResult(frontier, torch.where(select, objective, inf), guarded, pareto_count)


def select_set_preservation_frontier_batch(**kwargs: Any) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    scores = kwargs["scores"]
    bsz = int(scores.shape[0])
    frontier = torch.zeros_like(kwargs["base_mask"], dtype=torch.bool)
    adjusted = torch.full_like(scores, float("inf"))
    counts: list[int] = []
    for b in range(bsz):
        one = {k: (v[b] if torch.is_tensor(v) and v.ndim >= 2 else v) for k, v in kwargs.items()}
        result = select_set_preservation_frontier_1d(**one)
        frontier[b] = result.frontier
        adjusted[b] = result.adjusted_scores
        counts.append(result.pareto_candidates)
    return frontier, adjusted, counts
