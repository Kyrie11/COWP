"""Certificate-Guided Proposal Refinement (CGPR).

This source-independent module implements the proposed v16.8.4 refinement
interface.  It turns protected-pair certificate deficits into physically
screened longitudinal timing proposals.  It does not assert that a proposal is
safe; the normal COWP label/certificate pipeline must re-evaluate every output.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Protocol, Sequence
import math


PROTECTED_RELATIONS = {"AgentPriority", "EqualOrNegotiated"}


@dataclass(frozen=True)
class EgoState:
    speed_mps: float
    acceleration_mps2: float = 0.0


@dataclass(frozen=True)
class ConflictRegion:
    region_id: str
    ego_arc_distance_m: float
    agent_arrival_early_s: float
    agent_arrival_nominal_s: float
    agent_arrival_late_s: float

    def validate(self) -> None:
        if self.ego_arc_distance_m <= 0:
            raise ValueError("ego_arc_distance_m must be positive")
        if not (
            0 < self.agent_arrival_early_s
            <= self.agent_arrival_nominal_s
            <= self.agent_arrival_late_s
        ):
            raise ValueError("arrival envelope must be positive and ordered")


@dataclass(frozen=True)
class ProtectedPairCertificate:
    agent_id: str
    relation: str
    conflict_mass: float
    option_preservation: float
    tail_burden_excess: float
    uncertainty: float
    regions: tuple[ConflictRegion, ...]

    def is_protected(self) -> bool:
        return self.relation in PROTECTED_RELATIONS


@dataclass(frozen=True)
class RefinementConfig:
    option_threshold: float = 0.65
    conflict_mass_floor: float = 0.10
    gap_s: tuple[float, ...] = (0.8, 1.4, 2.0)
    min_target_time_s: float = 0.4
    min_accel_mps2: float = -3.5
    max_accel_mps2: float = 2.5
    max_abs_jerk_proxy_mps3: float = 10.0
    actuation_ramp_s: float = 0.4
    accel_dedup_mps2: float = 0.10
    max_pairs: int = 3
    max_regions_per_pair: int = 3
    max_candidates: int = 24
    stop_distance_margin_m: float = 1.0
    relation_weight_agent_priority: float = 1.0
    relation_weight_equal: float = 0.9


@dataclass(frozen=True)
class RefinementProposal:
    source: str
    agent_id: str
    relation: str
    region_id: str
    timing_side: str
    target_time_s: float
    gap_s: float
    solved_acceleration_mps2: float
    terminal_speed_mps: float
    ego_arc_distance_m: float
    certificate_deficit_score: float
    provenance: dict[str, float | str] = field(default_factory=dict)


class PathProfileAdapter(Protocol):
    def build_candidate(
        self,
        *,
        ego_state: EgoState,
        proposal: RefinementProposal,
    ) -> object:
        """Map a longitudinal timing proposal onto the planner's reference path."""


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def deficit_score(
    cert: ProtectedPairCertificate,
    cfg: RefinementConfig,
) -> float:
    relation_weight = (
        cfg.relation_weight_agent_priority
        if cert.relation == "AgentPriority"
        else cfg.relation_weight_equal
    )
    conflict = max(0.0, cert.conflict_mass - cfg.conflict_mass_floor)
    option = max(0.0, cfg.option_threshold - cert.option_preservation)
    burden = max(0.0, cert.tail_burden_excess)
    uncertainty = _clamp01(cert.uncertainty)
    return relation_weight * (
        1.25 * conflict + 1.50 * option + 1.00 * burden + 0.25 * uncertainty
    )


def solve_constant_acceleration(
    *,
    distance_m: float,
    initial_speed_mps: float,
    target_time_s: float,
    min_accel_mps2: float,
    max_accel_mps2: float,
    stop_distance_margin_m: float = 1.0,
) -> tuple[float, float] | None:
    """Solve d=v0*t+0.5*a*t^2 with a no-premature-stop constraint."""
    if distance_m <= 0 or initial_speed_mps < 0 or target_time_s <= 0:
        return None
    a = 2.0 * (distance_m - initial_speed_mps * target_time_s) / (
        target_time_s * target_time_s
    )
    if not (min_accel_mps2 <= a <= max_accel_mps2):
        return None

    terminal_speed = initial_speed_mps + a * target_time_s
    if a < 0:
        stop_time = -initial_speed_mps / a if a != 0 else math.inf
        if stop_time < target_time_s:
            stop_distance = (
                initial_speed_mps * stop_time + 0.5 * a * stop_time * stop_time
            )
            if stop_distance < distance_m - stop_distance_margin_m:
                return None
    if terminal_speed < -1e-6:
        return None
    return a, max(0.0, terminal_speed)


def _jerk_proxy_ok(a: float, cfg: RefinementConfig) -> bool:
    ramp = max(cfg.actuation_ramp_s, 1e-3)
    return abs(a) / ramp <= cfg.max_abs_jerk_proxy_mps3


def _candidate_key(p: RefinementProposal, cfg: RefinementConfig) -> tuple:
    quant = round(p.solved_acceleration_mps2 / cfg.accel_dedup_mps2)
    return (p.agent_id, p.region_id, p.timing_side, quant)


def generate_certificate_guided_refinements(
    *,
    ego_state: EgoState,
    certificates: Sequence[ProtectedPairCertificate],
    config: RefinementConfig = RefinementConfig(),
) -> list[RefinementProposal]:
    """Generate robust pass-before/pass-after repairs for top protected deficits.

    The routine deliberately does not generate a proposal from an unprotected
    relation. It uses the agent early arrival for pass-before and late arrival
    for pass-after, which makes timing robust to the supplied arrival envelope.
    """
    ranked = []
    for cert in certificates:
        if not cert.is_protected():
            continue
        if cert.conflict_mass <= config.conflict_mass_floor and (
            cert.option_preservation >= config.option_threshold
            and cert.tail_burden_excess <= 0
        ):
            continue
        for region in cert.regions:
            region.validate()
        ranked.append((deficit_score(cert, config), cert))
    ranked.sort(key=lambda x: x[0], reverse=True)

    proposals: list[RefinementProposal] = []
    seen = set()
    for score, cert in ranked[: config.max_pairs]:
        regions = sorted(
            cert.regions,
            key=lambda r: (r.agent_arrival_nominal_s, r.ego_arc_distance_m),
        )[: config.max_regions_per_pair]
        for region in regions:
            for gap in config.gap_s:
                targets = (
                    ("pass_before", region.agent_arrival_early_s - gap),
                    ("pass_after", region.agent_arrival_late_s + gap),
                )
                for side, target in targets:
                    if target < config.min_target_time_s:
                        continue
                    solution = solve_constant_acceleration(
                        distance_m=region.ego_arc_distance_m,
                        initial_speed_mps=ego_state.speed_mps,
                        target_time_s=target,
                        min_accel_mps2=config.min_accel_mps2,
                        max_accel_mps2=config.max_accel_mps2,
                        stop_distance_margin_m=config.stop_distance_margin_m,
                    )
                    if solution is None:
                        continue
                    accel, terminal_speed = solution
                    if not _jerk_proxy_ok(accel - ego_state.acceleration_mps2, config):
                        continue
                    proposal = RefinementProposal(
                        source="certificate_guided_rmr_bcte",
                        agent_id=cert.agent_id,
                        relation=cert.relation,
                        region_id=region.region_id,
                        timing_side=side,
                        target_time_s=target,
                        gap_s=gap,
                        solved_acceleration_mps2=accel,
                        terminal_speed_mps=terminal_speed,
                        ego_arc_distance_m=region.ego_arc_distance_m,
                        certificate_deficit_score=score,
                        provenance={
                            "conflict_mass": cert.conflict_mass,
                            "option_preservation": cert.option_preservation,
                            "tail_burden_excess": cert.tail_burden_excess,
                            "uncertainty": cert.uncertainty,
                            "agent_arrival_early_s": region.agent_arrival_early_s,
                            "agent_arrival_nominal_s": region.agent_arrival_nominal_s,
                            "agent_arrival_late_s": region.agent_arrival_late_s,
                        },
                    )
                    key = _candidate_key(proposal, config)
                    if key in seen:
                        continue
                    seen.add(key)
                    proposals.append(proposal)
                    if len(proposals) >= config.max_candidates:
                        return proposals
    return proposals


def sample_longitudinal_profile(
    *,
    ego_state: EgoState,
    proposal: RefinementProposal,
    dt_s: float = 0.1,
    horizon_s: float = 8.0,
) -> list[tuple[float, float, float]]:
    """Return (t, arc_distance, speed) for adapter/testing use."""
    if dt_s <= 0 or horizon_s <= 0:
        raise ValueError("dt_s and horizon_s must be positive")
    out = []
    steps = int(round(horizon_s / dt_s))
    a = proposal.solved_acceleration_mps2
    for step in range(steps + 1):
        t = step * dt_s
        v = max(0.0, ego_state.speed_mps + a * t)
        if a < 0 and ego_state.speed_mps + a * t < 0:
            stop_t = -ego_state.speed_mps / a
            s = ego_state.speed_mps * stop_t + 0.5 * a * stop_t * stop_t
        else:
            s = ego_state.speed_mps * t + 0.5 * a * t * t
        out.append((t, max(0.0, s), v))
    return out
