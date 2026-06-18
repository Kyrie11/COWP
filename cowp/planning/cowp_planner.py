from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cowp.core.constants import MacroType
from cowp.planning.fallback import conservative_fallback
from cowp.planning.ncf_filter import candidate_scores, hard_first_filter


@dataclass
class PlannerDecision:
    trajectory: np.ndarray
    candidate_index: int
    reason: str
    score: float
    accepted_mask: np.ndarray


class COWPPlanner:
    def __init__(self, cfg: dict, ablation: dict | None = None):
        self.cfg = cfg
        self.ablation = ablation or {}

    def select_from_labels(self, label: dict[str, np.ndarray], current_state: np.ndarray | None = None) -> PlannerDecision:
        cand_valid = label["cowp/candidates/valid"].astype(bool)
        conventional = label["cowp/candidates/conventional_safe"].astype(bool)
        witness = label["cowp/witness/exists"].astype(float)
        opr = label["cowp/witness/opr"].astype(float)
        c_i = label["cowp/witness/c_i"].astype(float)
        utility = label["cowp/candidates/ego_utility_prior"].astype(float)
        pcfg = self.cfg.get("planning", {})
        # True label-level branch ablations when source-resolved certificate
        # masses are available in newly built labels.  This keeps the evaluation
        # switch from being a no-op for w/o neutral or w/o priority variants.
        source_mass = label.get("cowp/witness/natural_conflict_mass_by_source")
        natural_mass_source = label.get("cowp/witness/natural_mass_by_source")
        low_safe_source = label.get("cowp/witness/low_safe_mass_by_source")
        branch_keys = ("use_obs_branch", "use_neutral_branch", "use_priority_branch")
        if source_mass is not None and low_safe_source is not None and any(not bool(self.ablation.get(k, True)) for k in branch_keys):
            enabled = np.asarray([
                bool(self.ablation.get("use_obs_branch", True)),
                bool(self.ablation.get("use_neutral_branch", True)),
                bool(self.ablation.get("use_priority_branch", True)),
                False,
            ], dtype=bool)
            source_mass = np.asarray(source_mass, dtype=float)
            low_safe_source = np.asarray(low_safe_source, dtype=float)
            conflict_mass = source_mass[..., enabled].sum(axis=-1)
            low_safe_mass = low_safe_source[..., enabled].sum(axis=-1)
            if natural_mass_source is not None:
                denom = np.asarray(natural_mass_source, dtype=float)[..., enabled].sum(axis=-1)
            else:
                # Backward-compatible fallback for labels built before
                # natural_mass_by_source existed: approximate the denominator by
                # the union of conflicted and preserved low-burden mass.
                denom = np.maximum(conflict_mass + low_safe_mass, 1.0)
            beta = label["cowp/natural/beta"].astype(float)[None, :]
            alpha = float(pcfg.get("alpha_opr_infer", self.cfg.get("ncf", {}).get("alpha_opr", 0.35)))
            min_safe = label["cowp/witness/min_safe_burden"].astype(float)
            use_option = bool(self.ablation.get("use_option_preservation", True))
            opr = np.where(denom > 1e-6, low_safe_mass / np.maximum(denom, 1e-6), 1.0) if use_option else np.ones_like(low_safe_mass)
            opr = np.clip(opr, 0.0, 1.0)
            option_collapsed = use_option & (opr < alpha)
            witness = ((conflict_mass >= float(self.cfg.get("ncf", {}).get("positive_min_natural_conflict_mass", 0.10))) & ((min_safe > beta) | option_collapsed)).astype(float)
        accepted = hard_first_filter(
            cand_valid,
            conventional,
            witness,
            opr,
            p_hard=float(pcfg.get("p_hard", 0.75)),
            alpha=float(pcfg.get("alpha_opr_infer", 0.35)),
            use_hard_witness_rejection=bool(self.ablation.get("use_hard_witness_rejection", True)),
            use_option_preservation=bool(self.ablation.get("use_option_preservation", True)),
        )
        scores = candidate_scores(
            utility,
            witness,
            opr,
            c_i,
            p_soft=float(pcfg.get("p_soft", 0.45)),
            alpha=float(pcfg.get("alpha_opr_infer", 0.35)),
            gamma=float(pcfg.get("ncf_gamma_infer", 0.10)),
            soft_burden_only=bool(self.ablation.get("soft_burden_cost_only", False)),
        )
        if np.any(accepted):
            masked = np.where(accepted, scores, np.inf)
            k = int(np.argmin(masked))
            return PlannerDecision(label["cowp/candidates/trajectory"][k], k, "selected_noncoercive", float(masked[k]), accepted)
        if current_state is None:
            # Offline labels do not include the current simulator state needed to
            # synthesize a true conservative stop.  Prefer explicitly neutral /
            # stop-like conventional candidates instead of silently selecting the
            # best false-safe conventional plan.
            valid = cand_valid & conventional
            is_neutral = label.get("cowp/candidates/is_neutral", np.zeros_like(cand_valid)).astype(bool)
            if np.any(valid & is_neutral):
                idx = np.where(valid & is_neutral)[0]
                k = int(idx[np.argmin(scores[idx])])
                return PlannerDecision(label["cowp/candidates/trajectory"][k], k, "fallback_neutral_conventional", float(scores[k]), accepted)
            macro = label.get("cowp/candidates/macro_type", np.full_like(scores, int(MacroType.PAD))).astype(int)
            stop_like = np.isin(macro, [int(MacroType.STOP_BEFORE_CONFLICT), int(MacroType.YIELD), int(MacroType.CREEP), int(MacroType.NEUTRAL_EGO)])
            if np.any(valid & stop_like):
                idx = np.where(valid & stop_like)[0]
                k = int(idx[np.argmin(scores[idx])])
                return PlannerDecision(label["cowp/candidates/trajectory"][k], k, "fallback_stop_like_conventional", float(scores[k]), accepted)
            if np.any(cand_valid):
                # Return a trajectory for downstream visualization, but mark the
                # candidate index as invalid so metric code treats this as fallback
                # rather than as accepting a coercive candidate.
                k_vis = int(np.where(cand_valid)[0][0])
                return PlannerDecision(label["cowp/candidates/trajectory"][k_vis], -1, "fallback_conservative_unavailable", float("inf"), accepted)
            T = int(self.cfg.get("time", {}).get("future_steps", 80))
            return PlannerDecision(np.zeros((T, 7), dtype=np.float32), -1, "fallback_no_candidate", float("inf"), accepted)
        return PlannerDecision(conservative_fallback(current_state, self.cfg), -1, "conservative_fallback", float("inf"), accepted)
