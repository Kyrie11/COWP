from __future__ import annotations

from dataclasses import dataclass

import numpy as np

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
            valid = np.where(cand_valid & conventional)[0]
            if len(valid):
                k = int(valid[np.argmin(scores[valid])])
                return PlannerDecision(label["cowp/candidates/trajectory"][k], k, "fallback_best_conventional", float(scores[k]), accepted)
            k = int(np.where(cand_valid)[0][0]) if np.any(cand_valid) else -1
            return PlannerDecision(label["cowp/candidates/trajectory"][k], k, "fallback_first_valid", float(scores[k]) if k >= 0 else float("inf"), accepted)
        return PlannerDecision(conservative_fallback(current_state, self.cfg), -1, "conservative_fallback", float("inf"), accepted)
