from __future__ import annotations

"""Literature/code provenance and fidelity contracts for external baselines.

The registry is intentionally explicit: an OC-RAP candidate-lattice adaptation
must never be reported as an author's official implementation.  `core_retained`
describes the paper mechanisms implemented in this repository; `known_gaps`
lists components that remain outside the fair executable-candidate protocol.
"""

from dataclasses import asdict, dataclass
from typing import Iterable


@dataclass(frozen=True)
class BaselineProvenance:
    canonical_name: str
    aliases: tuple[str, ...]
    regimes: tuple[str, ...]
    paper_title: str
    paper_year: int | None
    paper_url: str | None
    official_code_url: str | None
    implementation_kind: str
    fidelity: str
    core_retained: tuple[str, ...]
    known_gaps: tuple[str, ...]
    reporting_name: str

    def to_dict(self) -> dict:
        return asdict(self)


REGISTRY: tuple[BaselineProvenance, ...] = (
    BaselineProvenance(
        "nominal_replay", ("nominal", "nominal_replay", "log_replay"), ("safe",),
        "Logged trajectory replay (dataset control; not a paper implementation)", None, None, None,
        "dataset control", "exact",
        ("replays the logged nominal candidate",), (), "Logged nominal replay",
    ),
    BaselineProvenance(
        "wayformer_bc", ("route_bc", "route_bc_lite", "waymax_bc", "waymax_bc_lite", "wayformer_bc", "wayformer_style_bc", "route_bc_wayformer"), ("safe",),
        "Wayformer: Motion Forecasting via Simple & Efficient Attention Networks", 2022,
        "https://arxiv.org/abs/2207.05844", None,
        "native candidate-policy adaptation", "paper-core adaptation",
        ("factorized candidate tokens", "Transformer scene fusion", "latent-query compression and cross-attention", "route-conditioned candidate scoring"),
        ("not the authors' forecasting data pipeline", "no official released checkpoint/backend", "candidate classification replaces the original multimodal trajectory output"),
        "Wayformer-style route BC",
    ),
    BaselineProvenance(
        "gameformer_lite", ("gameformer", "gameformer_lite", "gameformer_levelk"), ("safe", "near"),
        "GameFormer: Game-theoretic Modeling and Learning of Transformer-based Interactive Prediction and Planning for Autonomous Driving", 2023,
        "https://arxiv.org/abs/2303.05760", "https://github.com/MCZhi/GameFormer-Planner",
        "native paper-core reproduction on OC-RAP tensors", "high paper-core fidelity",
        ("agent/history encoder", "multimodal initial decoder", "level-k interaction decoder", "previous-level future encoding", "deep trajectory supervision", "candidate-level planning decoder"),
        ("OC-RAP/WOMD preprocessing replaces the authors' nuPlan pipeline", "not checkpoint-compatible with the official repository", "finite candidate prefixes replace continuous trajectory sampling"),
        "GameFormer level-k adapter",
    ),
    BaselineProvenance(
        "plantf", ("plantf", "plan_tf", "plantf_adapter"), ("safe",),
        "Rethinking Imitation-based Planner for Autonomous Driving", 2024,
        "https://arxiv.org/abs/2309.10443", "https://github.com/jchengai/planTF",
        "WOMD candidate-lattice architecture adaptation", "paper-core adaptation",
        ("pure imitation policy", "compact state/vector attention", "state-dropout regularization", "closed-loop candidate policy scoring"),
        ("nuPlan feature builder/trajectory decoder replaced by OC-RAP tensors and executable prefixes", "not checkpoint-compatible with the official repository"),
        "PlanTF adapter",
    ),
    BaselineProvenance(
        "pluto", ("pluto", "pluto_adapter"), ("safe",),
        "PLUTO: Pushing the Limit of Imitation Learning-based Planning for Autonomous Driving", 2024,
        "https://arxiv.org/abs/2404.14327", "https://github.com/jchengai/pluto",
        "WOMD candidate-lattice architecture adaptation", "paper-core adaptation",
        ("longitudinal/lateral maneuver-aware queries", "imitation objective", "contrastive imitation signal", "candidate-level closed-loop selection"),
        ("nuPlan renderer/augmentations and native trajectory decoder are not reproduced", "contrastive learning is implemented within the common candidate set rather than the authors' full batch augmentation pipeline"),
        "PLUTO adapter",
    ),
    BaselineProvenance(
        "pdm_closed", ("pdm_closed", "pdm_closed_adapter"), ("safe",),
        "Parting with Misconceptions about Learning-based Vehicle Motion Planning", 2023,
        "https://arxiv.org/abs/2306.07962", "https://github.com/autonomousvision/tuplan_garage",
        "finite-lattice PDM-Closed adaptation", "paper-core adaptation",
        ("route-centered proposal scoring", "IDM-style longitudinal preference", "collision/TTC safety", "progress and comfort scoring"),
        ("native nuPlan centerline proposal generator and scorer geometry are replaced by the common OC-RAP executable candidate lattice",),
        "PDM-Closed adapter",
    ),
    BaselineProvenance(
        "pdm_hybrid", ("pdm_hybrid", "pdm_hybrid_adapter"), ("safe",),
        "Parting with Misconceptions about Learning-based Vehicle Motion Planning", 2023,
        "https://arxiv.org/abs/2306.07962", "https://github.com/autonomousvision/tuplan_garage",
        "finite-lattice PDM-Hybrid adaptation", "paper-core adaptation",
        ("PDM closed-loop rule score", "learned long-horizon ego refinement", "rule-plus-learned hybrid selection"),
        ("native nuPlan PDM proposal generator and ego-forecasting model are replaced by OC-RAP candidates and a regime-trained refinement head",),
        "PDM-Hybrid adapter",
    ),
    BaselineProvenance(
        "idm", ("idm", "idm_planner"), ("safe",),
        "Congested Traffic States in Empirical Observations and Microscopic Simulations (Intelligent Driver Model)", 2000,
        "https://doi.org/10.1103/PhysRevE.62.1805", None,
        "finite-lattice IDM control projection", "equation-core adaptation",
        ("IDM desired acceleration", "front-gap and relative-speed interaction", "comfortable headway/deceleration parameters"),
        ("continuous IDM acceleration is projected onto the nearest feasible executable candidate", "lateral path choice comes from the common candidate set"),
        "IDM projection",
    ),
    BaselineProvenance(
        "betopnet_lite", ("betop", "betop_lite", "betopnet", "betopnet_lite"), ("safe",),
        "Reasoning Multi-Agent Behavioral Topology for Interactive Autonomous Driving", 2024,
        "https://arxiv.org/abs/2409.18031", "https://github.com/OpenDriveLab/BeTop",
        "native paper-core adaptation", "paper-core adaptation",
        ("actor and map topology encoders", "iterative topology decoding", "topology-guided sparse attention", "topology auxiliary supervision", "candidate policy scoring"),
        ("not checkpoint-compatible with official code", "the public repository releases the WOMD prediction pipeline while its nuPlan planning release remains marked TODO", "OC-RAP topology labels replace official preprocessing", "candidate selector replaces the unreleased full planning stack"),
        "BeTop topology-aware adapter",
    ),
    BaselineProvenance(
        "marc_lite", ("marc", "marc_lite", "marc_contingency"), ("near",),
        "MARC: Multipolicy and Risk-aware Contingency Planning for Autonomous Driving", 2023,
        "https://arxiv.org/abs/2308.12021", None,
        "finite-lattice contingency reproduction", "paper-core adaptation",
        ("semantic multi-policy families", "dynamic non-anticipative shared prefix", "scenario-tail branching", "expected/CVaR risk tolerance", "constrained policy-family representative selection"),
        ("continuous nonlinear optimization is replaced by executable candidate prefixes", "no author code/checkpoint was identified", "actor prediction uses the shared observation-only mode bank"),
        "MARC candidate-lattice contingency",
    ),
    BaselineProvenance(
        "racp_lite", ("racp", "racp_lite", "risk_aware_contingency"), ("near",),
        "RACP: Risk-Aware Contingency Planning with Multi-Modal Predictions", 2024,
        "https://arxiv.org/abs/2402.17387", "https://github.com/KhMustafa/Risk-aware-contingency-planning-with-multi-modal-predictions",
        "finite-lattice belief/risk reproduction", "paper-core adaptation",
        ("belief-weighted multimodal prediction", "non-anticipative shared prefix", "branch risk with expected/CVaR mixture", "chance constraint", "utility-risk optimization"),
        ("continuous branch MPC is replaced by candidate enumeration", "not API/checkpoint compatible with official code", "belief update is prior-predictive until new online evidence arrives"),
        "RACP candidate-lattice planner",
    ),
    BaselineProvenance(
        "robust_scenario_mpc", ("robust_scenario_mpc", "scenario_mpc", "batkovic_scenario_mpc"), ("near",),
        "A Robust Scenario MPC Approach for Uncertain Multi-Modal Obstacles", 2021,
        "https://doi.org/10.1109/LCSYS.2020.3006819", None,
        "finite-lattice scenario MPC adaptation", "paper-core adaptation",
        ("multi-modal scenario costs", "worst-scenario robust satisfaction", "expected-cost term", "smooth executable control preference"),
        ("tube feedback policy and continuous MPC optimizer are replaced by explicit scoring of the shared executable candidate lattice",),
        "Robust scenario MPC adapter",
    ),
    BaselineProvenance(
        "dr_cvar_safety_filter", ("dr_cvar_safety_filter", "distributionally_robust_cvar_filter", "safaoui_dr_cvar_filter"), ("near",),
        "Distributionally Robust CVaR-Based Safety Filtering for Motion Planning in Uncertain Environments", 2024,
        "https://arxiv.org/abs/2309.08821", None,
        "finite-lattice DR-CVaR safety-filter adaptation", "paper-core approximation",
        ("sample-based obstacle uncertainty", "distributionally robust CVaR inflation", "minimal correction of a reference plan", "clearance admission"),
        ("paper's DRO safe-halfspace construction and continuous MPC correction are replaced by a conservative Wasserstein-CVaR bound and nearest admitted candidate",),
        "DR-CVaR safety filter adapter",
    ),
    BaselineProvenance(
        "conformal_predictive_safety_filter", ("conformal_predictive_safety_filter", "conformal_safety_filter", "cpsf"), ("near",),
        "Conformal Predictive Safety Filter for RL Controllers in Dynamic Environments", 2023,
        "https://arxiv.org/abs/2306.02551", None,
        "split-conformal finite-lattice safety-filter adaptation", "paper-core approximation",
        ("held-out calibration only", "observation-only uncertainty/risk score", "frozen conformal admission threshold", "minimal-deviation safety projection"),
        ("trajectory-wise conformal prediction intervals and learned RL safety filter are represented by binary risk admission over executable candidates",),
        "Conformal predictive safety filter adapter",
    ),
    BaselineProvenance(
        "expected_risk_filter", ("expected_risk", "expected_risk_filter", "expected_risk_planner"), ("near",),
        "Expected-risk constrained planning (generic control baseline)", None, None, None,
        "observation-only risk filter", "exact protocol definition",
        ("multimodal collision loss expectation", "risk-threshold admission", "utility-risk selection"), (), "Expected-risk filter",
    ),
    BaselineProvenance(
        "cvar_risk_filter", ("cvar_risk", "cvar_risk_filter", "cvar_planner"), ("near",),
        "CVaR-constrained planning (generic risk-sensitive baseline)", None, None, None,
        "observation-only risk filter", "exact protocol definition",
        ("weighted upper-tail CVaR", "risk-threshold admission", "utility-risk selection"), (), "CVaR risk filter",
    ),
    BaselineProvenance(
        "dro_cvar_filter", ("dro_cvar", "dro_cvar_filter", "dro_cvar_safety_filter", "dr_cvar_filter"), ("near",),
        "Distributionally robust CVaR planning (generic baseline)", None, None, None,
        "observation-only robust-risk surrogate", "explicit surrogate",
        ("CVaR tail risk", "ambiguity-radius dispersion penalty", "constrained selection"),
        ("dispersion penalty is a fast Wasserstein-inspired surrogate rather than a full inner ambiguity optimization",),
        "DRO-CVaR filter",
    ),
    BaselineProvenance(
        "predictive_safety_filter", ("predictive_safety_filter", "psf", "cbf_backup_filter", "predictive_cbf_backup", "backup_cbf_filter"), ("near",),
        "A Predictive Safety Filter for Learning-Based Control of Constrained Nonlinear Dynamical Systems", 2021,
        "https://arxiv.org/abs/1812.05506", None,
        "finite-lattice predictive safety filter", "paper-core adaptation",
        ("minimal intervention", "input feasibility", "stage backup-set margin", "terminal backup-set margin", "predictive barrier condition"),
        ("candidate enumeration replaces online nonlinear MPC", "geometric stopping set replaces a learned/verified terminal controller"),
        "Predictive safety filter",
    ),
    BaselineProvenance(
        "oracle_recovery_filter", ("oracle_filter", "oracle_recovery_filter", "branchwise_oracle_filter", "oracle_branchwise_recovery"), ("near",),
        "OC-RAP teacher-only oracle upper bound (not an external baseline)", None, None, None,
        "non-deployable audit upper bound", "exact teacher audit",
        ("branch-wise existential recovery option", "oracle order: option maximization before latent-root aggregation"),
        ("uses teacher tensors and must never be reported as deployable",), "Teacher oracle upper bound",
    ),
    BaselineProvenance(
        "postimpact_mpc_lite", ("postimpact_mpc", "postimpact_mpc_lite", "post_impact_mpc_lite", "postimpact_mpc_paper", "integrated_postimpact_mpc"), ("contact",),
        "Integrated Post-Impact Planning and Active Safety Control for Autonomous Vehicles", 2023,
        "https://doi.org/10.1109/TIV.2023.3236150", None, "finite-lattice post-impact objective", "paper-core adaptation",
        ("post-impact yaw/adhesion stability", "secondary-collision risk", "terminal motion objective", "control effort constraints"),
        ("no impact impulse/damage-state estimator", "continuous vehicle-dynamics MPC replaced by candidate costs"),
        "Post-impact MPC adapter",
    ),
    BaselineProvenance(
        "post_crash_braking", ("post_crash_braking", "post_crash_braking_rule", "stable_stop", "stable_stop_rule", "postcrash_stable_stop"), ("contact",),
        "A System for Autonomous Braking of a Vehicle Following Collision", 2017,
        "https://doi.org/10.4271/2017-01-1581", None,
        "finite-lattice autonomous post-impact braking adaptation", "paper-core adaptation",
        ("collision-triggered autonomous braking", "ABS-level stable-stop intent", "terminal-speed stabilization", "secondary-risk gate"),
        ("hydraulic/ABS actuator dynamics are not present in WOMD; braking is projected onto executable brake/stabilize candidates",),
        "Autonomous post-impact braking",
    ),
    BaselineProvenance(
        "post_collision_restoration", ("post_collision_restoration", "trajectory_restoration", "post_collision_trajectory_restoration", "post_collision_restoration_heuristic", "ackermann_restoration"), ("contact",),
        "Post-Collision Trajectory Restoration for a Single-track Ackermann Vehicle using Heuristic Steering and Tractive Force Functions", 2026,
        "https://arxiv.org/abs/2602.08444", None, "finite-lattice restoration objective", "paper-core adaptation",
        ("trajectory re-alignment", "yaw stabilization", "progress preservation", "clearance/risk objective", "Ackermann-feasibility proxies"),
        ("paper is recent/preprint and no official implementation was identified", "online state/impact estimator is not reproduced", "continuous restoration optimization replaced by candidate selection"),
        "Post-collision restoration",
    ),
    BaselineProvenance(
        "postimpact_motion_tvlqr", ("postimpact_motion_tvlqr", "postimpact_motion_planning", "wang2022_postimpact", "postimpact_tvlqr"), ("contact",),
        "Post-Impact Motion Planning and Tracking Control for Autonomous Vehicles", 2022,
        "https://doi.org/10.1186/s10033-022-00745-w", None,
        "finite-lattice post-impact planning/control adaptation", "paper-core adaptation",
        ("post-impact trajectory re-alignment", "obstacle-potential avoidance", "TVLQR-like tracking/stability objective", "control-effort allocation proxy"),
        ("polynomial/APF continuous planner, TVLQR generalized-force dynamics, and wheel torque allocator are replaced by candidate-level objectives because WOMD lacks wheel-force state",),
        "Post-impact APF/TVLQR adapter",
    ),
    BaselineProvenance(
        "compensatory_postimpact_mpc", ("compensatory_postimpact_mpc", "cao_postimpact_mpc"), ("contact",),
        "Compensatory Model Predictive Control for Post-impact Trajectory Tracking via Active Front Steering and Differential Torque Vectoring", 2021,
        "https://doi.org/10.1177/0954407020979087", None,
        "finite-lattice FCC-MPC adaptation", "paper-core adaptation",
        ("lateral/yaw deviation attenuation", "feedforward-feedback compensation objective", "input/rate and adhesion proxies", "secondary-collision risk term"),
        ("wheel-level active-front-steering and differential-torque-vectoring allocation is unavailable in WOMD and is represented through candidate feasibility/control costs",),
        "Compensatory post-impact MPC adapter",
    ),
    BaselineProvenance(
        "robust_postimpact_control", ("robust_postimpact_control", "postimpact_sliding_mode", "ao_postimpact_control"), ("contact",),
        "Advanced Post-impact Safety and Stability Control for Electric Vehicles", 2022,
        "https://doi.org/10.1049/itr2.12230", None,
        "finite-lattice sliding-mode/QP adaptation", "paper-core adaptation",
        ("course/lateral sliding surfaces", "yaw/stability recovery", "adhesion-aware robust allocation proxy", "fault-tolerant control-rate preference"),
        ("in-wheel-motor fault model and convex wheel-torque allocation cannot be reconstructed from WOMD and are represented by executable-candidate feasibility/adhesion costs",),
        "Robust post-impact stability-control adapter",
    ),
    BaselineProvenance(
        "severity_minimization", ("severity_minimization", "severity_minimization_planner", "unavoidable_collision_planner", "crash_mitigation_planner", "uc_severity_planner"), ("contact",),
        "Motion planning for autonomous vehicles with the inclusion of post-impact motions for minimising collision risk", 2023,
        "https://doi.org/10.1080/00423114.2022.2088396", None, "finite-lattice severity objective", "paper-core adaptation",
        ("relative-speed severity proxy", "collision probability", "penetration/clearance penalty", "post-impact stability and controllability"),
        ("detailed occupant injury/damage models are not available in WOMD", "continuous crash configuration optimization replaced by executable prefixes"),
        "Unavoidable-collision severity minimization",
    ),
)


MAIN_TABLE_BY_REGIME: dict[str, tuple[str, ...]] = {
    "safe": ("gameformer_lite", "plantf", "pluto", "pdm_closed", "pdm_hybrid", "idm"),
    "near": ("marc_lite", "racp_lite", "robust_scenario_mpc", "predictive_safety_filter", "dr_cvar_safety_filter", "conformal_predictive_safety_filter"),
    "contact": ("postimpact_mpc_lite", "post_crash_braking", "postimpact_motion_tvlqr", "post_collision_restoration", "compensatory_postimpact_mpc", "robust_postimpact_control"),
}

LEGACY_OR_DIAGNOSTIC_BY_REGIME: dict[str, tuple[str, ...]] = {
    "safe": ("nominal_replay", "wayformer_bc", "betopnet_lite"),
    "near": ("gameformer_lite", "expected_risk_filter", "cvar_risk_filter", "dro_cvar_filter", "oracle_recovery_filter"),
    "contact": ("severity_minimization",),
}


def find_provenance(name: str) -> BaselineProvenance | None:
    key = str(name).strip().lower()
    for item in REGISTRY:
        if key == item.canonical_name or key in item.aliases:
            return item
    return None


def registry_dict(regimes: Iterable[str] | None = None) -> list[dict]:
    selected = set(str(x).lower() for x in regimes) if regimes is not None else None
    return [x.to_dict() for x in REGISTRY if selected is None or selected.intersection(x.regimes)]
