from __future__ import annotations

"""Publication/reference metadata for the WOMD external-baseline adapters.

The names intentionally distinguish a source-faithful *cross-domain adaptation*
from an author's native benchmark implementation.  This is important for paper
claims: GameFormer does not release WOMD closed-loop planning code; DTPP/PLUTO/
PDM-Closed are native nuPlan planners; PlanT 2.0 is native CARLA.
"""

BASELINE_REFERENCE_METADATA = {
    "gameformer": {
        "paper": "GameFormer: Game-theoretic Modeling and Learning of Transformer-based Interactive Prediction and Planning for Autonomous Driving",
        "citation_key": "huang2023gameformer",
        "native_domain": "WOMD open-loop / interaction prediction; nuPlan planner exists separately",
        "adapter_label": "GameFormer-WOMD/Waymax adaptation",
        "fidelity": "high-model / adapted-closed-loop",
        "claim": "Hierarchical interactive GameFormer network and multimodal direct ego trajectory head, retrained on the matched WOMD split. The Waymax closed-loop wrapper is ours because the official WOMD repository explicitly does not release WOMD closed-loop planning code.",
    },
    "dtpp": {
        "paper": "DTPP: Differentiable Joint Conditional Prediction and Cost Evaluation for Tree Policy Planning in Autonomous Driving",
        "citation_key": "huang2024dtpp",
        "native_domain": "nuPlan",
        "adapter_label": "DTPP-WOMD shared-proposal-tree adaptation",
        "fidelity": "high-network / medium-planner",
        "claim": "DTPP encoder, ego-conditioned prediction/cost structure and learned score are retained, but the native nuPlan spline/tree proposal constructor is replaced by the matched WOMD/COWP proposal tree. Report this as an adaptation, not the authors' native DTPP planner.",
    },
    "pluto": {
        "paper": "PLUTO: Pushing the Limit of Imitation Learning-based Planning for Autonomous Driving",
        "citation_key": "cheng2024pluto",
        "native_domain": "nuPlan",
        "adapter_label": "PLUTO-style WOMD adaptation",
        "fidelity": "paper-idea / simplified-clean-room",
        "claim": "Clean-room WOMD implementation preserving vector scene encoding, factorized longitudinal/lateral queries, multimodal imitation, auxiliary imitation and optional contrastive consistency. It is not a line-by-line port of the official nuPlan PLUTO code and must be labeled PLUTO-style/adaptation in the paper.",
    },
    "plant2": {
        "paper": "PlanT 2.0: Exposing Biases and Structural Flaws in Closed-Loop Driving",
        "citation_key": "gerstenecker2025plant20exposingbiases",
        "native_domain": "CARLA Leaderboard 2.0 / Bench2Drive",
        "adapter_label": "PlanT2-style WOMD adaptation",
        "fidelity": "paper-idea / simplified-clean-room",
        "claim": "Object-centric transformer planning abstraction is retained with WOMD actor/route/map tokens, but CARLA-specific PlanT 2.0 data, tokenization, control and training stack are not reproduced. Label as PlanT2-style/adaptation.",
    },
    "pdm_closed": {
        "paper": "Parting with Misconceptions about Learning-based Vehicle Motion Planning",
        "citation_key": "dauner2023parting",
        "native_domain": "nuPlan",
        "adapter_label": "PDM-Closed-style WOMD adaptation",
        "fidelity": "medium-concept / adapted-proposals",
        "claim": "Predictive rule-based safety/progress/comfort scoring is applied to the matched WOMD local proposal bank. The official PDM-Closed centerline proposal/observation stack is nuPlan-specific and is not reproduced exactly; do not call this the native official PDM-Closed implementation.",
    },
}


def baseline_reference_metadata(name: str) -> dict:
    return dict(BASELINE_REFERENCE_METADATA.get(str(name).lower(), {}))
