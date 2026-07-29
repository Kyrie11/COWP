from __future__ import annotations

import numpy as np

from ocrap.data.schema import DatasetSample, SceneHistory
from ocrap.simulation.observation.bev import unknown_ratio_in_corridor
from ocrap.utils.geometry import agent_state_to_box, compute_ttc, min_box_clearance


def assign_regimes(samples: list[DatasetSample], history: SceneHistory, cfg: dict) -> None:
    if not samples:
        return
    thresholds = cfg.get("regime_thresholds", {})
    tau_high = float(thresholds.get("tau_high", 1.0))
    tau_d = float(thresholds.get("tau_d", 2.0))
    tau_ttc = float(thresholds.get("tau_ttc", 3.0))
    tau_occ = float(thresholds.get("tau_occ", 0.05))
    # ``tau_occ`` is intentionally strict for the occluded-regime tag.  In WOMD
    # BEV crops, however, an unknown corridor ratio above 0.05 is common even for
    # otherwise benign, uniformly sampled frames.  Using the same threshold for
    # normal/NUP tagging makes ``normal`` permanently false on real Waymax/WOMD
    # smoke sets.  Treat occlusion as an overlapping attribute and use a separate,
    # much looser threshold for normal samples.
    tau_normal_occ = float(thresholds.get("tau_normal_occ", 0.75))
    tau_normal_dep = float(thresholds.get("tau_normal_dep", 0.50))
    tau_prefix_hard = float(thresholds.get("tau_prefix_hard", 0.0))
    tau_prefix_harm = float(thresholds.get("tau_prefix_harm", 0.05))
    tau_contact = float(thresholds.get("tau_contact", 0.8))
    require_uniform_for_normal = bool(thresholds.get("require_uniform_for_normal", False))
    include_prefix_collision_in_near = bool(thresholds.get("include_prefix_collision_in_near", False))
    include_prefix_contact_in_post = bool(thresholds.get("include_prefix_contact_in_post", False))
    use_paper_regime_definitions = bool(thresholds.get("use_paper_regime_definitions", True))
    dep_vals = [float(s.r_dep_star) for s in samples]
    min_dep = min(dep_vals)
    max_dep = max(dep_vals)
    nominal_sample = next((s for s in samples if bool(s.is_nominal)), samples[0])
    nominal_dep = float(nominal_sample.r_dep_star)
    nominal_orc = float(nominal_sample.r_orc_star)
    boxes = np.asarray([agent_state_to_box(a) for a in history.agent_history[-1, 1:]], dtype=np.float32) if history.agent_history.shape[1] > 1 else np.zeros((0, 9), dtype=np.float32)
    valids = history.agent_valid[-1, 1:] if history.agent_history.shape[1] > 1 else np.zeros((0,), dtype=bool)
    ego_box = agent_state_to_box(history.agent_history[-1, 0])
    dmin = min_box_clearance(ego_box, boxes, valids) if len(boxes) else 99.0
    ttc = compute_ttc(history.ego_state, boxes, valids) if len(boxes) else 99.0
    occ_ratio = unknown_ratio_in_corridor(history.occ_mask)
    time_reasons = set(str(x) for x in history.metadata.get("time_sampling_reasons", []) if x is not None)
    history_near = bool(dmin < tau_d or ttc < tau_ttc)
    history_contact = bool(dmin < tau_contact)
    # Paper-level regimes are scenario/scene-time attributes.  Prefix-induced
    # collisions are still recorded as hard_violation / harm_proxy, but should
    # not by default turn an otherwise normal scene-time into a near-contact or
    # post-contact regime.  This keeps normal/safe background sets from being
    # mislabeled merely because one deliberately generated candidate prefix is
    # bad.  The legacy behavior can be restored with the include_* flags.
    nominal_prefix_collision = bool(nominal_sample.prefix.diagnostics.get("prefix_collision", False))
    nominal_prefix_contact = bool(nominal_sample.prefix.diagnostics.get("prefix_contact", False))
    nominal_prefix_safe = bool(
        nominal_sample.prefix.feasible
        and float(nominal_sample.prefix.hard_violation) <= tau_prefix_hard
        and float(nominal_sample.prefix.harm_proxy) <= tau_prefix_harm
        and not nominal_prefix_collision
        and not nominal_prefix_contact
    )
    uniform_ok = bool((not require_uniform_for_normal) or ("uniform" in time_reasons))
    nominal_headroom = bool(nominal_dep > tau_normal_dep and nominal_orc > tau_normal_dep)
    scene_normal_anchor = bool(
        (not bool(nominal_sample.i_art_star))
        and nominal_headroom
        and nominal_prefix_safe
        and not history_near
        and not history_contact
        and occ_ratio <= tau_normal_occ
        and uniform_ok
    )
    if use_paper_regime_definitions:
        scene_low_headroom = bool((not scene_normal_anchor) and max_dep > 0.0 and nominal_dep <= tau_high)
    else:
        scene_low_headroom = bool(nominal_dep <= tau_high and max_dep > 0.0)
    for s in samples:
        prefix_collision = bool(s.prefix.diagnostics.get("prefix_collision", False))
        prefix_contact = bool(s.prefix.diagnostics.get("prefix_contact", False))
        near = bool(history_near or (include_prefix_collision_in_near and prefix_collision))
        # Keep the public paper-level ``post_contact`` union for backward
        # compatibility, but expose its two semantically different sources.
        #
        # * observed: the logged history is already in contact (or, when the
        #   legacy flag is enabled, the candidate prefix itself makes contact);
        # * counterfactual: at least one generated future is explicitly tagged
        #   as a contact-surrogate recovery branch.
        #
        # This separation is required for clean contact-surrogate datasets:
        # they may require counterfactual contact while forbidding logged/prefix
        # contact, instead of silently accepting a mixture of both.
        post_observed = bool(
            history_contact or (include_prefix_contact_in_post and prefix_contact)
        )
        post_counterfactual = bool(
            any(bool(f.metadata.get("contact_surrogate", False)) for f in s.futures)
        )
        post = bool(post_observed or post_counterfactual)
        prefix_safe = bool(
            s.prefix.feasible
            and float(s.prefix.hard_violation) <= tau_prefix_hard
            and float(s.prefix.harm_proxy) <= tau_prefix_harm
        )
        sample_headroom = bool(float(s.r_dep_star) > tau_normal_dep and float(s.r_orc_star) > tau_normal_dep)
        normal = bool(
            scene_normal_anchor
            and (not s.i_art_star)
            and sample_headroom
            and prefix_safe
            and not near
            and not post
        )
        regimes = {
            "normal": normal,
            "low_headroom": scene_low_headroom,
            "occluded": bool(occ_ratio > tau_occ),
            "near_contact": near,
            "post_contact": post,
            "post_contact_observed": post_observed,
            "post_contact_counterfactual": post_counterfactual,
            "oracle_artifact": bool(s.i_art_star),
        }
        # Extra diagnostics for auditing; downstream diagnose ignores unknown
        # keys unless explicitly requested.
        regimes["prefix_collision"] = prefix_collision
        regimes["prefix_contact"] = prefix_contact
        s.regime_label.clear()
        s.regime_label.update(regimes)
