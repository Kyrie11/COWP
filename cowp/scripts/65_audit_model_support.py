from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from cowp.core.constants import MechanismToken, NaturalSource, PriorityRelation, ProposalSource, ResponseSource
from cowp.data.dataset import COWPNpzDataset

WANTED = {
    "cowp/candidates/valid", "cowp/candidates/conventional_safe", "cowp/candidates/false_safe",
    "cowp/candidates/noncoercive_feasible", "cowp/candidates/proposal_source", "cowp/candidates/certificate_valid",
    "cowp/critical/valid", "cowp/critical/mechanism_valid", "cowp/critical/agent_type", "cowp/critical/base_priority",
    "cowp/natural/valid", "cowp/natural/source", "cowp/natural/weight", "cowp/natural/traj",
    "cowp/natural/priority_preserved", "cowp/natural/burden_neutral", "cowp/natural/beta", "cowp/natural/map_evidence_mode",
    "cowp/response/valid", "cowp/response/source", "cowp/response/is_safe", "cowp/response/is_low_burden",
    "cowp/response/burden_total", "cowp/response/root_index",
    "cowp/transport/response_is_min_burden",
    "cowp/audit/pair_relevant", "cowp/audit/root_affected", "cowp/audit/root_unsafe",
    "cowp/audit/canonical_root_weight", "cowp/audit/root_event_interval",
    "cowp/witness/exists", "cowp/witness/token", "cowp/witness/opr", "cowp/witness/tail_burden_excess",
    "cowp/witness/min_safe_burden", "cowp/witness/burden_total", "cowp/witness/burden_components",
    "cowp/witness/pair_noncoercive_feasible", "cowp/witness/rho", "cowp/witness/conflict_interval",
    "cowp/transport/mode_valid", "cowp/transport/mode_conflict", "cowp/transport/mode_affected",
    "cowp/transport/mode_retained_low_safe",
    "cowp/transport/root_low_safe_score", "cowp/transport/root_target_confidence",
    "cowp/transport/root_min_safe_burden", "cowp/transport/canonical_root_weight",
}

VISIBILITY_CONTEXT = {
    # Optional context.  Label-only caches need not contain these fields, while
    # tensor caches do; loading them lets COWPNpzDataset perform exact
    # Scenario-track -> model-row alignment before the post-cache support audit.
    "cowp/critical/track_index", "cowp/critical/track_id",
    "state/id", "womd/state/id", "state/current/valid", "womd/state/current/valid",
    "state/is_sdc", "womd/state/is_sdc",
}



def _binary(c: Counter, name: str, target: np.ndarray, mask: np.ndarray) -> None:
    t = np.asarray(target, dtype=bool)
    m = np.asarray(mask, dtype=bool)
    if t.shape != m.shape:
        return
    c[f"{name}.total"] += int(m.sum())
    c[f"{name}.pos"] += int((m & t).sum())
    c[f"{name}.neg"] += int((m & ~t).sum())


def _finite_values(store: dict[str, list[float]], name: str, arr: np.ndarray, mask: np.ndarray | None = None) -> None:
    x = np.asarray(arr, dtype=np.float64)
    if mask is not None and np.asarray(mask).shape == x.shape:
        x = x[np.asarray(mask, dtype=bool)]
    else:
        x = x.reshape(-1)
    x = x[np.isfinite(x)]
    if x.size:
        # Bounded reservoir by deterministic striding: enough for range/std audit
        if x.size > 4096:
            x = x[np.linspace(0, x.size - 1, 4096, dtype=np.int64)]
        store[name].extend(float(v) for v in x.tolist())
        if len(store[name]) > 200000:
            store[name] = store[name][::2]


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit whether a fresh COWP v16.8.9 label/cache distribution supports the active learned objectives, not only the core binary heads.")
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--sample-scenes", type=int, default=0, help="0 scans all scenes")
    ap.add_argument("--min-class-examples", type=int, default=32)
    ap.add_argument("--min-source-examples", type=int, default=32)
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--max-unauditable-critical-rate", type=float, default=0.01)
    ap.add_argument("--min-certificate-complete-scene-rate", type=float, default=0.98)
    args = ap.parse_args()

    ds = COWPNpzDataset(args.cache_dir)
    if len(ds) == 0:
        raise FileNotFoundError(f"empty cache: {args.cache_dir}")
    if 0 < args.sample_scenes < len(ds):
        indices = sorted(set(np.linspace(0, len(ds) - 1, num=args.sample_scenes, dtype=np.int64).tolist()))
    else:
        indices = list(range(len(ds)))

    c = Counter()
    source = defaultdict(Counter)
    cont: dict[str, list[float]] = defaultdict(list)
    read_errors: list[dict[str, str]] = []
    integrity_errors: list[dict[str, str]] = []
    for i in indices:
        try:
            row = ds.load(i, WANTED | VISIBILITY_CONTEXT)
        except Exception as exc:
            if len(read_errors) < 20:
                read_errors.append({"file": ds.paths[i].name, "error": repr(exc)})
            continue
        missing = [k for k in WANTED if k not in row]
        if missing:
            if len(integrity_errors) < 20:
                integrity_errors.append({"file": ds.paths[i].name, "error": f"missing keys: {missing[:8]}"})
            continue
        cand = np.asarray(row["cowp/candidates/valid"], dtype=bool).reshape(-1)
        cert_cand = np.asarray(row["cowp/candidates/certificate_valid"], dtype=bool).reshape(-1)[:cand.size] & cand
        crit_model = np.asarray(row["cowp/critical/valid"], dtype=bool).reshape(-1)
        crit_selected = np.asarray(row.get("cowp/critical/selected_before_input_mask", crit_model), dtype=bool).reshape(-1)[:crit_model.size]
        mech_model = np.asarray(row["cowp/critical/mechanism_valid"], dtype=bool).reshape(-1)[:crit_model.size]
        mech_label = np.asarray(row.get("cowp/critical/mechanism_valid_before_input_mask", mech_model), dtype=bool).reshape(-1)[:crit_model.size]
        crit = crit_model & mech_model
        if not cand.size or not crit_model.size:
            continue
        c["scenes"] += 1
        c["critical_selected"] += int(crit_selected.sum())
        c["critical_mechanism_valid"] += int((crit_selected & mech_label).sum())
        c["critical_unauditable"] += int((crit_selected & ~mech_label).sum())
        c["critical_input_invisible"] += int((crit_selected & ~crit_model).sum())
        c["scenes_with_input_invisible_critical"] += int(np.any(crit_selected & ~crit_model))
        # Candidate-level certificate validity is the authoritative coverage
        # indicator.  Tensor-cache construction can invalidate a scene if a
        # Scenario-selected critical actor is absent from the model input.
        c["certificate_complete_scenes"] += int(np.all((~cand) | cert_cand))
        conv = np.asarray(row["cowp/candidates/conventional_safe"], dtype=bool)[:cand.size]
        ncf = np.asarray(row["cowp/candidates/noncoercive_feasible"], dtype=bool)[:cand.size]
        fs = np.asarray(row["cowp/candidates/false_safe"], dtype=bool)[:cand.size]
        _binary(c, "candidate_ncf", ncf, cert_cand)
        _binary(c, "candidate_false_safe", fs, cert_cand & conv)
        for x in np.asarray(row["cowp/candidates/proposal_source"], dtype=np.int64)[:cand.size][cand]:
            source["proposal"][int(x)] += 1

        base_pair = cand[:, None] & crit[None, :]
        rel = np.asarray(row["cowp/audit/pair_relevant"], dtype=bool)[:cand.size, :crit.size]
        wit = np.asarray(row["cowp/witness/exists"], dtype=bool)[:cand.size, :crit.size]
        pair_ncf = np.asarray(row["cowp/witness/pair_noncoercive_feasible"], dtype=bool)[:cand.size, :crit.size]
        _binary(c, "pair_relevance", rel, base_pair)
        _binary(c, "witness_on_relevant", wit, base_pair & rel)
        _binary(c, "pair_ncf_on_relevant", pair_ncf, base_pair & rel)
        rho = np.asarray(row["cowp/witness/rho"], dtype=np.int64)[:cand.size, :crit.size]
        protected = base_pair & rel & ((rho == int(PriorityRelation.AGENT_PRIORITY)) | (rho == int(PriorityRelation.EQUAL_OR_NEGOTIATED)))
        if np.any(protected):
            protected_bad = np.any(protected & ~pair_ncf, axis=1)
            protected_audited = np.any(protected, axis=1)
            _binary(c, "protected_candidate_feasible", ~protected_bad, cert_cand & conv & protected_audited)

        token = np.asarray(row["cowp/witness/token"], dtype=np.int64)[:cand.size, :crit.size]
        for x in token[base_pair & rel & wit]:
            source["witness_token"][int(x)] += 1
        _finite_values(cont, "witness_opr", np.asarray(row["cowp/witness/opr"], dtype=np.float32)[:cand.size, :crit.size], base_pair & rel)
        _finite_values(cont, "witness_tail", np.asarray(row["cowp/witness/tail_burden_excess"], dtype=np.float32)[:cand.size, :crit.size], base_pair & rel)
        _finite_values(cont, "witness_min_safe_burden", np.asarray(row["cowp/witness/min_safe_burden"], dtype=np.float32)[:cand.size, :crit.size], base_pair & rel)
        _finite_values(cont, "witness_burden_total", np.asarray(row["cowp/witness/burden_total"], dtype=np.float32)[:cand.size, :crit.size], base_pair & rel & wit)
        wcomp = np.asarray(row["cowp/witness/burden_components"], dtype=np.float32)[:cand.size, :crit.size]
        if wcomp.ndim >= 3:
            comp_mask = np.broadcast_to((base_pair & rel & wit)[..., None], wcomp.shape)
            _finite_values(cont, "witness_burden_components", wcomp, comp_mask)

        nv = np.asarray(row["cowp/natural/valid"], dtype=bool)[:crit.size]
        crit_type = np.asarray(row["cowp/critical/agent_type"], dtype=np.int64).reshape(-1)[:crit.size]
        ns = np.asarray(row["cowp/natural/source"], dtype=np.int64)[:crit.size]
        nw = np.asarray(row["cowp/natural/weight"], dtype=np.float32)[:crit.size]
        nt = np.asarray(row["cowp/natural/traj"], dtype=np.float32)[:crit.size]
        pp = np.asarray(row["cowp/natural/priority_preserved"], dtype=bool)[:crit.size]
        for x in ns[nv & crit[:, None]]:
            source["natural"][int(x)] += 1
        _binary(c, "natural_priority_preserved", pp, nv & crit[:, None] & (ns == int(NaturalSource.PRIO)))
        base_rho = np.asarray(row["cowp/critical/base_priority"], dtype=np.int64).reshape(-1)[:crit.size]
        protected_crit = crit & ((base_rho == int(PriorityRelation.AGENT_PRIORITY)) | (base_rho == int(PriorityRelation.EQUAL_OR_NEGOTIATED)))
        for a in np.where(protected_crit)[0]:
            c["protected_critical_agents"] += 1
            prio_mask = nv[a] & (ns[a] == int(NaturalSource.PRIO))
            if not np.any(prio_mask):
                c["protected_without_prio_root"] += 1
            elif not np.any(pp[a] & prio_mask):
                c["protected_without_priority_preserved_prio_root"] += 1
        evidence = np.asarray(row["cowp/natural/map_evidence_mode"], dtype=np.int64)[:crit.size]
        c["empirical_corridor_roots"] += int((nv & crit[:, None] & (evidence == 2)).sum())
        natural_active = nv & crit[:, None]
        if np.any(natural_active) and not np.all(np.isfinite(nt[natural_active])):
            if len(integrity_errors) < 20:
                integrity_errors.append({"file": ds.paths[i].name, "error": "non-finite valid natural trajectory"})

        # Audit every critical agent, including scenes where *all* critical
        # agents are rootless.  The old np.any(natural_active) guard enclosed
        # this loop and therefore under-counted the exact failure mode the gate
        # was intended to catch.
        nb = np.asarray(row["cowp/natural/burden_neutral"], dtype=np.float32)[:crit.size]
        beta = np.asarray(row["cowp/natural/beta"], dtype=np.float32).reshape(-1)[:crit.size]
        low_eps = 1.0e-6
        for a in np.where(crit)[0]:
            mask = nv[a]
            if not np.any(mask):
                c["critical_without_natural_roots"] += 1
                c["critical_without_low_burden_natural_roots"] += 1
                c["critical_with_lt2_low_burden_natural_roots"] += 1
                source["critical_without_natural_roots_by_type"][int(crit_type[a])] += 1
                source["critical_without_low_burden_natural_roots_by_type"][int(crit_type[a])] += 1
                source["critical_with_lt2_low_burden_natural_roots_by_type"][int(crit_type[a])] += 1
                continue
            w = np.asarray(nw[a, mask], dtype=np.float64)
            if (
                np.any(w < -1e-8)
                or not np.all(np.isfinite(w))
                or not np.isclose(float(w.sum()), 1.0, rtol=1.0e-4, atol=1.0e-4)
            ):
                c["invalid_natural_weight_agents"] += 1
            if int(mask.sum()) < 2:
                c["critical_with_lt2_natural_roots"] += 1
                source["critical_with_lt2_natural_roots_by_type"][int(crit_type[a])] += 1

            # COWP's retained-mass/OPR object is explicitly a *low-burden*
            # natural option set.  Counting arbitrary valid roots is therefore
            # insufficient support: every critical agent needs low-burden
            # natural mass, and at least two roots to avoid an accidental
            # singleton certificate.
            low_mask = mask & np.isfinite(nb[a]) & (nb[a] <= float(beta[a]) + low_eps)
            if not np.any(low_mask):
                c["critical_without_low_burden_natural_roots"] += 1
                source["critical_without_low_burden_natural_roots_by_type"][int(crit_type[a])] += 1
            if int(low_mask.sum()) < 2:
                c["critical_with_lt2_low_burden_natural_roots"] += 1
                source["critical_with_lt2_low_burden_natural_roots_by_type"][int(crit_type[a])] += 1

        rv = np.asarray(row["cowp/response/valid"], dtype=bool)[:cand.size, :crit.size]
        rs = np.asarray(row["cowp/response/source"], dtype=np.int64)[:cand.size, :crit.size]
        safe = np.asarray(row["cowp/response/is_safe"], dtype=bool)[:cand.size, :crit.size]
        low = np.asarray(row["cowp/response/is_low_burden"], dtype=bool)[:cand.size, :crit.size]
        rbur = np.asarray(row["cowp/response/burden_total"], dtype=np.float32)[:cand.size, :crit.size]
        pair_response_mask = rel[..., None] & base_pair[..., None]
        response_mask = rv & pair_response_mask
        # response/valid is an occupancy mask for a fixed-cardinality decoded set,
        # not an active BCE target in the mainline model.  Audit coverage instead:
        # every causally relevant valid pair must populate every response slot so
        # inference can safely treat the R decoded slots as active hypotheses.
        expected_response_slots = np.broadcast_to(pair_response_mask, rv.shape)
        c["response_slots_expected"] += int(expected_response_slots.sum())
        c["response_slots_present"] += int((rv & expected_response_slots).sum())
        c["relevant_pairs_with_incomplete_response_bank"] += int(
            np.sum(pair_response_mask[..., 0] & (rv.sum(axis=-1) != rv.shape[-1]))
        )
        _binary(c, "response_safe", safe, response_mask)
        _binary(c, "response_low_burden", low, response_mask)
        rminflag = np.asarray(row["cowp/transport/response_is_min_burden"], dtype=bool)[:cand.size, :crit.size]
        _binary(c, "response_min_burden", rminflag, response_mask)
        for x in rs[response_mask]:
            source["response"][int(x)] += 1
        _finite_values(cont, "response_burden", rbur, response_mask)
        root_idx = np.asarray(row["cowp/response/root_index"], dtype=np.int64)[:cand.size, :crit.size]
        c["root_indexed_responses"] += int((response_mask & (root_idx >= 0)).sum())

        mv = np.asarray(row["cowp/transport/mode_valid"], dtype=bool)[:cand.size, :crit.size]
        mc = np.asarray(row["cowp/transport/mode_conflict"], dtype=bool)[:cand.size, :crit.size]
        ma = np.asarray(row["cowp/transport/mode_affected"], dtype=bool)[:cand.size, :crit.size]
        mr = np.asarray(row["cowp/transport/mode_retained_low_safe"], dtype=bool)[:cand.size, :crit.size]
        q = np.asarray(row["cowp/transport/root_low_safe_score"], dtype=np.float32)[:cand.size, :crit.size]
        conf = np.asarray(row["cowp/transport/root_target_confidence"], dtype=np.float32)[:cand.size, :crit.size]
        rmin = np.asarray(row["cowp/transport/root_min_safe_burden"], dtype=np.float32)[:cand.size, :crit.size]
        root_mask = mv & rel[..., None] & base_pair[..., None]
        _binary(c, "mode_conflict", mc, root_mask)
        _binary(c, "mode_affected", ma, root_mask)
        _binary(c, "mode_retain", mr, root_mask)
        _binary(c, "root_recovery", q >= 0.5, root_mask & ma)
        confident = root_mask & ma & (conf >= 0.25)
        _finite_values(cont, "root_min_safe_burden", rmin, confident)
        _finite_values(cont, "root_recovery_score", q, root_mask & ma)
        c["confident_affected_roots"] += int(confident.sum())
        if np.any(mc & ~ma & root_mask):
            c["conflict_not_affected_violations"] += int((mc & ~ma & root_mask).sum())

        # Fresh audit interval must agree with unsafe support for explanation labels.
        au = np.asarray(row["cowp/audit/root_unsafe"], dtype=bool)[:cand.size, :crit.size]
        iv = np.asarray(row["cowp/audit/root_event_interval"], dtype=np.int64)[:cand.size, :crit.size]
        if iv.shape[:-1] == au.shape and iv.shape[-1] == 2:
            bad = root_mask & au & ((iv[..., 0] < 0) | (iv[..., 1] < iv[..., 0]))
            c["unsafe_missing_event_interval"] += int(bad.sum())

    minc = max(int(args.min_class_examples), 1)
    mins = max(int(args.min_source_examples), 1)
    binary_names = (
        "candidate_ncf", "candidate_false_safe", "pair_relevance", "witness_on_relevant",
        "pair_ncf_on_relevant", "response_safe", "response_low_burden", "response_min_burden",
        "mode_conflict", "mode_affected", "mode_retain", "root_recovery", "protected_candidate_feasible",
    )
    checks: dict[str, bool] = {
        "no_read_errors": not read_errors,
        "no_integrity_errors": not integrity_errors,
        "auditability_coverage": (float(c["critical_unauditable"]) / max(int(c["critical_selected"]), 1)) <= float(args.max_unauditable_critical_rate),
        "certificate_complete_scene_coverage": (float(c["certificate_complete_scenes"]) / max(int(c["scenes"]), 1)) >= float(args.min_certificate_complete_scene_rate),
        "every_auditable_critical_has_natural_root": int(c["critical_without_natural_roots"]) == 0,
        "every_auditable_critical_has_multi_root_support": int(c["critical_with_lt2_natural_roots"]) == 0,
        "every_auditable_critical_has_low_burden_natural_root": int(c["critical_without_low_burden_natural_roots"]) == 0,
        "every_auditable_critical_has_multi_low_burden_root_support": int(c["critical_with_lt2_low_burden_natural_roots"]) == 0,
        "every_protected_auditable_critical_has_prio_root": int(c["protected_without_prio_root"]) == 0,
        "every_protected_prio_root_is_priority_preserved": int(c["protected_without_priority_preserved_prio_root"]) == 0,
        "natural_weights_valid": int(c["invalid_natural_weight_agents"]) == 0,
        "conflict_subset_affected": int(c["conflict_not_affected_violations"]) == 0,
        "unsafe_event_intervals_complete": int(c["unsafe_missing_event_interval"]) == 0,
        "root_indexed_response_support": int(c["root_indexed_responses"]) >= mins,
        "confident_affected_root_support": int(c["confident_affected_roots"]) >= mins,
        "response_slot_full_coverage": int(c["response_slots_present"]) == int(c["response_slots_expected"]),
        "every_relevant_pair_has_full_response_bank": int(c["relevant_pairs_with_incomplete_response_bank"]) == 0,
    }
    binary_support = {}
    for name in binary_names:
        total, pos, neg = int(c[f"{name}.total"]), int(c[f"{name}.pos"]), int(c[f"{name}.neg"])
        binary_support[name] = {"total": total, "positive": pos, "negative": neg, "positive_rate": float(pos / max(total, 1))}
        checks[f"{name}_non_degenerate"] = pos >= minc and neg >= minc

    natural_sources = {NaturalSource.OBS, NaturalSource.NEU, NaturalSource.PRIO}
    response_sources = {ResponseSource.PRED, ResponseSource.OPT, ResponseSource.EMG}
    for src in natural_sources:
        checks[f"natural_source_{src.name}"] = int(source["natural"][int(src)]) >= mins
    for src in response_sources:
        checks[f"response_source_{src.name}"] = int(source["response"][int(src)]) >= mins
    # Token CE has NONE negatives plus positive mechanism tokens. Require at least
    # two positive mechanisms so the head is not reduced to a binary shortcut;
    # per-token counts remain visible for scientific reporting.
    positive_token_classes = [t for t in MechanismToken if t != MechanismToken.NONE and int(source["witness_token"][int(t)]) >= mins]
    checks["multiple_positive_witness_mechanisms"] = len(positive_token_classes) >= 2

    continuous = {}
    for name, vals in cont.items():
        arr = np.asarray(vals, dtype=np.float64)
        if arr.size:
            continuous[name] = {"count": int(arr.size), "min": float(arr.min()), "max": float(arr.max()), "mean": float(arr.mean()), "std": float(arr.std())}
        else:
            continuous[name] = {"count": 0, "min": None, "max": None, "mean": None, "std": None}
        checks[f"{name}_finite_variable"] = bool(arr.size >= mins and np.all(np.isfinite(arr)) and float(arr.max() - arr.min()) > 1e-6)

    proposal_counts = {p.name: int(source["proposal"][int(p)]) for p in ProposalSource}
    natural_counts = {p.name: int(source["natural"][int(p)]) for p in NaturalSource}
    response_counts = {p.name: int(source["response"][int(p)]) for p in ResponseSource}
    token_counts = {p.name: int(source["witness_token"][int(p)]) for p in MechanismToken}

    passed = all(checks.values()) and not read_errors and not integrity_errors
    report = {
        "schema_version": "cowp_v16_8_13_model_support_audit_v1",
        "cache_dir": str(Path(args.cache_dir).resolve()),
        "inspected_scenes": int(c["scenes"]),
        "min_class_examples": minc,
        "min_source_examples": mins,
        "strict": bool(args.strict),
        "max_unauditable_critical_rate": float(args.max_unauditable_critical_rate),
        "min_certificate_complete_scene_rate": float(args.min_certificate_complete_scene_rate),
        "pass": bool(passed),
        "checks": checks,
        "binary_support": binary_support,
        "source_support": {
            "proposal": proposal_counts, "natural": natural_counts, "response": response_counts,
            "witness_token_positive_pairs": token_counts,
            "critical_without_natural_roots_by_type": dict(source["critical_without_natural_roots_by_type"]),
            "critical_with_lt2_natural_roots_by_type": dict(source["critical_with_lt2_natural_roots_by_type"]),
            "critical_without_low_burden_natural_roots_by_type": dict(source["critical_without_low_burden_natural_roots_by_type"]),
            "critical_with_lt2_low_burden_natural_roots_by_type": dict(source["critical_with_lt2_low_burden_natural_roots_by_type"]),
        },
        "continuous_support": continuous,
        "auxiliary_counts": {
            "critical_selected": int(c["critical_selected"]),
            "critical_mechanism_valid": int(c["critical_mechanism_valid"]),
            "critical_unauditable": int(c["critical_unauditable"]),
            "critical_unauditable_rate": float(c["critical_unauditable"] / max(int(c["critical_selected"]), 1)),
            "critical_input_invisible": int(c["critical_input_invisible"]),
            "critical_input_invisible_rate": float(c["critical_input_invisible"] / max(int(c["critical_selected"]), 1)),
            "scenes_with_input_invisible_critical": int(c["scenes_with_input_invisible_critical"]),
            "certificate_complete_scenes": int(c["certificate_complete_scenes"]),
            "certificate_complete_scene_rate": float(c["certificate_complete_scenes"] / max(int(c["scenes"]), 1)),
            "protected_critical_agents": int(c["protected_critical_agents"]),
            "protected_without_prio_root": int(c["protected_without_prio_root"]),
            "protected_without_priority_preserved_prio_root": int(c["protected_without_priority_preserved_prio_root"]),
            "empirical_corridor_roots": int(c["empirical_corridor_roots"]),
            "root_indexed_responses": int(c["root_indexed_responses"]),
            "confident_affected_roots": int(c["confident_affected_roots"]),
            "critical_without_natural_roots": int(c["critical_without_natural_roots"]),
            "critical_with_lt2_natural_roots": int(c["critical_with_lt2_natural_roots"]),
            "critical_without_low_burden_natural_roots": int(c["critical_without_low_burden_natural_roots"]),
            "critical_with_lt2_low_burden_natural_roots": int(c["critical_with_lt2_low_burden_natural_roots"]),
            "invalid_natural_weight_agents": int(c["invalid_natural_weight_agents"]),
            "conflict_not_affected_violations": int(c["conflict_not_affected_violations"]),
            "unsafe_missing_event_interval": int(c["unsafe_missing_event_interval"]),
            "response_slots_expected": int(c["response_slots_expected"]),
            "response_slots_present": int(c["response_slots_present"]),
            "relevant_pairs_with_incomplete_response_bank": int(c["relevant_pairs_with_incomplete_response_bank"]),
        },
        "read_errors": read_errors,
        "integrity_errors": integrity_errors,
        "interpretation": "Pass means all *auditable* selected critical relations have complete low-burden typed natural support, protected relations carry a true PRIO root, and the explicitly reported unauditable fraction stays below the preregistered coverage cap. Candidate-level certificate labels are evaluated only where certificate_valid=true. This does not replace held-out evaluation or multi-seed training validation.",
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.strict and not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
