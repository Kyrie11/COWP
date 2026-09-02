from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from cowp.data.dataset import COWPNpzDataset


WANTED = {
    "cowp/candidates/valid",
    "cowp/candidates/certificate_valid",
    "cowp/candidates/conventional_safe",
    "cowp/candidates/priority_eligible",
    "cowp/candidates/priority_noncoercive_feasible",
    "cowp/candidates/priority_false_safe",
    "cowp/critical/valid",
    "cowp/critical/mechanism_valid",
    "cowp/witness/rho",
    "cowp/witness/opr",
    "cowp/transport/transported_opr",
    "cowp/transport/mode_valid",
    "cowp/transport/mode_affected",
    "cowp/transport/root_low_safe_score",
}


def _sample_indices(n: int, limit: int) -> list[int]:
    if limit <= 0 or limit >= n:
        return list(range(n))
    return sorted(set(np.linspace(0, n - 1, num=limit, dtype=np.int64).tolist()))


def _bool1(row: dict, key: str, n: int, default: bool = False) -> np.ndarray:
    arr = row.get(key)
    if arr is None:
        return np.full(n, default, dtype=bool)
    return np.asarray(arr, dtype=bool).reshape(-1)[:n]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Audit within-scene/root contrast needed by the six-layer COWP mechanism. "
            "Unlike prevalence gates, this asks whether the cache contains the actual "
            "intervention contrasts learned by root viability/recovery/option transport."
        )
    )
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--sample-scenes", type=int, default=0)
    ap.add_argument("--opr-alpha", type=float, default=0.35)
    ap.add_argument("--recovery-threshold", type=float, default=0.50)
    ap.add_argument("--partial-opr-low", type=float, default=0.05)
    ap.add_argument("--partial-opr-high", type=float, default=0.95)
    ap.add_argument("--min-rankable-scenes", type=int, default=128)
    ap.add_argument("--min-rank-pairs", type=int, default=2048)
    ap.add_argument("--min-viability-switch-scenes", type=int, default=128)
    ap.add_argument("--min-viability-switch-roots", type=int, default=1024)
    ap.add_argument("--min-recovery-switch-scenes", type=int, default=32)
    ap.add_argument("--min-recovery-switch-roots", type=int, default=128)
    ap.add_argument("--min-opr-switch-scenes", type=int, default=128)
    ap.add_argument("--min-partial-opr-pairs", type=int, default=512)
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    ds = COWPNpzDataset(args.cache_dir)
    if len(ds) == 0:
        raise FileNotFoundError(f"empty cache: {args.cache_dir}")
    indices = _sample_indices(len(ds), int(args.sample_scenes))
    c: Counter[str] = Counter()
    read_errors: list[dict[str, str]] = []
    missing_keys: Counter[str] = Counter()

    for i in indices:
        try:
            row = ds.load(i, WANTED)
        except Exception as exc:
            if len(read_errors) < 20:
                read_errors.append({"file": ds.paths[i].name, "error": repr(exc)})
            continue

        cand = np.asarray(row.get("cowp/candidates/valid", []), dtype=bool).reshape(-1)
        crit_sel = np.asarray(row.get("cowp/critical/valid", []), dtype=bool).reshape(-1)
        if cand.size == 0 or crit_sel.size == 0:
            c["empty_shape_scenes"] += 1
            continue
        K, A = cand.size, crit_sel.size
        cert = _bool1(row, "cowp/candidates/certificate_valid", K, True) & cand
        conv = _bool1(row, "cowp/candidates/conventional_safe", K, False)
        pelig = _bool1(row, "cowp/candidates/priority_eligible", K, False)
        pncf = _bool1(row, "cowp/candidates/priority_noncoercive_feasible", K, False)
        pfs = _bool1(row, "cowp/candidates/priority_false_safe", K, False)
        primary = cert & conv & pelig
        pos = primary & pncf & ~pfs
        neg = primary & pfs
        npos, nneg = int(pos.sum()), int(neg.sum())
        c["scenes"] += 1
        c["protected_disc_candidates"] += npos + nneg
        c["protected_ncf_candidates"] += npos
        c["protected_false_safe_candidates"] += nneg
        if npos and nneg:
            c["rankable_scenes"] += 1
            c["rank_pairs"] += npos * nneg

        crit = crit_sel & np.asarray(
            row.get("cowp/critical/mechanism_valid", crit_sel), dtype=bool
        ).reshape(-1)[:A]
        if not crit.any():
            continue

        mv = np.asarray(row.get("cowp/transport/mode_valid", []), dtype=bool)
        ma = np.asarray(row.get("cowp/transport/mode_affected", []), dtype=bool)
        q = np.asarray(row.get("cowp/transport/root_low_safe_score", []), dtype=np.float32)
        if mv.ndim != 3 or mv.shape[:2] != (K, A):
            missing_keys["cowp/transport/mode_valid_shape"] += 1
        elif ma.shape != mv.shape:
            missing_keys["cowp/transport/mode_affected_shape"] += 1
        else:
            base = cand[:, None, None] & crit[None, :, None] & mv
            has_affected = (base & ma).any(axis=0)
            has_unaffected = (base & ~ma).any(axis=0)
            viability_switch = has_affected & has_unaffected
            vs = int(viability_switch.sum())
            c["viability_switch_roots"] += vs
            c["root_support_total"] += int(base.any(axis=0).sum())
            if vs:
                c["viability_switch_scenes"] += 1

            if q.shape == mv.shape:
                affected_support = base & ma
                rec_pos = (affected_support & (q >= float(args.recovery_threshold))).any(axis=0)
                rec_neg = (affected_support & (q < float(args.recovery_threshold))).any(axis=0)
                recovery_switch = rec_pos & rec_neg
                rs = int(recovery_switch.sum())
                c["recovery_switch_roots"] += rs
                c["affected_root_support_total"] += int(affected_support.any(axis=0).sum())
                if rs:
                    c["recovery_switch_scenes"] += 1
            else:
                missing_keys["cowp/transport/root_low_safe_score_shape"] += 1

        rho = np.asarray(row.get("cowp/witness/rho", []), dtype=np.int64)
        opr_raw = row.get("cowp/transport/transported_opr")
        if opr_raw is None:
            opr_raw = row.get("cowp/witness/opr")
        opr = np.asarray(opr_raw if opr_raw is not None else [], dtype=np.float32)
        if rho.shape[:2] != (K, A):
            missing_keys["cowp/witness/rho_shape"] += 1
            continue
        if opr.shape[:2] != (K, A):
            missing_keys["cowp/witness/opr_shape"] += 1
            continue
        protected_pair = cand[:, None] & crit[None, :] & ((rho == 2) | (rho == 3))
        finite = protected_pair & np.isfinite(opr)
        low = finite & (opr < float(args.opr_alpha))
        high = finite & (opr >= float(args.opr_alpha))
        opr_switch_agent = low.any(axis=0) & high.any(axis=0)
        osw = int(opr_switch_agent.sum())
        c["opr_switch_agents"] += osw
        if osw:
            c["opr_switch_scenes"] += 1
        partial = finite & (opr > float(args.partial_opr_low)) & (opr < float(args.partial_opr_high))
        c["partial_opr_pairs"] += int(partial.sum())
        c["protected_opr_pairs"] += int(finite.sum())

    checks = {
        "no_read_errors": not read_errors,
        "required_shapes_complete": not missing_keys,
        "protected_rankable_scene_support": c["rankable_scenes"] >= int(args.min_rankable_scenes),
        "protected_rank_pair_support": c["rank_pairs"] >= int(args.min_rank_pairs),
        "candidate_induced_viability_switch_scene_support": c["viability_switch_scenes"] >= int(args.min_viability_switch_scenes),
        "candidate_induced_viability_switch_root_support": c["viability_switch_roots"] >= int(args.min_viability_switch_roots),
        "same_root_recovery_switch_scene_support": c["recovery_switch_scenes"] >= int(args.min_recovery_switch_scenes),
        "same_root_recovery_switch_root_support": c["recovery_switch_roots"] >= int(args.min_recovery_switch_roots),
        "option_mass_switch_scene_support": c["opr_switch_scenes"] >= int(args.min_opr_switch_scenes),
        "partial_option_mass_support": c["partial_opr_pairs"] >= int(args.min_partial_opr_pairs),
    }
    passed = all(checks.values())
    scenes = max(int(c["scenes"]), 1)
    result = {
        "schema_version": "cowp_v16_8_22_mechanism_contrast_audit_v1",
        "cache_dir": str(Path(args.cache_dir).resolve()),
        "inspected_scenes": int(c["scenes"]),
        "read_errors": read_errors,
        "missing_or_bad_shapes": dict(missing_keys),
        "thresholds": {
            "opr_alpha": float(args.opr_alpha),
            "recovery_threshold": float(args.recovery_threshold),
            "min_rankable_scenes": int(args.min_rankable_scenes),
            "min_rank_pairs": int(args.min_rank_pairs),
            "min_viability_switch_scenes": int(args.min_viability_switch_scenes),
            "min_viability_switch_roots": int(args.min_viability_switch_roots),
            "min_recovery_switch_scenes": int(args.min_recovery_switch_scenes),
            "min_recovery_switch_roots": int(args.min_recovery_switch_roots),
            "min_opr_switch_scenes": int(args.min_opr_switch_scenes),
            "min_partial_opr_pairs": int(args.min_partial_opr_pairs),
        },
        "counts": {k: int(v) for k, v in sorted(c.items())},
        "rates": {
            "rankable_scene_rate": float(c["rankable_scenes"] / scenes),
            "viability_switch_scene_rate": float(c["viability_switch_scenes"] / scenes),
            "recovery_switch_scene_rate": float(c["recovery_switch_scenes"] / scenes),
            "opr_switch_scene_rate": float(c["opr_switch_scenes"] / scenes),
            "partial_opr_pair_rate": float(c["partial_opr_pairs"] / max(c["protected_opr_pairs"], 1)),
        },
        "checks": checks,
        "pass": bool(passed),
        "interpretation": (
            "PASS: the cache contains within-scene/root intervention contrasts for Layers 2-5; "
            "training readiness is not inferred from population prevalence alone."
            if passed else
            "FAIL: one or more six-layer mechanisms lack within-scene/root contrast support. "
            "Do not compensate by changing a population-rate threshold; inspect the failed contrast."
        ),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.strict and not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
