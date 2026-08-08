from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from cowp.data.dataset import COWPNpzDataset

WANTED = {
    "cowp/candidates/valid",
    "cowp/candidates/conventional_safe",
    "cowp/candidates/noncoercive_feasible",
    "cowp/critical/valid",
    "cowp/audit/pair_relevant",
    "cowp/witness/exists",
    "cowp/witness/pair_noncoercive_feasible",
    "cowp/witness/rho",
    "cowp/transport/mode_valid",
    "cowp/transport/mode_conflict",
    "cowp/transport/mode_affected",
    "cowp/transport/root_low_safe_score",
    "cowp/transport/root_target_confidence",
    "cowp/audit/root_burden_only_affected",
}


def _binary(counter: Counter, name: str, target: np.ndarray, mask: np.ndarray) -> None:
    x = np.asarray(target, dtype=bool)
    m = np.asarray(mask, dtype=bool)
    if x.shape != m.shape:
        return
    counter[f"{name}_total"] += int(m.sum())
    counter[f"{name}_positive"] += int((m & x).sum())
    counter[f"{name}_negative"] += int((m & ~x).sum())


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit whether a fresh COWP cache contains non-degenerate supervision for every core learned head.")
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--sample-scenes", type=int, default=0, help="0 scans all files")
    ap.add_argument("--min-class-examples", type=int, default=32)
    ap.add_argument("--strict", action="store_true", help="exit non-zero when a core head lacks both classes")
    args = ap.parse_args()

    ds = COWPNpzDataset(args.cache_dir)
    if len(ds) == 0:
        raise FileNotFoundError(f"empty cache: {args.cache_dir}")
    if args.sample_scenes > 0 and args.sample_scenes < len(ds):
        indices = sorted(set(np.linspace(0, len(ds) - 1, num=args.sample_scenes, dtype=np.int64).tolist()))
    else:
        indices = list(range(len(ds)))

    c = Counter()
    read_errors: list[dict[str, str]] = []
    for i in indices:
        try:
            row = ds.load(i, WANTED)
        except Exception as exc:
            if len(read_errors) < 20:
                read_errors.append({"file": ds.paths[i].name, "error": repr(exc)})
            continue
        cand = np.asarray(row.get("cowp/candidates/valid", []), dtype=bool).reshape(-1)
        crit = np.asarray(row.get("cowp/critical/valid", []), dtype=bool).reshape(-1)
        if not len(cand) or not len(crit):
            continue
        c["scenes"] += 1
        conv = np.asarray(row.get("cowp/candidates/conventional_safe", np.zeros_like(cand)), dtype=bool)[: len(cand)]
        ncf = np.asarray(row.get("cowp/candidates/noncoercive_feasible", np.zeros_like(cand)), dtype=bool)[: len(cand)]
        _binary(c, "candidate_conventional_safe", conv, cand)
        _binary(c, "candidate_ncf", ncf, cand)

        base_pair = cand[:, None] & crit[None, :]
        rel = np.asarray(row.get("cowp/audit/pair_relevant", np.zeros_like(base_pair)), dtype=bool)[: len(cand), : len(crit)]
        _binary(c, "pair_relevance", rel, base_pair)
        witness = np.asarray(row.get("cowp/witness/exists", np.zeros_like(base_pair)), dtype=bool)[: len(cand), : len(crit)]
        pair_ncf = np.asarray(row.get("cowp/witness/pair_noncoercive_feasible", np.ones_like(base_pair)), dtype=bool)[: len(cand), : len(crit)]
        _binary(c, "witness_on_relevant", witness, base_pair & rel)
        _binary(c, "pair_ncf_on_relevant", pair_ncf, base_pair & rel)

        rho = np.asarray(row.get("cowp/witness/rho", np.zeros_like(base_pair, dtype=np.int32)), dtype=np.int64)[: len(cand), : len(crit)]
        protected = base_pair & ((rho == 2) | (rho == 3))
        c["protected_pair_total"] += int(protected.sum())
        c["protected_relevant_total"] += int((protected & rel).sum())

        mv = np.asarray(row.get("cowp/transport/mode_valid", []), dtype=bool)
        mc = np.asarray(row.get("cowp/transport/mode_conflict", []), dtype=bool)
        ma = np.asarray(row.get("cowp/transport/mode_affected", []), dtype=bool)
        q = np.asarray(row.get("cowp/transport/root_low_safe_score", []), dtype=np.float32)
        conf = np.asarray(row.get("cowp/transport/root_target_confidence", []), dtype=np.float32)
        bo = np.asarray(row.get("cowp/audit/root_burden_only_affected", []), dtype=bool)
        if mv.ndim == 3 and mc.shape == mv.shape and ma.shape == mv.shape:
            rel_mode = mv & rel[..., None]
            _binary(c, "mode_conflict_on_relevant", mc, rel_mode)
            _binary(c, "mode_affected_on_relevant", ma, rel_mode)
            if bo.shape == mv.shape:
                c["burden_only_affected_roots"] += int((bo & rel_mode).sum())
            if q.shape == mv.shape:
                affected_mode = rel_mode & ma
                # q is a soft [0,1] root recovery score.  Split at 0.5 only for
                # class-support diagnostics; the actual loss keeps the continuous target.
                _binary(c, "root_recovery_on_affected", q >= 0.5, affected_mode)
                if conf.shape == mv.shape:
                    c["confident_affected_roots"] += int((affected_mode & (conf >= 0.25)).sum())

    min_examples = int(max(args.min_class_examples, 1))
    core = (
        "candidate_ncf",
        "pair_relevance",
        "witness_on_relevant",
        "pair_ncf_on_relevant",
        "mode_conflict_on_relevant",
        "mode_affected_on_relevant",
        "root_recovery_on_affected",
    )
    checks = {}
    rates = {}
    for name in core:
        pos = int(c[f"{name}_positive"])
        neg = int(c[f"{name}_negative"])
        total = int(c[f"{name}_total"])
        checks[f"{name}_non_degenerate"] = pos >= min_examples and neg >= min_examples
        rates[name] = {
            "total": total,
            "positive": pos,
            "negative": neg,
            "positive_rate": float(pos / max(total, 1)),
        }
    # A tiny smoke should not be rejected merely because a rare auxiliary subset
    # has <32 examples.  Full-data strict mode is where class support becomes a
    # training-readiness hard gate.
    passed = (not read_errors) and (all(checks.values()) if args.strict else True)
    result = {
        "schema_version": "cowp_v16_8_9_training_supervision_audit_v1",
        "cache_dir": str(Path(args.cache_dir).resolve()),
        "inspected_scenes": int(c["scenes"]),
        "read_errors": read_errors,
        "min_class_examples": min_examples,
        "strict": bool(args.strict),
        "class_support": rates,
        "auxiliary_counts": {
            "protected_pair_total": int(c["protected_pair_total"]),
            "protected_relevant_total": int(c["protected_relevant_total"]),
            "burden_only_affected_roots": int(c["burden_only_affected_roots"]),
            "confident_affected_roots": int(c["confident_affected_roots"]),
        },
        "checks": checks,
        "pass": bool(passed),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.strict and not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
