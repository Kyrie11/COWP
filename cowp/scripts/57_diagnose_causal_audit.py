from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from cowp.core.constants import ProposalSource
from cowp.data.dataset import COWPNpzDataset

WANTED = {
    "scenario/id",
    "cowp/candidates/valid",
    "cowp/candidates/conventional_safe",
    "cowp/candidates/noncoercive_feasible",
    "cowp/candidates/proposal_source",
    "cowp/candidates/audited_pair_count",
    "cowp/candidates/ncf_blocker_count",
    "cowp/critical/valid",
    "cowp/audit/pair_relevant",
    "cowp/audit/relevance_mass",
    "cowp/audit/root_affected",
    "cowp/audit/root_unsafe",
    "cowp/witness/exists",
    "cowp/witness/pair_noncoercive_feasible",
    "cowp/witness/blocker_code",
    "cowp/witness/opr",
    "cowp/witness/tail_burden_excess",
    "cowp/witness/rho",
    "cowp/transport/mode_affected",
    "cowp/transport/mode_conflict",
    "cowp/response/valid",
}


def _source_name(value: int) -> str:
    try:
        return ProposalSource(int(value)).name
    except Exception:
        return f"UNKNOWN_{int(value)}"


def _rate(n: int, d: int) -> float:
    return float(n / max(d, 1))


def main() -> None:
    ap = argparse.ArgumentParser(description="Diagnose v16.8.9 candidate-conditioned causal-audit supervision.")
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--scene-ids", default=None)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    allow = None
    if args.scene_ids:
        allow = {x.strip() for x in Path(args.scene_ids).read_text(encoding="utf-8").splitlines() if x.strip()}
    ds = COWPNpzDataset(args.cache_dir)
    pair = Counter()
    roots = Counter()
    scenes = Counter()
    blocker_codes = Counter()
    source_stats: dict[str, Counter] = defaultdict(Counter)
    relevance_masses: list[float] = []
    audited_counts: list[int] = []
    blocker_counts: list[int] = []
    read_errors: list[dict[str, str]] = []

    for i in range(len(ds)):
        if args.limit > 0 and scenes["rows"] >= args.limit:
            break
        try:
            row = ds.load(i, WANTED)
        except Exception as exc:
            if len(read_errors) < 20:
                read_errors.append({"file": ds.paths[i].name, "error": repr(exc)})
            continue
        sid_arr = np.asarray(row.get("scenario/id", ds.paths[i].stem))
        sid = str(sid_arr.item()) if sid_arr.size == 1 else ds.paths[i].stem
        if allow is not None and sid not in allow:
            continue
        scenes["rows"] += 1
        valid = np.asarray(row.get("cowp/candidates/valid", []), dtype=bool).reshape(-1)
        crit = np.asarray(row.get("cowp/critical/valid", []), dtype=bool).reshape(-1)
        if not len(valid) or not len(crit):
            continue
        base = valid[:, None] & crit[None, :]
        rel = np.asarray(row.get("cowp/audit/pair_relevant", np.ones_like(base)), dtype=bool)[:len(valid), :len(crit)] & base
        exists = np.asarray(row.get("cowp/witness/exists", np.zeros_like(base)), dtype=bool)[:len(valid), :len(crit)]
        pncf = np.asarray(row.get("cowp/witness/pair_noncoercive_feasible", (~exists)), dtype=bool)[:len(valid), :len(crit)]
        code = np.asarray(row.get("cowp/witness/blocker_code", np.zeros_like(base, dtype=np.int32)), dtype=np.int64)[:len(valid), :len(crit)]
        rmass = np.asarray(row.get("cowp/audit/relevance_mass", np.zeros_like(base, dtype=np.float32)), dtype=np.float32)[:len(valid), :len(crit)]
        opr = np.asarray(row.get("cowp/witness/opr", np.ones_like(base, dtype=np.float32)), dtype=np.float32)[:len(valid), :len(crit)]
        tail = np.asarray(row.get("cowp/witness/tail_burden_excess", np.zeros_like(base, dtype=np.float32)), dtype=np.float32)[:len(valid), :len(crit)]
        rho = np.asarray(row.get("cowp/witness/rho", np.zeros_like(base, dtype=np.int32)), dtype=np.int64)[:len(valid), :len(crit)]
        blocker = base & ~pncf
        silent = rel & blocker & ~exists
        irrelevant_blocker = base & ~rel & blocker
        protected = ((rho == 2) | (rho == 3)) & base

        pair.update(
            base=int(base.sum()), relevant=int(rel.sum()), irrelevant=int((base & ~rel).sum()),
            witness=int((exists & base).sum()), blocker=int(blocker.sum()), silent_blocker=int(silent.sum()),
            irrelevant_blocker=int(irrelevant_blocker.sum()), protected=int(protected.sum()),
            protected_relevant=int((protected & rel).sum()), protected_blocker=int((protected & blocker).sum()),
        )
        relevance_masses.extend(rmass[base].astype(float).tolist())
        for c in code[blocker].tolist(): blocker_codes[str(int(c))] += 1

        affected = np.asarray(row.get("cowp/audit/root_affected", []), dtype=bool)
        unsafe = np.asarray(row.get("cowp/audit/root_unsafe", []), dtype=bool)
        if affected.ndim == 3:
            affected = affected[:len(valid), :len(crit)]
            unsafe = unsafe[:len(valid), :len(crit)] if unsafe.shape == affected.shape else np.zeros_like(affected)
            roots["affected"] += int(affected.sum())
            roots["unsafe"] += int((affected & unsafe).sum())
            roots["burden_only"] += int((affected & ~unsafe).sum())
        t_aff = np.asarray(row.get("cowp/transport/mode_affected", []), dtype=bool)
        if affected.ndim == 3 and t_aff.shape == affected.shape:
            roots["transport_affected_mismatch"] += int(np.logical_xor(t_aff, affected).sum())

        resp = np.asarray(row.get("cowp/response/valid", []), dtype=bool)
        if resp.ndim == 4:
            # no current schema uses 4-D valid; retained defensively
            resp = resp.any(axis=-1)
        if resp.ndim == 3:
            resp_pair = resp[:len(valid), :len(crit)].any(axis=-1)
            pair["relevant_with_response"] += int((rel & resp_pair).sum())
            pair["irrelevant_with_response"] += int((base & ~rel & resp_pair).sum())

        ncf = np.asarray(row.get("cowp/candidates/noncoercive_feasible", np.zeros_like(valid)), dtype=bool)[:len(valid)] & valid
        conv = np.asarray(row.get("cowp/candidates/conventional_safe", np.zeros_like(valid)), dtype=bool)[:len(valid)] & valid
        scenes["any_ncf"] += int(ncf.any())
        scenes["any_conv"] += int(conv.any())
        scenes["false_safe_floor"] += int(conv.any() and not ncf.any())
        aud = np.asarray(row.get("cowp/candidates/audited_pair_count", rel.sum(axis=1)), dtype=np.int64)[:len(valid)]
        blk = np.asarray(row.get("cowp/candidates/ncf_blocker_count", blocker.sum(axis=1)), dtype=np.int64)[:len(valid)]
        audited_counts.extend(aud[valid].tolist())
        blocker_counts.extend(blk[valid].tolist())
        src = np.asarray(row.get("cowp/candidates/proposal_source", np.zeros_like(valid)), dtype=np.int64)[:len(valid)]
        for k in np.where(valid)[0]:
            st = source_stats[_source_name(int(src[k]))]
            st["valid"] += 1
            st["ncf"] += int(ncf[k])
            st["conventional_safe"] += int(conv[k])
            st["audited_pairs"] += int(aud[k])
            st["blockers"] += int(blk[k])

    rows = int(scenes["rows"])
    result = {
        "schema_version": "cowp_v16_8_9_causal_audit_diagnostic_v1",
        "cache_dir": str(Path(args.cache_dir).resolve()),
        "num_scenes": rows,
        "read_errors": read_errors,
        "scene_rates": {
            "any_ncf": _rate(scenes["any_ncf"], rows),
            "any_conventional_safe": _rate(scenes["any_conv"], rows),
            "false_safe_lower_bound": _rate(scenes["false_safe_floor"], rows),
        },
        "pair_counts": dict(pair),
        "pair_rates": {
            "relevant": _rate(pair["relevant"], pair["base"]),
            "witness_given_relevant": _rate(pair["witness"], pair["relevant"]),
            "blocker_given_relevant": _rate(pair["blocker"], pair["relevant"]),
            "silent_blocker": _rate(pair["silent_blocker"], pair["base"]),
            "irrelevant_blocker": _rate(pair["irrelevant_blocker"], pair["base"]),
            "burden_only_root_fraction": _rate(roots["burden_only"], roots["affected"]),
            "response_on_relevant_pair": _rate(pair["relevant_with_response"], pair["relevant"]),
            "response_on_irrelevant_pair": _rate(pair["irrelevant_with_response"], pair["irrelevant"]),
        },
        "root_counts": dict(roots),
        "blocker_code_counts": dict(blocker_codes),
        "candidate_stats": {
            "mean_audited_pairs": float(np.mean(audited_counts)) if audited_counts else 0.0,
            "p90_audited_pairs": float(np.percentile(audited_counts, 90)) if audited_counts else 0.0,
            "mean_blockers": float(np.mean(blocker_counts)) if blocker_counts else 0.0,
            "p90_blockers": float(np.percentile(blocker_counts, 90)) if blocker_counts else 0.0,
        },
        "relevance_mass": {
            "mean": float(np.mean(relevance_masses)) if relevance_masses else 0.0,
            "p50": float(np.percentile(relevance_masses, 50)) if relevance_masses else 0.0,
            "p90": float(np.percentile(relevance_masses, 90)) if relevance_masses else 0.0,
        },
        "proposal_source_stats": {k: dict(v) for k, v in sorted(source_stats.items())},
        "integrity": {
            "no_read_errors": not read_errors,
            "no_silent_blockers": pair["silent_blocker"] == 0,
            "no_irrelevant_blockers": pair["irrelevant_blocker"] == 0,
            "transport_affected_matches_audit": roots["transport_affected_mismatch"] == 0,
            "no_responses_for_irrelevant_pairs": pair["irrelevant_with_response"] == 0,
        },
    }
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
