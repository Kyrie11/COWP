from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable


def _read_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    if path.is_dir():
        ids.update(p.stem for p in path.glob("*.npz"))
        return ids
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                row = None
            if isinstance(row, dict):
                sid = row.get("scenario_id", row.get("id"))
                if sid is not None:
                    ids.add(str(sid))
                    continue
            ids.add(line.split()[0])
    return ids


def _iter_index_ids(index_jsonl: Path) -> Iterable[str]:
    with index_jsonl.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sid = row.get("scenario_id")
            if sid is None:
                raise ValueError(f"{index_jsonl}:{line_no}: missing scenario_id")
            yield str(sid)


def _rank(seed: str, scenario_id: str) -> bytes:
    return hashlib.sha256(f"{seed}\0{scenario_id}".encode("utf-8")).digest()


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Create a deterministic uniform-by-scenario hash holdout from a complete WOMD index. "
            "Use --exclude repeatedly to keep development/probe/cache scenes out of the frozen evaluation set."
        )
    )
    ap.add_argument("--index-jsonl", required=True)
    ap.add_argument("--output-ids", required=True)
    ap.add_argument("--output-manifest", required=True)
    ap.add_argument("--count", type=int, required=True)
    ap.add_argument("--seed", default="cowp-publication-holdout-v1")
    ap.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="txt/jsonl ID file or cache/label directory containing *.npz; repeat as needed.",
    )
    args = ap.parse_args()

    if args.count <= 0:
        raise SystemExit("--count must be positive")

    index_path = Path(args.index_jsonl)
    seen: set[str] = set()
    universe: list[str] = []
    duplicate_index_ids = 0
    for sid in _iter_index_ids(index_path):
        if sid in seen:
            duplicate_index_ids += 1
            continue
        seen.add(sid)
        universe.append(sid)

    excluded: set[str] = set()
    exclude_breakdown: dict[str, int] = {}
    for raw in args.exclude:
        p = Path(raw)
        ids = _read_ids(p)
        exclude_breakdown[str(p)] = len(ids)
        excluded.update(ids)

    eligible = [sid for sid in universe if sid not in excluded]
    eligible.sort(key=lambda sid: (_rank(str(args.seed), sid), sid))
    if len(eligible) < args.count:
        raise SystemExit(
            f"not enough eligible scenarios: requested={args.count}, eligible={len(eligible)}, "
            f"index_unique={len(universe)}, excluded_overlap={len(set(universe) & excluded)}"
        )
    selected = eligible[: args.count]

    output_ids = Path(args.output_ids)
    output_ids.parent.mkdir(parents=True, exist_ok=True)
    output_ids.write_text("".join(f"{sid}\n" for sid in selected), encoding="utf-8")

    selected_digest = hashlib.sha256("\n".join(selected).encode("utf-8")).hexdigest()
    payload = {
        "schema_version": "cowp_hash_holdout_manifest_v1",
        "selection": "sha256(seed\\0scenario_id), ascending; uniform deterministic sample over unique index rows",
        "seed": str(args.seed),
        "index_jsonl": str(index_path),
        "index_unique_scenarios": len(universe),
        "duplicate_index_ids": duplicate_index_ids,
        "exclude_sources": exclude_breakdown,
        "excluded_unique_total": len(excluded),
        "excluded_overlap_with_index": len(set(universe) & excluded),
        "eligible_scenarios": len(eligible),
        "selected_count": len(selected),
        "selected_ids_sha256": selected_digest,
        "output_ids": str(output_ids),
        "leakage_checks": {
            "selected_intersects_excluded": bool(set(selected) & excluded),
            "selected_unique": len(set(selected)) == len(selected),
        },
    }
    if payload["leakage_checks"]["selected_intersects_excluded"] or not payload["leakage_checks"]["selected_unique"]:
        raise RuntimeError("holdout leakage/uniqueness invariant failed")

    manifest = Path(args.output_manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
