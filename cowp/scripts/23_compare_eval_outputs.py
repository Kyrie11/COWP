from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


_DEFAULT_IGNORE = {
    "prefilter_waymax_shards",
    "jit_waymax_env",
    "jit_waymax_metrics",
    "reuse_waymax_env",
    "shared_model_pass",
    "methods",
}


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _compare(
    reference: Any,
    candidate: Any,
    *,
    path: str,
    atol: float,
    rtol: float,
    ignore_keys: set[str],
    mismatches: list[str],
) -> None:
    if isinstance(reference, dict) and isinstance(candidate, dict):
        ref_keys = {k for k in reference if k not in ignore_keys}
        cand_keys = {k for k in candidate if k not in ignore_keys}
        for key in sorted(ref_keys - cand_keys):
            mismatches.append(f"{path}/{key}: missing from candidate")
        for key in sorted(cand_keys - ref_keys):
            mismatches.append(f"{path}/{key}: extra in candidate")
        for key in sorted(ref_keys & cand_keys):
            _compare(
                reference[key],
                candidate[key],
                path=f"{path}/{key}",
                atol=atol,
                rtol=rtol,
                ignore_keys=ignore_keys,
                mismatches=mismatches,
            )
        return
    if isinstance(reference, list) and isinstance(candidate, list):
        if len(reference) != len(candidate):
            mismatches.append(f"{path}: list length {len(reference)} != {len(candidate)}")
            return
        for index, (ref_item, cand_item) in enumerate(zip(reference, candidate)):
            _compare(
                ref_item,
                cand_item,
                path=f"{path}[{index}]",
                atol=atol,
                rtol=rtol,
                ignore_keys=ignore_keys,
                mismatches=mismatches,
            )
        return
    if _is_number(reference) and _is_number(candidate):
        ref_value = float(reference)
        cand_value = float(candidate)
        if math.isnan(ref_value) and math.isnan(cand_value):
            return
        if not math.isclose(ref_value, cand_value, rel_tol=rtol, abs_tol=atol):
            mismatches.append(f"{path}: {ref_value!r} != {cand_value!r}")
        return
    if reference != candidate:
        mismatches.append(f"{path}: {reference!r} != {candidate!r}")


def compare_eval_outputs(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    *,
    atol: float = 1e-6,
    rtol: float = 1e-6,
    ignore_keys: set[str] | None = None,
) -> list[str]:
    mismatches: list[str] = []
    _compare(
        reference,
        candidate,
        path="$",
        atol=float(atol),
        rtol=float(rtol),
        ignore_keys=set(_DEFAULT_IGNORE if ignore_keys is None else ignore_keys),
        mismatches=mismatches,
    )
    return mismatches


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two COWP evaluation JSON files recursively.")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--ignore-keys", default=",".join(sorted(_DEFAULT_IGNORE)))
    parser.add_argument("--max-mismatches", type=int, default=50)
    args = parser.parse_args()

    with Path(args.reference).open("r", encoding="utf-8") as f:
        reference = json.load(f)
    with Path(args.candidate).open("r", encoding="utf-8") as f:
        candidate = json.load(f)
    ignore_keys = {x.strip() for x in str(args.ignore_keys).split(",") if x.strip()}
    mismatches = compare_eval_outputs(
        reference,
        candidate,
        atol=args.atol,
        rtol=args.rtol,
        ignore_keys=ignore_keys,
    )
    if mismatches:
        print(f"FAIL: {len(mismatches)} mismatches")
        for row in mismatches[: max(int(args.max_mismatches), 1)]:
            print(row)
        raise SystemExit(1)
    print("PASS: evaluation outputs are equivalent within tolerance")


if __name__ == "__main__":
    main()
