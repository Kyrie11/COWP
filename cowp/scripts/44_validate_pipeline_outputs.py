from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric_ok(delta: dict[str, Any], names: tuple[str, ...]) -> bool:
    for name in names:
        row = delta.get(name)
        if isinstance(row, dict) and row.get("reference") is not None and row.get("candidate") is not None:
            return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate that a COWP run produced usable checkpoints and closed-loop metrics.")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--level", choices=("natural", "planner", "probe", "full"), required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    root = Path(args.out_root)
    checks: dict[str, bool] = {}
    errors: list[str] = []

    def require_file(key: str, rel: str) -> Path:
        path = root / rel
        ok = path.is_file() and path.stat().st_size > 0
        checks[key] = ok
        if not ok:
            errors.append(f"missing or empty: {path}")
        return path

    require_file("natural_checkpoint", "checkpoints/natural/cowp_natural_best.pt")
    require_file("natural_history", "checkpoints/natural/history_natural.json")
    require_file("natural_basis_gate", "eval/learned_offline/natural_basis_gate.json")
    require_file("natural_effectiveness_gate", "eval/learned_offline/natural_effectiveness_gate.json")

    if args.level in {"planner", "probe", "full"}:
        require_file("transport_checkpoint", "checkpoints/transport/cowp_witness_best.pt")
        require_file("planner_checkpoint", "checkpoints/planner/cowp_planner_best.pt")
        require_file("mechanism_verification", "eval/learned_offline/mechanism_verification.json")

    if args.level == "probe":
        probe_delta = require_file("probe_delta", "eval/probe/delta_conventional_vs_root_transport.json")
        if probe_delta.is_file():
            try:
                d = _load(probe_delta)
                for metric, aliases in {
                    "CR": ("CR",),
                    "offroad": ("OffroadRate", "Offroad"),
                    "progress": ("EP", "ProgressionMetric"),
                }.items():
                    ok = _metric_ok(d, aliases)
                    checks[f"probe_metric_{metric}"] = ok
                    if not ok:
                        errors.append(f"probe delta lacks usable {metric}: {probe_delta}")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"invalid probe delta JSON: {exc}")

    if args.level == "full":
        full_delta = require_file("full_delta", "eval/waymax/delta_conventional_vs_cowp.json")
        require_file("full_delta_vs_planner", "eval/waymax/delta_planner_vs_cowp.json")
        if full_delta.is_file():
            try:
                d = _load(full_delta)
                for metric, aliases in {
                    "CR": ("CR",),
                    "offroad": ("OffroadRate", "Offroad"),
                    "progress": ("EP", "ProgressionMetric"),
                }.items():
                    ok = _metric_ok(d, aliases)
                    checks[f"full_metric_{metric}"] = ok
                    if not ok:
                        errors.append(f"full delta lacks usable {metric}: {full_delta}")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"invalid full delta JSON: {exc}")

    report = {"pass": not errors, "level": args.level, "checks": checks, "errors": errors}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
