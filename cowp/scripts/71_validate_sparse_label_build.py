from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


REQUIRED_KEYS = (
    "scenario/id",
    "cowp/candidates/trajectory",
    "cowp/candidates/valid",
    "cowp/critical/track_index",
    "cowp/natural/traj",
    "cowp/response/traj",
    "cowp/witness/exists",
)


def _read_ids(path: Path) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
            sid = row.get("scenario_id", row.get("id")) if isinstance(row, dict) else None
            value = str(sid) if sid is not None else line.split()[0]
        except Exception:
            value = line.split()[0]
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _latest_profile(profile: Path | None) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    by_sid: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, Any]] = []
    if profile is None or not profile.is_file():
        return by_sid, errors
    with profile.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception as exc:
                if len(errors) < 20:
                    errors.append({"line": lineno, "error": repr(exc)})
                continue
            sid = str(row.get("scenario_id", ""))
            if sid:
                by_sid[sid] = row
    return by_sid, errors


def _check_npz(path: Path, expected_sid: str) -> tuple[bool, str]:
    if not path.is_file():
        return False, "missing"
    try:
        with np.load(path, allow_pickle=True) as data:
            keys = set(data.files)
            missing = [k for k in REQUIRED_KEYS if k not in keys]
            if missing:
                return False, "missing_keys:" + ",".join(missing[:4])
            sid_arr = data["scenario/id"]
            sid = str(sid_arr.item() if getattr(sid_arr, "shape", ()) == () else sid_arr)
            if sid != expected_sid:
                return False, f"scenario_id_mismatch:{sid}"
            # Touch representative small arrays so corrupt/truncated members are detected.
            _ = data["cowp/candidates/valid"].shape
            _ = data["cowp/witness/exists"].shape
    except Exception as exc:
        return False, f"load_error:{type(exc).__name__}:{exc}"
    return True, "ok"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Validate that a sparse allow-list label build reached a terminal state for every requested scene "
            "and, for proposal probes, produced a complete NPZ for every requested scene. This distinguishes "
            "an interrupted/unresolved build from a genuine downstream semantic/model-support failure."
        )
    )
    ap.add_argument("--labels-dir", required=True)
    ap.add_argument("--scene-ids", required=True)
    ap.add_argument("--profile-jsonl", default=None)
    ap.add_argument("--allow-terminal-filtered", action="store_true")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    labels_dir = Path(args.labels_dir)
    scene_ids = _read_ids(Path(args.scene_ids))
    profile_path = Path(args.profile_jsonl) if args.profile_jsonl else None
    profile_by_sid, profile_parse_errors = _latest_profile(profile_path)

    missing_npz: list[str] = []
    corrupt_npz: list[dict[str, str]] = []
    not_seen_in_profile: list[str] = []
    terminal_filtered: list[dict[str, Any]] = []
    terminal_error: list[dict[str, Any]] = []
    terminal_existing_or_written = 0

    for sid in scene_ids:
        row = profile_by_sid.get(sid)
        if row is None:
            not_seen_in_profile.append(sid)
        else:
            status = str(row.get("status", "unknown"))
            if status in {"written", "existing"}:
                terminal_existing_or_written += 1
            elif status == "filtered":
                terminal_filtered.append({
                    "scenario_id": sid,
                    "filter_reason": row.get("filter_reason"),
                    "seconds": row.get("seconds"),
                })
            elif status == "error":
                terminal_error.append({"scenario_id": sid, "error": row.get("error")})
        ok, reason = _check_npz(labels_dir / f"{sid}.npz", sid)
        if not ok:
            if reason == "missing":
                missing_npz.append(sid)
            else:
                corrupt_npz.append({"scenario_id": sid, "reason": reason})

    unexpected_npz = sorted(
        p.stem for p in labels_dir.glob("*.npz")
        if p.is_file() and p.stem not in set(scene_ids)
    )
    unresolved = sorted(set(not_seen_in_profile))
    filtered_ids = {str(x["scenario_id"]) for x in terminal_filtered}
    missing_without_terminal_explanation = sorted(set(missing_npz) - filtered_ids - {str(x["scenario_id"]) for x in terminal_error})

    pipeline_complete = not unresolved and not terminal_error and not corrupt_npz and not missing_without_terminal_explanation
    npz_complete = not missing_npz and not corrupt_npz
    passed = pipeline_complete and (npz_complete or bool(args.allow_terminal_filtered))

    if unresolved:
        failure_class = "pipeline_incomplete_or_requested_scene_not_resolved"
        interpretation = (
            "FAIL: requested scene ids were never observed in the build profile. The sparse build was interrupted, "
            "used the wrong Scenario source/index, or did not scan far enough. This is NOT evidence of a label semantic change."
        )
    elif terminal_error or corrupt_npz:
        failure_class = "build_or_artifact_error"
        interpretation = "FAIL: at least one requested scene reached a build error or produced a corrupt NPZ."
    elif missing_npz:
        failure_class = "terminal_filtered_or_missing_npz"
        interpretation = (
            "FAIL: every requested scene was resolved, but at least one requested proposal-probe scene has no NPZ. "
            "Inspect terminal filter reasons; this is a scientific/semantic probe outcome, not an interrupted scan."
        )
    else:
        failure_class = "none"
        interpretation = "PASS: every requested sparse-probe scene resolved and has an integrity-checked NPZ."

    report = {
        "schema_version": "cowp_sparse_label_build_integrity_v1",
        "pass": passed,
        "pipeline_complete": pipeline_complete,
        "npz_complete": npz_complete,
        "failure_class": failure_class,
        "labels_dir": str(labels_dir.resolve()),
        "scene_ids": str(Path(args.scene_ids).resolve()),
        "profile_jsonl": str(profile_path.resolve()) if profile_path is not None else None,
        "counts": {
            "requested": len(scene_ids),
            "profile_terminal_written_or_existing": terminal_existing_or_written,
            "profile_terminal_filtered": len(terminal_filtered),
            "profile_terminal_error": len(terminal_error),
            "not_seen_in_profile": len(not_seen_in_profile),
            "missing_npz": len(missing_npz),
            "corrupt_npz": len(corrupt_npz),
            "unexpected_npz": len(unexpected_npz),
        },
        "not_seen_in_profile": not_seen_in_profile[:100],
        "terminal_filtered": terminal_filtered[:100],
        "terminal_error": terminal_error[:100],
        "missing_npz": missing_npz[:100],
        "corrupt_npz": corrupt_npz[:100],
        "unexpected_npz": unexpected_npz[:100],
        "profile_parse_errors": profile_parse_errors,
        "interpretation": interpretation,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
