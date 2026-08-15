from __future__ import annotations

import argparse
import importlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

EXPECTED_V16_8_16_CODE_FINGERPRINT = "f8ce814303af601e56be4ac830a93fc167931fa51ceb5ae2c153502bd7d808a3"
EXPECTED_V16_8_16_LABEL_SEMANTIC_FINGERPRINT = "adcea5cb927d4c06c7f667725ce1c5b7b62808d6bd2e84244149d01ab25a1fa0"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _run(cmd: list[str]) -> int:
    print("[reaudit]", " ".join(cmd), flush=True)
    return int(subprocess.run(cmd, check=False).returncode)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Re-audit a v16.8.16 smoke label cache under the v16.8.18 support/promotion policy without rebuilding labels."
    )
    ap.add_argument("--source-smoke-root", required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--max-unauditable-critical-rate", type=float, default=0.05)
    ap.add_argument("--min-certificate-complete-scene-rate", type=float, default=0.75)
    ap.add_argument("--min-protected-prio-coverage", type=float, default=0.95)
    ap.add_argument("--max-auditability-stratum-gap", type=float, default=0.03)
    ap.add_argument("--max-certificate-stratum-gap", type=float, default=0.10)
    args = ap.parse_args()

    src = Path(args.source_smoke_root).resolve()
    out = Path(args.output_root).resolve()
    labels = src / "labels_val_v16_8_16"
    source_verdict = src / "v16_8_16_smoke_verdict.json"
    source_fp_file = src / "v16_8_16_code_fingerprint.sha256"
    hard_ids = src / "hard_scene_ids.txt"
    random_ids = src / "random_scene_ids.txt"
    profile = src / "fresh_profile.jsonl"
    base_screen = src / "base_screen_verdict.json"
    supervision = src / "training_supervision_audit.json"
    for path in (labels, source_verdict, source_fp_file, hard_ids, random_ids, profile, base_screen, supervision):
        if not path.exists():
            raise FileNotFoundError(f"required v16.8.16 smoke artifact missing: {path}")

    old = _load(source_verdict)
    stored_fp = source_fp_file.read_text(encoding="utf-8").strip()
    if stored_fp != EXPECTED_V16_8_16_CODE_FINGERPRINT or old.get("code_fingerprint_sha256") != stored_fp:
        raise RuntimeError(
            "Source smoke is not the reviewed v16.8.16 label build; do not policy-reuse it. "
            f"stored={stored_fp!r}, verdict={old.get('code_fingerprint_sha256')!r}"
        )

    gate = importlib.import_module("cowp.scripts.59_gate_fresh_v16_8_9_cache_protocol")
    code_root = Path.cwd()
    label_fp = gate.current_label_semantic_fingerprint(code_root)
    if label_fp != EXPECTED_V16_8_16_LABEL_SEMANTIC_FINGERPRINT:
        raise RuntimeError(
            "Current label-producing semantics differ from reviewed v16.8.16; a fresh smoke rebuild is required. "
            f"current_label_fp={label_fp}"
        )
    current_fp = gate.current_fingerprint(code_root)

    out.mkdir(parents=True, exist_ok=True)
    model_support = out / "model_support_audit.json"
    natural_diag = out / "natural_support_diagnostic.json"
    verdict = out / "v16_8_18_smoke_verdict.json"

    model_support_rc = _run([
        sys.executable, "-m", "cowp.scripts.65_audit_model_support",
        "--cache-dir", str(labels), "--sample-scenes", "0",
        "--min-class-examples", "8", "--min-source-examples", "8",
        "--max-unauditable-critical-rate", str(args.max_unauditable_critical_rate),
        "--min-certificate-complete-scene-rate", str(args.min_certificate_complete_scene_rate),
        "--min-protected-prio-coverage", str(args.min_protected_prio_coverage),
        "--coverage-gate-mode", "wilson_gross_failure",
        "--hard-scene-ids", str(hard_ids), "--random-scene-ids", str(random_ids),
        "--max-auditability-stratum-gap", str(args.max_auditability_stratum_gap),
        "--max-certificate-stratum-gap", str(args.max_certificate_stratum_gap),
        "--output", str(model_support), "--strict",
    ])
    natural_diag_rc = _run([
        sys.executable, "-m", "cowp.scripts.68_summarize_natural_support_diagnostics",
        "--input", str(profile), "--output", str(natural_diag),
    ])

    for name in ("hard_scene_ids.txt", "random_scene_ids.txt", "union_scene_ids.txt", "base_screen_verdict.json", "training_supervision_audit.json", "fastpath_ab"):
        source = src / name
        target = out / name
        if source.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(source, target)
        elif source.exists():
            shutil.copy2(source, target)

    base = _load(base_screen)
    sup = _load(supervision)
    ms = _load(model_support) if model_support.is_file() else {}
    nat = _load(natural_diag) if natural_diag.is_file() else {}
    msc = ms.get("checks", {}) or {}
    checks = {
        "source_v16_8_16_label_semantics_verified": True,
        "model_support_audit_completed": bool(model_support.is_file()),
        "natural_support_diagnostic_completed": bool(natural_diag.is_file()) and natural_diag_rc == 0,
        "proposal_causal_screen_pass": bool(base.get("screen_pass", False)),
        "training_supervision_pass": bool(sup.get("pass", False)),
        "model_support_pass": bool(ms.get("pass", False)),
        "auditability_coverage": bool(msc.get("auditability_coverage", False)),
        "certificate_complete_scene_coverage": bool(msc.get("certificate_complete_scene_coverage", False)),
        "auditability_stratum_balance": bool(msc.get("auditability_stratum_balance", False)),
        "certificate_stratum_balance": bool(msc.get("certificate_stratum_balance", False)),
        "natural_rootless_zero_on_auditable": int(nat.get("rootless_critical_agents", -1)) == 0,
        "natural_lt2_low_burden_zero_on_auditable": int(nat.get("critical_agents_with_lt2_low_burden_roots", -1)) == 0,
        "protected_prio_coverage": float(nat.get("protected_prio_root_coverage", 0.0)) >= float(args.min_protected_prio_coverage),
    }
    passed = all(checks.values())
    payload = {
        "schema_version": "cowp_v16_8_18_smoke_policy_reaudit_v2",
        "code_fingerprint_sha256": current_fp,
        "label_semantic_fingerprint_sha256": label_fp,
        "source_v16_8_16_code_fingerprint_sha256": stored_fp,
        "reused_labels_from": str(labels),
        "pass": passed,
        "recommend_strict_probe": passed,
        "recommend_full_rebuild": False,
        "checks": checks,
        "semantic_returncodes": {"model_support": model_support_rc, "natural_diagnostic": natural_diag_rc},
        "base_screen": base,
        "model_support_coverage": {
            "thresholds": {k: ms.get(k) for k in (
                "max_unauditable_critical_rate", "min_certificate_complete_scene_rate",
                "min_protected_prio_coverage", "coverage_gate_mode",
                "max_auditability_stratum_gap", "max_certificate_stratum_gap",
            )},
            "statistics": ms.get("coverage_statistics", {}),
            "by_stratum": ms.get("coverage_by_stratum", {}),
        },
        "natural_support": {k: nat.get(k) for k in (
            "critical_agents_selected", "mechanism_auditable_critical_agents",
            "mechanism_unauditable_critical_agents", "mechanism_unauditable_rate",
            "mechanism_unauditable_future_support", "mechanism_unauditable_finalizer_counts",
            "mechanism_unauditable_with_sufficient_future_but_no_substantial_route_geometry",
            "mechanism_unauditable_with_insufficient_future", "rootless_critical_agents",
            "critical_agents_with_lt2_low_burden_roots", "protected_auditable_critical_agents",
            "protected_without_prio_root", "protected_prio_root_coverage",
        )},
        "next_action": (
            "Policy re-audit PASS: run a fresh 400-hard + 800-random v16.8.18 strict probe; do not full-rebuild yet."
            if passed else
            "Policy re-audit FAIL: do not run strict/full rebuild; inspect coverage-by-stratum and natural-support diagnostics."
        ),
    }
    verdict.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "v16_8_18_code_fingerprint.sha256").write_text(current_fp + "\n", encoding="utf-8")
    (out / "v16_8_18_label_semantic_fingerprint.sha256").write_text(label_fp + "\n", encoding="utf-8")
    (out / "smoke_pipeline_status.json").write_text(json.dumps({
        "schema_version": "cowp_v16_8_18_smoke_pipeline_status_v1",
        "pipeline_complete": True,
        "mode": "policy_reaudit_v16_8_16_labels",
        "composite_verdict_written": True,
        "recommend_strict_probe": bool(payload.get("recommend_strict_probe", False)),
        "semantic_returncodes": payload.get("semantic_returncodes", {}),
        "next_action": payload.get("next_action"),
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
