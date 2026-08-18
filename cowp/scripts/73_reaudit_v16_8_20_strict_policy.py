from __future__ import annotations

import argparse
import importlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

EXPECTED_V16_8_20_CODE_FINGERPRINT = "e9fcfab92ed8a24cac3215e6ca037897231ce59e74fd186ddd8200c6338b8172"
EXPECTED_V16_8_20_LABEL_SEMANTIC_FINGERPRINT = "c7f8a33f5e9fef04ac009d41806173369ddbfef6ac0b7e7c4ac0ca1edfc0af51"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _copy_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _run(cmd: list[str]) -> int:
    print("[v16.8.21 strict policy re-audit]", " ".join(cmd), flush=True)
    return int(subprocess.run(cmd, check=False).returncode)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Re-audit a reviewed v16.8.20 400-hard+800-random strict probe under "
            "the v16.8.21 evidence-aligned promotion policy without rebuilding labels."
        )
    )
    ap.add_argument("--source-strict-root", required=True)
    ap.add_argument("--output-root", required=True)
    args = ap.parse_args()

    src = Path(args.source_strict_root).resolve()
    out = Path(args.output_root).resolve()

    required = {
        "source_verdict": src / "v16_8_18_strict_verdict.json",
        "source_fp": src / "v16_8_18_code_fingerprint.sha256",
        "paired": src / "paired_proposal_probe.json",
        "ablation": src / "proposal_source_ablation.json",
        "profile": src / "fresh_probe_profile_summary.json",
        "audit": src / "causal_audit_diagnostic.json",
        "supervision": src / "training_supervision_audit.json",
        "model_support": src / "model_support_audit.json",
        "natural": src / "natural_support_diagnostic.json",
        "sparse": src / "sparse_label_build_integrity.json",
    }
    missing = [str(p) for p in required.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError("required v16.8.20 strict artifacts missing: " + ", ".join(missing))

    source_verdict = _load(required["source_verdict"])
    source_fp = required["source_fp"].read_text(encoding="utf-8").strip()
    if source_fp != EXPECTED_V16_8_20_CODE_FINGERPRINT:
        raise RuntimeError(
            "Source probe fingerprint is not the reviewed v16.8.20 build; policy reuse is forbidden. "
            f"stored={source_fp!r}"
        )
    if source_verdict.get("code_fingerprint_sha256") != source_fp:
        raise RuntimeError("Source strict verdict/fingerprint mismatch; do not policy-reuse this probe.")
    if not bool(source_verdict.get("checks", {}).get("sparse_label_build_complete", False)):
        raise RuntimeError("Source strict probe did not complete the sparse label build.")

    gate = importlib.import_module("cowp.scripts.59_gate_fresh_v16_8_9_cache_protocol")
    code_root = Path(__file__).resolve().parents[2]
    label_fp = gate.current_label_semantic_fingerprint(code_root)
    if label_fp != EXPECTED_V16_8_20_LABEL_SEMANTIC_FINGERPRINT:
        raise RuntimeError(
            "Current Scenario->label semantics differ from reviewed v16.8.20. "
            "A fresh smoke/strict label rebuild is required instead of policy re-audit. "
            f"current_label_fp={label_fp}"
        )
    current_fp = gate.current_fingerprint(code_root)

    out.mkdir(parents=True, exist_ok=True)
    screen_path = out / "base_screen_verdict.json"
    screen_rc = _run([
        sys.executable, "-m", "cowp.scripts.58_screen_v16_8_9_causal_audit_probe",
        "--paired-probe", str(required["paired"]),
        "--source-ablation", str(required["ablation"]),
        "--profile-summary", str(required["profile"]),
        "--audit-diagnostic", str(required["audit"]),
        "--output", str(screen_path),
        "--strict",
    ])

    # Copy the immutable evidence bundle used by the re-audit.  NPZ files are not
    # copied: the source full fingerprint plus the unchanged label-semantic hash
    # is the provenance bridge authorizing policy-only reuse.
    for name in (
        "paired_proposal_probe.json", "proposal_source_ablation.json",
        "fresh_probe_profile_summary.json", "causal_audit_diagnostic.json",
        "training_supervision_audit.json", "model_support_audit.json",
        "natural_support_diagnostic.json", "sparse_label_build_integrity.json",
        "probe_manifest_audit.json", "hard_scene_ids.txt", "representative_random_scene_ids.txt",
        "probe_union_scene_ids.txt", "fresh_probe_profile.jsonl",
    ):
        _copy_if_exists(src / name, out / name)

    screen = _load(screen_path) if screen_path.is_file() else {}
    sup = _load(required["supervision"])
    model = _load(required["model_support"])
    nat = _load(required["natural"])
    sparse = _load(required["sparse"])
    mchecks = model.get("checks", {}) or {}

    checks = {
        "source_v16_8_20_full_fingerprint_verified": True,
        "source_v16_8_20_label_semantics_verified": True,
        "sparse_label_build_complete": bool(sparse.get("pass", False)) and bool(sparse.get("pipeline_complete", False)),
        "proposal_evidence_aligned_screen_pass": bool(screen.get("screen_pass", False)) and screen_rc == 0,
        "training_supervision_pass": bool(sup.get("pass", False)),
        "model_support_pass": bool(model.get("pass", False)),
        "auditability_coverage": bool(mchecks.get("auditability_coverage", False)),
        "certificate_complete_scene_coverage": bool(mchecks.get("certificate_complete_scene_coverage", False)),
        "auditability_stratum_balance": bool(mchecks.get("auditability_stratum_balance", False)),
        "certificate_stratum_balance": bool(mchecks.get("certificate_stratum_balance", False)),
        "natural_rootless_zero_on_auditable": int(nat.get("rootless_critical_agents", -1)) == 0,
        "natural_lt2_low_burden_zero_on_auditable": int(nat.get("critical_agents_with_lt2_low_burden_roots", -1)) == 0,
        "protected_prio_coverage": float(nat.get("protected_prio_root_coverage", 0.0)) >= 0.95,
    }
    passed = all(checks.values())
    failed = [k for k, v in checks.items() if not v]

    payload = {
        "schema_version": "cowp_v16_8_21_strict_policy_reaudit_v1",
        "strict": True,
        "code_fingerprint_sha256": current_fp,
        "label_semantic_fingerprint_sha256": label_fp,
        "source_v16_8_20_code_fingerprint_sha256": source_fp,
        "source_strict_root": str(src),
        "pass": bool(passed),
        "recommend_full_rebuild": bool(passed),
        "checks": checks,
        "failed_checks": failed,
        "semantic_returncodes": {"screen": screen_rc},
        "base_screen": screen,
        "support_reuse_contract": {
            "label_tensors_rebuilt": False,
            "label_semantics_changed": False,
            "promotion_policy_changed": True,
            "source_evidence_reused": True,
            "why_reuse_is_valid": (
                "v16.8.21 changes only the statistical/promotion interpretation of protected-scene prevalence. "
                "The Scenario->label semantic fingerprint is exactly the reviewed v16.8.20 fingerprint."
            ),
        },
        "next_action": (
            "STRICT POLICY RE-AUDIT PASS: run a fresh train-pilot under the current code fingerprint; "
            "only then authorize the full rebuild."
            if passed else
            "STRICT POLICY RE-AUDIT FAIL: do not full-rebuild; inspect failed_checks and base_screen."
        ),
    }

    compat = out / "v16_8_18_strict_verdict.json"
    named = out / "v16_8_21_strict_verdict.json"
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    compat.write_text(text, encoding="utf-8")
    named.write_text(text, encoding="utf-8")
    (out / "v16_8_18_code_fingerprint.sha256").write_text(current_fp + "\n", encoding="utf-8")
    (out / "v16_8_21_label_semantic_fingerprint.sha256").write_text(label_fp + "\n", encoding="utf-8")
    (out / "strict_pipeline_status.json").write_text(json.dumps({
        "schema_version": "cowp_v16_8_21_strict_policy_reaudit_status_v1",
        "pipeline_complete": True,
        "mode": "policy_reaudit_v16_8_20_strict_labels",
        "composite_verdict_written": True,
        "recommend_full_rebuild": bool(passed),
        "failed_checks": failed,
        "next_action": payload["next_action"],
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
