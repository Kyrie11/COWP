from __future__ import annotations

import argparse
import importlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

EXPECTED_V16_8_21_CODE_FINGERPRINT = "8227755941c577f69c03c7e44aae6010a67a23f1a557539480d6766512121e9c"
EXPECTED_LABEL_SEMANTIC_FINGERPRINT = "c7f8a33f5e9fef04ac009d41806173369ddbfef6ac0b7e7c4ac0ca1edfc0af51"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _copy(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)


def _run(cmd: list[str]) -> int:
    print("[v16.8.22 train-pilot re-audit]", " ".join(cmd), flush=True)
    return int(subprocess.run(cmd, check=False).returncode)


def _resolve_cache(src: Path, sparse: dict, override: str) -> Path:
    candidates: list[Path] = []
    if override:
        candidates.append(Path(override).expanduser())
    candidates.append(src / "labels_train_v16_8_18")
    old = sparse.get("labels_dir")
    if old:
        candidates.append(Path(str(old)).expanduser())
    for p in candidates:
        if p.is_dir() and any(p.glob("*.npz")):
            return p.resolve()
    raise FileNotFoundError(
        "No train-pilot NPZ cache found. The compact archive uploaded for review intentionally omits NPZ files; "
        "run this command on the machine that still has formal_v16_8_21_train_pilot/labels_train_v16_8_18, "
        "or pass --cache-dir explicitly. Tried: " + ", ".join(str(p) for p in candidates)
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Re-audit a completed v16.8.21 train-pilot under the v16.8.22 six-layer training-support contract. "
            "Scenario->label semantics are unchanged; model primary/auxiliary semantics and the train promotion gate change."
        )
    )
    ap.add_argument("--source-train-root", required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--cache-dir", default="")
    args = ap.parse_args()

    src = Path(args.source_train_root).resolve()
    out = Path(args.output_root).resolve()
    if src == out:
        raise RuntimeError("output-root must differ from source-train-root")
    required = {
        "source_verdict": src / "v16_8_18_train_pilot_verdict.json",
        "source_fp": src / "v16_8_18_code_fingerprint.sha256",
        "sparse": src / "sparse_label_build_integrity.json",
        "manifest": src / "pilot_manifest_audit.json",
        "profile": src / "profile_train_pilot.jsonl",
        "hard_ids": src / "hard_scene_ids.txt",
        "random_ids": src / "random_scene_ids.txt",
        "union_ids": src / "pilot_union_scene_ids.txt",
    }
    missing = [str(p) for p in required.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError("required v16.8.21 train-pilot artifacts missing: " + ", ".join(missing))

    source_fp = required["source_fp"].read_text(encoding="utf-8").strip()
    source_verdict = _load(required["source_verdict"])
    if source_verdict.get("code_fingerprint_sha256") != source_fp:
        raise RuntimeError("source train-pilot verdict/fingerprint mismatch")
    sparse = _load(required["sparse"])
    if not (bool(sparse.get("pass", False)) and bool(sparse.get("pipeline_complete", False))):
        raise RuntimeError("source train-pilot sparse build was incomplete")

    gate = importlib.import_module("cowp.scripts.59_gate_fresh_v16_8_9_cache_protocol")
    code_root = Path(__file__).resolve().parents[2]
    label_fp = gate.current_label_semantic_fingerprint(code_root)
    if label_fp != EXPECTED_LABEL_SEMANTIC_FINGERPRINT:
        raise RuntimeError(
            "Current Scenario->label semantics differ from v16.8.21; a fresh pilot label build is required. "
            f"current_label_fp={label_fp}"
        )
    current_fp = gate.current_fingerprint(code_root)
    if source_fp not in {EXPECTED_V16_8_21_CODE_FINGERPRINT, current_fp}:
        raise RuntimeError(
            "Source train-pilot is neither the reviewed v16.8.21 build nor a fresh v16.8.22 build. "
            f"stored={source_fp!r}, current={current_fp!r}"
        )
    cache = _resolve_cache(src, sparse, args.cache_dir)

    out.mkdir(parents=True, exist_ok=True)
    for name in (
        "sparse_label_build_integrity.json", "pilot_manifest_audit.json", "profile_train_pilot.jsonl",
        "profile_train_pilot_summary.json", "hard_scene_ids.txt", "random_scene_ids.txt", "pilot_union_scene_ids.txt",
        "old_train_proposal_ceiling.json",
    ):
        _copy(src / name, out / name)

    sup_path = out / "training_supervision_audit.json"
    model_path = out / "model_support_audit.json"
    causal_path = out / "causal_audit_diagnostic.json"
    natural_path = out / "natural_support_diagnostic.json"
    ceiling_path = out / "fresh_train_proposal_ceiling.json"
    contrast_path = out / "mechanism_contrast_audit.json"

    rc_sup = _run([sys.executable, "-m", "cowp.scripts.62_audit_training_supervision",
                   "--cache-dir", str(cache), "--output", str(sup_path), "--sample-scenes", "0", "--strict"])
    rc_model = _run([
        sys.executable, "-m", "cowp.scripts.65_audit_model_support",
        "--cache-dir", str(cache), "--output", str(model_path), "--sample-scenes", "0", "--strict",
        "--max-unauditable-critical-rate", "0.05", "--min-certificate-complete-scene-rate", "0.75",
        "--min-protected-prio-coverage", "0.98",
        "--hard-scene-ids", str(required["hard_ids"]), "--random-scene-ids", str(required["random_ids"]),
        "--max-auditability-stratum-gap", "0.03", "--max-certificate-stratum-gap", "0.10",
    ])
    rc_causal = _run([sys.executable, "-m", "cowp.scripts.57_diagnose_causal_audit",
                      "--cache-dir", str(cache), "--scene-ids", str(required["union_ids"]), "--output", str(causal_path)])
    rc_natural = _run([sys.executable, "-m", "cowp.scripts.68_summarize_natural_support_diagnostics",
                       "--input", str(required["profile"]), "--output", str(natural_path)])
    rc_ceiling = _run([sys.executable, "-m", "cowp.scripts.45_diagnose_proposal_ceiling",
                       "--cache-dir", str(cache), "--output", str(ceiling_path), "--limit", "0"])
    rc_contrast = _run([sys.executable, "-m", "cowp.scripts.74_audit_mechanism_contrast",
                        "--cache-dir", str(cache), "--output", str(contrast_path), "--sample-scenes", "0", "--strict"])

    sup = _load(sup_path) if sup_path.is_file() else {}
    model = _load(model_path) if model_path.is_file() else {}
    causal = _load(causal_path) if causal_path.is_file() else {}
    natural = _load(natural_path) if natural_path.is_file() else {}
    ceiling = _load(ceiling_path) if ceiling_path.is_file() else {}
    contrast = _load(contrast_path) if contrast_path.is_file() else {}
    manifest = _load(required["manifest"])
    mchecks = model.get("checks", {}) or {}
    cintegrity = causal.get("integrity", {}) or {}
    class_support = sup.get("class_support", {}) or {}
    global_ncf_class = class_support.get("candidate_ncf", {}) or {}
    p_ncf_class = class_support.get("priority_candidate_ncf", {}) or {}
    p_fs_class = class_support.get("priority_candidate_false_safe", {}) or {}
    scount = ceiling.get("scene_counts", {}) or {}
    srates = ceiling.get("scene_rates", {}) or {}

    global_ncf_candidate_positive = int(global_ncf_class.get("positive", 0))
    global_ncf_scene_count = int(scount.get("any_ncf", 0))
    priority_positive = int(p_ncf_class.get("positive", 0))
    priority_negative = int(p_fs_class.get("positive", 0))

    checks = {
        "source_v16_8_21_full_fingerprint_verified": True,
        "label_semantics_unchanged": True,
        "sparse_label_build_complete": bool(sparse.get("pass", False)) and bool(sparse.get("pipeline_complete", False)),
        "pilot_manifest_pass": bool(manifest.get("pass", False)),
        "training_supervision_pass": bool(sup.get("pass", False)) and rc_sup == 0,
        "model_support_pass": bool(model.get("pass", False)) and rc_model == 0,
        "auditability_coverage": bool(mchecks.get("auditability_coverage", False)),
        "certificate_complete_scene_coverage": bool(mchecks.get("certificate_complete_scene_coverage", False)),
        "auditability_stratum_balance": bool(mchecks.get("auditability_stratum_balance", False)),
        "certificate_stratum_balance": bool(mchecks.get("certificate_stratum_balance", False)),
        "natural_rootless_zero_on_auditable": int(natural.get("rootless_critical_agents", -1)) == 0,
        "natural_lt2_low_burden_zero_on_auditable": int(natural.get("critical_agents_with_lt2_low_burden_roots", -1)) == 0,
        "protected_prio_coverage": float(natural.get("protected_prio_root_coverage", 0.0)) >= 0.98,
        "causal_no_read_errors": not causal.get("read_errors") and rc_causal == 0,
        "causal_no_silent_blockers": bool(cintegrity.get("no_silent_blockers", False)),
        "causal_no_irrelevant_blockers": bool(cintegrity.get("no_irrelevant_blockers", False)),
        "candidate_any_valid": float(srates.get("any_valid", 0.0)) >= 0.99 and rc_ceiling == 0,
        # Evidence-count gates, not population prevalence gates.  The pilot is deliberately
        # 400 hard + 800 random and should not be required to reproduce the population rate.
        "global_aux_candidate_ncf_count": global_ncf_candidate_positive >= 1024,
        "global_aux_ncf_scene_count": global_ncf_scene_count >= 128,
        "protected_candidate_positive_count": priority_positive >= 1024,
        "protected_candidate_negative_count": priority_negative >= 1024,
        "six_layer_mechanism_contrast_pass": bool(contrast.get("pass", False)) and rc_contrast == 0,
    }
    passed = all(checks.values())
    failed = [k for k, v in checks.items() if not v]

    priority_eligible_scenes = int(scount.get("any_priority_eligible", 0))
    priority_ncf_scenes = int(scount.get("any_priority_ncf", 0))
    conditional_priority_ncf = float(priority_ncf_scenes / max(priority_eligible_scenes, 1))

    payload = {
        "schema_version": "cowp_v16_8_22_train_pilot_verdict_v1",
        "code_fingerprint_sha256": current_fp,
        "label_semantic_fingerprint_sha256": label_fp,
        "source_v16_8_21_code_fingerprint_sha256": source_fp,
        "source_train_root": str(src),
        "source_cache_dir": str(cache),
        "pass": bool(passed),
        "recommend_full_rebuild": bool(passed),
        "checks": checks,
        "failed_checks": failed,
        "semantic_returncodes": {
            "supervision": rc_sup, "model_support": rc_model, "causal": rc_causal,
            "natural": rc_natural, "ceiling": rc_ceiling, "mechanism_contrast": rc_contrast,
        },
        "scene_rates": srates,
        "scene_counts": scount,
        "evidence_counts": {
            "global_ncf_candidate_positive": global_ncf_candidate_positive,
            "global_ncf_scene_count": global_ncf_scene_count,
            "priority_ncf_candidate_positive": priority_positive,
            "priority_false_safe_candidate_positive": priority_negative,
            "priority_eligible_scenes": priority_eligible_scenes,
            "priority_ncf_scenes": priority_ncf_scenes,
            "priority_ncf_given_eligible": conditional_priority_ncf,
        },
        "mechanism_contrast": contrast,
        "support_contract": {
            "label_tensors_rebuilt": False,
            "label_semantics_changed": False,
            "model_primary_semantics_changed": True,
            "train_promotion_policy_changed": True,
            "removed_gate": "all-critical any-NCF scene prevalence >= 0.30",
            "replacement": (
                "absolute global-auxiliary class/scene support + protected class support + within-scene/root "
                "mechanism contrast for viability, same-root recovery, option mass and protected ranking"
            ),
            "why": (
                "The pilot composition intentionally oversamples hard scenes. Population prevalence is not a "
                "training-identifiability condition and conflicts with the protected-priority primary certificate."
            ),
        },
        "next_action": (
            "TRAIN-PILOT RE-AUDIT PASS: v16.8.22 strict + train-pilot may authorize full-core rebuild."
            if passed else
            "TRAIN-PILOT RE-AUDIT FAIL: do not full-rebuild. Inspect failed_checks and mechanism_contrast; "
            "only rebuild proposal labels if a within-scene/root contrast check fails."
        ),
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    (out / "v16_8_22_train_pilot_verdict.json").write_text(text, encoding="utf-8")
    # Compatibility path consumed by the v16.8.18 full-core driver.
    (out / "v16_8_18_train_pilot_verdict.json").write_text(text, encoding="utf-8")
    (out / "v16_8_18_code_fingerprint.sha256").write_text(current_fp + "\n", encoding="utf-8")
    (out / "v16_8_22_label_semantic_fingerprint.sha256").write_text(label_fp + "\n", encoding="utf-8")
    (out / "train_pilot_pipeline_status.json").write_text(json.dumps({
        "schema_version": "cowp_v16_8_22_train_pilot_pipeline_status_v1",
        "pipeline_complete": True,
        "mode": "mechanism_support_reaudit_v16_8_21_labels",
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
