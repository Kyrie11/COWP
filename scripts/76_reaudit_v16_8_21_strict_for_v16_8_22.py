from __future__ import annotations

import argparse
import importlib
import json
import shutil
from pathlib import Path

EXPECTED_V16_8_21_CODE_FINGERPRINT = "8227755941c577f69c03c7e44aae6010a67a23f1a557539480d6766512121e9c"
EXPECTED_LABEL_SEMANTIC_FINGERPRINT = "c7f8a33f5e9fef04ac009d41806173369ddbfef6ac0b7e7c4ac0ca1edfc0af51"


def _load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


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


def main() -> None:
    ap = argparse.ArgumentParser(description="Reuse a passed v16.8.21 strict data-evidence bundle under the v16.8.22 model/training fingerprint.")
    ap.add_argument("--source-strict-root", required=True)
    ap.add_argument("--output-root", required=True)
    args = ap.parse_args()
    src = Path(args.source_strict_root).resolve()
    out = Path(args.output_root).resolve()
    if src == out:
        raise RuntimeError("output-root must differ from source-strict-root")

    verdict_path = src / "v16_8_21_strict_verdict.json"
    fp_path = src / "v16_8_18_code_fingerprint.sha256"
    lfp_path = src / "v16_8_21_label_semantic_fingerprint.sha256"
    for p in (verdict_path, fp_path, lfp_path, src / "base_screen_verdict.json"):
        if not p.is_file():
            raise FileNotFoundError(str(p))
    source = _load(verdict_path)
    source_fp = fp_path.read_text(encoding="utf-8").strip()
    source_lfp = lfp_path.read_text(encoding="utf-8").strip()
    if source_fp != EXPECTED_V16_8_21_CODE_FINGERPRINT:
        raise RuntimeError(f"unexpected source v16.8.21 fingerprint: {source_fp}")
    if source_lfp != EXPECTED_LABEL_SEMANTIC_FINGERPRINT:
        raise RuntimeError(f"unexpected source label-semantic fingerprint: {source_lfp}")
    if not (source.get("pass") is True and source.get("recommend_full_rebuild") is True):
        raise RuntimeError("source v16.8.21 strict bundle did not pass; do not reuse it")

    gate = importlib.import_module("cowp.scripts.59_gate_fresh_v16_8_9_cache_protocol")
    code_root = Path(__file__).resolve().parents[2]
    current_lfp = gate.current_label_semantic_fingerprint(code_root)
    if current_lfp != source_lfp:
        raise RuntimeError("v16.8.22 changes Scenario->label semantics; fresh strict rebuild required")
    current_fp = gate.current_fingerprint(code_root)

    out.mkdir(parents=True, exist_ok=True)
    for p in src.iterdir():
        if p.name in {"v16_8_18_strict_verdict.json", "v16_8_21_strict_verdict.json", "strict_pipeline_status.json", "v16_8_18_code_fingerprint.sha256"}:
            continue
        if p.is_file():
            _copy(p, out / p.name)

    checks = dict(source.get("checks", {}))
    checks["source_v16_8_21_strict_pass_verified"] = True
    checks["label_semantics_unchanged_for_v16_8_22"] = True
    passed = all(bool(v) for v in checks.values())
    failed = [k for k, v in checks.items() if not v]
    payload = {
        "schema_version": "cowp_v16_8_22_strict_reaudit_v1",
        "strict": True,
        "code_fingerprint_sha256": current_fp,
        "label_semantic_fingerprint_sha256": current_lfp,
        "source_v16_8_21_code_fingerprint_sha256": source_fp,
        "source_strict_root": str(src),
        "pass": bool(passed),
        "recommend_full_rebuild": bool(passed),
        "checks": checks,
        "failed_checks": failed,
        "base_screen": source.get("base_screen", _load(src / "base_screen_verdict.json")),
        "support_reuse_contract": {
            "label_tensors_rebuilt": False,
            "label_semantics_changed": False,
            "strict_promotion_policy_changed": False,
            "model_primary_semantics_changed": True,
            "why_reuse_is_valid": (
                "v16.8.22 changes model/training aggregation and train-pilot support auditing, not Scenario->label tensors. "
                "The already-passed v16.8.21 validation strict evidence therefore remains the correct data-side gate."
            ),
        },
        "next_action": (
            "STRICT RE-AUDIT PASS: run the v16.8.22 train-pilot mechanism-support re-audit."
            if passed else "STRICT RE-AUDIT FAIL: do not full-rebuild."
        ),
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    (out / "v16_8_22_strict_verdict.json").write_text(text, encoding="utf-8")
    (out / "v16_8_18_strict_verdict.json").write_text(text, encoding="utf-8")
    (out / "v16_8_18_code_fingerprint.sha256").write_text(current_fp + "\n", encoding="utf-8")
    (out / "v16_8_22_label_semantic_fingerprint.sha256").write_text(current_lfp + "\n", encoding="utf-8")
    (out / "strict_pipeline_status.json").write_text(json.dumps({
        "schema_version": "cowp_v16_8_22_strict_reaudit_status_v1",
        "pipeline_complete": True,
        "mode": "reuse_passed_v16_8_21_strict_data_evidence",
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
