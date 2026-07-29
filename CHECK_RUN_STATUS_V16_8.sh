#!/usr/bin/env bash
set -euo pipefail

OUT_ROOT="${OUT_ROOT:-outputs/cowp_v16_8_mechanism_v9labels_seed2026}"

echo "COWP v16.8 run: $OUT_ROOT"
echo "----------------------------------------"
found_pid=0
for kind in driver full_driver smoke_driver; do
  pid_file="$OUT_ROOT/logs/${kind}.pid"
  log_file="$OUT_ROOT/logs/${kind}.nohup.log"
  if [[ -s "$pid_file" ]]; then
    found_pid=1
    pid="$(cat "$pid_file")"
    if kill -0 "$pid" 2>/dev/null; then state=RUNNING; else state=EXITED; fi
    echo "$kind: $state pid=$pid log=$log_file"
  fi
done
[[ "$found_pid" == "1" ]] || echo "pid: no launcher pid file found"

for manifest in \
  "$OUT_ROOT/configs/data_protocol_manifest.json" \
  "$OUT_ROOT/configs/precision_manifest.json" \
  "$OUT_ROOT/configs/natural_attribution_transfer_manifest.json"; do
  if [[ -s "$manifest" ]]; then
    echo
    echo "[$(basename "$manifest" .json)]"
    python -m json.tool "$manifest"
  fi
done

QUALITY_BYPASS_FILE=""
for candidate in "$OUT_ROOT/eval/QUALITY_GATES_BYPASSED.txt" "$OUT_ROOT/QUALITY_GATES_BYPASSED.txt"; do
  if [[ -s "$candidate" ]]; then QUALITY_BYPASS_FILE="$candidate"; break; fi
done
if [[ -n "$QUALITY_BYPASS_FILE" ]]; then
  echo
  echo "STATUS: ENGINEERING-ONLY (quality gate bypass marker exists)"
  echo "marker: $QUALITY_BYPASS_FILE"
  sed -n '1,30p' "$QUALITY_BYPASS_FILE"
fi

python - "$OUT_ROOT" <<'PY'
from __future__ import annotations
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
print("\n[stage histories]")
for stage, folder in (("natural", "natural"), ("witness", "transport"), ("planner", "planner")):
    p = root / "checkpoints" / folder / f"history_{stage}.json"
    if not p.exists():
        print(f"{stage}: missing")
        continue
    try:
        rows = json.loads(p.read_text(encoding="utf-8"))
        row = rows[-1] if rows else {}
        print(
            f"{stage}: rows={len(rows)} last_epoch={row.get('epoch')} "
            f"optimizer_steps={row.get('train/runtime/optimizer_steps')} "
            f"amp_skips={row.get('train/runtime/amp_skipped_steps')} "
            f"improved={row.get('checkpoint/improved')}"
        )
    except Exception as exc:
        print(f"{stage}: invalid history ({exc})")

print("\n[quality gates]")
for p in (
    root / "eval" / "learned_offline" / "natural_basis_gate.json",
    root / "eval" / "learned_offline" / "natural_effectiveness_gate.json",
    root / "eval" / "learned_offline" / "mechanism_verification.json",
):
    if not p.exists():
        print(f"{p.name}: missing")
        continue
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        passed = data.get("pass", data.get("passed"))
        role = data.get("gate_role", "")
        print(f"{p.stem}: pass={passed} role={role}")
        if p.stem == "mechanism_verification":
            for key in (
                "priority_accept_ncf_recall",
                "priority_accept_ncf_precision",
                "priority_bcot_false_safe_auprc",
                "priority_root_transport_auprc",
                "learned_accepted_candidate_rate",
                "fallback_rate",
                "priority_burden_transfer_rate",
                "priority_transfer_improvement",
                "global_false_safe_improvement",
            ):
                if key in data:
                    print(f"  {key}: {data[key]}")
    except Exception as exc:
        print(f"{p.name}: invalid ({exc})")

shared = root / "eval" / "learned_offline" / "_shared_model_pass.json"
if shared.exists():
    try:
        payload = json.loads(shared.read_text(encoding="utf-8"))
        cowp = payload.get("cowp", {})
        print("\n[held-out COWP mechanism metrics]")
        for key in (
            "PriorityBurdenTransferRate",
            "PriorityCertificate/AcceptNCFRecall",
            "PriorityCertificate/AcceptNCFPrecision",
            "PriorityCertificate/NCFSceneRetention",
            "PriorityCertificate/NonCoerciveProgressRegret",
            "LearnedAcceptedCandidateRate",
            "LearnedAcceptNCFRecall",
            "LearnedAcceptNCFPrecision",
            "FallbackRate",
            "EP",
            "EP_m",
            "BCOT/PriorityFalseSafe_AUPRC",
            "RootTransport/PriorityConflict_AUPRC",
            "PriorityBurden/BTE_CVaR_25",
        ):
            if key in cowp:
                print(f"{key}: {cowp[key]}")
    except Exception as exc:
        print(f"shared model pass invalid ({exc})")
PY

echo
if [[ -s "$OUT_ROOT/eval/pipeline_completion_report.json" ]]; then
  echo "[pipeline completion]"
  python -m json.tool "$OUT_ROOT/eval/pipeline_completion_report.json"
else
  echo "pipeline_completion_report.json: missing"
  last_line=""
  for log in "$OUT_ROOT/logs/full_driver.nohup.log" "$OUT_ROOT/logs/smoke_driver.nohup.log" "$OUT_ROOT/logs/driver.nohup.log"; do
    if [[ -s "$log" ]]; then
      last_line="$(tail -n 1 "$log")"
      echo "last launcher line: $last_line"
      break
    fi
  done
  if [[ "$last_line" == *"stopped after"* ]]; then
    echo "STATUS: INTENTIONAL_PARTIAL_RUN"
  elif [[ "$last_line" == *"complete:"* ]]; then
    echo "STATUS: COMPLETION_VALIDATOR_MISSING"
  else
    echo "STATUS: INCOMPLETE_OR_FAILED (inspect logs)"
  fi
fi
