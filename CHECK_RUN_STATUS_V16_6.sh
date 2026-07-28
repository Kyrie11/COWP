#!/usr/bin/env bash
set -euo pipefail

OUT_ROOT="${OUT_ROOT:-outputs/cowp_v16_6_pipeline_v9labels_seed2026}"

echo "COWP run: $OUT_ROOT"
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

if [[ -s "$OUT_ROOT/configs/data_protocol_manifest.json" ]]; then
  echo
  echo "[data protocol]"
  python -m json.tool "$OUT_ROOT/configs/data_protocol_manifest.json"
fi
if [[ -s "$OUT_ROOT/configs/precision_manifest.json" ]]; then
  echo
  echo "[precision policy]"
  python -m json.tool "$OUT_ROOT/configs/precision_manifest.json"
fi

QUALITY_BYPASS_FILE=""
for candidate in "$OUT_ROOT/eval/QUALITY_GATES_BYPASSED.txt" "$OUT_ROOT/QUALITY_GATES_BYPASSED.txt"; do
  if [[ -s "$candidate" ]]; then QUALITY_BYPASS_FILE="$candidate"; break; fi
done
if [[ -n "$QUALITY_BYPASS_FILE" ]]; then
  echo
  echo "STATUS: ENGINEERING-ONLY (one or more quality gates were bypassed)"
  echo "marker: $QUALITY_BYPASS_FILE"
  sed -n '1,20p' "$QUALITY_BYPASS_FILE"
fi

python - "$OUT_ROOT" <<'PY'
from __future__ import annotations
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
print("\n[stage histories]")
for stage in ("natural", "witness", "planner"):
    p = root / "checkpoints" / ("transport" if stage == "witness" else stage) / f"history_{stage}.json"
    if not p.exists():
        print(f"{stage}: missing")
        continue
    try:
        rows = json.loads(p.read_text(encoding="utf-8"))
        row = rows[-1] if rows else {}
        epoch = row.get("epoch")
        steps = row.get("train/runtime/optimizer_steps")
        skips = row.get("train/runtime/amp_skipped_steps")
        improved = row.get("checkpoint/improved")
        print(f"{stage}: rows={len(rows)} last_epoch={epoch} optimizer_steps={steps} amp_skips={skips} improved={improved}")
    except Exception as exc:
        print(f"{stage}: invalid history ({exc})")

print("\n[quality gates]")
for name in (
    "natural_basis_gate",
    "natural_effectiveness_gate",
    "mechanism_verification",
):
    p = root / "eval" / "learned_offline" / f"{name}.json"
    if not p.exists():
        print(f"{name}: missing")
        continue
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        passed = data.get("pass", data.get("passed"))
        print(f"{name}: pass={passed}")
    except Exception as exc:
        print(f"{name}: invalid ({exc})")
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
    echo "STATUS: INTENTIONAL_PARTIAL_RUN (the launcher stopped at the requested stage; this is not a full pipeline result)"
  elif [[ "$last_line" == *"complete:"* ]]; then
    echo "STATUS: COMPLETION_VALIDATOR_MISSING (inspect validator logs)"
  else
    echo "STATUS: INCOMPLETE_OR_FAILED (inspect the last non-empty log)"
  fi
fi
