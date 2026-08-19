#!/usr/bin/env bash
set -euo pipefail

cmd="${1:-check}"
PYTHON_BIN="${PYTHON_BIN:-python}"
BENCH_ROOT="${BENCH_ROOT:-/data0/senzeyu2/dataset/COWP/formal_v16_8_24_fast_benchmark}"
COWP_ROOT="${COWP_ROOT:-/data0/senzeyu2/dataset/COWP/formal_v16_8_24_compact_full}"
PREFLIGHT_ROOT="${PREFLIGHT_ROOT:-$COWP_ROOT/preflight}"

preflight(){
  mkdir -p "$PREFLIGHT_ROOT"
  "$PYTHON_BIN" -m cowp.scripts.77_audit_active_execution_chain \
    --repo-root . --output "$PREFLIGHT_ROOT/active_execution_chain.json"
  if [[ -n "${WOMD_ROOT:-}" ]]; then
    "$PYTHON_BIN" -m cowp.scripts.69_audit_womd_split_layout \
      --womd-root "$WOMD_ROOT" --sample-scenario-shards "${WOMD_LAYOUT_SAMPLE_SHARDS:-8}" \
      --output "$PREFLIGHT_ROOT/womd_v131_split_layout.json"
  else
    echo "WOMD_ROOT not set: skipped WOMD split-layout audit." >&2
  fi
}

case "$cmd" in
  preflight)
    preflight
    ;;
  benchmark)
    preflight
    REPORT_ONLY=0 bash BENCHMARK_V16_8_24_FASTPATHS_CN.sh
    ;;
  recover-benchmark)
    # Use this after the V23 run that completed labels + semantic equivalence
    # but failed only because it called the nonexistent module 44.  No labels
    # are rebuilt; the report is regenerated from the existing artifacts.
    preflight
    REPORT_ONLY=1 bash BENCHMARK_V16_8_24_FASTPATHS_CN.sh
    ;;
  full-core)
    preflight
    if [[ "${SKIP_SPEED_GATE:-0}" != "1" ]]; then
      [[ -f "$BENCH_ROOT/benchmark_report.json" ]] || {
        echo "Missing $BENCH_ROOT/benchmark_report.json." >&2
        echo "Run: bash NEXT_EXECUTION_V16_8_24_CN.sh recover-benchmark (for your completed V23 benchmark)" >&2
        echo "or : bash NEXT_EXECUTION_V16_8_24_CN.sh benchmark" >&2
        exit 2
      }
      "$PYTHON_BIN" - "$BENCH_ROOT/benchmark_report.json" <<'PY'
import json,os,sys
r=json.load(open(sys.argv[1],encoding='utf-8'))
if not r.get('semantic_equivalence_pass'):
    raise SystemExit('semantic equivalence did not pass; full build blocked')
sp=r.get('matched_worker_time_speedup')
if not r.get('recommended_full_build') and os.environ.get('ALLOW_SLOW_FULL','0') != '1':
    raise SystemExit('benchmark did not recommend full build; inspect report or explicitly set ALLOW_SLOW_FULL=1')
print('semantic equivalence: PASS')
print('matched worker-time speedup:',sp)
print('recommended_full_build: PASS')
PY
    fi
    bash PREPARE_COWP_V16_8_24_FAST_DATA_CN.sh
    ;;
  outcomes)
    COWP_ROOT="$COWP_ROOT" bash ATTACH_WAYMAX_OUTCOMES_V16_8_24_CN.sh
    ;;
  check)
    echo "=== active execution chain ==="
    if [[ -f "$PREFLIGHT_ROOT/active_execution_chain.json" ]]; then cat "$PREFLIGHT_ROOT/active_execution_chain.json"; else echo "not run"; fi
    echo "=== WOMD layout ==="
    if [[ -f "$PREFLIGHT_ROOT/womd_v131_split_layout.json" ]]; then cat "$PREFLIGHT_ROOT/womd_v131_split_layout.json"; else echo "not run"; fi
    echo "=== benchmark ==="
    if [[ -f "$BENCH_ROOT/benchmark_report.json" ]]; then cat "$BENCH_ROOT/benchmark_report.json"; else echo "not run/recovered"; fi
    echo "=== compact full ==="
    if [[ -f "$COWP_ROOT/full_core_support_verdict_v16_8_24.json" ]]; then cat "$COWP_ROOT/full_core_support_verdict_v16_8_24.json"; else echo "not built"; fi
    echo "=== Waymax outcomes ==="
    if [[ -f "$COWP_ROOT/waymax_outcome_support_v16_8_24.json" ]]; then cat "$COWP_ROOT/waymax_outcome_support_v16_8_24.json"; else echo "not attached"; fi
    ;;
  *)
    echo "Usage: $0 {preflight|benchmark|recover-benchmark|full-core|outcomes|check}" >&2
    exit 2
    ;;
esac
