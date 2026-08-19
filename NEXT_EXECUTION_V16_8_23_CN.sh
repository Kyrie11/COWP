#!/usr/bin/env bash
set -euo pipefail
cmd="${1:-check}"
BENCH_ROOT="${BENCH_ROOT:-/tmp/cowp_v16_8_23_fast_benchmark}"
COWP_ROOT="${COWP_ROOT:-/data0/senzeyu2/dataset/COWP/formal_v16_8_23_compact_full}"
case "$cmd" in
  benchmark)
    bash BENCHMARK_V16_8_23_FASTPATHS_CN.sh
    ;;
  full-core)
    if [[ "${SKIP_SPEED_GATE:-0}" != "1" ]]; then
      [[ -f "$BENCH_ROOT/benchmark_report.json" ]] || {
        echo "Missing $BENCH_ROOT/benchmark_report.json. Run: bash NEXT_EXECUTION_V16_8_23_CN.sh benchmark" >&2; exit 2; }
      python - "$BENCH_ROOT/benchmark_report.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
if not r.get('semantic_equivalence_pass'):
    raise SystemExit('semantic equivalence did not pass; full build blocked')
if not r.get('recommended_full_build') and __import__('os').environ.get('ALLOW_SLOW_FULL','0') != '1':
    raise SystemExit('benchmark did not recommend full build (speedup < 1.25 when measurable); tune LABEL_WORKERS/BENCH_ROOT or set ALLOW_SLOW_FULL=1 explicitly')
print('semantic equivalence: PASS')
print('matched worker-time speedup:',r.get('matched_worker_time_speedup'))
print('recommended_full_build: PASS')
PY
    fi
    bash PREPARE_COWP_V16_8_23_FAST_DATA_CN.sh
    ;;
  outcomes)
    COWP_ROOT="$COWP_ROOT" bash ATTACH_WAYMAX_OUTCOMES_V16_8_10_CN.sh
    ;;
  check)
    echo "=== benchmark ==="
    [[ -f "$BENCH_ROOT/benchmark_report.json" ]] && cat "$BENCH_ROOT/benchmark_report.json" || echo "not run"
    echo "=== compact full ==="
    [[ -f "$COWP_ROOT/full_core_support_verdict_v16_8_23.json" ]] && cat "$COWP_ROOT/full_core_support_verdict_v16_8_23.json" || echo "not built"
    ;;
  *)
    echo "Usage: $0 {benchmark|full-core|outcomes|check}" >&2; exit 2;;
esac
