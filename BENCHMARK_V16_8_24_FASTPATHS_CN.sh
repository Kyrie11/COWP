#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
WOMD_ROOT="${WOMD_ROOT:-}"
SOURCE_TRAIN_PILOT_ROOT="${SOURCE_TRAIN_PILOT_ROOT:?export SOURCE_TRAIN_PILOT_ROOT=/path/to/formal_v16_8_21_train_pilot}"
REFERENCE_LABEL_CACHE="${REFERENCE_LABEL_CACHE:-$SOURCE_TRAIN_PILOT_ROOT/labels_train_v16_8_18}"
REFERENCE_PROFILE_JSONL="${REFERENCE_PROFILE_JSONL:-$SOURCE_TRAIN_PILOT_ROOT/profile_train_pilot.jsonl}"
BENCH_ROOT="${BENCH_ROOT:-/tmp/cowp_v16_8_24_fast_benchmark}"
BENCH_SCENES="${BENCH_SCENES:-64}"
LABEL_WORKERS="${LABEL_WORKERS:-40}"
LABEL_CFG="${LABEL_CFG:-configs/label_cowp_v16_8.yaml}"
REPORT_ONLY="${REPORT_ONLY:-0}"
COWP_CPUSET="${COWP_CPUSET:-}"

SCENARIO_TRAIN="${WOMD_ROOT:+$WOMD_ROOT/uncompressed/scenario/training/*.tfrecord*}"
INDEX_CACHE_ROOT="${INDEX_CACHE_ROOT:-${WOMD_ROOT:+$WOMD_ROOT/.cowp_v131_indices}}"
SCENARIO_INDEX_TRAIN="${SCENARIO_INDEX_TRAIN:-${INDEX_CACHE_ROOT:+$INDEX_CACHE_ROOT/scenario_location_index_train.jsonl}}"
SCENARIO_INDEX_TRAIN_META="${SCENARIO_INDEX_TRAIN_META:-${INDEX_CACHE_ROOT:+$INDEX_CACHE_ROOT/scenario_location_index_train.meta.json}}"
IDS="$BENCH_ROOT/benchmark_scene_ids.txt"
FAST="$BENCH_ROOT/labels_fast"
PROFILE="$BENCH_ROOT/profile_fast.jsonl"
SEMANTIC="$BENCH_ROOT/semantic_equivalence.json"
WALL_FILE="$BENCH_ROOT/fast_build_wall_seconds.txt"
mkdir -p "$BENCH_ROOT"
[[ -z "$INDEX_CACHE_ROOT" ]] || mkdir -p "$INDEX_CACHE_ROOT"

# Every process gets one BLAS/OpenMP thread.  The label engine is scene-level
# multiprocessing with many small geometry/NumPy kernels; nested BLAS threads
# cause severe oversubscription on dual-socket machines.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
export CUDA_VISIBLE_DEVICES=-1

run_cpu_bound(){
  if [[ -n "$COWP_CPUSET" ]] && command -v taskset >/dev/null 2>&1; then
    taskset -c "$COWP_CPUSET" "$@"
  else
    "$@"
  fi
}

make_report(){
  "$PYTHON_BIN" -m cowp.scripts.49_summarize_label_build_profile \
    --input "$PROFILE" --output "$BENCH_ROOT/profile_fast_summary.json" --top-slow 20

  "$PYTHON_BIN" - "$IDS" "$REFERENCE_PROFILE_JSONL" "$PROFILE" "$SEMANTIC" "$WALL_FILE" "$BENCH_ROOT/benchmark_report.json" <<'PY'
import json,statistics,sys
from pathlib import Path
ids=set(x.strip() for x in Path(sys.argv[1]).read_text().splitlines() if x.strip())
def rows(path):
    p=Path(path)
    if not p.is_file(): return []
    out=[]
    for line in p.read_text().splitlines():
        if not line.strip(): continue
        try:r=json.loads(line)
        except Exception:continue
        if str(r.get('scenario_id','')) in ids and r.get('status')=='written': out.append(r)
    return out
old=rows(sys.argv[2]); new=rows(sys.argv[3])
sem=json.loads(Path(sys.argv[4]).read_text())
def stats(rs):
    x=[float(r.get('seconds',0.0)) for r in rs]
    if not x:return {'n':0}
    x=sorted(x)
    q=lambda p:x[min(len(x)-1,int(round((len(x)-1)*p)))]
    return {'n':len(x),'sum_worker_s':sum(x),'mean_scene_s':statistics.mean(x),
            'p50_scene_s':q(.5),'p90_scene_s':q(.9),'max_scene_s':max(x)}
a,b=stats(old),stats(new)
speedup=(a.get('sum_worker_s',0)/b.get('sum_worker_s',1)) if a.get('n')==b.get('n') and a.get('n',0)>0 else None
wall_path=Path(sys.argv[5]); wall=None
if wall_path.is_file():
    try: wall=float(wall_path.read_text().strip())
    except Exception: wall=None
report={
    'schema_version':'cowp_v16_8_24_fastpath_benchmark_v1',
    'semantic_equivalence_pass':bool(sem.get('pass')),
    'reference':a,
    'fast':b,
    'fast_wall_seconds':wall,
    'matched_worker_time_speedup':speedup,
    'recommended_full_build':bool(sem.get('pass')) and (speedup is None or speedup>=1.25),
    'interpretation':(
      'Semantic parity is the hard correctness gate. Worker-time speedup is reported only when the historical per-scene profile is available; wall time is measured for newly-run benchmarks.'
    ),
}
Path(sys.argv[6]).write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(report,indent=2,ensure_ascii=False))
if not report['semantic_equivalence_pass']: raise SystemExit(2)
PY
}

if [[ "$REPORT_ONLY" == "1" ]]; then
  [[ -f "$IDS" && -f "$PROFILE" && -f "$SEMANTIC" ]] || {
    echo "REPORT_ONLY=1 requires existing benchmark_scene_ids.txt, profile_fast.jsonl, and semantic_equivalence.json under $BENCH_ROOT" >&2
    exit 3
  }
  make_report
  echo "Recovered benchmark report without rebuilding labels: $BENCH_ROOT/benchmark_report.json"
  exit 0
fi

[[ -n "$WOMD_ROOT" ]] || { echo "export WOMD_ROOT=... is required for a fresh benchmark" >&2; exit 3; }
SCENARIO_TRAIN="$WOMD_ROOT/uncompressed/scenario/training/*.tfrecord*"
INDEX_CACHE_ROOT="${INDEX_CACHE_ROOT:-$WOMD_ROOT/.cowp_v131_indices}"
SCENARIO_INDEX_TRAIN="${SCENARIO_INDEX_TRAIN:-$INDEX_CACHE_ROOT/scenario_location_index_train.jsonl}"
SCENARIO_INDEX_TRAIN_META="${SCENARIO_INDEX_TRAIN_META:-$INDEX_CACHE_ROOT/scenario_location_index_train.meta.json}"
mkdir -p "$INDEX_CACHE_ROOT"

"$PYTHON_BIN" - "$REFERENCE_LABEL_CACHE" "$IDS" "$BENCH_SCENES" <<'PY'
from pathlib import Path
import hashlib,sys
root=Path(sys.argv[1]); out=Path(sys.argv[2]); n=int(sys.argv[3])
ids=[p.stem for p in root.glob('*.npz') if p.is_file()]
ids=sorted(set(ids), key=lambda x:(hashlib.sha256(('v16.8.24-bench|'+x).encode()).hexdigest(),x))[:n]
if len(ids)<n: raise SystemExit(f'reference cache has only {len(ids)} scenes, need {n}')
out.write_text('\n'.join(ids)+'\n',encoding='utf-8')
print(f'benchmark scenes={len(ids)}')
PY

# IMPORTANT: this index is persistent and shared with full-core.  The v23
# benchmark put a ~95 MB training index under BENCH_ROOT and full-core then
# rebuilt it under COWP_ROOT.  That duplicated an O(full-WOMD-training) scan.
"$PYTHON_BIN" -m cowp.scripts.72_build_scenario_location_index \
  --proto-glob "$SCENARIO_TRAIN" --output "$SCENARIO_INDEX_TRAIN" \
  --meta-output "$SCENARIO_INDEX_TRAIN_META" --reuse-if-valid
printf '%s\n' "$SCENARIO_INDEX_TRAIN" > "$BENCH_ROOT/scenario_index_used.txt"

rm -rf "$FAST"; mkdir -p "$FAST"
start_ns=$(date +%s%N)
run_cpu_bound "$PYTHON_BIN" -m cowp.scripts.01_build_labels_from_proto \
  --data-config configs/data.yaml --label-config "$LABEL_CFG" --proto-glob "$SCENARIO_TRAIN" \
  --output-dir "$FAST" --allow-scenario-ids "$IDS" --index-jsonl "$SCENARIO_INDEX_TRAIN" --require-all-allowed-resolved \
  --num-workers "$LABEL_WORKERS" --start-method forkserver --max-pending-multiplier 2 \
  --no-compress --skip-diagnostics --cpu-only --profile-jsonl "$PROFILE"
end_ns=$(date +%s%N)
"$PYTHON_BIN" - "$start_ns" "$end_ns" "$WALL_FILE" <<'PY'
from pathlib import Path
import sys
s,e=int(sys.argv[1]),int(sys.argv[2])
Path(sys.argv[3]).write_text(f'{(e-s)/1e9:.6f}\n')
PY

"$PYTHON_BIN" -m cowp.scripts.66_compare_label_semantic_equivalence \
  --reference-dir "$REFERENCE_LABEL_CACHE" --candidate-dir "$FAST" --scene-ids "$IDS" \
  --output "$SEMANTIC"
make_report

echo "Benchmark complete: $BENCH_ROOT/benchmark_report.json"
