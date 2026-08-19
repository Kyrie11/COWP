#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${PYTHON_BIN:-python}"
WOMD_ROOT="${WOMD_ROOT:?export WOMD_ROOT=...}"
SOURCE_TRAIN_PILOT_ROOT="${SOURCE_TRAIN_PILOT_ROOT:?export SOURCE_TRAIN_PILOT_ROOT=/path/to/formal_v16_8_21_train_pilot}"
REFERENCE_LABEL_CACHE="${REFERENCE_LABEL_CACHE:-$SOURCE_TRAIN_PILOT_ROOT/labels_train_v16_8_18}"
REFERENCE_PROFILE_JSONL="${REFERENCE_PROFILE_JSONL:-$SOURCE_TRAIN_PILOT_ROOT/profile_train_pilot.jsonl}"
BENCH_ROOT="${BENCH_ROOT:-/tmp/cowp_v16_8_23_fast_benchmark}"
BENCH_SCENES="${BENCH_SCENES:-64}"
LABEL_WORKERS="${LABEL_WORKERS:-16}"
LABEL_CFG="${LABEL_CFG:-configs/label_cowp_v16_8.yaml}"
SCENARIO_TRAIN="$WOMD_ROOT/uncompressed/scenario/training/*.tfrecord*"
INDEX="$BENCH_ROOT/scenario_location_index_train.jsonl"
IDS="$BENCH_ROOT/benchmark_scene_ids.txt"
FAST="$BENCH_ROOT/labels_fast"
PROFILE="$BENCH_ROOT/profile_fast.jsonl"
mkdir -p "$BENCH_ROOT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1

$PYTHON_BIN - "$REFERENCE_LABEL_CACHE" "$IDS" "$BENCH_SCENES" <<'PY'
from pathlib import Path
import hashlib,sys
root=Path(sys.argv[1]); out=Path(sys.argv[2]); n=int(sys.argv[3])
ids=[p.stem for p in root.glob('*.npz') if p.is_file()]
ids=sorted(set(ids), key=lambda x:(hashlib.sha256(('v16.8.23-bench|'+x).encode()).hexdigest(),x))[:n]
if len(ids)<n: raise SystemExit(f'reference cache has only {len(ids)} scenes, need {n}')
out.write_text('\n'.join(ids)+'\n',encoding='utf-8')
print(f'benchmark scenes={len(ids)}')
PY

$PYTHON_BIN -m cowp.scripts.72_build_scenario_location_index --proto-glob "$SCENARIO_TRAIN" --output "$INDEX" --meta-output "$BENCH_ROOT/index.meta.json" --reuse-if-valid
rm -rf "$FAST"; mkdir -p "$FAST"
$PYTHON_BIN -m cowp.scripts.01_build_labels_from_proto \
  --data-config configs/data.yaml --label-config "$LABEL_CFG" --proto-glob "$SCENARIO_TRAIN" \
  --output-dir "$FAST" --allow-scenario-ids "$IDS" --index-jsonl "$INDEX" --require-all-allowed-resolved \
  --num-workers "$LABEL_WORKERS" --start-method forkserver --max-pending-multiplier 2 \
  --no-compress --skip-diagnostics --cpu-only --profile-jsonl "$PROFILE"

$PYTHON_BIN -m cowp.scripts.66_compare_label_semantic_equivalence \
  --reference-dir "$REFERENCE_LABEL_CACHE" --candidate-dir "$FAST" --scene-ids "$IDS" \
  --output "$BENCH_ROOT/semantic_equivalence.json"
$PYTHON_BIN -m cowp.scripts.49_summarize_label_build_profile --input "$PROFILE" --output "$BENCH_ROOT/profile_fast_summary.json"

$PYTHON_BIN - "$IDS" "$REFERENCE_PROFILE_JSONL" "$PROFILE" "$BENCH_ROOT/semantic_equivalence.json" "$BENCH_ROOT/benchmark_report.json" <<'PY'
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
    return {'n':len(x),'sum_worker_s':sum(x),'mean_scene_s':statistics.mean(x),'p50_scene_s':q(.5),'p90_scene_s':q(.9),'max_scene_s':max(x)}
a,b=stats(old),stats(new)
speedup=(a.get('sum_worker_s',0)/b.get('sum_worker_s',1)) if a.get('n')==b.get('n') and a.get('n',0)>0 else None
report={'schema_version':'cowp_v16_8_23_fastpath_benchmark_v1','semantic_equivalence_pass':bool(sem.get('pass')),
        'reference':a,'fast':b,'matched_worker_time_speedup':speedup,
        'recommended_full_build':bool(sem.get('pass')) and (speedup is None or speedup>=1.25),
        'interpretation':'Full build is semantically authorized only after exact label parity. Speedup is based on matched per-scene worker time when the historical profile is available.'}
Path(sys.argv[5]).write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(report,indent=2,ensure_ascii=False))
if not report['semantic_equivalence_pass']: raise SystemExit(2)
PY

echo "Benchmark complete: $BENCH_ROOT/benchmark_report.json"
