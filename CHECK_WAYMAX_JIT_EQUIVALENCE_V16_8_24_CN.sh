#!/usr/bin/env bash
set -euo pipefail

# Exact semantic/speed gate for the optimized Waymax execution path.
# It deliberately keeps ALL dataset-defining parameters identical and changes
# only JAX execution:
#   reference  : eager env.step + eager Waymax safety metrics
#   optimized  : JIT env.step + JIT of the unchanged Waymax
#                OverlapMetric/OffroadMetric + SDC running max
# env.reset JIT is intentionally NOT enabled because the probe showed reset is
# ~0.02 s/candidate and therefore not worth extra compilation complexity.

PYTHON_BIN="${PYTHON_BIN:-python}"
COWP_ROOT="${COWP_ROOT:-/data0/senzeyu2/dataset/COWP/formal_v16_8_24_compact_full_5k}"
CACHE_DIR="${WAYMAX_JIT_CHECK_CACHE:-$COWP_ROOT/tensor_cache_train}"
OUT_ROOT="${WAYMAX_JIT_CHECK_ROOT:-$COWP_ROOT/waymax_jit_equivalence_v16_8_24}"
SCENES="${WAYMAX_JIT_CHECK_SCENES:-8}"
MAX_CANDIDATES="${MAX_REPLAY_CANDIDATES:-24}"
HORIZON="${REPLAY_HORIZON:-80}"
GPUS_CSV="${WAYMAX_JIT_CHECK_GPUS:-0,1}"
JAX_CACHE_ROOT="${WAYMAX_JAX_COMPILATION_CACHE_DIR:-$COWP_ROOT/.jax_compilation_cache_waymax_v16_8_24}"

IFS=',' read -r -a GPUS <<< "$GPUS_CSV"
[[ "${#GPUS[@]}" -ge 1 ]] || { echo "WAYMAX_JIT_CHECK_GPUS is empty" >&2; exit 2; }
REF_GPU="${GPUS[0]}"
OPT_GPU="${GPUS[1]:-${GPUS[0]}}"

[[ -d "$CACHE_DIR" ]] || { echo "Missing cache: $CACHE_DIR" >&2; exit 3; }
mkdir -p "$OUT_ROOT" "$JAX_CACHE_ROOT/gpu${REF_GPU}" "$JAX_CACHE_ROOT/gpu${OPT_GPU}"
EAGER="$OUT_ROOT/eager.jsonl"
OPT="$OUT_ROOT/optimized.jsonl"
EAGER_PROFILE="$OUT_ROOT/eager_profile.jsonl"
OPT_PROFILE="$OUT_ROOT/optimized_profile.jsonl"
EAGER_LOG="$OUT_ROOT/eager.log"
OPT_LOG="$OUT_ROOT/optimized.log"
REPORT="$OUT_ROOT/compare.json"
rm -f "$EAGER" "$OPT" "$EAGER_PROFILE" "$OPT_PROFILE" "$EAGER_LOG" "$OPT_LOG" "$REPORT"

common=(
  --data-config configs/data.yaml
  --label-config configs/label_cowp_v16_8.yaml
  --eval-config configs/eval_cowp_v16_8.yaml
  --cache-dir "$CACHE_DIR"
  --state-source cache
  --candidate-selection balanced
  --max-candidates-per-scene "$MAX_CANDIDATES"
  --rollout-horizon-steps "$HORIZON"
  --waymax-device gpu
  --waymax-action-mode absolute_xy_yaw
  --metric-set safety
  --metric-eval-mode step
  --metric-eval-interval 1
  --done-check-interval 1
  --limit-scenes "$SCENES"
  --gc-every-scenes 0
  --profile-detail scene
  --overwrite
)

run_ref(){
  (
    export CUDA_VISIBLE_DEVICES="$REF_GPU"
    export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
    export JAX_COMPILATION_CACHE_DIR="$JAX_CACHE_ROOT/gpu${REF_GPU}"
    "$PYTHON_BIN" -u -m cowp.scripts.13_replay_waymax_candidates \
      "${common[@]}" \
      --outcomes-jsonl "$EAGER" \
      --profile-replay-jsonl "$EAGER_PROFILE" \
      --progress-desc "equivalence eager GPU=$REF_GPU"
  ) > >(tee "$EAGER_LOG") 2>&1
}

run_opt(){
  (
    export CUDA_VISIBLE_DEVICES="$OPT_GPU"
    export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
    export JAX_COMPILATION_CACHE_DIR="$JAX_CACHE_ROOT/gpu${OPT_GPU}"
    "$PYTHON_BIN" -u -m cowp.scripts.13_replay_waymax_candidates \
      "${common[@]}" \
      --outcomes-jsonl "$OPT" \
      --profile-replay-jsonl "$OPT_PROFILE" \
      --jit-env-step \
      --jit-safety-metrics \
      --progress-desc "equivalence optimized GPU=$OPT_GPU"
  ) > >(tee "$OPT_LOG") 2>&1
}

if [[ "$REF_GPU" != "$OPT_GPU" ]]; then
  echo "=== Running eager and optimized gates concurrently on GPUs $REF_GPU/$OPT_GPU ==="
  run_ref & p0=$!
  run_opt & p1=$!
  rc=0
  wait "$p0" || rc=$?
  wait "$p1" || rc=$?
  [[ "$rc" -eq 0 ]] || { echo "equivalence worker failed rc=$rc" >&2; exit "$rc"; }
else
  echo "=== One GPU supplied; running gate sequentially on GPU $REF_GPU ==="
  run_ref
  run_opt
fi

"$PYTHON_BIN" - "$EAGER" "$OPT" "$EAGER_PROFILE" "$OPT_PROFILE" "$REPORT" <<'PY'
import json, math, statistics, sys
from pathlib import Path
ref_path,opt_path,ref_prof,opt_prof,report_path=map(Path,sys.argv[1:6])

def rows(p):
    d={}
    for line in p.read_text(encoding='utf-8').splitlines():
        if not line.strip(): continue
        r=json.loads(line)
        k=(str(r.get('scenario_id','')),int(r.get('candidate_index',-1)))
        d[k]=r
    return d

def profiles(p):
    out=[]
    if not p.exists(): return out
    for line in p.read_text(encoding='utf-8').splitlines():
        if line.strip(): out.append(json.loads(line))
    return out

def timing(ps):
    rollout=sum(float(r.get('rollout_candidates_s',0.0) or 0.0) for r in ps if r.get('status')=='ok')
    n=sum(int(r.get('new_rows',0) or 0)+int(r.get('failed_rows',0) or 0) for r in ps if r.get('status')=='ok')
    return {'rollout_s':rollout,'candidates':n,'s_per_candidate':rollout/max(n,1)}

a=rows(ref_path); b=rows(opt_path)
keys=sorted(set(a)|set(b))
missing_ref=[k for k in keys if k not in a]
missing_opt=[k for k in keys if k not in b]
mismatches=[]
for k in keys:
    if k not in a or k not in b: continue
    ra,rb=a[k],b[k]
    fields=('rollout_valid','collision','offroad','steps')
    diff={f:(ra.get(f),rb.get(f)) for f in fields if ra.get(f)!=rb.get(f)}
    if bool(ra.get('error')) != bool(rb.get('error')):
        diff['error_presence']=(ra.get('error'),rb.get('error'))
    if diff:
        mismatches.append({'key':list(k),'diff':diff})

rp=profiles(ref_prof); op=profiles(opt_prof)
rt=timing(rp); ot=timing(op)
speedup=(rt['s_per_candidate']/ot['s_per_candidate']) if ot['s_per_candidate']>0 else None
jit_step_active=bool(op) and all(bool(x.get('jit_env_step_active',x.get('jit_env_step',False))) for x in op if x.get('status')=='ok')
jit_metric_active=bool(op) and all(bool(x.get('jit_safety_metrics_active',False)) for x in op if x.get('status')=='ok')
payload={
  'schema_version':'cowp_v16_8_24_waymax_accel_equivalence_v2',
  'reference_rows':len(a),'optimized_rows':len(b),
  'missing_in_reference':len(missing_ref),'missing_in_optimized':len(missing_opt),
  'label_or_step_mismatches':len(mismatches),'mismatch_examples':mismatches[:20],
  'reference_timing':rt,'optimized_timing':ot,'speedup_x':speedup,
  'optimized_jit_env_step_active':jit_step_active,
  'optimized_jit_safety_metrics_active':jit_metric_active,
}
payload['semantic_pass']=(len(a)>0 and len(b)>0 and not missing_ref and not missing_opt and not mismatches)
payload['jit_active_pass']=bool(jit_step_active and jit_metric_active)
payload['pass']=bool(payload['semantic_pass'] and payload['jit_active_pass'])
report_path.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps(payload,indent=2,ensure_ascii=False))
if not payload['semantic_pass']:
    raise SystemExit('SEMANTIC EQUIVALENCE FAILED: do not resume full build with JIT.')
if not payload['jit_active_pass']:
    raise SystemExit('Outputs matched but one JIT path fell back to eager; inspect logs before full build.')
PY

echo "WAYMAX JIT SEMANTIC GATE: PASS"
echo "Recommended full-run acceleration flags:"
echo "  export WAYMAX_JIT_ENV_STEP=1"
echo "  export WAYMAX_JIT_SAFETY_METRICS=1"
echo "  export WAYMAX_JIT_ENV_RESET=0"
echo "Report: $REPORT"
