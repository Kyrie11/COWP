#!/usr/bin/env bash
set -Eeuo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

MAIN_RUN="${MAIN_RUN:-runs/ocrap_v48_35_continuous_frontier_dedicated_4835}"
ROOT="${ABLATION_ROOT:-runs/ocrap_v48_35_continuous_frontier_ablations_4835}"
SOURCE_RUN="${SOURCE_RUN:-runs/ocrap_v48_13_terra_proxy_4801}"
OCRAP_ROOT="${OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}"
PROTOCOL_ROOT="${PROTOCOL_ROOT:-$OCRAP_ROOT/calibration_v48_14_prism_4814}"
GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"
mkdir -p "$ROOT/tasks" "$ROOT/logs"

python - "$MAIN_RUN/V48_35_COMPLETE.json" <<'PY_AUTH'
import json,pathlib,sys
p=pathlib.Path(sys.argv[1])
if not p.is_file(): raise SystemExit(f'missing main completion contract: {p}')
d=json.load(open(p,encoding='utf-8'))
if not (d.get('pipeline_valid') and d.get('certificate_executed') and d.get('gate_evaluated')):
    raise SystemExit('ablations require a valid evaluated v48.35 main run')
if int(d.get('pipeline_exit_code',-1)) not in (0,20):
    raise SystemExit('main run is not an algorithm result (expected RC 0 or 20)')
if d.get('test_roots_read',True): raise SystemExit('main run reports test-root access')
PY_AUTH

copy_indices() {
  local out="$1"
  mkdir -p "$out"
  for f in evidence_adapt_teacher_pcd_index.jsonl evidence_adapt_teacher_pcd_index_summary.json \
           evidence_adapt_dev_teacher_pcd_index.jsonl evidence_adapt_dev_teacher_pcd_index_summary.json; do
    [[ -s "$MAIN_RUN/$f" ]] || { echo "missing main index $MAIN_RUN/$f" >&2; exit 30; }
    cp -a "$MAIN_RUN/$f" "$out/$f"
  done
}

run_task() {
  local name="$1" context="$2" prior="$3" factor_root="${4:-}"
  local out="$ROOT/tasks/$name"
  rm -rf "$out"; copy_indices "$out"
  local cache_args=()
  if [[ -n "$factor_root" ]]; then
    cache_args+=(
      "V4835_FACTOR_CACHE_BALANCED=$factor_root/candidates/balanced/factor_stage"
      "V4835_FACTOR_CACHE_PRECISION=$factor_root/candidates/precision/factor_stage"
    )
  fi
  set +e
  env "${cache_args[@]}" \
    OUTPUTDIR="$out" SOURCE_RUN="$SOURCE_RUN" OCRAP_ROOT="$OCRAP_ROOT" PROTOCOL_ROOT="$PROTOCOL_ROOT" \
    GPU0="$GPU0" GPU1="$GPU1" \
    EVIDENCE_CALIBRATOR_CONTEXT_SOURCE="$context" \
    EVIDENCE_ADMISSION_PRIOR_MODE="$prior" \
    bash scripts/run_v48_35_continuous_frontier_dedicated.sh \
      >"$ROOT/logs/${name}.log" 2>&1
  local rc=$?
  set -e
  if [[ "$rc" != 0 && "$rc" != 20 ]]; then
    echo "$name failed with pipeline RC=$rc" >&2
    return 30
  fi
  python - "$out" "$name" "$context" "$prior" "$rc" <<'PY_TASK'
import json,pathlib,sys,time
out=pathlib.Path(sys.argv[1]); status=json.load(open(out/'V48_35_COMPLETE.json',encoding='utf-8'))
doc={'event':'v48_35_ablation_task_complete','created_unix':time.time(),
     'name':sys.argv[2],'context_source':sys.argv[3],'admission_prior_mode':sys.argv[4],
     'pipeline_exit_code':int(sys.argv[5]),'pipeline_valid':status.get('pipeline_valid'),
     'gate_evaluated':status.get('gate_evaluated'),'gate_passed':status.get('gate_passed'),
     'test_roots_read':False}
(out/'ABLATION_TASK_COMPLETE.json').write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
PY_TASK
}

# D is the main run. A isolates legacy compact context + compensatory slack.
# B changes representation only (reuses D's physical factor stage).
# C changes the non-compensatory cap only relative to A (reuses A's factor stage).
run_task A_legacy_context_soft_slack relative safety_slack
run_task B_physical_context_soft_slack physical_relative safety_slack "$MAIN_RUN"
run_task C_legacy_context_frontier_cap relative frontier_capped_slack "$ROOT/tasks/A_legacy_context_soft_slack"

python - "$ROOT" "$MAIN_RUN" <<'PY_STATUS'
import json,pathlib,sys,time
root=pathlib.Path(sys.argv[1]); main=pathlib.Path(sys.argv[2])
tasks={}
for name in ('A_legacy_context_soft_slack','B_physical_context_soft_slack','C_legacy_context_frontier_cap'):
    tasks[name]=json.load(open(root/'tasks'/name/'ABLATION_TASK_COMPLETE.json',encoding='utf-8'))
main_status=json.load(open(main/'V48_35_COMPLETE.json',encoding='utf-8'))
tasks['D_physical_context_frontier_cap_main']={
  'context_source':'physical_relative','admission_prior_mode':'frontier_capped_slack',
  'pipeline_exit_code':main_status.get('pipeline_exit_code'),'pipeline_valid':main_status.get('pipeline_valid'),
  'gate_evaluated':main_status.get('gate_evaluated'),'gate_passed':main_status.get('gate_passed'),
  'source':str(main),'test_roots_read':False}
doc={'event':'v48_35_ablation_suite_complete','created_unix':time.time(),
     'design':'2x2 representation (legacy relative vs executable physical-relative) x admission (soft compensatory slack vs non-compensatory frontier cap); one shared rule in every task',
     'tasks':tasks,'complete':True,'test_roots_read':False}
(root/'ABLATIONS_COMPLETE.json').write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
PY_STATUS
