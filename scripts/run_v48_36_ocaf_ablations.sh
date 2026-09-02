#!/usr/bin/env bash
set -Eeuo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"; export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"; export PYTHONUNBUFFERED=1
MAIN_RUN="${MAIN_RUN:-runs/ocrap_v48_36_ocaf_dedicated_4836}"
ROOT="${ABLATION_ROOT:-runs/ocrap_v48_36_ocaf_ablations_4836}"
SOURCE_RUN="${SOURCE_RUN:-runs/ocrap_v48_13_terra_proxy_4801}"
OCRAP_ROOT="${OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}"
PROTOCOL_ROOT="${PROTOCOL_ROOT:-$OCRAP_ROOT/calibration_v48_14_prism_4814}"
GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"
mkdir -p "$ROOT/tasks" "$ROOT/logs"
python tools/resolve_v48_36_authoritative_result.py --run "$MAIN_RUN" --output "$ROOT/MAIN_RUN_AUTHORIZATION.json" >/dev/null
python - "$ROOT/MAIN_RUN_AUTHORIZATION.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); rc=x.get('authoritative_exit_code')
if not x.get('valid') or rc not in (0,20): raise SystemExit('ablations require a valid evaluated v48.36 algorithm result (RC 0 or 20)')
PY
copy_indices(){ local out="$1"; mkdir -p "$out"; for f in evidence_adapt_teacher_pcd_index.jsonl evidence_adapt_teacher_pcd_index_summary.json evidence_adapt_dev_teacher_pcd_index.jsonl evidence_adapt_dev_teacher_pcd_index_summary.json; do [[ -s "$MAIN_RUN/$f" ]] || { echo "missing $MAIN_RUN/$f" >&2; return 30; }; cp -a "$MAIN_RUN/$f" "$out/$f"; done; }
run_task(){
  # Keep nounset-safe local initialization: later RHS values must not depend on
  # names assigned by the same `local` command.
  local name context prior out rc
  name="$1"
  context="$2"
  prior="$3"
  out="$ROOT/tasks/$name"
  rc=30
  rm -rf "$out"; mkdir -p "$out"
  if copy_indices "$out"; then
    set +e
    OUTPUTDIR="$out" SOURCE_RUN="$SOURCE_RUN" OCRAP_ROOT="$OCRAP_ROOT" PROTOCOL_ROOT="$PROTOCOL_ROOT" GPU0="$GPU0" GPU1="$GPU1" \
      EVIDENCE_CALIBRATOR_CONTEXT_SOURCE="$context" EVIDENCE_ADMISSION_PRIOR_MODE="$prior" \
      bash scripts/run_v48_36_ocaf_dedicated.sh >"$ROOT/logs/${name}.log" 2>&1
    rc=$?; set -e
  fi
  python - "$out" "$name" "$context" "$prior" "$rc" <<'PY'
import json,os,pathlib,sys,time
out=pathlib.Path(sys.argv[1]); rc=int(sys.argv[5]); status={}
try: status=json.load(open(out/'V48_36_COMPLETE.json'))
except Exception: pass
doc={'event':'v48_36_ablation_task_complete','version':'v48.36-OCAF','created_unix':time.time(),'name':sys.argv[2],
     'context_source':sys.argv[3],'admission_prior_mode':sys.argv[4],'raw_exit_code':rc,
     'algorithm_result':rc in (0,20) and status.get('pipeline_valid') is True,
     'pipeline_exit_code':status.get('pipeline_exit_code',30),'gate_evaluated':status.get('gate_evaluated',False),
     'gate_passed':status.get('gate_passed',False),'test_roots_read':False}
tmp=out/f'.ABLATION_TASK_COMPLETE.json.tmp.{os.getpid()}.{time.time_ns()}'; tmp.write_text(json.dumps(doc,indent=2)+'\n'); os.replace(tmp,out/'ABLATION_TASK_COMPLETE.json')
PY
}
# 2x2: representation (action-only vs observation-conditioned action) x admission (soft slack vs non-compensatory frontier).
# Tasks are independent; one engineering failure cannot suppress the remaining scientific controls.
run_task A_action_only_soft_slack physical_relative safety_slack
run_task B_ocaf_soft_slack physical_interaction safety_slack
run_task C_action_only_frontier physical_relative frontier_capped_slack
python - "$ROOT" "$MAIN_RUN" <<'PY'
import json,os,pathlib,sys,time
root=pathlib.Path(sys.argv[1]); main=pathlib.Path(sys.argv[2]); tasks={}
for n in ('A_action_only_soft_slack','B_ocaf_soft_slack','C_action_only_frontier'):
    tasks[n]=json.load(open(root/'tasks'/n/'ABLATION_TASK_COMPLETE.json'))
status=json.load(open(main/'V48_36_COMPLETE.json'))
tasks['D_ocaf_frontier_main']={'context_source':'physical_interaction','admission_prior_mode':'frontier_capped_slack',
 'pipeline_exit_code':status.get('pipeline_exit_code'),'pipeline_valid':status.get('pipeline_valid'),'gate_evaluated':status.get('gate_evaluated'),
 'gate_passed':status.get('gate_passed'),'source':str(main),'test_roots_read':False}
doc={'event':'v48_36_ocaf_ablation_suite_complete','version':'v48.36-OCAF','created_unix':time.time(),'complete':True,
 'design':'2x2 action-only versus observation-conditioned action interaction x compensatory slack versus non-compensatory frontier; one shared rule and no regime input in every task',
 'tasks':tasks,'engineering_failures_do_not_abort_independent_tasks':True,'test_roots_read':False}
tmp=root/f'.ABLATIONS_COMPLETE.json.tmp.{os.getpid()}.{time.time_ns()}'; tmp.write_text(json.dumps(doc,indent=2)+'\n'); os.replace(tmp,root/'ABLATIONS_COMPLETE.json')
PY
