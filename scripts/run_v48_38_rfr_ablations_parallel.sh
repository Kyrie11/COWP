#!/usr/bin/env bash
set -Eeuo pipefail
# Run the three *additional* v48.38 ablations concurrently. D is the main RFR
# experiment and is not redundantly rerun by default. Set RERUN_D=1 if desired.
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
BASE_OUT="${BASE_OUT:-runs}"
RERUN_D="${RERUN_D:-0}"

# Three controllers mean up to three training processes per GPU. RAM is not the
# bottleneck; cap CPU thread/data-loader fan-out so concurrent GPU jobs do not
# spend wall time fighting over host scheduling and disk queues.
export OMP_NUM_THREADS="${ABLATION_OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${ABLATION_MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${ABLATION_OPENBLAS_NUM_THREADS:-1}"
export NUM_WORKERS="${ABLATION_NUM_WORKERS:-2}"
export PREFETCH_FACTOR="${ABLATION_PREFETCH_FACTOR:-2}"

arms=(A B C)
[[ "$RERUN_D" == 1 ]] && arms+=(D)
mkdir -p "$BASE_OUT/v48_38_parallel_logs"
declare -A pids
for arm in "${arms[@]}"; do
  out="$BASE_OUT/ocrap_v48_38_rfr_ablation_${arm}"
  log="$BASE_OUT/v48_38_parallel_logs/arm_${arm}.launcher.log"
  echo "launch arm=$arm output=$out GPUs=$GPU0,$GPU1" | tee "$log"
  (
    set +e
    OUTPUTDIR="$out" GPU0="$GPU0" GPU1="$GPU1" \
      bash scripts/run_v48_38_rfr_ablation_arm.sh "$arm" >>"$log" 2>&1
    rc=$?
    printf '%s\n' "$rc" > "$BASE_OUT/v48_38_parallel_logs/arm_${arm}.rc"
    exit "$rc"
  ) &
  pids[$arm]=$!
done

# RC=20 is an expected valid algorithmic result and must not cause the launcher
# to kill the other arms. Wait for all jobs and report each authoritative code.
launcher_failure=0
set +e
for arm in "${arms[@]}"; do
  wait "${pids[$arm]}"
  wait_rc=$?
  rc_file="$BASE_OUT/v48_38_parallel_logs/arm_${arm}.rc"
  rc="$wait_rc"
  [[ -f "$rc_file" ]] && rc="$(cat "$rc_file")"
  printf 'arm=%s controller_rc=%s\n' "$arm" "$rc"
  if [[ "$rc" != 0 && "$rc" != 20 ]]; then launcher_failure=1; fi
done
set -e

python - "$BASE_OUT" "${arms[@]}" <<'PY_SUMMARY'
import json,pathlib,sys
base=pathlib.Path(sys.argv[1]); arms=sys.argv[2:]; out={}
for arm in arms:
    run=base/f'ocrap_v48_38_rfr_ablation_{arm}'
    p=run/'AUTHORITATIVE_RUN_STATUS.json'
    try:
        doc=json.load(open(p)); out[arm]={
            'valid':doc.get('valid'),'pipeline_valid':doc.get('pipeline_valid'),
            'authoritative_exit_code':doc.get('authoritative_exit_code'),
            'run':str(run),
        }
    except Exception as exc:
        out[arm]={'run':str(run),'status_error':repr(exc)}
summary=base/'v48_38_parallel_logs'/'ABLATION_PARALLEL_SUMMARY.json'
summary.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(out,ensure_ascii=False,indent=2))
PY_SUMMARY

if [[ "$launcher_failure" == 1 ]]; then
  echo "at least one arm had an engineering/non-gate exit code; inspect launcher logs" >&2
  exit 30
fi
exit 0
