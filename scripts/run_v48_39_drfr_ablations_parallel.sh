#!/usr/bin/env bash
set -Eeuo pipefail
# Run A/B/C concurrently; D is the main v48.39 experiment and is not rerun by
# default. Each controller uses GPU0 for balanced and GPU1 for precision.
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
BASE_OUT="${BASE_OUT:-runs}"
RERUN_D="${RERUN_D:-0}"

# Three concurrent controllers => up to three processes per GPU. Limit host
# thread/data-loader fan-out so wall time is GPU-bound rather than CPU/I/O-bound.
export OMP_NUM_THREADS="${ABLATION_OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${ABLATION_MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${ABLATION_OPENBLAS_NUM_THREADS:-1}"
export NUM_WORKERS="${ABLATION_NUM_WORKERS:-2}"
export PREFETCH_FACTOR="${ABLATION_PREFETCH_FACTOR:-2}"

arms=(A B C)
[[ "$RERUN_D" == 1 ]] && arms+=(D)
logdir="$BASE_OUT/v48_39_parallel_logs"
mkdir -p "$logdir"
declare -A pids
for arm in "${arms[@]}"; do
  out="$BASE_OUT/ocrap_v48_39_drfr_ablation_${arm}"
  log="$logdir/arm_${arm}.launcher.log"
  echo "launch arm=$arm output=$out GPUs=$GPU0,$GPU1" | tee "$log"
  (
    set +e
    OUTPUTDIR="$out" GPU0="$GPU0" GPU1="$GPU1" \
      bash scripts/run_v48_39_drfr_ablation_arm.sh "$arm" >>"$log" 2>&1
    rc=$?
    printf '%s\n' "$rc" > "$logdir/arm_${arm}.rc"
    exit "$rc"
  ) &
  pids[$arm]=$!
done

launcher_failure=0
set +e
for arm in "${arms[@]}"; do
  wait "${pids[$arm]}"; wait_rc=$?
  rc="$wait_rc"; [[ -f "$logdir/arm_${arm}.rc" ]] && rc="$(cat "$logdir/arm_${arm}.rc")"
  printf 'arm=%s controller_rc=%s\n' "$arm" "$rc"
  # 0 and 20 are scientifically valid terminal codes. Any other code indicates
  # an engineering/protocol failure and the launcher reports RC30 after all arms
  # have been allowed to finish.
  if [[ "$rc" != 0 && "$rc" != 20 ]]; then launcher_failure=1; fi
done
set -e

python - "$BASE_OUT" "${arms[@]}" <<'PY_SUMMARY'
import json,pathlib,sys
base=pathlib.Path(sys.argv[1]); arms=sys.argv[2:]; out={}
for arm in arms:
    run=base/f'ocrap_v48_39_drfr_ablation_{arm}'
    p=run/'AUTHORITATIVE_RUN_STATUS.json'
    try:
        doc=json.load(open(p)); out[arm]={
            'valid':doc.get('valid'),'pipeline_valid':doc.get('pipeline_valid'),
            'authoritative_exit_code':doc.get('authoritative_exit_code'),'run':str(run),
        }
    except Exception as exc:
        out[arm]={'run':str(run),'status_error':repr(exc)}
summary=base/'v48_39_parallel_logs'/'ABLATION_PARALLEL_SUMMARY.json'
summary.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(out,ensure_ascii=False,indent=2))
PY_SUMMARY

[[ "$launcher_failure" == 0 ]] || exit 30
exit 0
