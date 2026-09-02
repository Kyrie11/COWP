#!/usr/bin/env bash
set -Eeuo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
BASE_OUT="${BASE_OUT:-runs}"
GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"
if [[ "$GPU0" == "$GPU1" && "${V4850_ALLOW_SHARED_GPU_IDS:-0}" != 1 ]]; then
  echo "v48.50 2x2 requires two distinct GPU ids (GPU0=$GPU0 GPU1=$GPU1); set V4850_ALLOW_SHARED_GPU_IDS=1 only for explicit debugging" >&2
  exit 2
fi
export OCRAP_ROOT="${OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}"
export PROTOCOL_ROOT="${PROTOCOL_ROOT:-$OCRAP_ROOT/calibration_v48_14_prism_4814}"
export CAL_NEAR="${CAL_NEAR:-$OCRAP_ROOT/calibration_near_contact}"
export CAL_CONTACT="${CAL_CONTACT:-$OCRAP_ROOT/calibration_contact}"
export CAL_SAFE="${CAL_SAFE:-$OCRAP_ROOT/calibration_safe}"
export OMP_NUM_THREADS="${ABLATION_OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS" OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export NUM_WORKERS="${ABLATION_NUM_WORKERS:-3}" PREFETCH_FACTOR="${ABLATION_PREFETCH_FACTOR:-3}"
export CACHE_SAMPLES_IN_MEMORY=false
export PERSISTENT_TENSOR_CACHE=true
export PERSISTENT_TENSOR_CACHE_DIR="${OCRAP_TENSOR_CACHE_DIR:-$OCRAP_ROOT/.ocrap_tensor_cache_v4846}"
# Reduces allocator fragmentation; it cannot make an occupied GPU appear free.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$BASE_OUT" "$PERSISTENT_TENSOR_CACHE_DIR"

# v48.47-C showed that free *capacity*, not model tensor size, caused the OOM:
# another process held ~20.9 GiB on a 23.6 GiB A30.  Keep two arms concurrent
# (one per physical GPU) but fail/wait before leasing a card to an arm.
GPU_WAIT_TIMEOUT_MIN="${V4850_GPU_WAIT_TIMEOUT_MIN:-240}"
GPU_POLL_SECONDS="${V4850_GPU_POLL_SECONDS:-30}"
VARIANT_MODE="${V4850_VARIANT_MODE:-serial}"
case "$VARIANT_MODE" in
  serial) SERIAL_VARIANTS=1; default_min_free_mb=12000 ;;
  parallel) SERIAL_VARIANTS=0; default_min_free_mb=20000 ;;
  *) echo "V4850_VARIANT_MODE must be serial|parallel" >&2; exit 2 ;;
esac
GPU_MIN_FREE_MB="${V4850_GPU_MIN_FREE_MB:-$default_min_free_mb}"
# The scientifically clean default is serial Balanced/Precision *within* each
# arm. A/B (and later C/D) still run at the same time on GPU0/GPU1, so both A30s
# remain occupied without placing four training processes on the two cards.

free_mb() {
  local gpu="$1"
  nvidia-smi --id="$gpu" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc '0-9.'
}
compute_apps_json() {
  local gpu="$1"
  nvidia-smi --id="$gpu" --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null | \
    python -c 'import json,sys; rows=[]
for line in sys.stdin:
 p=[x.strip() for x in line.split(",")]
 if p and p[0].isdigit(): rows.append({"pid":int(p[0]),"used_memory_mb":float(p[1]) if len(p)>1 and p[1].replace(".","",1).isdigit() else None})
print(json.dumps(rows,separators=(",",":")))' || printf '[]\n'
}
wait_for_gpu_lease() {
  local gpu="$1" arm="$2" start now free apps
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi missing; cannot establish v48.50 GPU memory lease" >&2
    return 30
  fi
  start="$(date +%s)"
  while true; do
    free="$(free_mb "$gpu")"
    apps="$(compute_apps_json "$gpu")"
    if [[ "$free" =~ ^[0-9]+([.][0-9]+)?$ ]] && awk -v f="$free" -v m="$GPU_MIN_FREE_MB" 'BEGIN{exit !(f>=m)}'; then
      GPU="$gpu" ARM="$arm" FREE="$free" APPS="$apps" MINFREE="$GPU_MIN_FREE_MB" python - <<'PY'
import json,os,time
print(json.dumps({'event':'v48_50_gpu_lease_granted','unix':time.time(),'gpu':int(os.environ['GPU']),
 'arm':os.environ['ARM'],'free_mb':float(os.environ['FREE']),'minimum_free_mb':float(os.environ['MINFREE']),
 'preexisting_compute_apps':json.loads(os.environ['APPS'])},separators=(',',':')))
PY
      return 0
    fi
    now="$(date +%s)"
    if (( now - start >= GPU_WAIT_TIMEOUT_MIN * 60 )); then
      echo "GPU lease timeout arm=$arm gpu=$gpu free_mb=${free:-unknown} required=$GPU_MIN_FREE_MB apps=$apps" >&2
      return 30
    fi
    echo "waiting for GPU lease: arm=$arm gpu=$gpu free_mb=${free:-unknown} required=$GPU_MIN_FREE_MB apps=$apps" >&2
    sleep "$GPU_POLL_SECONDS"
  done
}

PERF_LOG="$BASE_OUT/OC-RAP-v48.50-runtime-telemetry.jsonl"
: > "$PERF_LOG"
perf_pid=""
if command -v nvidia-smi >/dev/null 2>&1; then
  (
    while true; do
      ts="$(date +%s)"; load="$(awk '{print $1","$2","$3}' /proc/loadavg 2>/dev/null || true)"
      mem_kb="$(awk '/MemAvailable:/{print $2}' /proc/meminfo 2>/dev/null || true)"
      nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
        --format=csv,noheader,nounits 2>/dev/null | \
      TS="$ts" LOAD="$load" MEMKB="$mem_kb" python -c 'import json,sys,os
for line in sys.stdin:
 p=[x.strip() for x in line.split(",")]
 print(json.dumps({"unix":float(os.environ["TS"]),"loadavg":os.environ.get("LOAD",""),"mem_available_kb":int(os.environ["MEMKB"]) if os.environ.get("MEMKB") else None,"gpu":int(p[0]),"gpu_util_pct":float(p[1]),"gpu_mem_util_pct":float(p[2]),"gpu_mem_used_mb":float(p[3]),"gpu_mem_total_mb":float(p[4]),"power_w":float(p[5]) if p[5] not in {"N/A","[N/A]"} else None}))' >> "$PERF_LOG" || true
      sleep "${V4850_TELEMETRY_INTERVAL_S:-30}"
    done
  ) & perf_pid=$!
fi
cleanup_perf(){ [[ -z "$perf_pid" ]] || kill "$perf_pid" 2>/dev/null || true; }
trap cleanup_perf EXIT INT TERM

bash scripts/prepare_v48_45_protocol.sh
export V4845_SKIP_PROTOCOL_PREPARE=1
[[ -s "$PROTOCOL_ROOT/V48_45_PROTOCOL_SEAL.json" ]] || { echo "missing shared protocol seal" >&2; exit 30; }

arm_out(){
  case "$1" in
    D) printf '%s/ocrap_v48_50_dcp_de_main' "$BASE_OUT" ;;
    A|B|C) printf '%s/ocrap_v48_50_dcp_de_ablation_%s' "$BASE_OUT" "$1" ;;
    *) return 2 ;;
  esac
}
run_arm(){
  local arm="$1" gpu="$2" out log rc free apps
  out="$(arm_out "$arm")"; rm -rf "$out"; mkdir -p "$out/logs"
  wait_for_gpu_lease "$gpu" "$arm" >"$out/logs/v48_50_gpu_lease.json" || return $?
  free="$(free_mb "$gpu")"; apps="$(compute_apps_json "$gpu")"
  GPU="$gpu" ARM="$arm" FREE="$free" APPS="$apps" SERIAL="$SERIAL_VARIANTS" MODE="$VARIANT_MODE" python - "$out/V48_50_GPU_SCHEDULER_DECISION.json" <<'PY'
import json,os,pathlib,sys,time
pathlib.Path(sys.argv[1]).write_text(json.dumps({'event':'v48_50_gpu_scheduler_decision','created_unix':time.time(),
 'arm':os.environ['ARM'],'gpu':int(os.environ['GPU']),'free_mb_at_launch':float(os.environ['FREE']),
 'preexisting_compute_apps':json.loads(os.environ['APPS']),'variant_mode':os.environ['MODE'],
 'serial_variants_on_one_gpu':os.environ['SERIAL']=='1','strategy_regime_conditioning':False},indent=2)+'\n')
PY
  date +%s.%N >"$out/logs/v48_50_launcher.start_unix"; log="$out/logs/v48_50_launcher.log"
  set +e
  BASE_OUT="$BASE_OUT" GPU0="$gpu" GPU1="$gpu" SERIAL_VARIANTS_ON_ONE_GPU="$SERIAL_VARIANTS" \
    bash scripts/run_v48_50_dcp_de_ablation_arm.sh "$arm" >"$log" 2>&1
  rc=$?
  set -e
  printf '%s\n' "$rc" >"$out/logs/v48_50_launcher.rc"; date +%s.%N >"$out/logs/v48_50_launcher.end_unix"
  python - "$out" "$arm" "$rc" <<'PY'
import json,pathlib,sys
out=pathlib.Path(sys.argv[1]); logs=out/'logs'; start=float((logs/'v48_50_launcher.start_unix').read_text()); end=float((logs/'v48_50_launcher.end_unix').read_text())
(out/'V48_50_RUNTIME.json').write_text(json.dumps({'arm':sys.argv[2],'launcher_start_unix':start,'launcher_end_unix':end,'wall_seconds':end-start,'exit_code':int(sys.argv[3])},indent=2)+'\n')
PY
  return "$rc"
}
wait_one(){
  local pid="$1" arm="$2" rc
  set +e; wait "$pid"; rc=$?; set -e
  case "$rc" in
    0) echo "arm $arm: RC=0 gate passed" ;;
    20) echo "arm $arm: RC=20 pipeline-valid algorithm failure" ;;
    *) echo "arm $arm: RC=$rc ENGINEERING FAILURE" >&2; return 1 ;;
  esac
}

engineering_failed=0
# Run both waves even if one arm in the first wave has an engineering failure.
# Attribution remains fail-closed below, but completing C/D gives a full
# engineering diagnosis instead of silently losing half of the 2x2 evidence.
for pair in "A B" "C D"; do
  read -r left right <<<"$pair"
  run_arm "$left" "$GPU0" & p0=$!
  run_arm "$right" "$GPU1" & p1=$!
  wait_one "$p0" "$left" || engineering_failed=1
  wait_one "$p1" "$right" || engineering_failed=1
done

python tools/summarize_v48_46_runtime_telemetry.py \
  --input "$PERF_LOG" --output "$BASE_OUT/OC-RAP-v48.50-runtime-telemetry-summary.json" || true

if [[ "$engineering_failed" == 0 ]]; then
  python tools/compare_v48_50_dcp_de_2x2.py \
    --a "$(arm_out A)" --b "$(arm_out B)" --c "$(arm_out C)" --d "$(arm_out D)" \
    --output "$BASE_OUT/OC-RAP-v48.50-DCP-DRFC-DE-2x2-audit.json"
else
  echo "one or more v48.50 arms had an engineering failure; all four waves were attempted but attribution is blocked" >&2
  exit 1
fi
