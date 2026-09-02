#!/usr/bin/env bash
set -Eeuo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
BASE_OUT="${BASE_OUT:-runs}"
GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"
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
mkdir -p "$BASE_OUT" "$PERSISTENT_TENSOR_CACHE_DIR"

# Lightweight performance telemetry.  It is read-only and never changes CUDA,
# data-loader, sampler, model or gate state.  The next result package can thus
# distinguish GPU under-utilisation from CPU/storage stalls quantitatively.
PERF_LOG="$BASE_OUT/OC-RAP-v48.46-runtime-telemetry.jsonl"
perf_pid=""
if command -v nvidia-smi >/dev/null 2>&1; then
  (
    while true; do
      ts="$(date +%s)"
      load="$(cat /proc/loadavg 2>/dev/null | awk '{print $1","$2","$3}' || true)"
      mem_kb="$(awk '/MemAvailable:/{print $2}' /proc/meminfo 2>/dev/null || true)"
      nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
        --format=csv,noheader,nounits 2>/dev/null | \
      TS="$ts" LOAD="$load" MEMKB="$mem_kb" python -c 'import json,sys,os; ts=float(os.environ["TS"]); load=os.environ.get("LOAD",""); mem=os.environ.get("MEMKB","");
for line in sys.stdin:
 p=[x.strip() for x in line.split(",")];
 print(json.dumps({"unix":ts,"loadavg":load,"mem_available_kb":int(mem) if mem else None,"gpu":int(p[0]),"gpu_util_pct":float(p[1]),"gpu_mem_util_pct":float(p[2]),"gpu_mem_used_mb":float(p[3]),"gpu_mem_total_mb":float(p[4]),"power_w":float(p[5]) if p[5] not in {"[N/A]","N/A"} else None}))' \
        >>"$PERF_LOG" || true
      sleep "${V4846_TELEMETRY_INTERVAL_S:-30}"
    done
  ) & perf_pid=$!
fi
cleanup_perf() { [[ -z "$perf_pid" ]] || kill "$perf_pid" 2>/dev/null || true; }
trap cleanup_perf EXIT INT TERM

# One deterministic calibration protocol for every arm; never touch test roots.
bash scripts/prepare_v48_45_protocol.sh
export V4845_SKIP_PROTOCOL_PREPARE=1
[[ -s "$PROTOCOL_ROOT/V48_45_PROTOCOL_SEAL.json" ]] || { echo "missing shared protocol seal" >&2; exit 30; }

# Two ablations run concurrently, one assigned per physical GPU.  Within one
# ablation, balanced+precision can either share that GPU concurrently (fast,
# default based on the user's observed memory headroom) or run serially.
V4846_VARIANT_MODE="${V4846_VARIANT_MODE:-parallel}"
case "$V4846_VARIANT_MODE" in
  parallel) serial_variants=0 ;;
  serial) serial_variants=1 ;;
  *) echo "V4846_VARIANT_MODE must be parallel|serial" >&2; exit 2 ;;
esac

arm_out() {
  case "$1" in
    D) printf '%s/ocrap_v48_46_ocswic_main' "$BASE_OUT" ;;
    A|B|C) printf '%s/ocrap_v48_46_ocswic_ablation_%s' "$BASE_OUT" "$1" ;;
    *) return 2 ;;
  esac
}
run_arm() {
  local arm="$1" gpu="$2" out log
  out="$(arm_out "$arm")"; rm -rf "$out"; mkdir -p "$out/logs"
  date +%s.%N >"$out/logs/v48_46_launcher.start_unix"
  log="$out/logs/v48_46_launcher.log"
  BASE_OUT="$BASE_OUT" GPU0="$gpu" GPU1="$gpu" SERIAL_VARIANTS_ON_ONE_GPU="$serial_variants" \
    bash scripts/run_v48_46_ocswic_ablation_arm.sh "$arm" >"$log" 2>&1
}
wait_one() {
  local pid="$1" arm="$2" out rc
  set +e; wait "$pid"; rc=$?; set -e
  out="$(arm_out "$arm")"; printf '%s\n' "$rc" >"$out/logs/v48_46_launcher.rc"
  date +%s.%N >"$out/logs/v48_46_launcher.end_unix"
  python - "$out" "$arm" "$rc" <<'PY_TIMING'
import json,pathlib,sys
out=pathlib.Path(sys.argv[1]); arm=sys.argv[2]; rc=int(sys.argv[3]); logs=out/'logs'
start=float((logs/'v48_46_launcher.start_unix').read_text()); end=float((logs/'v48_46_launcher.end_unix').read_text())
(out/'V48_46_RUNTIME.json').write_text(json.dumps({'arm':arm,'launcher_start_unix':start,'launcher_end_unix':end,'wall_seconds':end-start,'exit_code':rc},indent=2)+'\n')
PY_TIMING
  case "$rc" in
    0) echo "arm $arm: RC=0 gate passed" ;;
    20) echo "arm $arm: RC=20 pipeline-valid algorithm failure" ;;
    *) echo "arm $arm: RC=$rc ENGINEERING FAILURE" >&2; return 1 ;;
  esac
}

engineering_failed=0
# A/B first: they warm the persistent tensor cache. C/D then reuse it during
# both sequential witness stages and factor adaptation.
for pair in "A B" "C D"; do
  read -r left right <<<"$pair"
  run_arm "$left" "$GPU0" & p0=$!
  run_arm "$right" "$GPU1" & p1=$!
  wait_one "$p0" "$left" || engineering_failed=1
  wait_one "$p1" "$right" || engineering_failed=1
  [[ "$engineering_failed" == 0 ]] || break
done

if [[ "$engineering_failed" == 0 ]]; then
  python tools/compare_v48_46_ocswic_2x2.py \
    --a "$(arm_out A)" --b "$(arm_out B)" --c "$(arm_out C)" --d "$(arm_out D)" \
    --output "$BASE_OUT/OC-RAP-v48.46-OC-SWIC-2x2-audit.json"
else
  exit 1
fi
