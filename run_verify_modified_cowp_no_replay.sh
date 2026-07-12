#!/usr/bin/env bash
set -euo pipefail

# Verify the modified COWP/RC-NCF code using the EXISTING attached Waymax caches.
# This script intentionally does NOT replay or attach candidate outcomes.
# It disables log-divergence supervision/risk because the diagnosed caches contain
# no finite log-divergence labels, while retaining collision/offroad auxiliary loss.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$SCRIPT_DIR}"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# ------------------------- Paths -------------------------
export WOMD_ROOT="${WOMD_ROOT:-/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1}"
export COWP_ROOT="${COWP_ROOT:-/data0/senzeyu2/dataset/COWP/formal}"
TRAIN_CACHE="${TRAIN_CACHE:-$COWP_ROOT/tensor_cache_train_waymax}"
VAL_CACHE="${VAL_CACHE:-$COWP_ROOT/tensor_cache_val_waymax}"
WAYMAX_VAL="${WAYMAX_VAL:-$WOMD_ROOT/uncompressed/tf_example/validation/validation_tfexample.tfrecord@150}"
OUT_ROOT="${OUT_ROOT:-outputs/verify_modified_no_replay}"

# ------------------------- Runtime -------------------------
NUM_GPUS=2
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256}"

# MODE=smoke first verifies code and closed-loop plumbing cheaply.
# MODE=full performs the meaningful training/evaluation comparison.
MODE="${MODE:-smoke}"
RUN_TESTS="${RUN_TESTS:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_OFFLINE_EVAL="${RUN_OFFLINE_EVAL:-1}"
RUN_ONLINE_EVAL="${RUN_ONLINE_EVAL:-1}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"
FORCE_EVAL="${FORCE_EVAL:-0}"
STRICT_GATES="${STRICT_GATES:-0}"

TRAIN_WORKERS="${TRAIN_WORKERS:-6}"
PREFETCH="${PREFETCH:-1}"
EVAL_BATCH="${EVAL_BATCH:-64}"
CALIB_PERCENT="${CALIB_PERCENT:-20}"

case "$MODE" in
  smoke)
    EPOCH_NATURAL="${EPOCH_NATURAL:-2}"
    EPOCH_RESPONSE="${EPOCH_RESPONSE:-2}"
    EPOCH_WITNESS="${EPOCH_WITNESS:-4}"
    EPOCH_PLANNER="${EPOCH_PLANNER:-4}"
    TOTAL_ONLINE_SCENARIOS="${TOTAL_ONLINE_SCENARIOS:-100}"
    ;;
  full)
    EPOCH_NATURAL="${EPOCH_NATURAL:-20}"
    EPOCH_RESPONSE="${EPOCH_RESPONSE:-16}"
    EPOCH_WITNESS="${EPOCH_WITNESS:-28}"
    EPOCH_PLANNER="${EPOCH_PLANNER:-32}"
    TOTAL_ONLINE_SCENARIOS="${TOTAL_ONLINE_SCENARIOS:-1000}"
    ;;
  *) echo "MODE must be smoke or full" >&2; exit 2 ;;
esac

PER_GPU_BATCH_NATURAL="${PER_GPU_BATCH_NATURAL:-24}"
PER_GPU_BATCH_RESPONSE="${PER_GPU_BATCH_RESPONSE:-12}"
PER_GPU_BATCH_WITNESS="${PER_GPU_BATCH_WITNESS:-20}"
PER_GPU_BATCH_PLANNER="${PER_GPU_BATCH_PLANNER:-16}"

mkdir -p "$OUT_ROOT" "$OUT_ROOT/configs" "$OUT_ROOT/checkpoints" \
  "$OUT_ROOT/eval/learned_offline" "$OUT_ROOT/eval/waymax" "$OUT_ROOT/cache_splits"

json_ok() {
  local f="$1"
  [[ -s "$f" ]] && python - "$f" <<'PY' >/dev/null 2>&1
import json, sys
json.load(open(sys.argv[1], encoding="utf-8"))
PY
}

wait_all() {
  local status=0 pid
  for pid in "$@"; do wait "$pid" || status=$?; done
  return "$status"
}

require_file() { [[ -f "$1" ]] || { echo "Missing file: $1" >&2; exit 2; }; }
require_dir()  { [[ -d "$1" ]] || { echo "Missing directory: $1" >&2; exit 2; }; }

require_dir "$TRAIN_CACHE"
require_dir "$VAL_CACHE"
require_file "configs/train.yaml"
require_file "configs/model.yaml"
require_file "cowp/waymax_eval/policy_wrapper.py"

# Confirm that this is the modified code rather than the original implementation.
grep -q "ego_centric_normalization" configs/model.yaml
grep -q "natural_latent" cowp/models/cowp_model.py
grep -q "epistemic_uncertainty" cowp/models/witness_decoder.py
grep -q "_consistent_one_step_target" cowp/waymax_eval/policy_wrapper.py
grep -q "standard_metrics" cowp/scripts/17_merge_waymax_shards.py

echo "[0/8] Modified-code signatures found"

if [[ "$RUN_TESTS" == 1 ]]; then
  echo "[1/8] Regression tests"
  pytest --rootdir="$REPO_ROOT" -q
fi

# Build a safe training config for the diagnosed caches.  The original config has
# outcome_logdiv=0.5, but the current caches contain zero finite logdiv labels.
TRAIN_CFG="$OUT_ROOT/configs/train_no_logdiv.yaml"
python - "configs/train.yaml" "$TRAIN_CFG" <<'PY'
import sys, yaml
src, dst = sys.argv[1:3]
with open(src, encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
w = cfg.setdefault("loss_weights", {})
w["outcome_logdiv"] = 0.0
# Prevent an absent/zero-filled logdiv target from contributing to unsafe labels.
w["outcome_logdiv_unsafe_threshold"] = 1.0e9
# Collision/offroad classification, pair ranking and expected physical cost remain active.
with open(dst, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
print(dst)
PY

# Deterministically split validation cache into calibration/early-stop and held-out test.
# Symlinks avoid copying the large NPZ cache.
CALIB_CACHE="$OUT_ROOT/cache_splits/val_calib"
TEST_CACHE="$OUT_ROOT/cache_splits/val_test"
python - "$VAL_CACHE" "$CALIB_CACHE" "$TEST_CACHE" "$CALIB_PERCENT" <<'PY'
from pathlib import Path
import hashlib, os, shutil, sys
src, calib, test = map(Path, sys.argv[1:4])
pct = int(sys.argv[4])
if not (5 <= pct <= 50):
    raise SystemExit("CALIB_PERCENT must be between 5 and 50")
for dst in (calib, test):
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
files = sorted(src.glob("*.npz"))
if not files:
    raise SystemExit(f"No NPZ files found in {src}")
n_cal = 0
for p in files:
    h = int(hashlib.sha1(p.name.encode()).hexdigest()[:8], 16) % 100
    dst = calib if h < pct else test
    os.symlink(p.resolve(), dst / p.name)
    n_cal += int(dst == calib)
print({"total": len(files), "calib": n_cal, "test": len(files)-n_cal})
if n_cal < 100 or len(files)-n_cal < 100:
    raise SystemExit("Calibration/test split is unexpectedly small")
PY

echo "[2/8] Existing caches reused; no replay/attach. Logdiv disabled."

train_stage() {
  local stage="$1" epochs="$2" lr="$3" batch="$4" output="$5" resume="${6:-}"
  local best="$output/cowp_${stage}_best.pt"
  if [[ "$FORCE_TRAIN" != 1 && -s "$best" ]]; then
    echo "[train/$stage] keep existing: $best"
    return
  fi
  mkdir -p "$output"
  local args=(
    torchrun --standalone --nproc_per_node=2 -m cowp.scripts.03_train
    --data-config configs/data.yaml
    --model-config configs/model.yaml
    --train-config "$TRAIN_CFG"
    --cache-dir "$TRAIN_CACHE"
    --val-cache-dir "$CALIB_CACHE"
    --stage "$stage" --epochs "$epochs" --lr "$lr" --batch-size "$batch"
    --num-workers "$TRAIN_WORKERS" --prefetch-factor "$PREFETCH"
    --amp --fused-adamw --val-every 1 --output-dir "$output"
  )
  [[ -z "$resume" ]] || args+=(--resume "$resume")
  # Dense response trajectory labels are not needed to verify the cross-world witness
  # and are the largest RAM consumer. Keep response burden/components enabled.
  [[ "$stage" != response ]] || args+=(--no-response-traj)
  [[ "$stage" != planner ]] || args+=(--with-waymax-outcome-labels)
  "${args[@]}"
}

if [[ "$RUN_TRAIN" == 1 ]]; then
  echo "[3/8] Staged two-GPU training"
  train_stage natural  "$EPOCH_NATURAL"  3e-4   "$PER_GPU_BATCH_NATURAL"  "$OUT_ROOT/checkpoints/natural"
  NAT="$OUT_ROOT/checkpoints/natural/cowp_natural_best.pt"
  train_stage response "$EPOCH_RESPONSE" 2.5e-4 "$PER_GPU_BATCH_RESPONSE" "$OUT_ROOT/checkpoints/response" "$NAT"
  RESP="$OUT_ROOT/checkpoints/response/cowp_response_best.pt"
  train_stage witness  "$EPOCH_WITNESS"  2e-4   "$PER_GPU_BATCH_WITNESS"  "$OUT_ROOT/checkpoints/witness" "$RESP"
  WIT="$OUT_ROOT/checkpoints/witness/cowp_witness_best.pt"
  train_stage planner  "$EPOCH_PLANNER"  1.5e-4 "$PER_GPU_BATCH_PLANNER"  "$OUT_ROOT/checkpoints/planner" "$WIT"
fi

CKPT="${CKPT:-$OUT_ROOT/checkpoints/planner/cowp_planner_best.pt}"
require_file "$CKPT"

# Calibrate only on val_calib. The held-out val_test is not used for threshold choice.
CAL_SWEEP="$OUT_ROOT/eval/learned_offline/calibration_sweep.json"
CAL_JSON="$OUT_ROOT/eval/witness_calibration.json"
if [[ "$RUN_OFFLINE_EVAL" == 1 ]]; then
  echo "[4/8] Witness threshold calibration on calibration split"
  if [[ "$FORCE_EVAL" == 1 ]] || ! json_ok "$CAL_SWEEP"; then
    CUDA_VISIBLE_DEVICES=0 python -m cowp.scripts.04_eval_closed_loop \
      --data-config configs/data.yaml --label-config configs/label.yaml --eval-config configs/eval.yaml \
      --cache-dir "$CALIB_CACHE" --mode learned_offline --method cowp \
      --checkpoint "$CKPT" --batch-size "$EVAL_BATCH" --device cuda \
      --witness-threshold 0.50 \
      --witness-threshold-sweep 0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85 \
      --ncf-gate-mode priority --priority-hard-threshold 0.55 \
      --secondary-witness-threshold 0.85 --secondary-opr-alpha 0.10 \
      --offline-fallback stop_like --adaptive-frontier-margin 0.15 \
      --outcome-risk-penalty 0.0 \
      --output "$CAL_SWEEP"
  fi
  python -m cowp.scripts.18_calibrate_witness_threshold \
    --input "$CAL_SWEEP" --output "$CAL_JSON" \
    --min-ncf-recall 0.90 --max-fallback 0.25
fi

WITNESS_THRESHOLD="${WITNESS_THRESHOLD:-$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["witness_threshold"])' "$CAL_JSON")}" 
echo "Calibrated witness threshold: $WITNESS_THRESHOLD"

run_learned_eval() {
  local method="$1" gpu="$2" output="$3"
  if [[ "$FORCE_EVAL" != 1 ]] && json_ok "$output"; then
    echo "[learned/$method] keep existing"
    return
  fi
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    python -m cowp.scripts.04_eval_closed_loop \
      --data-config configs/data.yaml --label-config configs/label.yaml --eval-config configs/eval.yaml \
      --cache-dir "$TEST_CACHE" --mode learned_offline --method "$method" \
      --checkpoint "$CKPT" --batch-size "$EVAL_BATCH" --device cuda \
      --witness-threshold "$WITNESS_THRESHOLD" --ncf-gate-mode priority \
      --priority-hard-threshold 0.55 --secondary-witness-threshold 0.85 \
      --secondary-opr-alpha 0.10 --soft-ncf-penalty 1.5 \
      --offline-fallback stop_like --adaptive-frontier-margin 0.15 \
      --outcome-risk-penalty 0.0 \
      --output "$output"
  )
}

if [[ "$RUN_OFFLINE_EVAL" == 1 ]]; then
  echo "[5/8] Held-out learned-offline comparison"
  METHODS=(planner_score_only conventional_safety soft_burden_cost_only universal_ncf cowp)
  for ((i=0; i<${#METHODS[@]}; i+=2)); do
    pids=()
    for offset in 0 1; do
      j=$((i+offset)); ((j < ${#METHODS[@]})) || continue
      m="${METHODS[$j]}"
      run_learned_eval "$m" "$offset" "$OUT_ROOT/eval/learned_offline/${m}.json" & pids+=("$!")
    done
    wait_all "${pids[@]}"
  done
fi

run_waymax_method() {
  local method="$1"
  local per_shard=$(( (TOTAL_ONLINE_SCENARIOS + NUM_GPUS - 1) / NUM_GPUS ))
  local pids=() shard out
  for shard in 0 1; do
    out="$OUT_ROOT/eval/waymax/${method}_shard${shard}.json"
    if [[ "$FORCE_EVAL" != 1 ]] && json_ok "$out"; then
      echo "[waymax/$method] keep shard $shard"
      continue
    fi
    (
      export CUDA_VISIBLE_DEVICES="$shard"
      export XLA_PYTHON_CLIENT_PREALLOCATE=false
      python -m cowp.scripts.04_eval_closed_loop \
        --data-config configs/data.yaml --label-config configs/label.yaml --eval-config configs/eval.yaml \
        --mode waymax --waymax-split validation --tfexample-glob "$WAYMAX_VAL" \
        --method "$method" --checkpoint "$CKPT" \
        --num-scenarios "$per_shard" --num-shards 2 --shard-index "$shard" \
        --rollout-horizon-steps 80 --waymax-standard-metrics \
        --waymax-device gpu --jax-visible-devices 0 --jax-preallocate false \
        --waymax-action-mode absolute_xy_yaw --device cuda \
        --witness-threshold "$WITNESS_THRESHOLD" --ncf-gate-mode priority \
        --priority-hard-threshold 0.55 --secondary-witness-threshold 0.85 \
        --secondary-opr-alpha 0.10 --soft-ncf-penalty 1.5 \
        --adaptive-frontier-margin 0.15 \
        --outcome-risk-penalty 0.0 \
        --clear-accelerator-cache --output "$out"
    ) & pids+=("$!")
  done
  ((${#pids[@]} == 0)) || wait_all "${pids[@]}"
  python -m cowp.scripts.17_merge_waymax_shards \
    --output "$OUT_ROOT/eval/waymax/${method}_merged.json" \
    "$OUT_ROOT/eval/waymax/${method}_shard0.json" \
    "$OUT_ROOT/eval/waymax/${method}_shard1.json"
}

if [[ "$RUN_ONLINE_EVAL" == 1 ]]; then
  echo "[6/8] Real Waymax closed-loop comparison on identical sharded scenarios"
  for m in planner_score_only conventional_safety cowp; do
    run_waymax_method "$m"
  done
fi

# Produce a concise machine-readable and Markdown report. Logdiv is intentionally omitted.
echo "[7/8] Build verification report"
python - "$OUT_ROOT" "$MODE" "$TOTAL_ONLINE_SCENARIOS" "$STRICT_GATES" <<'PY'
from pathlib import Path
import json, math, sys
root = Path(sys.argv[1]); mode=sys.argv[2]; expected=int(sys.argv[3]); strict=int(sys.argv[4])
methods = ["planner_score_only","conventional_safety","soft_burden_cost_only","universal_ncf","cowp"]

def load(p):
    p=Path(p)
    return json.loads(p.read_text()) if p.is_file() else None

def learned(m):
    d=load(root/"eval/learned_offline"/f"{m}.json")
    return (d or {}).get(m,{})

def online(m):
    return load(root/"eval/waymax"/f"{m}_merged.json") or {}

L={m:learned(m) for m in methods}
O={m:online(m) for m in ("planner_score_only","conventional_safety","cowp")}

report={"mode":mode,"expected_online_scenarios":expected,"learned_offline":L,"online":{},"gates":{}}
for m,d in O.items():
    report["online"][m]={
        "num_rollouts":d.get("num_rollouts"),
        "standard":d.get("standard_metric_summary",{}),
        "diagnostic":d.get("policy_diagnostic_summary",{}),
        "episode":d.get("closed_loop_cowp_metric_summary",{}),
    }

cowp=L.get("cowp",{}); base=L.get("planner_score_only",{})
spread=float(cowp.get("WitnessProb/p90",0))-float(cowp.get("WitnessProb/p10",0))
report["gates"]["witness_auprc"]={"value":cowp.get("WitnessQuality/AUPRC"),"target":0.50 if mode=="full" else 0.40}
report["gates"]["witness_spread_p90_p10"]={"value":spread,"target":0.25 if mode=="full" else 0.10}
report["gates"]["offline_fsr_improves"]={"value":cowp.get("FSR"),"baseline":base.get("FSR"),"pass":float(cowp.get("FSR",1)) < float(base.get("FSR",1))}
report["gates"]["offline_ep_preserved"]={"value":cowp.get("EP"),"baseline":base.get("EP"),"tolerance":0.05,"pass":float(cowp.get("EP",0)) >= float(base.get("EP",0))-0.05}
report["gates"]["offline_fallback"]={"value":cowp.get("FallbackRate"),"target":0.15 if mode=="full" else 0.30}

hard_fail=[]
for m,d in report["online"].items():
    n=int(d.get("num_rollouts") or 0)
    # Because each shard receives ceil(total/2), odd totals can produce total+1.
    count_ok=n in {expected, expected+1}
    report["gates"][f"{m}/rollout_count"]={"value":n,"target":expected,"pass":count_ok}
    if not count_ok: hard_fail.append(f"{m}: rollout_count={n}")
    std=d.get("standard",{}); diag=d.get("diagnostic",{})
    kin=float(std.get("KinematicsInfeasibilityRate",1.0))
    kin_target=0.05 if mode=="full" else 0.10
    ok=kin <= kin_target
    report["gates"][f"{m}/kinematics"]={"value":kin,"target":kin_target,"pass":ok}
    if not ok: hard_fail.append(f"{m}: kinematics={kin:.4f}")
    crit=float(diag.get("ClosedLoopMean/critical_agents",0.0))
    tok=float(diag.get("ClosedLoopMean/conflict_tokens",0.0))
    report["gates"][f"{m}/critical_agents_not_saturated"]={"value":crit,"target_max":4.5,"pass":crit<=4.5}
    report["gates"][f"{m}/conflict_tokens_not_saturated"]={"value":tok,"target_max":48.0,"pass":tok<48.0}

# Online Pareto check: COWP should not buy social safety through a large physical-safety/progress loss.
cs=report["online"].get("cowp",{}).get("standard",{})
bs=report["online"].get("planner_score_only",{}).get("standard",{})
if cs and bs:
    report["gates"]["online_cr_noninferior"]={"value":cs.get("CR"),"baseline":bs.get("CR"),"margin":0.03,"pass":float(cs.get("CR",1))<=float(bs.get("CR",1))+0.03}
    report["gates"]["online_ep_noninferior"]={"value":cs.get("EP"),"baseline":bs.get("EP"),"margin":0.05,"pass":float(cs.get("EP",0))>=float(bs.get("EP",0))-0.05}

(root/"verification_report.json").write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")

lines=["# Modified COWP no-replay verification", "", f"Mode: `{mode}`", "", "## Held-out learned-offline", "", "| Method | FSR | EP | CBS | OPR | HBCR | Fallback | Witness AUPRC |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
for m in methods:
    x=L.get(m,{})
    def f(k):
        v=x.get(k)
        return "-" if v is None else f"{float(v):.4f}"
    lines.append(f"| {m} | {f('FSR')} | {f('EP')} | {f('CBS')} | {f('OPR')} | {f('HBCR')} | {f('FallbackRate')} | {f('WitnessQuality/AUPRC')} |")
lines += ["", "## Real Waymax closed loop", "", "| Method | N | CR | Collision | Offroad | EP | Kinematic infeasible | Mean critical | Mean conflict tokens |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
for m in ("planner_score_only","conventional_safety","cowp"):
    x=report["online"].get(m,{}); s=x.get("standard",{}); d=x.get("diagnostic",{})
    def g(v): return "-" if v is None else f"{float(v):.4f}"
    lines.append(f"| {m} | {x.get('num_rollouts','-')} | {g(s.get('CR'))} | {g(s.get('CollisionRate'))} | {g(s.get('OffroadRate'))} | {g(s.get('EP'))} | {g(s.get('KinematicsInfeasibilityRate'))} | {g(d.get('ClosedLoopMean/critical_agents'))} | {g(d.get('ClosedLoopMean/conflict_tokens'))} |")
lines += ["", "## Gates", ""]
for k,v in report["gates"].items():
    passed=v.get("pass")
    if passed is None and "target" in v and v.get("value") is not None:
        passed=float(v["value"]) >= float(v["target"]) if "auprc" in k or "spread" in k else float(v["value"]) <= float(v["target"])
    mark="PASS" if passed else ("CHECK" if passed is None else "FAIL")
    lines.append(f"- **{mark}** `{k}`: {v}")
lines += ["", "Log-divergence is intentionally excluded because the existing attached cache has no finite logdiv labels."]
(root/"verification_report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
print(json.dumps({"report":str(root/"verification_report.md"),"hard_fail":hard_fail},indent=2,ensure_ascii=False))
if strict and hard_fail:
    raise SystemExit(3)
PY

echo "[8/8] Complete"
echo "Checkpoint: $CKPT"
echo "Report: $OUT_ROOT/verification_report.md"
echo "JSON:   $OUT_ROOT/verification_report.json"
echo
if [[ "$MODE" == smoke ]]; then
  echo "Smoke mode only proves the modified pipeline works. For a meaningful result comparison run:"
  echo "  MODE=full FORCE_TRAIN=1 FORCE_EVAL=1 bash $0"
fi
