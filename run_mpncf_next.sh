#!/usr/bin/env bash
set -euo pipefail

# Robust risk-calibrated Mechanism-aware P-NCF/COWP next experiment.
# Key fixes vs run_verify_modified_cowp_no_replay.sh:
# 1) never calls pytest through a Python that cannot parse `from __future__ import annotations`;
# 2) trains/evaluates with label.yaml loaded into the model checkpoint, so planning/ncf semantics match eval;
# 3) disables degenerate log-divergence labels in the current cache;
# 4) logs every stage under $OUT_ROOT/logs and can self-detach through nohup;
# 5) runs independent offline/online shards in parallel while keeping sequential training dependencies.

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
OUT_ROOT="${OUT_ROOT:-outputs/rc_mpncf_next}"
LOG_DIR="$OUT_ROOT/logs"

mkdir -p "$OUT_ROOT" "$LOG_DIR" "$OUT_ROOT/configs" "$OUT_ROOT/checkpoints" \
  "$OUT_ROOT/eval/learned_offline" "$OUT_ROOT/eval/waymax" "$OUT_ROOT/cache_splits"

if [[ "${DETACH:-0}" == "1" && -z "${COWP_ALREADY_DETACHED:-}" ]]; then
  export COWP_ALREADY_DETACHED=1
  if command -v setsid >/dev/null 2>&1; then
    nohup setsid bash "$0" "$@" > "$LOG_DIR/driver.nohup.log" 2>&1 &
  else
    nohup bash "$0" "$@" > "$LOG_DIR/driver.nohup.log" 2>&1 &
  fi
  echo "Detached PID: $!"
  echo "Driver log: $LOG_DIR/driver.nohup.log"
  exit 0
fi

# ------------------------- Python/runtime -------------------------
supports_future_annotations() {
  "$1" - <<'PY' >/dev/null 2>&1
from __future__ import annotations
print("ok")
PY
}

choose_python() {
  local candidates=()
  [[ -n "${PYTHON_BIN:-}" ]] && candidates+=("$PYTHON_BIN")
  candidates+=(python python3 python3.12 python3.11 python3.10 python3.9 python3.8 python3.7)
  local c
  for c in "${candidates[@]}"; do
    command -v "$c" >/dev/null 2>&1 || continue
    if supports_future_annotations "$c"; then
      echo "$c"
      return 0
    fi
  done
  echo "No Python >=3.7-like interpreter found. Set PYTHON_BIN to the conda/env Python used for COWP." >&2
  return 2
}

PYTHON_BIN="$(choose_python)"
echo "Using Python: $($PYTHON_BIN -c 'import sys; print(sys.executable, sys.version.split()[0])')"

json_ok() {
  local f="$1"
  [[ -s "$f" ]] && "$PYTHON_BIN" - "$f" <<'PY' >/dev/null 2>&1
import json, sys
json.load(open(sys.argv[1], encoding="utf-8"))
PY
}

json_matches_ckpt() {
  local f="$1" ckpt="$2" method="${3:-}" mode="${4:-}"
  [[ -s "$f" ]] || return 1
  "$PYTHON_BIN" - "$f" "$ckpt" "$method" "$mode" <<'PY' >/dev/null 2>&1
import json, sys
path, ckpt, method, mode = sys.argv[1:5]
try:
    d = json.load(open(path, encoding="utf-8"))
except Exception:
    raise SystemExit(1)
if ckpt and str(d.get("checkpoint", "")) != ckpt:
    raise SystemExit(1)
if mode and str(d.get("mode", "")) != mode:
    raise SystemExit(1)
if method:
    if mode == "waymax" and str(d.get("method", d.get("baseline", ""))) != method:
        raise SystemExit(1)
    if mode == "learned_offline" and method not in d:
        raise SystemExit(1)
raise SystemExit(0)
PY
}

wait_all() {
  local status=0 pid
  for pid in "$@"; do
    if ! wait "$pid"; then status=1; fi
  done
  return "$status"
}

run_logged() {
  local name="$1"; shift
  local log="$LOG_DIR/${name}.log"
  echo "[$name] start -> $log"
  "$@" > "$log" 2>&1
  local rc=$?
  if [[ "$rc" -ne 0 ]]; then
    echo "[$name] failed with exit code $rc. Last 80 log lines:" >&2
    tail -80 "$log" >&2 || true
    return "$rc"
  fi
  echo "[$name] done"
}

run_bg_logged() {
  local name="$1"; shift
  local log="$LOG_DIR/${name}.log"
  echo "[$name] start bg -> $log"
  ( "$@" > "$log" 2>&1 ) &
  RUN_BG_PID=$!
}


checkpoint_epoch() {
  local ckpt="$1"
  [[ -s "$ckpt" ]] || { echo -1; return 0; }
  "$PYTHON_BIN" - "$ckpt" <<'PYCKPT'
import re, sys
from pathlib import Path
path = Path(sys.argv[1])
try:
    import torch
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    print(int(obj.get("epoch", -1)) if isinstance(obj, dict) else -1)
except Exception:
    m = re.search(r"_epoch(\d+)\.pt$", path.name)
    print(int(m.group(1)) if m else -1)
PYCKPT
}

latest_valid_stage_checkpoint() {
  local output="$1" stage="$2"
  "$PYTHON_BIN" - "$output" "$stage" <<'PYLATEST'
from pathlib import Path
import re, sys
out = Path(sys.argv[1]); stage = sys.argv[2]

def file_epoch(p: Path) -> int:
    m = re.search(r"_epoch(\d+)\.pt$", p.name)
    return int(m.group(1)) if m else -1
cands = sorted(out.glob(f"cowp_{stage}_epoch*.pt"), key=file_epoch, reverse=True)
if not cands:
    print("\t-1")
    raise SystemExit(0)
try:
    import torch
except Exception:
    p = cands[0]
    print(f"{p}\t{file_epoch(p)}")
    raise SystemExit(0)
for p in cands:
    try:
        try:
            obj = torch.load(p, map_location="cpu", weights_only=False)
        except TypeError:
            obj = torch.load(p, map_location="cpu")
        epoch = int(obj.get("epoch", file_epoch(p))) if isinstance(obj, dict) else file_epoch(p)
        if epoch >= 0:
            print(f"{p}\t{epoch}")
            raise SystemExit(0)
    except Exception as exc:
        print(f"Warning: ignoring unreadable checkpoint {p}: {exc}", file=sys.stderr)
print("\t-1")
PYLATEST
}

require_file() { [[ -f "$1" ]] || { echo "Missing file: $1" >&2; exit 2; }; }
require_dir()  { [[ -d "$1" ]] || { echo "Missing directory: $1" >&2; exit 2; }; }

require_dir "$TRAIN_CACHE"
require_dir "$VAL_CACHE"
require_file "configs/train.yaml"
require_file "configs/model.yaml"
require_file "configs/label.yaml"
require_file "cowp/waymax_eval/policy_wrapper.py"

grep -q "witness_scene_prior" cowp/models/losses.py
grep -q "witness_probability_source" cowp/waymax_eval/rollout.py
grep -q -- "--label-config" cowp/scripts/03_train.py
grep -q "standard_metrics" cowp/scripts/17_merge_waymax_shards.py
grep -q "candidate_certificate" cowp/models/cowp_model.py
grep -q "candidate_certificate_loss" cowp/models/losses.py

echo "[0/8] Calibrated P-NCF code signatures found"

# ------------------------- Runtime knobs -------------------------
NUM_GPUS="${NUM_GPUS:-2}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

MODE="${MODE:-full}"          # smoke | full
RUN_TESTS="${RUN_TESTS:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_OFFLINE_EVAL="${RUN_OFFLINE_EVAL:-1}"
RUN_ONLINE_EVAL="${RUN_ONLINE_EVAL:-1}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"
FORCE_EVAL="${FORCE_EVAL:-0}"
STRICT_GATES="${STRICT_GATES:-0}"
CLEAR_ACCELERATOR_CACHE="${CLEAR_ACCELERATOR_CACHE:-0}"
WAYMAX_STANDARD_METRICS="${WAYMAX_STANDARD_METRICS:-1}"
WAYMAX_STANDARD_METRIC_NAMES="${WAYMAX_STANDARD_METRIC_NAMES:-}"
WAYMAX_STATUS_EVERY="${WAYMAX_STATUS_EVERY:-10}"
REUSE_WAYMAX_ENV="${REUSE_WAYMAX_ENV:-1}"
NO_PROGRESS="${NO_PROGRESS:-0}"
ROLLOUT_HORIZON_STEPS="${ROLLOUT_HORIZON_STEPS:-80}"
OUTCOME_RISK_PENALTY="${OUTCOME_RISK_PENALTY:-0.75}"
OUTCOME_RISK_THRESHOLD="${OUTCOME_RISK_THRESHOLD:-1.10}"
COWP_FAST_DIAGNOSTICS="${COWP_FAST_DIAGNOSTICS:-0}"
PLANNER_ONLY_RETRAIN="${PLANNER_ONLY_RETRAIN:-0}"
WITNESS_CKPT="${WITNESS_CKPT:-}"
# Use latest planner checkpoint after planner-only retrain.  Best-by-val can
# silently keep an old checkpoint when the validation loss is dominated by
# saturated witness terms; this was the main reason retraining appeared to have
# no effect. Set CKPT_SELECTION=best to recover the old behavior.
CKPT_SELECTION="${CKPT_SELECTION:-latest}"

TRAIN_WORKERS="${TRAIN_WORKERS:-6}"
PREFETCH="${PREFETCH:-1}"
EVAL_BATCH="${EVAL_BATCH:-64}"
CALIB_PERCENT="${CALIB_PERCENT:-20}"

case "$MODE" in
  smoke)
    EPOCH_WITNESS="${EPOCH_WITNESS:-6}"
    EPOCH_PLANNER="${EPOCH_PLANNER:-6}"
    TOTAL_ONLINE_SCENARIOS="${TOTAL_ONLINE_SCENARIOS:-100}"
    ;;
  full)
    EPOCH_WITNESS="${EPOCH_WITNESS:-32}"
    EPOCH_PLANNER="${EPOCH_PLANNER:-36}"
    TOTAL_ONLINE_SCENARIOS="${TOTAL_ONLINE_SCENARIOS:-1000}"
    ;;
  *) echo "MODE must be smoke or full" >&2; exit 2 ;;
esac

PER_GPU_BATCH_WITNESS="${PER_GPU_BATCH_WITNESS:-20}"
PER_GPU_BATCH_PLANNER="${PER_GPU_BATCH_PLANNER:-16}"

# Optional warm start from the previous response stage.  If absent, train natural/response first.
WARM_RESPONSE_CKPT="${WARM_RESPONSE_CKPT:-outputs/verify_modified_no_replay/checkpoints/response/cowp_response_best.pt}"
NATURAL_EPOCHS="${NATURAL_EPOCHS:-20}"
RESPONSE_EPOCHS="${RESPONSE_EPOCHS:-16}"
PER_GPU_BATCH_NATURAL="${PER_GPU_BATCH_NATURAL:-24}"
PER_GPU_BATCH_RESPONSE="${PER_GPU_BATCH_RESPONSE:-12}"

# Runtime config: preserve current labels, but make witness certificate logit-calibrated and disable missing logdiv.
TRAIN_CFG="$OUT_ROOT/configs/train_rc_mpncf_no_logdiv.yaml"
LABEL_CFG="$OUT_ROOT/configs/label_rc_mpncf.yaml"
run_logged make_runtime_configs "$PYTHON_BIN" - "configs/train.yaml" "$TRAIN_CFG" "configs/label.yaml" "$LABEL_CFG" <<'PY'
import sys, yaml
train_src, train_dst, label_src, label_dst = sys.argv[1:5]
with open(train_src, encoding="utf-8") as f:
    train = yaml.safe_load(f)
w = train.setdefault("loss_weights", {})
w["outcome_logdiv"] = 0.0
w["outcome_logdiv_unsafe_threshold"] = 1.0e9
w["witness_scene_prior"] = float(w.get("witness_scene_prior", 1.5))
w["witness_scene_prior_min_pairs"] = int(w.get("witness_scene_prior_min_pairs", 4))
w["witness_logit_l2"] = float(w.get("witness_logit_l2", 0.001))
w["witness_evidential"] = float(w.get("witness_evidential", 0.05))
w["witness_mined_fraction"] = float(w.get("witness_mined_fraction", 0.25))
w["witness_balanced_fraction"] = float(w.get("witness_balanced_fraction", 0.75))
# Pairwise witness is still trained, but planner fine-tuning should not be
# dominated by the saturated pair certificate.  Shift the planner stage toward
# candidate-level NCF/false-safe discrimination.
w["planner_witness_scale"] = float(w.get("planner_witness_scale", 0.005))
w["candidate_certificate"] = float(w.get("candidate_certificate", 4.0))
w["candidate_certificate_ncf"] = float(w.get("candidate_certificate_ncf", 2.5))
w["candidate_certificate_false_safe"] = float(w.get("candidate_certificate_false_safe", 3.5))
w["candidate_certificate_quality"] = float(w.get("candidate_certificate_quality", 2.0))
w["candidate_certificate_prior"] = float(w.get("candidate_certificate_prior", 1.25))
w["candidate_certificate_rank"] = float(w.get("candidate_certificate_rank", 3.0))
w["candidate_certificate_risk_bce"] = float(w.get("candidate_certificate_risk_bce", 3.0))
w["candidate_certificate_risk_rank"] = float(w.get("candidate_certificate_risk_rank", 3.0))
w["candidate_certificate_spread"] = float(w.get("candidate_certificate_spread", 0.50))
w["candidate_cert_min_logit_spread"] = float(w.get("candidate_cert_min_logit_spread", 0.50))
w["candidate_cert_pair_margin"] = float(w.get("candidate_cert_pair_margin", 1.25))
w["candidate_outcome_logdiv_unsafe_threshold"] = float(w.get("candidate_outcome_logdiv_unsafe_threshold", 8.0))
w["candidate_outcome_safe_ncf_target"] = float(w.get("candidate_outcome_safe_ncf_target", 0.75))
w["closed_loop"] = float(w.get("closed_loop", 2.0))
w["outcome_cls"] = float(w.get("outcome_cls", 2.0))
w["outcome_pair_rank"] = float(w.get("outcome_pair_rank", 2.0))
w["outcome_expected_cost"] = float(w.get("outcome_expected_cost", 1.0))
w["candidate_ncf_cls"] = float(w.get("candidate_ncf_cls", 2.0))
w["candidate_false_safe_cls"] = float(w.get("candidate_false_safe_cls", 1.5))
w["ranking"] = float(w.get("ranking", 1.0))
w["imitation"] = float(w.get("imitation", 0.25))
with open(train_dst, "w", encoding="utf-8") as f:
    yaml.safe_dump(train, f, sort_keys=False, allow_unicode=True)
with open(label_src, encoding="utf-8") as f:
    label = yaml.safe_load(f)
pcfg = label.setdefault("planning", {})
pcfg["witness_probability_source"] = "logit"
pcfg["witness_temperature"] = float(pcfg.get("witness_temperature", 1.0))
pcfg["witness_logit_bias"] = float(pcfg.get("witness_logit_bias", 0.0))
pcfg["evidential_probability_mix"] = 0.0
pcfg["evidential_ucb_scale"] = 0.0
pcfg["adaptive_frontier_margin"] = float(pcfg.get("adaptive_frontier_margin", 0.25))
pcfg["candidate_certificate_penalty"] = float(pcfg.get("candidate_certificate_penalty", 1.5))
pcfg["candidate_min_ncf_prob"] = float(pcfg.get("candidate_min_ncf_prob", 0.05))
pcfg["candidate_max_false_safe_prob"] = float(pcfg.get("candidate_max_false_safe_prob", 0.95))
pcfg["candidate_risk_ncf_weight"] = float(pcfg.get("candidate_risk_ncf_weight", 1.0))
pcfg["candidate_risk_false_safe_weight"] = float(pcfg.get("candidate_risk_false_safe_weight", 2.0))
pcfg["candidate_risk_quality_weight"] = float(pcfg.get("candidate_risk_quality_weight", 0.75))
# Hybrid certificate fallback: if the learned candidate head is flat, make
# outcome-risk ranking the certificate used by the COWP frontier.
pcfg["candidate_cert_fallback_min_std"] = float(pcfg.get("candidate_cert_fallback_min_std", 2.0e-3))
pcfg["candidate_cert_flat_fallback_mix"] = float(pcfg.get("candidate_cert_flat_fallback_mix", 0.90))
pcfg["candidate_cert_hybrid_fallback_mix"] = float(pcfg.get("candidate_cert_hybrid_fallback_mix", 0.25))
pcfg["candidate_cert_fallback_outcome_mix"] = float(pcfg.get("candidate_cert_fallback_outcome_mix", 0.55))
pcfg["candidate_cert_fallback_action_mix"] = float(pcfg.get("candidate_cert_fallback_action_mix", 0.20))
pcfg["candidate_cert_fallback_rule_mix"] = float(pcfg.get("candidate_cert_fallback_rule_mix", 0.15))
pcfg["candidate_cert_fallback_pressure_mix"] = float(pcfg.get("candidate_cert_fallback_pressure_mix", 0.10))
pcfg["candidate_pair_risk_mix"] = float(pcfg.get("candidate_pair_risk_mix", 0.08))
pcfg["candidate_frontier_keep_fraction"] = float(pcfg.get("candidate_frontier_keep_fraction", 0.50))
pcfg["candidate_frontier_min_keep"] = int(pcfg.get("candidate_frontier_min_keep", 2))
pcfg["candidate_frontier_max_keep"] = int(pcfg.get("candidate_frontier_max_keep", 8))
pcfg["online_ignore_initial_jerk_steps"] = int(pcfg.get("online_ignore_initial_jerk_steps", 3))
pcfg["online_candidate_dedup_endpoint_m"] = float(pcfg.get("online_candidate_dedup_endpoint_m", 0.08))
pcfg["online_extra_accel_values_mps2"] = pcfg.get("online_extra_accel_values_mps2", [-4.0, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.25, 0.75, 1.25, 2.0, 3.0])
pcfg["target_online_candidates"] = int(pcfg.get("target_online_candidates", 32))
pcfg["online_candidate_dedup_endpoint_m"] = float(pcfg.get("online_candidate_dedup_endpoint_m", 0.08))
pcfg["online_candidate_dedup_speed_mps"] = float(pcfg.get("online_candidate_dedup_speed_mps", 0.15))
pcfg["online_dedup_same_macro_only"] = bool(pcfg.get("online_dedup_same_macro_only", True))
pcfg["online_terminal_speed_offsets_mps"] = pcfg.get("online_terminal_speed_offsets_mps", [-5.0, -3.0, -1.5, 0.0, 1.5, 3.0])
pcfg["online_terminal_s_offsets_m"] = pcfg.get("online_terminal_s_offsets_m", [-16.0, -8.0, 0.0, 8.0, 16.0])
pcfg["online_terminal_lateral_offsets_m"] = pcfg.get("online_terminal_lateral_offsets_m", [])
pcfg["candidate_pressure_prior_penalty"] = float(pcfg.get("candidate_pressure_prior_penalty", 0.75))
pcfg["candidate_pressure_prior_mix"] = float(pcfg.get("candidate_pressure_prior_mix", 0.45))
pcfg["candidate_rule_risk_penalty"] = float(pcfg.get("candidate_rule_risk_penalty", 2.5))
pcfg["candidate_rule_risk_mix"] = float(pcfg.get("candidate_rule_risk_mix", 2.5))
pcfg["candidate_action_risk_penalty"] = float(pcfg.get("candidate_action_risk_penalty", 3.0))
pcfg["candidate_action_risk_mix"] = float(pcfg.get("candidate_action_risk_mix", 3.0))
pcfg["online_action_risk_horizon_steps"] = int(pcfg.get("online_action_risk_horizon_steps", 8))
pcfg["candidate_outcome_risk_mix"] = float(pcfg.get("candidate_outcome_risk_mix", 1.25))
pcfg["decision_risk_min_std"] = float(pcfg.get("decision_risk_min_std", 1e-3))
pcfg["decision_risk_min_spread"] = float(pcfg.get("decision_risk_min_spread", 1e-3))
pcfg["candidate_frontier_keep_fraction"] = float(pcfg.get("candidate_frontier_keep_fraction", 0.25))
pcfg["candidate_frontier_min_keep"] = int(pcfg.get("candidate_frontier_min_keep", 2))
pcfg["candidate_frontier_max_keep"] = int(pcfg.get("candidate_frontier_max_keep", 4))
pcfg["online_collision_check_stride"] = int(pcfg.get("online_collision_check_stride", 4))
pcfg["online_collision_check_horizon_steps"] = int(pcfg.get("online_collision_check_horizon_steps", 60))
pcfg["online_rule_risk_stride"] = int(pcfg.get("online_rule_risk_stride", 4))
pcfg["online_rule_risk_horizon_steps"] = int(pcfg.get("online_rule_risk_horizon_steps", 60))
pcfg["online_require_cv_for_priority_agents"] = bool(pcfg.get("online_require_cv_for_priority_agents", True))
pcfg["online_logged_collision_buffer_m"] = float(pcfg.get("online_logged_collision_buffer_m", 0.15))
pcfg["online_priority_cv_collision_buffer_m"] = float(pcfg.get("online_priority_cv_collision_buffer_m", 0.35))
pcfg["online_collision_max_agents"] = int(pcfg.get("online_collision_max_agents", 24))
pcfg["online_rule_risk_max_agents"] = int(pcfg.get("online_rule_risk_max_agents", 8))
pcfg["online_pressure_prior_max_agents"] = int(pcfg.get("online_pressure_prior_max_agents", 8))
pcfg["online_pressure_prior_stride"] = int(pcfg.get("online_pressure_prior_stride", 4))
pcfg["online_pressure_prior_horizon_steps"] = int(pcfg.get("online_pressure_prior_horizon_steps", 60))
ccfg = label.setdefault("candidate", {})
ccfg["ignore_initial_jerk_steps"] = int(ccfg.get("ignore_initial_jerk_steps", 3))
ccfg["jerk_check_percentile"] = float(ccfg.get("jerk_check_percentile", 99.0))
ccfg["max_jerk_mps3"] = float(max(float(ccfg.get("max_jerk_mps3", 6.0)), 10.0))
with open(label_dst, "w", encoding="utf-8") as f:
    yaml.safe_dump(label, f, sort_keys=False, allow_unicode=True)
print({"train_cfg": train_dst, "label_cfg": label_dst})
PY

if [[ "$RUN_TESTS" == 1 ]]; then
  echo "[1/8] Regression tests"
  run_logged tests env PYTHONDONTWRITEBYTECODE=1 "$PYTHON_BIN" -m pytest --rootdir="$REPO_ROOT" -q
else
  echo "[1/8] Regression tests skipped"
fi

# Deterministic validation split for calibration vs held-out test.
CALIB_CACHE="$OUT_ROOT/cache_splits/val_calib"
TEST_CACHE="$OUT_ROOT/cache_splits/val_test"
run_logged split_val_cache "$PYTHON_BIN" - "$VAL_CACHE" "$CALIB_CACHE" "$TEST_CACHE" "$CALIB_PERCENT" <<'PY'
from pathlib import Path
import hashlib, os, shutil, sys
src, calib, test = map(Path, sys.argv[1:4])
pct = int(sys.argv[4])
if not (5 <= pct <= 50):
    raise SystemExit("CALIB_PERCENT must be between 5 and 50")
for dst in (calib, test):
    if dst.exists(): shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
files = sorted(src.glob("*.npz"))
if not files: raise SystemExit(f"No NPZ files found in {src}")
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

echo "[2/8] Existing caches reused. Logdiv disabled; collision/offroad kept as auxiliary labels."

train_stage() {
  local stage="$1" epochs="$2" lr="$3" batch="$4" output="$5" upstream_resume="${6:-}"
  local best="$output/cowp_${stage}_best.pt"
  mkdir -p "$output"

  local latest_info latest latest_epoch best_epoch resume resume_training target_last_epoch
  latest_info="$(latest_valid_stage_checkpoint "$output" "$stage")"
  IFS=$'\t' read -r latest latest_epoch <<< "$latest_info"
  latest="${latest:-}"
  latest_epoch="${latest_epoch:--1}"
  best_epoch="$(checkpoint_epoch "$best")"
  resume="${upstream_resume:-}"
  resume_training=0
  target_last_epoch=$((epochs - 1))

  if [[ "$FORCE_TRAIN" != 1 ]]; then
    if [[ -n "$latest" && -s "$latest" && "$latest_epoch" =~ ^-?[0-9]+$ && $latest_epoch -ge $target_last_epoch ]]; then
      if [[ ! -s "$best" ]]; then
        cp -f "$latest" "$best"
        echo "[train/$stage] completed $epochs epochs; copied latest checkpoint to missing best: $best"
      else
        echo "[train/$stage] completed $epochs epochs; keep existing best: $best"
      fi
      return
    fi
    if [[ -n "$latest" && -s "$latest" && "$latest_epoch" =~ ^-?[0-9]+$ && $latest_epoch -ge 0 ]]; then
      resume="$latest"
      resume_training=1
      echo "[train/$stage] resume from latest checkpoint epoch=$latest_epoch -> target_epochs=$epochs: $latest"
    elif [[ -s "$best" && "$best_epoch" =~ ^-?[0-9]+$ && $best_epoch -ge $target_last_epoch ]]; then
      echo "[train/$stage] best checkpoint already reaches target epoch=$best_epoch/$target_last_epoch: $best"
      return
    elif [[ -s "$best" && "$best_epoch" =~ ^-?[0-9]+$ && $best_epoch -ge 0 ]]; then
      resume="$best"
      resume_training=1
      echo "[train/$stage] no readable epoch checkpoint; resume from best epoch=$best_epoch -> target_epochs=$epochs: $best"
    elif [[ -n "$resume" ]]; then
      echo "[train/$stage] warm start from upstream checkpoint: $resume"
    else
      echo "[train/$stage] start from scratch"
    fi
  else
    echo "[train/$stage] FORCE_TRAIN=1, ignoring same-stage checkpoints and retraining this stage"
  fi

  local args=(
    "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="$NUM_GPUS" -m cowp.scripts.03_train
    --data-config configs/data.yaml
    --model-config configs/model.yaml
    --label-config "$LABEL_CFG"
    --train-config "$TRAIN_CFG"
    --cache-dir "$TRAIN_CACHE"
    --val-cache-dir "$CALIB_CACHE"
    --stage "$stage" --epochs "$epochs" --lr "$lr" --batch-size "$batch"
    --num-workers "$TRAIN_WORKERS" --prefetch-factor "$PREFETCH"
    --amp --fused-adamw --val-every 1 --output-dir "$output"
  )
  [[ -z "$resume" ]] || args+=(--resume "$resume")
  [[ "$resume_training" != 1 ]] || args+=(--resume-training)
  [[ "$stage" != response ]] || args+=(--no-response-traj)
  [[ "$stage" != planner ]] || args+=(--with-waymax-outcome-labels)
  run_logged "train_${stage}" "${args[@]}"
}

if [[ "$RUN_TRAIN" == 1 ]]; then
  echo "[3/8] Staged two-GPU training"
  if [[ "$PLANNER_ONLY_RETRAIN" == 1 ]]; then
    WIT="${WITNESS_CKPT:-$OUT_ROOT/checkpoints/witness/cowp_witness_best.pt}"
    require_file "$WIT"
    train_stage planner "$EPOCH_PLANNER" 1.5e-4 "$PER_GPU_BATCH_PLANNER" "$OUT_ROOT/checkpoints/planner" "$WIT"
  else
    if [[ -s "$WARM_RESPONSE_CKPT" && "$FORCE_TRAIN" != 1 ]]; then
      echo "Using warm response checkpoint: $WARM_RESPONSE_CKPT"
      RESP="$WARM_RESPONSE_CKPT"
    else
      train_stage natural "$NATURAL_EPOCHS" 3e-4 "$PER_GPU_BATCH_NATURAL" "$OUT_ROOT/checkpoints/natural"
      NAT="$OUT_ROOT/checkpoints/natural/cowp_natural_best.pt"
      require_file "$NAT"
      train_stage response "$RESPONSE_EPOCHS" 2.5e-4 "$PER_GPU_BATCH_RESPONSE" "$OUT_ROOT/checkpoints/response" "$NAT"
      RESP="$OUT_ROOT/checkpoints/response/cowp_response_best.pt"
      require_file "$RESP"
    fi
    train_stage witness "$EPOCH_WITNESS" 2e-4 "$PER_GPU_BATCH_WITNESS" "$OUT_ROOT/checkpoints/witness" "$RESP"
    WIT="$OUT_ROOT/checkpoints/witness/cowp_witness_best.pt"
    require_file "$WIT"
    train_stage planner "$EPOCH_PLANNER" 1.5e-4 "$PER_GPU_BATCH_PLANNER" "$OUT_ROOT/checkpoints/planner" "$WIT"
  fi
fi

if [[ -z "${CKPT:-}" ]]; then
  if [[ "$CKPT_SELECTION" == "latest" ]]; then
    latest_info="$(latest_valid_stage_checkpoint "$OUT_ROOT/checkpoints/planner" planner)"
    IFS=$'\t' read -r latest_ckpt latest_epoch <<< "$latest_info"
    if [[ -n "${latest_ckpt:-}" && -s "$latest_ckpt" ]]; then
      CKPT="$latest_ckpt"
      echo "Using latest planner checkpoint for eval: $CKPT (epoch=$latest_epoch)"
    else
      CKPT="$OUT_ROOT/checkpoints/planner/cowp_planner_best.pt"
    fi
  else
    CKPT="$OUT_ROOT/checkpoints/planner/cowp_planner_best.pt"
  fi
fi
require_file "$CKPT"

CAL_SWEEP="$OUT_ROOT/eval/learned_offline/calibration_sweep.json"
CAL_JSON="$OUT_ROOT/eval/witness_calibration.json"
if [[ "$RUN_OFFLINE_EVAL" == 1 ]]; then
  echo "[4/8] Witness threshold calibration on calibration split"
  if [[ "$FORCE_EVAL" == 1 ]] || ! json_matches_ckpt "$CAL_SWEEP" "$CKPT" "cowp" "learned_offline"; then
    run_logged calibrate_sweep env CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" -m cowp.scripts.04_eval_closed_loop \
      --data-config configs/data.yaml --label-config "$LABEL_CFG" --eval-config configs/eval.yaml \
      --cache-dir "$CALIB_CACHE" --mode learned_offline --method cowp \
      --checkpoint "$CKPT" --batch-size "$EVAL_BATCH" --device cuda \
      --witness-threshold 0.50 \
      --witness-threshold-sweep 0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90 \
      --ncf-gate-mode priority --priority-hard-threshold 0.55 \
      --secondary-witness-threshold 0.85 --secondary-opr-alpha 0.10 \
      --offline-fallback stop_like --adaptive-frontier-margin 0.25 \
      --soft-ncf-penalty 2.0 --outcome-risk-penalty "$OUTCOME_RISK_PENALTY" --outcome-risk-threshold "$OUTCOME_RISK_THRESHOLD" \
      --output "$CAL_SWEEP"
  fi
  run_logged select_threshold "$PYTHON_BIN" -m cowp.scripts.18_calibrate_witness_threshold \
    --input "$CAL_SWEEP" --output "$CAL_JSON" \
    --min-ncf-recall 0.90 --max-fallback 0.25
fi

WITNESS_THRESHOLD="${WITNESS_THRESHOLD:-$($PYTHON_BIN -c 'import json,sys; print(json.load(open(sys.argv[1]))["witness_threshold"])' "$CAL_JSON")}" 
echo "Calibrated witness threshold: $WITNESS_THRESHOLD"

run_learned_eval() {
  local method="$1" gpu="$2" output="$3"
  if [[ "$FORCE_EVAL" != 1 ]] && json_matches_ckpt "$output" "$CKPT" "$method" "learned_offline"; then
    echo "[learned/$method] keep existing for checkpoint $CKPT"
    return
  fi
  env CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -m cowp.scripts.04_eval_closed_loop \
    --data-config configs/data.yaml --label-config "$LABEL_CFG" --eval-config configs/eval.yaml \
    --cache-dir "$TEST_CACHE" --mode learned_offline --method "$method" \
    --checkpoint "$CKPT" --batch-size "$EVAL_BATCH" --device cuda \
    --witness-threshold "$WITNESS_THRESHOLD" --ncf-gate-mode priority \
    --priority-hard-threshold 0.55 --secondary-witness-threshold 0.85 \
    --secondary-opr-alpha 0.10 --soft-ncf-penalty 2.0 \
    --offline-fallback stop_like --adaptive-frontier-margin 0.25 \
    --outcome-risk-penalty "$OUTCOME_RISK_PENALTY" --outcome-risk-threshold "$OUTCOME_RISK_THRESHOLD" \
    --output "$output"
}

if [[ "$RUN_OFFLINE_EVAL" == 1 ]]; then
  echo "[5/8] Held-out learned-offline comparison"
  METHODS=(planner_score_only conventional_safety soft_burden_cost_only universal_ncf cowp)
  for ((i=0; i<${#METHODS[@]}; i+=2)); do
    pids=()
    for offset in 0 1; do
      j=$((i+offset)); ((j < ${#METHODS[@]})) || continue
      m="${METHODS[$j]}"
      run_bg_logged "learned_${m}" run_learned_eval "$m" "$offset" "$OUT_ROOT/eval/learned_offline/${m}.json"
      pids+=("$RUN_BG_PID")
    done
    wait_all "${pids[@]}"
  done
fi

run_waymax_method() {
  local method="$1"
  local per_shard=$(( (TOTAL_ONLINE_SCENARIOS + NUM_GPUS - 1) / NUM_GPUS ))
  local pids=() shard out
  local shards=()
  for shard in $(seq 0 $((NUM_GPUS-1))); do
    out="$OUT_ROOT/eval/waymax/${method}_shard${shard}.json"
    if [[ "$FORCE_EVAL" != 1 ]] && json_matches_ckpt "$out" "$CKPT" "$method" "waymax"; then
      echo "[waymax/$method] keep shard $shard for checkpoint $CKPT"
      continue
    fi
    local metric_args=() clear_args=() progress_args=() env_reuse_args=()
    [[ "$WAYMAX_STANDARD_METRICS" == 1 ]] && metric_args+=(--waymax-standard-metrics)
    [[ -z "$WAYMAX_STANDARD_METRIC_NAMES" ]] || metric_args+=(--waymax-standard-metric-names "$WAYMAX_STANDARD_METRIC_NAMES")
    [[ "$CLEAR_ACCELERATOR_CACHE" == 1 ]] && clear_args+=(--clear-accelerator-cache)
    [[ "$NO_PROGRESS" == 1 ]] && progress_args+=(--no-progress --status-every "$WAYMAX_STATUS_EVERY")
    [[ "$REUSE_WAYMAX_ENV" == 1 ]] && env_reuse_args+=(--reuse-waymax-env) || env_reuse_args+=(--no-reuse-waymax-env)
    run_bg_logged "waymax_${method}_shard${shard}" env CUDA_VISIBLE_DEVICES="$shard" XLA_PYTHON_CLIENT_PREALLOCATE=false "$PYTHON_BIN" -m cowp.scripts.04_eval_closed_loop \
      --data-config configs/data.yaml --label-config "$LABEL_CFG" --eval-config configs/eval.yaml \
      --mode waymax --waymax-split validation --tfexample-glob "$WAYMAX_VAL" \
      --method "$method" --checkpoint "$CKPT" \
      --num-scenarios "$per_shard" --num-shards "$NUM_GPUS" --shard-index "$shard" \
      --rollout-horizon-steps "$ROLLOUT_HORIZON_STEPS" "${metric_args[@]}" \
      --waymax-device gpu --jax-visible-devices 0 --jax-preallocate false \
      --waymax-action-mode absolute_xy_yaw --device cuda \
      --witness-threshold "$WITNESS_THRESHOLD" --ncf-gate-mode priority \
      --priority-hard-threshold 0.55 --secondary-witness-threshold 0.85 \
      --secondary-opr-alpha 0.10 --soft-ncf-penalty 2.0 \
      --adaptive-frontier-margin 0.25 \
      --outcome-risk-penalty "$OUTCOME_RISK_PENALTY" --outcome-risk-threshold "$OUTCOME_RISK_THRESHOLD" "${clear_args[@]}" "${progress_args[@]}" "${env_reuse_args[@]}" \
      --output "$out"
    pids+=("$RUN_BG_PID")
    shards+=("$out")
  done
  ((${#pids[@]} == 0)) || wait_all "${pids[@]}"
  # Also merge already-existing shards when no new shard was launched.
  if ((${#shards[@]} == 0)); then
    for shard in $(seq 0 $((NUM_GPUS-1))); do
      shards+=("$OUT_ROOT/eval/waymax/${method}_shard${shard}.json")
    done
  fi
  run_logged "merge_waymax_${method}" "$PYTHON_BIN" -m cowp.scripts.17_merge_waymax_shards \
    --output "$OUT_ROOT/eval/waymax/${method}_merged.json" \
    "${shards[@]}"
}

if [[ "$RUN_ONLINE_EVAL" == 1 ]]; then
  echo "[6/8] Real Waymax closed-loop comparison on identical sharded scenarios"
  read -r -a ONLINE_METHOD_ARRAY <<< "${ONLINE_METHODS:-planner_score_only conventional_safety cowp}"
  for m in "${ONLINE_METHOD_ARRAY[@]}"; do
    run_waymax_method "$m"
  done
fi

echo "[7/8] Build verification report"
run_logged build_report "$PYTHON_BIN" - "$OUT_ROOT" "$MODE" "$TOTAL_ONLINE_SCENARIOS" "$STRICT_GATES" <<'PY'
from pathlib import Path
import json, sys
root = Path(sys.argv[1]); mode=sys.argv[2]; expected=int(sys.argv[3]); strict=int(sys.argv[4])
methods = ["planner_score_only","conventional_safety","soft_burden_cost_only","universal_ncf","cowp"]

def load(p):
    p=Path(p)
    return json.loads(p.read_text()) if p.is_file() else None

def learned(m):
    d=load(root/"eval/learned_offline"/f"{m}.json")
    return (d or {}).get(m,{})

def online(m):
    d=load(root/"eval/waymax"/f"{m}_merged.json")
    return d or {}

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
report["gates"]["offline_ep_noninferior"]={"value":cowp.get("EP"),"baseline":base.get("EP"),"margin":0.05,"pass":float(cowp.get("EP",0)) >= float(base.get("EP",0))-0.05}
report["gates"]["offline_false_safe_selected_reduced"]={"value":cowp.get("SelectedFalseSafeRate"),"baseline":base.get("SelectedFalseSafeRate"),"pass":float(cowp.get("SelectedFalseSafeRate",1)) < float(base.get("SelectedFalseSafeRate",1))}
report["gates"]["offline_fallback"]={"value":cowp.get("FallbackRate"),"target":0.25 if mode=="full" else 0.35}

hard_fail=[]
for m,d in report["online"].items():
    n=int(d.get("num_rollouts") or 0)
    count_ok=n in {expected, expected+1}
    report["gates"][f"{m}/rollout_count"]={"value":n,"target":expected,"pass":count_ok}
    if not count_ok: hard_fail.append(f"{m}: rollout_count={n}")
    std=d.get("standard",{}); diag=d.get("diagnostic",{})
    kin=float(std.get("KinematicsInfeasibilityRate",1.0))
    kin_target=0.05 if mode=="full" else 0.10
    ok=kin <= kin_target
    report["gates"][f"{m}/kinematics"]={"value":kin,"target":kin_target,"pass":ok}
    if not ok: hard_fail.append(f"{m}: kinematics={kin:.4f}")
    report["gates"][f"{m}/critical_agents_not_saturated"]={"value":float(diag.get("ClosedLoopMean/critical_agents",0.0)),"target_max":4.5,"pass":float(diag.get("ClosedLoopMean/critical_agents",0.0))<=4.5}

cs=report["online"].get("cowp",{}).get("standard",{})
bs=report["online"].get("planner_score_only",{}).get("standard",{})
if cs and bs:
    report["gates"]["online_cr_noninferior"]={"value":cs.get("CR"),"baseline":bs.get("CR"),"margin":0.03,"pass":float(cs.get("CR",1))<=float(bs.get("CR",1))+0.03}
    report["gates"]["online_ep_noninferior"]={"value":cs.get("EP"),"baseline":bs.get("EP"),"margin":0.05,"pass":float(cs.get("EP",0))>=float(bs.get("EP",0))-0.05}

(root/"verification_report.json").write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
lines=["# Mechanism-aware P-NCF/COWP verification", "", f"Mode: `{mode}`", "", "## Held-out learned-offline", "", "| Method | FSR | EP | CBS | OPR | HBCR | Selected FS | Fallback | Witness AUPRC | Cert FS AUPRC | Cert NCF AUPRC | Cert Risk Rank | Sel Cert Risk | Sel Pressure |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
for m in methods:
    x=L.get(m,{})
    def f(k):
        v=x.get(k)
        return "-" if v is None else f"{float(v):.4f}"
    lines.append(f"| {m} | {f('FSR')} | {f('EP')} | {f('CBS')} | {f('OPR')} | {f('HBCR')} | {f('SelectedFalseSafeRate')} | {f('FallbackRate')} | {f('WitnessQuality/AUPRC')} | {f('CandidateCertificate/FalseSafe_AUPRC')} | {f('CandidateCertificate/NCF_AUPRC')} | {f('CandidateCertificate/RiskRankingPairAccuracy')} | {f('CandidateCertificate/SelectedRiskMean')} | {f('CandidateCertificate/SelectedPressurePriorMean')} |")
lines += ["", "## Real Waymax closed loop", "", "| Method | N | CR | Collision | Offroad | EP | Kin infeasible | Valid cand | Conv cand | Accepted | Fallback step | Sel cert risk | Sel pressure | Rule risk | Action risk | Outcome risk |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
for m in ("planner_score_only","conventional_safety","cowp"):
    x=report["online"].get(m,{}); s=x.get("standard",{}); d=x.get("diagnostic",{})
    def g(v): return "-" if v is None else f"{float(v):.4f}"
    lines.append(f"| {m} | {x.get('num_rollouts','-')} | {g(s.get('CR'))} | {g(s.get('CollisionRate'))} | {g(s.get('OffroadRate'))} | {g(s.get('EP'))} | {g(s.get('KinematicsInfeasibilityRate'))} | {g(d.get('ClosedLoopMean/valid_candidates'))} | {g(d.get('ClosedLoopMean/conventional_candidates'))} | {g(d.get('ClosedLoopMean/accepted_candidates'))} | {g(d.get('ClosedLoopFallbackStepRate'))} | {g(d.get('ClosedLoopMean/selected_candidate_cert_risk'))} | {g(d.get('ClosedLoopMean/selected_candidate_pressure_prior'))} | {g(d.get('ClosedLoopMean/selected_candidate_rule_risk'))} | {g(d.get('ClosedLoopMean/selected_candidate_action_risk'))} | {g(d.get('ClosedLoopMean/selected_outcome_decision_risk'))} |")
lines += ["", "## Gates", ""]
for k,v in report["gates"].items():
    passed=v.get("pass")
    if passed is None and "target" in v and v.get("value") is not None:
        passed=float(v["value"]) >= float(v["target"]) if "auprc" in k or "spread" in k else float(v["value"]) <= float(v["target"])
    mark="PASS" if passed else ("CHECK" if passed is None else "FAIL")
    lines.append(f"- **{mark}** `{k}`: {v}")
lines += ["", "Log-divergence is intentionally excluded because the attached cache has no finite logdiv labels."]
(root/"verification_report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
print(json.dumps({"report":str(root/"verification_report.md"),"hard_fail":hard_fail},indent=2,ensure_ascii=False))
if strict and hard_fail:
    raise SystemExit(3)
PY

echo "[8/8] Complete"
echo "Checkpoint: $CKPT"
echo "Report: $OUT_ROOT/verification_report.md"
echo "JSON:   $OUT_ROOT/verification_report.json"
echo "Logs:   $LOG_DIR"
