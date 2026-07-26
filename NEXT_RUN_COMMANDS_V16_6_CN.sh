#!/usr/bin/env bash
set -euo pipefail

COWP_CODE_ROOT="${COWP_CODE_ROOT:-$(cd "$(dirname "$0")" && pwd)}"
cd "$COWP_CODE_ROOT"

# 本脚本默认复用已经存在的 formal/tensor_cache_*_waymax 与 transport_v9。
# 不执行 index、build labels、build tensor cache、Waymax replay 或 transport augment。
export COWP_ROOT="${COWP_ROOT:-/data0/senzeyu2/dataset/COWP/formal}"
export RAW_TRAIN_CACHE="${RAW_TRAIN_CACHE:-$COWP_ROOT/tensor_cache_train_waymax}"
export RAW_VAL_CACHE="${RAW_VAL_CACHE:-$COWP_ROOT/tensor_cache_val_waymax}"
export TRAIN_CACHE="${TRAIN_CACHE:-$COWP_ROOT/tensor_cache_train_waymax_transport_v9}"
export VAL_CACHE="${VAL_CACHE:-$COWP_ROOT/tensor_cache_val_waymax_transport_v9}"
export OUT_ROOT="${OUT_ROOT:-outputs/cowp_v16_6_pipeline_v9labels_seed2026}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

mkdir -p "$OUT_ROOT/eval" "$OUT_ROOT/logs"

# 默认把整个流程放到后台；所有总输出进入 driver.nohup.log，各阶段仍保留独立日志。
# 前台调试可显式设置 BACKGROUND=0。
export BACKGROUND="${BACKGROUND:-1}"
if [[ "$BACKGROUND" == "1" && "${COWP_V16_6_BACKGROUND_CHILD:-0}" != "1" ]]; then
  export COWP_V16_6_BACKGROUND_CHILD=1
  nohup env BACKGROUND=0 bash "$0" > "$OUT_ROOT/logs/driver.nohup.log" 2>&1 < /dev/null &
  pid=$!
  printf '%s\n' "$pid" > "$OUT_ROOT/logs/driver.pid"
  echo "[cowp_v16.6] background pid=$pid"
  echo "[cowp_v16.6] log=$OUT_ROOT/logs/driver.nohup.log"
  exit 0
fi

# 环境、依赖、配置和真实 A=6/M=24 forward-loss-backward 合同预检。
preflight_args=(--require-cuda)
[[ "${REQUIRE_WAYMAX_PREFLIGHT:-0}" == "1" ]] && preflight_args+=(--require-waymax)
python -u -m cowp.scripts.43_pipeline_preflight \
  --model-config configs/model_cowp_v16.yaml \
  --label-config configs/label_cowp_v16.yaml \
  --train-config configs/train_cowp_v16.yaml \
  --eval-config configs/eval_cowp_v16.yaml \
  --output "$OUT_ROOT/eval/pipeline_preflight.json" \
  "${preflight_args[@]}" | tee "$OUT_ROOT/logs/pipeline_preflight.log"

# 代码回归测试。真实维度的 natural-loss 回归测试会在启动 GPU 训练前捕获广播错误。
pytest -q

# 先扫描服务器上“当前实际存在”的 raw cache。上传的旧报告中 train=14640，
# 而 v14 训练日志加载了 20440，因此不能直接假定旧报告代表当前目录。
CACHE_SUFF_REPORT="$OUT_ROOT/eval/cache_sufficiency_current.json"
if [[ "${FORCE_CACHE_AUDIT:-0}" == "1" || ! -s "$CACHE_SUFF_REPORT" ]]; then
  cache_audit_sample_args=()
  if [[ "${FULL_CACHE_AUDIT:-0}" != "1" ]]; then
    cache_audit_sample_args=(--sample-scenes "${CACHE_AUDIT_SAMPLE_SCENES:-2048}")
  fi
  python -u -m cowp.scripts.19_diagnose_waymax_cache_sufficiency \
    --train-cache "$RAW_TRAIN_CACHE" \
    --val-cache "$RAW_VAL_CACHE" \
    --workers "${CACHE_AUDIT_WORKERS:-8}" \
    "${cache_audit_sample_args[@]}" \
    --output-json "$CACHE_SUFF_REPORT" \
    | tee "$OUT_ROOT/logs/cache_sufficiency_current.log"
fi

# 对 raw/v9 overlay 做独立门禁：数量、关键字段、SDC、critical mapping、
# response-root 范围、train/val 重叠及 logdiv 状态。
CACHE_REUSE_REPORT="$OUT_ROOT/eval/cache_reuse_gate_v9.json"
if [[ "${FORCE_CACHE_AUDIT:-0}" == "1" || ! -s "$CACHE_REUSE_REPORT" ]]; then
  python -u -m cowp.scripts.38_gate_cache_reuse \
    --raw-train "$RAW_TRAIN_CACHE" \
    --raw-val "$RAW_VAL_CACHE" \
    --transport-train "$TRAIN_CACHE" \
    --transport-val "$VAL_CACHE" \
    --sample-scenes "${CACHE_GATE_SAMPLE_SCENES:-1024}" \
    --min-train-scenes "${MIN_TRAIN_SCENES:-20000}" \
    --min-val-scenes "${MIN_VAL_SCENES:-5000}" \
    --output "$CACHE_REUSE_REPORT" \
    | tee "$OUT_ROOT/logs/cache_reuse_gate_v9.log"
fi

# v9 safety replay 没有有限 logdiv；禁止误用它作为真零标签。
# 本轮首先验证 CNOB dynamics decoder 与新 loss，本脚本不会重建数据。
python - <<'PY'
from pathlib import Path
import yaml
cfg=yaml.safe_load(Path('configs/train_cowp_v16.yaml').read_text())
assert abs(float(cfg['loss_weights'].get('outcome_logdiv', 0.0))) <= 1e-12
PY

export DATA_PROTOCOL=v9_reuse
export CACHE_REUSE_REPORT
export TRAIN_VISIBLE_DEVICES="${TRAIN_VISIBLE_DEVICES:-0,1}"
export TRAIN_NPROC="${TRAIN_NPROC:-2}"
export AMP_DTYPE="${AMP_DTYPE:-auto}"
export NATURAL_AMP="${NATURAL_AMP:-0}"
export TRANSPORT_AMP="${TRANSPORT_AMP:-1}"
export PLANNER_AMP="${PLANNER_AMP:-1}"
export RUN_AUGMENT=0
export RUN_DIAGNOSE="${RUN_DIAGNOSE:-1}"
export DIAG_PROFILE="${DIAG_PROFILE:-fast}"
export RUN_NATURAL="${RUN_NATURAL:-1}"
export RUN_TRANSPORT="${RUN_TRANSPORT:-1}"
export RUN_PLANNER="${RUN_PLANNER:-1}"
export RUN_OFFLINE="${RUN_OFFLINE:-1}"
export RUN_PROBE="${RUN_PROBE:-1}"
export RUN_FULL="${RUN_FULL:-0}"
export FORCE_TRAIN="${FORCE_TRAIN:-0}"
export FORCE_EVAL="${FORCE_EVAL:-0}"
# 首次运行默认只到 natural：只有绝对 gate 和“相对解析基线有效性 gate”都通过，
# 才值得继续消耗 transport/planner/Waymax GPU 时间。
export STOP_AFTER_STAGE="${STOP_AFTER_STAGE:-natural}"

bash run_cowp_v16_6_dual_gpu.sh
