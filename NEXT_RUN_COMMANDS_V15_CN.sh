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
export OUT_ROOT="${OUT_ROOT:-outputs/cowp_v15_model_v9labels_seed2026}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

mkdir -p "$OUT_ROOT/eval" "$OUT_ROOT/logs"

# 代码回归测试。
pytest -q

# 先扫描服务器上“当前实际存在”的 raw cache。上传的旧报告中 train=14640，
# 而 v14 训练日志加载了 20440，因此不能直接假定旧报告代表当前目录。
CACHE_SUFF_REPORT="$OUT_ROOT/eval/cache_sufficiency_current.json"
if [[ "${FORCE_CACHE_AUDIT:-0}" == "1" || ! -s "$CACHE_SUFF_REPORT" ]]; then
  python -u -m cowp.scripts.19_diagnose_waymax_cache_sufficiency \
    --train-cache "$RAW_TRAIN_CACHE" \
    --val-cache "$RAW_VAL_CACHE" \
    --workers "${CACHE_AUDIT_WORKERS:-8}" \
    --output-json "$CACHE_SUFF_REPORT" \
    | tee "$OUT_ROOT/logs/cache_sufficiency_current.log"
fi

# 对 raw/v9 overlay 做独立门禁：数量、关键字段、SDC、critical mapping、
# response-root 范围、train/val 重叠及 logdiv 状态。
CACHE_REUSE_REPORT="$OUT_ROOT/eval/cache_reuse_gate_v9.json"
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

# v9 safety replay没有有限 logdiv；禁止误用它作为真零标签。
python - <<'PY'
from pathlib import Path
import yaml
cfg=yaml.safe_load(Path('configs/train_cowp_v15.yaml').read_text())
assert abs(float(cfg['loss_weights'].get('outcome_logdiv', 0.0))) <= 1e-12
PY

export DATA_PROTOCOL=v9_reuse
export CACHE_REUSE_REPORT
export TRAIN_VISIBLE_DEVICES="${TRAIN_VISIBLE_DEVICES:-0,1}"
export TRAIN_NPROC="${TRAIN_NPROC:-2}"
export RUN_AUGMENT=0
export RUN_DIAGNOSE="${RUN_DIAGNOSE:-1}"
export RUN_NATURAL="${RUN_NATURAL:-1}"
export RUN_TRANSPORT="${RUN_TRANSPORT:-1}"
export RUN_PLANNER="${RUN_PLANNER:-1}"
export RUN_OFFLINE="${RUN_OFFLINE:-1}"
export RUN_PROBE="${RUN_PROBE:-1}"
export RUN_FULL="${RUN_FULL:-0}"
export FORCE_TRAIN="${FORCE_TRAIN:-0}"
export FORCE_EVAL="${FORCE_EVAL:-0}"

bash run_cowp_v15_dual_gpu.sh
