#!/usr/bin/env bash
set -u -o pipefail
ROOT="${COWP_CODE_ROOT:-$(cd "$(dirname "$0")" && pwd)}"; cd "$ROOT"
DATA_ROOT="${DATA_ROOT:-/data0/senzeyu2/dataset/COWP/formal_v16_8_9_causal_audit}"
SEED="${TRAIN_SEED:-2026}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
BASE_OUT="${OUT_BASE:-outputs/cowp_v16_8_9_ablation}"

run_one() {
  local tag="$1" model_cfg="$2" train_cfg="$3"
  local out="${BASE_OUT}_${tag}_seed${SEED}"
  echo "===== learned ablation: $tag ====="
  set +e
  DATA_ROOT="$DATA_ROOT" OUT_ROOT="$out" TRAIN_SEED="$SEED" CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    MODEL_CONFIG_SOURCE="$model_cfg" TRAIN_CONFIG_SOURCE="$train_cfg" BACKGROUND=0 FORCE_TRAIN=1 FORCE_EVAL=1 \
    bash "$ROOT/NEXT_RUN_COMMANDS_V16_8_9_MECHANISM_CN.sh"
  local status=$?
  set -e
  # A mechanism gate may correctly reject an ablation. Preserve the output and
  # continue, but require that the learned-offline directory was actually made.
  if [[ ! -d "$out/eval/learned_offline" ]]; then
    echo "ablation $tag failed before learned evaluation (status=$status)" >&2
    return "$status"
  fi
  echo "ablation $tag finished with status=$status; gate failure is an experimental result, not a launcher error."
}
set -e
run_one no_causal_relevance \
  configs/model_cowp_v16_8_9_no_causal_relevance.yaml \
  configs/train_cowp_v16_8_9_no_causal_relevance.yaml
run_one conflict_only_transport \
  configs/model_cowp_v16_8_9_conflict_only_transport.yaml \
  configs/train_cowp_v16_8_9_conflict_only_transport.yaml
