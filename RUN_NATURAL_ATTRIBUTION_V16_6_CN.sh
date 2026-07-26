#!/usr/bin/env bash
set -euo pipefail

ROOT="${COWP_CODE_ROOT:-$(cd "$(dirname "$0")" && pwd)}"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
COWP_ROOT="${COWP_ROOT:-/data0/senzeyu2/dataset/COWP/formal}"
RAW_TRAIN_CACHE="${RAW_TRAIN_CACHE:-$COWP_ROOT/tensor_cache_train_waymax}"
RAW_VAL_CACHE="${RAW_VAL_CACHE:-$COWP_ROOT/tensor_cache_val_waymax}"
INIT_CKPT="${INIT_CKPT:-outputs/cowp_v10_gct_probe100_seed2026/checkpoints/planner/cowp_planner_best.pt}"
# By default, reuse the v16.5 checkpoints already produced on the server.  The
# uploaded result archives intentionally omitted large checkpoint files.
MAIN_OUT_ROOT="${MAIN_OUT_ROOT:-outputs/cowp_v16_5_natural_recovery_v9labels_seed2026}"
SOURCE_ABL_ROOT="${SOURCE_ABL_ROOT:-outputs/cowp_v16_5_natural_ablations_v9labels_seed2026}"
ATTR_OUT_ROOT="${ATTR_OUT_ROOT:-outputs/cowp_v16_6_natural_attribution_aligned_v9labels_seed2026}"
TRAIN_VISIBLE_DEVICES="${TRAIN_VISIBLE_DEVICES:-0,1}"
TRAIN_NPROC="${TRAIN_NPROC:-2}"
BATCH="${BATCH_PER_GPU:-5}"
WORKERS="${NATURAL_NUM_WORKERS:-8}"
VAL_WORKERS="${NATURAL_VAL_NUM_WORKERS:-2}"
PREFETCH="${NATURAL_PREFETCH_FACTOR:-2}"
VAL_PREFETCH="${NATURAL_VAL_PREFETCH_FACTOR:-1}"
VAL_EVERY="${NATURAL_VAL_EVERY:-2}"
LR="${NATURAL_LR:-3.0e-5}"
DIAG_SCENES="${LEARNED_NATURAL_DIAG_SCENES:-2000}"
DIAG_BATCH="${LEARNED_NATURAL_DIAG_BATCH:-8}"
DIAG_WORKERS="${DIAG_WORKERS:-8}"
DIAG_DEVICE="${LEARNED_NATURAL_DIAG_DEVICE:-cuda}"
mkdir -p "$ATTR_OUT_ROOT" "$ATTR_OUT_ROOT/logs" "$ATTR_OUT_ROOT/aligned_reports"

MAIN_REPORT="$MAIN_OUT_ROOT/eval/learned_offline/learned_natural_effectiveness.json"
MAIN_GATE="$MAIN_OUT_ROOT/eval/learned_offline/natural_effectiveness_gate.json"
[[ -s "$MAIN_REPORT" && -s "$MAIN_GATE" ]] || {
  echo "missing strict natural recovery reports under MAIN_OUT_ROOT=$MAIN_OUT_ROOT" >&2; exit 2;
}
"$PYTHON_BIN" - "$MAIN_GATE" <<'PY'
import json, sys
x=json.load(open(sys.argv[1], encoding='utf-8'))
assert bool(x.get('pass', False)), f"main natural effectiveness gate failed: {sys.argv[1]}"
PY

TARGET_EPOCH="$($PYTHON_BIN - "$MAIN_REPORT" <<'PY'
import json,sys
x=json.load(open(sys.argv[1],encoding='utf-8'))
e=x.get('checkpoint_epoch')
assert isinstance(e,int) and e >= 0, f'invalid checkpoint_epoch={e!r}'
print(e)
PY
)"
TARGET_EPOCHS=$((TARGET_EPOCH + 1))
printf -v TARGET_TAG '%03d' "$TARGET_EPOCH"
echo "[v16.6 attribution] aligned checkpoint epoch=$TARGET_EPOCH"

resolve_checkpoint() {
  local dir="$1"
  local exact="$dir/cowp_natural_epoch${TARGET_TAG}.pt"
  local best="$dir/cowp_natural_best.pt"
  if [[ -s "$exact" ]]; then printf '%s\n' "$exact"; return 0; fi
  if [[ -s "$best" ]]; then
    if "$PYTHON_BIN" - "$best" "$TARGET_EPOCH" <<'PY' >/dev/null 2>&1
import sys, torch
x=torch.load(sys.argv[1],map_location='cpu')
raise SystemExit(0 if int(x.get('epoch',-999)) == int(sys.argv[2]) else 1)
PY
    then printf '%s\n' "$best"; return 0; fi
  fi
  return 1
}

MAIN_CKPT_DIR="$MAIN_OUT_ROOT/checkpoints/natural"
if ! MAIN_CKPT="$(resolve_checkpoint "$MAIN_CKPT_DIR")"; then
  echo "Cannot find main checkpoint at selected epoch $TARGET_EPOCH in $MAIN_CKPT_DIR." >&2
  echo "Expected cowp_natural_epoch${TARGET_TAG}.pt or a best checkpoint whose metadata epoch matches." >&2
  echo "Re-run v16.6 strict recovery with --save-every 1, or point MAIN_OUT_ROOT to the server run containing checkpoints." >&2
  exit 2
fi

run_or_resolve_arm() {
  local name="$1" model_cfg="$2" train_cfg="$3"
  local source_dir="$SOURCE_ABL_ROOT/$name/checkpoints"
  local train_dir="$ATTR_OUT_ROOT/retrained/$name/checkpoints"
  local ckpt=""
  if [[ "${FORCE_RETRAIN_ABLATIONS:-0}" != "1" ]] && ckpt="$(resolve_checkpoint "$source_dir" 2>/dev/null)"; then
    printf '%s\n' "$ckpt"; return 0
  fi
  mkdir -p "$train_dir" "$ATTR_OUT_ROOT/retrained/$name/logs" "$ATTR_OUT_ROOT/retrained/$name/configs"
  [[ -s "$INIT_CKPT" ]] || { echo "missing INIT_CKPT=$INIT_CKPT" >&2; exit 2; }
  cp "$model_cfg" "$train_cfg" "$ATTR_OUT_ROOT/retrained/$name/configs/"
  echo "[v16.6 attribution] exact epoch checkpoint missing for $name; retraining only to epoch $TARGET_EPOCH" >&2
  env CUDA_VISIBLE_DEVICES="$TRAIN_VISIBLE_DEVICES" \
    "$TORCHRUN_BIN" --standalone --nproc_per_node="$TRAIN_NPROC" -m cowp.scripts.03_train \
      --data-config configs/data.yaml --model-config "$model_cfg" \
      --label-config configs/label_cowp_v16.yaml --train-config "$train_cfg" \
      --cache-dir "$RAW_TRAIN_CACHE" --val-cache-dir "$RAW_VAL_CACHE" \
      --stage natural --epochs "$TARGET_EPOCHS" --batch-size "$BATCH" --lr "$LR" \
      --num-workers "$WORKERS" --prefetch-factor "$PREFETCH" \
      --val-num-workers "$VAL_WORKERS" --val-prefetch-factor "$VAL_PREFETCH" \
      --val-every "$VAL_EVERY" --device cuda --output-dir "$train_dir" \
      --early-stop-patience 0 --early-stop-min-delta 1e-4 \
      --lr-scheduler plateau --min-lr 2e-6 --save-every 1 --no-positive-oversampling \
      --eval-before-train --reset-checkpoint-prefix natural_decoder \
      --natural-graph-unfreeze-epoch -1 --grad-clip 1.0 --amp --amp-dtype auto --fused-adamw \
      --resume "$INIT_CKPT" 2>&1 | tee "$ATTR_OUT_ROOT/retrained/$name/logs/train.log" >&2
  ckpt="$(resolve_checkpoint "$train_dir")" || {
    echo "retraining did not produce aligned epoch checkpoint for $name" >&2; exit 2;
  }
  printf '%s\n' "$ckpt"
}

NO_CAP_CKPT="$(run_or_resolve_arm no_obs_capacity_boost configs/model_cowp_v16_no_obs_capacity.yaml configs/train_cowp_v16.yaml)"
NO_ENV_CKPT="$(run_or_resolve_arm no_mass_aware_root_envelope configs/model_cowp_v16.yaml configs/train_cowp_v16_no_mass_trust.yaml)"

run_diagnostic() {
  local name="$1" ckpt="$2" model_cfg="$3" train_cfg="$4"
  local report="$ATTR_OUT_ROOT/aligned_reports/$name.json"
  "$PYTHON_BIN" -u -m cowp.scripts.39_diagnose_learned_natural \
    --data-config configs/data.yaml --model-config "$model_cfg" \
    --label-config configs/label_cowp_v16.yaml --train-config "$train_cfg" \
    --cache-dir "$RAW_VAL_CACHE" --checkpoint "$ckpt" \
    --max-scenes "$DIAG_SCENES" --batch-size "$DIAG_BATCH" \
    --num-workers "$DIAG_WORKERS" --device "$DIAG_DEVICE" \
    --output "$report" 2>&1 | tee "$ATTR_OUT_ROOT/logs/diagnose_${name}.log"
}

# Re-diagnose all three checkpoints with one code path and the exact same sampled
# scene indices.  Never reuse old JSON summaries because v16.6 adds the exact
# squared-excess objective and paired scene-level statistics.
run_diagnostic main "$MAIN_CKPT" configs/model_cowp_v16.yaml configs/train_cowp_v16.yaml
run_diagnostic no_obs_capacity_boost "$NO_CAP_CKPT" configs/model_cowp_v16_no_obs_capacity.yaml configs/train_cowp_v16.yaml
run_diagnostic no_mass_aware_root_envelope "$NO_ENV_CKPT" configs/model_cowp_v16.yaml configs/train_cowp_v16_no_mass_trust.yaml

"$PYTHON_BIN" - "$ATTR_OUT_ROOT/attribution_protocol_manifest.json" "$MAIN_CKPT" "$NO_CAP_CKPT" "$NO_ENV_CKPT" "$TARGET_EPOCH" "$DIAG_SCENES" <<'PY'
import json, pathlib, sys
out, main, cap, env, epoch, scenes = sys.argv[1:]
x={
  'version':'v16.6', 'checkpoint_policy':'same_main_selected_epoch',
  'target_epoch':int(epoch), 'diagnostic_scenes':int(scenes),
  'checkpoints':{'main':main,'no_obs_capacity_boost':cap,'no_mass_aware_root_envelope':env},
  'paper_claim_note':'This single-seed gate permits downstream evidence collection; it is not a publication-level component claim.'
}
pathlib.Path(out).write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')
PY

"$PYTHON_BIN" -u -m cowp.scripts.41_compare_natural_ablations \
  --main "$ATTR_OUT_ROOT/aligned_reports/main.json" \
  --no-obs-capacity "$ATTR_OUT_ROOT/aligned_reports/no_obs_capacity_boost.json" \
  --no-mass-trust "$ATTR_OUT_ROOT/aligned_reports/no_mass_aware_root_envelope.json" \
  --output "$ATTR_OUT_ROOT/natural_component_attribution_gate.json" \
  --bootstrap-samples "${ATTR_BOOTSTRAP_SAMPLES:-2000}" \
  2>&1 | tee "$ATTR_OUT_ROOT/natural_component_attribution_gate.log"

echo "v16.6 aligned attribution complete: $ATTR_OUT_ROOT/natural_component_attribution_gate.json"
