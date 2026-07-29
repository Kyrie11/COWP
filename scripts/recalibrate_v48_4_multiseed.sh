#!/usr/bin/env bash
set -euo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

SOURCE_RUN="${SOURCE_RUN:?Set SOURCE_RUN to a completed v48.4 output directory}"
EVAL_OCRAP_ROOT="${EVAL_OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}"
MULTISEED_ROOT="${MULTISEED_ROOT:-${SOURCE_RUN}_calibration_multiseed}"
SEEDS="${SEEDS:-4801,4802,4803}"
CALIBRATION_FRACTION="${CALIBRATION_FRACTION:-0.50}"
GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"
mkdir -p "$MULTISEED_ROOT"

IFS=',' read -r -a seed_array <<< "$SEEDS"
for raw_seed in "${seed_array[@]}"; do
  seed="${raw_seed//[[:space:]]/}"
  [[ -n "$seed" ]] || continue
  out="$MULTISEED_ROOT/seed_$seed"
  split="$out/dataset_splits"
  mkdir -p "$split" "$out/logs"
  for regime in safe near_contact contact; do
    input="$EVAL_OCRAP_ROOT/val_$regime"
    python tools/split_calibration_by_scene_v48.py \
      --input "$input" \
      --calibration-output "$split/calibration_$regime" \
      --validation-output "$split/development_$regime" \
      --calibration-fraction "$CALIBRATION_FRACTION" --seed "$seed" \
      --link-mode hardlink --overwrite \
      > "$out/logs/split_${regime}.log" 2>&1
  done

  calibrate_variant() {
    local variant="$1" gpu="$2"
    local ckpt="$SOURCE_RUN/candidates/$variant/model_v48_trac_sr/best.pt"
    local vout="$out/candidates/$variant"
    [[ -f "$ckpt" ]] || { echo "skip missing $ckpt"; return 0; }
    mkdir -p "$vout/calibration" "$vout/logs"
    set +e
    CUDA_VISIBLE_DEVICES="$gpu" python -u tools/calibrate_policy_risk_v48.py \
      --dataset "$split/calibration_near_contact" --checkpoint "$ckpt" --bucket near \
      --output "$vout/calibration/direct_value_risk_near_v48.json" \
      --rows-output "$vout/calibration/direct_value_risk_near_v48.rows.jsonl" \
      --required-min-groups="${NEAR_MIN_GROUPS:-80}" --required-min-scenes="${NEAR_MIN_SCENES:-40}" \
      --min-fit-selected="${NEAR_MIN_FIT_SELECTED:-8}" --min-verify-selected="${NEAR_MIN_VERIFY_SELECTED:-6}" \
      --max-fit-harmful-group-ucb="${NEAR_MAX_FIT_HARM_UCB:-0.16}" \
      --max-verify-harmful-group-ucb="${NEAR_MAX_VERIFY_HARM_UCB:-0.18}" \
      > "$vout/logs/calibrate_near.log" 2>&1
    echo $? > "$vout/calibration/near.exit_code"
    CUDA_VISIBLE_DEVICES="$gpu" python -u tools/calibrate_policy_risk_v48.py \
      --dataset "$split/calibration_contact" --checkpoint "$ckpt" --bucket contact \
      --output "$vout/calibration/direct_value_risk_contact_v48.json" \
      --rows-output "$vout/calibration/direct_value_risk_contact_v48.rows.jsonl" \
      --required-min-groups="${CONTACT_MIN_GROUPS:-100}" --required-min-scenes="${CONTACT_MIN_SCENES:-45}" \
      --min-fit-selected="${CONTACT_MIN_FIT_SELECTED:-9}" --min-verify-selected="${CONTACT_MIN_VERIFY_SELECTED:-6}" \
      --max-fit-harmful-group-ucb="${CONTACT_MAX_FIT_HARM_UCB:-0.18}" \
      --max-verify-harmful-group-ucb="${CONTACT_MAX_VERIFY_HARM_UCB:-0.20}" \
      > "$vout/logs/calibrate_contact.log" 2>&1
    echo $? > "$vout/calibration/contact.exit_code"
    set -e
  }
  calibrate_variant balanced "$GPU0" & p0=$!
  calibrate_variant precision "$GPU1" & p1=$!
  wait "$p0" || true; wait "$p1" || true
  echo "completed calibration seed=$seed output=$out"
done

python tools/summarize_v48_4_multiseed.py --root "$MULTISEED_ROOT" \
  --output "$MULTISEED_ROOT/multiseed_summary.json"
echo "Multi-seed summary: $MULTISEED_ROOT/multiseed_summary.json"
