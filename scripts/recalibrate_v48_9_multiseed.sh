#!/usr/bin/env bash
set -euo pipefail

REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

SOURCE_RUN="${SOURCE_RUN:?Set SOURCE_RUN to a completed v48.9 training run}"
EVAL_OCRAP_ROOT="${EVAL_OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}"
MULTISEED_ROOT="${MULTISEED_ROOT:-${SOURCE_RUN}_multiseed}"
SEEDS="${SEEDS:-4801,4802,4803}"
CALIBRATION_FRACTION="${CALIBRATION_FRACTION:-0.50}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"

[[ -f "$SOURCE_RUN/TRAINING_COMPLETE.json" ]] || {
  echo "SOURCE_RUN lacks TRAINING_COMPLETE.json: $SOURCE_RUN" >&2
  exit 2
}
python tools/audit_v48_7_completion.py \
  --root "$SOURCE_RUN" \
  --output "$SOURCE_RUN/source_completion_audit_v48_7.json"

mkdir -p "$MULTISEED_ROOT"
python - "$SOURCE_RUN/TRAINING_COMPLETE.json" "$MULTISEED_ROOT/source_checkpoint_manifest.json" <<'PY'
import json, pathlib, sys
src=json.load(open(sys.argv[1])); pathlib.Path(sys.argv[2]).write_text(json.dumps(src,indent=2)+'\n')
PY

IFS=',' read -r -a seed_array <<< "$SEEDS"
for raw_seed in "${seed_array[@]}"; do
  seed="${raw_seed//[[:space:]]/}"
  [[ -n "$seed" ]] || continue
  out="$MULTISEED_ROOT/seed_$seed"
  split="$out/dataset_splits"
  mkdir -p "$split" "$out/logs"
  for regime in safe near_contact contact; do
    python tools/split_calibration_by_scene_v48.py \
      --input "$EVAL_OCRAP_ROOT/val_$regime" \
      --calibration-output "$split/calibration_$regime" \
      --validation-output "$split/development_$regime" \
      --calibration-fraction "$CALIBRATION_FRACTION" \
      --seed "$seed" --link-mode hardlink --overwrite \
      >"$out/logs/split_${regime}.log" 2>&1
  done

  calibrate_variant() {
    local variant="$1" gpu="$2"
    local checkpoint="$SOURCE_RUN/candidates/$variant/model_v48_trac_sr/best.pt"
    local variant_out="$out/candidates/$variant"
    [[ -f "$checkpoint" ]] || return 0
    mkdir -p "$variant_out/calibration" "$variant_out/logs"
    set +e
    CUDA_VISIBLE_DEVICES="$gpu" python -u tools/calibrate_policy_risk_v48.py \
      --dataset "$split/calibration_near_contact" --checkpoint "$checkpoint" \
      --bucket near --risk-source "${RISK_SOURCE:-direct_delta}" --conformal-alpha="${CONFORMAL_ALPHA:-0.10}" --conformal-temperature="${CONFORMAL_TEMPERATURE:-0.02}" --conformal-scope="${CONFORMAL_SCOPE:-policy_top1}" \
      --output "$variant_out/calibration/direct_value_risk_near_v48.json" \
      --rows-output "$variant_out/calibration/direct_value_risk_near_v48.rows.jsonl" \
      --required-min-groups="${NEAR_MIN_GROUPS:-80}" \
      --required-min-scenes="${NEAR_MIN_SCENES:-40}" \
      --min-fit-selected="${NEAR_MIN_FIT_SELECTED:-8}" \
      --min-verify-selected="${NEAR_MIN_VERIFY_SELECTED:-6}" \
      --max-fit-harmful-group-ucb="${NEAR_MAX_FIT_HARM_UCB:-0.16}" \
      --max-verify-harmful-group-ucb="${NEAR_MAX_VERIFY_HARM_UCB:-0.18}" \
      --max-selected-macro-share="${MAX_SELECTED_MACRO_SHARE:-0.85}" \
      >"$variant_out/logs/calibrate_near.log" 2>&1
    echo $? >"$variant_out/calibration/near.exit_code"

    CUDA_VISIBLE_DEVICES="$gpu" python -u tools/calibrate_policy_risk_v48.py \
      --dataset "$split/calibration_contact" --checkpoint "$checkpoint" \
      --bucket contact --risk-source "${RISK_SOURCE:-direct_delta}" --conformal-alpha="${CONFORMAL_ALPHA:-0.10}" --conformal-temperature="${CONFORMAL_TEMPERATURE:-0.02}" --conformal-scope="${CONFORMAL_SCOPE:-policy_top1}" \
      --output "$variant_out/calibration/direct_value_risk_contact_v48.json" \
      --rows-output "$variant_out/calibration/direct_value_risk_contact_v48.rows.jsonl" \
      --required-min-groups="${CONTACT_MIN_GROUPS:-100}" \
      --required-min-scenes="${CONTACT_MIN_SCENES:-45}" \
      --min-fit-selected="${CONTACT_MIN_FIT_SELECTED:-9}" \
      --min-verify-selected="${CONTACT_MIN_VERIFY_SELECTED:-6}" \
      --max-fit-harmful-group-ucb="${CONTACT_MAX_FIT_HARM_UCB:-0.18}" \
      --max-verify-harmful-group-ucb="${CONTACT_MAX_VERIFY_HARM_UCB:-0.20}" \
      --max-selected-macro-share="${MAX_SELECTED_MACRO_SHARE:-0.85}" \
      >"$variant_out/logs/calibrate_contact.log" 2>&1
    echo $? >"$variant_out/calibration/contact.exit_code"
    set -e
  }

  calibrate_variant balanced "$GPU0" & p0=$!
  calibrate_variant precision "$GPU1" & p1=$!
  wait "$p0" || true
  wait "$p1" || true
  echo "seed $seed complete"
done

# Prove that every calibration used immutable source bytes.
python - "$SOURCE_RUN/TRAINING_COMPLETE.json" <<'PY'
import hashlib,json,pathlib,sys
manifest=json.load(open(sys.argv[1])); bad=[]
for name,item in (manifest.get('variants') or {}).items():
    path=pathlib.Path(item['checkpoint'])
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != item['sha256']:
        bad.append(name)
if bad:
    raise SystemExit('source checkpoint changed during multi-seed calibration: '+','.join(bad))
PY

python tools/summarize_v48_7_multiseed.py \
  --root "$MULTISEED_ROOT" \
  --output "$MULTISEED_ROOT/multiseed_summary_v48_9.json"
printf '{"complete":true,"seeds":"%s","risk_source":"%s"}\n' "$SEEDS" "${RISK_SOURCE:-direct_delta}" \
  > "$MULTISEED_ROOT/MULTISEED_COMPLETE.json"
