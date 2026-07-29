#!/usr/bin/env bash
set -euo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"; cd "$REPO"; export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
SOURCE_RUN="${SOURCE_RUN:?Set SOURCE_RUN to a completed v48.5 training run}"
EVAL_OCRAP_ROOT="${EVAL_OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}"
MULTISEED_ROOT="${MULTISEED_ROOT:-${SOURCE_RUN}_multiseed}"
SEEDS="${SEEDS:-4801,4802,4803}"; CALIBRATION_FRACTION="${CALIBRATION_FRACTION:-0.50}"; GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"
[[ -f "$SOURCE_RUN/TRAINING_COMPLETE.json" ]] || { echo "SOURCE_RUN lacks TRAINING_COMPLETE.json" >&2; exit 2; }
python tools/audit_v48_5_completion.py --root "$SOURCE_RUN" --output "$SOURCE_RUN/source_completion_audit.json"
mkdir -p "$MULTISEED_ROOT"
python - "$SOURCE_RUN/TRAINING_COMPLETE.json" "$MULTISEED_ROOT/source_checkpoint_manifest.json" <<'PY'
import json,sys,pathlib
src=json.load(open(sys.argv[1])); pathlib.Path(sys.argv[2]).write_text(json.dumps(src,indent=2)+'\n')
PY
IFS=',' read -r -a arr <<< "$SEEDS"
for raw in "${arr[@]}"; do
 seed="${raw//[[:space:]]/}"; [[ -n "$seed" ]] || continue; out="$MULTISEED_ROOT/seed_$seed"; split="$out/dataset_splits"; mkdir -p "$split" "$out/logs"
 for regime in safe near_contact contact; do
  python tools/split_calibration_by_scene_v48.py --input "$EVAL_OCRAP_ROOT/val_$regime" --calibration-output "$split/calibration_$regime" --validation-output "$split/development_$regime" --calibration-fraction "$CALIBRATION_FRACTION" --seed "$seed" --link-mode hardlink --overwrite >"$out/logs/split_${regime}.log" 2>&1
 done
 cal(){ local v="$1" gpu="$2"; local ck="$SOURCE_RUN/candidates/$v/model_v48_trac_sr/best.pt" vo="$out/candidates/$v"; [[ -f "$ck" ]] || return 0; mkdir -p "$vo/calibration" "$vo/logs"; set +e
  CUDA_VISIBLE_DEVICES="$gpu" python -u tools/calibrate_policy_risk_v48.py --dataset "$split/calibration_near_contact" --checkpoint "$ck" --bucket near --risk-source delta_distribution --output "$vo/calibration/direct_value_risk_near_v48.json" --rows-output "$vo/calibration/direct_value_risk_near_v48.rows.jsonl" --required-min-groups="${NEAR_MIN_GROUPS:-80}" --required-min-scenes="${NEAR_MIN_SCENES:-40}" --min-fit-selected="${NEAR_MIN_FIT_SELECTED:-8}" --min-verify-selected="${NEAR_MIN_VERIFY_SELECTED:-6}" --max-fit-harmful-group-ucb="${NEAR_MAX_FIT_HARM_UCB:-0.16}" --max-verify-harmful-group-ucb="${NEAR_MAX_VERIFY_HARM_UCB:-0.18}" >"$vo/logs/calibrate_near.log" 2>&1; echo $? >"$vo/calibration/near.exit_code"
  CUDA_VISIBLE_DEVICES="$gpu" python -u tools/calibrate_policy_risk_v48.py --dataset "$split/calibration_contact" --checkpoint "$ck" --bucket contact --risk-source delta_distribution --output "$vo/calibration/direct_value_risk_contact_v48.json" --rows-output "$vo/calibration/direct_value_risk_contact_v48.rows.jsonl" --required-min-groups="${CONTACT_MIN_GROUPS:-100}" --required-min-scenes="${CONTACT_MIN_SCENES:-45}" --min-fit-selected="${CONTACT_MIN_FIT_SELECTED:-9}" --min-verify-selected="${CONTACT_MIN_VERIFY_SELECTED:-6}" --max-fit-harmful-group-ucb="${CONTACT_MAX_FIT_HARM_UCB:-0.18}" --max-verify-harmful-group-ucb="${CONTACT_MAX_VERIFY_HARM_UCB:-0.20}" >"$vo/logs/calibrate_contact.log" 2>&1; echo $? >"$vo/calibration/contact.exit_code"; set -e
 }
 cal balanced "$GPU0" & p0=$!; cal precision "$GPU1" & p1=$!; wait "$p0" || true; wait "$p1" || true
 echo "seed $seed complete"
done
# Verify source checkpoint bytes did not change while calibration ran.
python - "$SOURCE_RUN/TRAINING_COMPLETE.json" <<'PY'
import hashlib,json,pathlib,sys
m=json.load(open(sys.argv[1])); bad=[]
for n,x in (m.get('variants') or {}).items():
 p=pathlib.Path(x['checkpoint']); h=hashlib.sha256(p.read_bytes()).hexdigest()
 if h!=x['sha256']: bad.append(n)
if bad: raise SystemExit('source checkpoint changed during multi-seed: '+','.join(bad))
PY
python tools/summarize_v48_4_multiseed.py --root "$MULTISEED_ROOT" --output "$MULTISEED_ROOT/multiseed_summary.json"
printf '{"complete":true,"seeds":"%s"}\n' "$SEEDS" > "$MULTISEED_ROOT/MULTISEED_COMPLETE.json"
