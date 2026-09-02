#!/usr/bin/env bash
# V48.84 OC-SAOP: Stage-I/root action-observability adjudication after V48.83 STOP.
set -Eeuo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"; cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}"; export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"; export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
BASE_OUT="${BASE_OUT:-/home/senzeyu2/code/OC-RAP/runs}"; GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"
export OCRAP_ROOT="${OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}"; PROTOCOL_ROOT="${PROTOCOL_ROOT:-$OCRAP_ROOT/calibration_v48_14_prism_4814}"
REFERENCE_A="${V4884_REFERENCE_A:-$BASE_OUT/ocrap_v48_56_dcp_drfc_bcde_drac_ablation_A}"
V83_COMPARE="${V4884_V83_COMPARE:-$BASE_OUT/OC-RAP-v48.83-DCP-DRFC-BCDE-RIFA-OC-CRTF-comparison.json}"
RUNTIME="$BASE_OUT/OC-RAP-v48.84-runtime-code-contract.json"; COMPARE="$BASE_OUT/OC-RAP-v48.84-DCP-DRFC-BCDE-RIFA-OC-SAOP-comparison.json"; COMPLETE="$BASE_OUT/OC-RAP-v48.84-PIPELINE_COMPLETE.json"
BOUT="$BASE_OUT/OC-RAP-v48.84-SAOP-balanced.json"; POUT="$BASE_OUT/OC-RAP-v48.84-SAOP-precision.json"; CERT_INDEX="$BASE_OUT/OC-RAP-v48.84-certificate-teacher-pcd-index.jsonl"; CERT_SUM="$BASE_OUT/OC-RAP-v48.84-certificate-teacher-pcd-index-summary.json"; CACHE="${V4884_TENSOR_CACHE_DIR:-$BASE_OUT/.ocrap_v48_84_probe_tensor_cache}"
rm -f "$RUNTIME" "$COMPARE" "$COMPLETE" "$BOUT" "$POUT" "$CERT_INDEX" "$CERT_SUM"; mkdir -p "$CACHE"
python tools/check_v48_84_runtime_code_contract.py --repo "$REPO" --output "$RUNTIME"
python - "$V83_COMPARE" <<'PY'
import json,pathlib,sys
p=json.loads(pathlib.Path(sys.argv[1]).read_text());d=p.get('preregistered_decision') or {}
if not(p.get('valid') and p.get('attribution_ready') and d.get('status')=='COUNTERFACTUAL_RECOVERY_TAIL_FIELD_STOP'):
 raise SystemExit('V48.83 STOP prerequisite missing')
PY
TRAIN_NEAR="$PROTOCOL_ROOT/evidence_adapt_train_near_contact"; TRAIN_CONTACT="$PROTOCOL_ROOT/evidence_adapt_train_contact"; DEV_NEAR="$PROTOCOL_ROOT/evidence_adapt_dev_near_contact"; DEV_CONTACT="$PROTOCOL_ROOT/evidence_adapt_dev_contact"; CERT_NEAR="$PROTOCOL_ROOT/certificate_pool_near_contact"; CERT_CONTACT="$PROTOCOL_ROOT/certificate_pool_contact"
TRAIN_INDEX="$REFERENCE_A/evidence_adapt_teacher_pcd_index.jsonl"; DEV_INDEX="$REFERENCE_A/evidence_adapt_dev_teacher_pcd_index.jsonl"
python tools/build_teacher_pcd_index_v48.py --dataset "$CERT_NEAR,$CERT_CONTACT" --output "$CERT_INDEX" --summary-output "$CERT_SUM" --alpha 0.2 --beta 0.2 --top-m 8 --option-execution-semantics observation_class --positive-gain 0.015 --quality-mode off --workers "${V4884_LABEL_WORKERS:-8}"
run_probe(){ local v="$1" gpu="$2" out="$3"; CUDA_VISIBLE_DEVICES="$gpu" python tools/run_v48_84_stage_i_action_observability_probe.py --checkpoint "$REFERENCE_A/candidates/$v/model_v48_trac_sr/best.pt" --train-dataset "$TRAIN_NEAR,$TRAIN_CONTACT" --dev-dataset "$DEV_NEAR,$DEV_CONTACT" --certificate-dataset "$CERT_NEAR,$CERT_CONTACT" --train-index "$TRAIN_INDEX" --dev-index "$DEV_INDEX" --certificate-index "$CERT_INDEX" --cache-dir "$CACHE/$v" --device cuda --variant "$v" --output "$out"; }
set +e; run_probe balanced "$GPU0" "$BOUT" & p0=$!; run_probe precision "$GPU1" "$POUT" & p1=$!; wait "$p0"; r0=$?; wait "$p1"; r1=$?; set -e
[[ $r0 == 0 && $r1 == 0 ]] || exit 30
python tools/compare_v48_84_saop.py --balanced "$BOUT" --precision "$POUT" --output "$COMPARE"
python tools/check_v48_84_pipeline_complete.py --runtime "$RUNTIME" --balanced "$BOUT" --precision "$POUT" --comparison "$COMPARE" --output "$COMPLETE"
cd "$BASE_OUT"; zip -qj OC-RAP-v48.84-OC-SAOP-audits.zip "$RUNTIME" "$BOUT" "$POUT" "$CERT_SUM" "$COMPARE" "$COMPLETE"
echo "V48.84 complete: upload OC-RAP-v48.84-OC-SAOP-audits.zip"
