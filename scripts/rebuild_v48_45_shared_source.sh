#!/usr/bin/env bash
set -Eeuo pipefail

# Rebuild a fixed v48.45 source after historical warm-start checkpoints were lost.
# This is a two-stage shared-source construction:
#   S0) learn one recovery backbone/witness from pooled Safe/Near/Contact train data;
#   S1) freeze S0 and fit the balanced/precision direct proposal/evidence heads on
#       Near/Contact candidate groups.  Regime strata are data/audit strata; no new
#       v48.45 SOWR router or regime-specific admission rule is introduced here.
# The resulting two checkpoints are immutable common inputs to A/B/C/D.
# No calibration/certificate/test roots are read here.

REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"

OCRAP_ROOT="${OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}"
SOURCE_OUT="${SOURCE_OUT:-$REPO/runs/ocrap_v48_45_source_rebuild_s7}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
SEED="${SOURCE_SEED:-7}"
TRAIN_SAFE="${SOURCE_TRAIN_SAFE:-$OCRAP_ROOT/train_safe}"
TRAIN_NEAR="${SOURCE_TRAIN_NEAR:-$OCRAP_ROOT/train_near_contact}"
TRAIN_CONTACT="${SOURCE_TRAIN_CONTACT:-$OCRAP_ROOT/train_contact}"
DEV_SAFE="${SOURCE_DEV_SAFE:-$OCRAP_ROOT/val_safe}"
DEV_NEAR="${SOURCE_DEV_NEAR:-$OCRAP_ROOT/val_near_contact}"
DEV_CONTACT="${SOURCE_DEV_CONTACT:-$OCRAP_ROOT/val_contact}"
# S0 learns recovery semantics from all three regimes. S1 intentionally fits only
# the Near/Contact proposal/evidence heads, matching the current recovery-candidate
# policy scope while keeping the S0 witness shared and fixed for every ablation arm.
BACKBONE_TRAIN_MIX="$TRAIN_SAFE,$TRAIN_NEAR,$TRAIN_CONTACT"
BACKBONE_VAL_MIX="$DEV_SAFE,$DEV_NEAR,$DEV_CONTACT"
POLICY_TRAIN_MIX="$TRAIN_NEAR,$TRAIN_CONTACT"
POLICY_VAL_MIX="$DEV_NEAR,$DEV_CONTACT"
BACKBONE_GROUP_INDEX="$SOURCE_OUT/teacher_pcd_backbone_train_index.jsonl"
BACKBONE_GROUP_SUMMARY="$SOURCE_OUT/teacher_pcd_backbone_train_index_summary.json"
BACKBONE_VAL_GROUP_INDEX="$SOURCE_OUT/teacher_pcd_backbone_dev_index.jsonl"
BACKBONE_VAL_GROUP_SUMMARY="$SOURCE_OUT/teacher_pcd_backbone_dev_index_summary.json"
POLICY_GROUP_INDEX="$SOURCE_OUT/teacher_pcd_policy_train_index.jsonl"
POLICY_GROUP_SUMMARY="$SOURCE_OUT/teacher_pcd_policy_train_index_summary.json"
POLICY_VAL_GROUP_INDEX="$SOURCE_OUT/teacher_pcd_policy_dev_index.jsonl"
POLICY_VAL_GROUP_SUMMARY="$SOURCE_OUT/teacher_pcd_policy_dev_index_summary.json"
BACKBONE_RUN="$SOURCE_OUT/shared_recovery_backbone"
BACKBONE_CKPT="$BACKBONE_RUN/model_v48_trac_sr/best.pt"
BACKBONE_DONE="$BACKBONE_RUN/TRAINING_COMPLETE.json"

if [[ "${ALLOW_SOURCE_REBUILD_OVERWRITE:-0}" != 1 && -e "$SOURCE_OUT/SOURCE_REBUILD_COMPLETE.json" ]]; then
  echo "SOURCE_OUT already contains a completed source: $SOURCE_OUT" >&2
  echo "Use that immutable source, or set ALLOW_SOURCE_REBUILD_OVERWRITE=1 for an intentional new source identity." >&2
  exit 73
fi
if [[ "${ALLOW_SOURCE_REBUILD_OVERWRITE:-0}" == 1 ]]; then
  rm -rf "$SOURCE_OUT"
fi
mkdir -p "$SOURCE_OUT/logs"
# A resumed unsealed source may contain a marker from the previous failed attempt.
# Remove only status markers; never remove a completed S0 backbone here.
rm -f "$SOURCE_OUT/SOURCE_REBUILD_FAILED.json" "$SOURCE_OUT/S1_SOURCE_POLICY_STATUS.json"
SOURCE_REBUILD_STAGE="preflight"
source_rebuild_failure_marker() {
  local rc=$?
  if [[ $rc -ne 0 && ! -f "$SOURCE_OUT/SOURCE_REBUILD_COMPLETE.json" ]]; then
    SOURCE_REBUILD_FAILURE_RC="$rc" SOURCE_REBUILD_FAILURE_STAGE="$SOURCE_REBUILD_STAGE" SOURCE_REBUILD_FAILURE_OUT="$SOURCE_OUT" python - <<'PYFAIL' || true
import json, os, pathlib, time
out = pathlib.Path(os.environ["SOURCE_REBUILD_FAILURE_OUT"]) / "SOURCE_REBUILD_FAILED.json"
doc = {
    "event": "v48_45_source_rebuild_failed",
    "implementation_version": "v48.45.4-s1-nounset-hotfix",
    "created_unix": time.time(),
    "stage": os.environ.get("SOURCE_REBUILD_FAILURE_STAGE", "unknown"),
    "raw_exit_code": int(os.environ.get("SOURCE_REBUILD_FAILURE_RC", "1")),
    "source_rebuild_complete": False,
    "test_roots_read": False,
}
tmp = out.with_name(f".{out.name}.tmp")
tmp.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
os.replace(tmp, out)
PYFAIL
  fi
  return "$rc"
}
trap source_rebuild_failure_marker EXIT

for d in "$TRAIN_SAFE" "$TRAIN_NEAR" "$TRAIN_CONTACT" "$DEV_SAFE" "$DEV_NEAR" "$DEV_CONTACT"; do
  [[ -d "$d" ]] || { echo "missing source dataset root: $d" >&2; exit 2; }
done

# Fail closed on train/development leakage. Calibration/certificate/test roots are
# deliberately absent from this script.
python tools/check_scene_overlap_v48.py \
  --train-root "$TRAIN_SAFE" --train-root "$TRAIN_NEAR" --train-root "$TRAIN_CONTACT" \
  --development-root "$DEV_SAFE" --development-root "$DEV_NEAR" --development-root "$DEV_CONTACT" \
  --output "$SOURCE_OUT/source_train_dev_overlap_audit.json" \
  --fail-on-train-development-overlap \
  2>&1 | tee "$SOURCE_OUT/logs/source_train_dev_overlap_audit.log"

# Build exact teacher-PCD group indexes once and freeze them for the source run.
SOURCE_REBUILD_STAGE="teacher_index_build"
build_index() {
  local dataset="$1" index="$2" summary="$3" log="$4"
  if [[ ! -s "$index" || ! -s "$summary" || "${REBUILD_SOURCE_INDEX:-0}" == 1 ]]; then
    python tools/build_teacher_pcd_index_v48.py \
      --dataset "$dataset" --output "$index" --summary-output "$summary" \
      --alpha="${ALPHA:-0.2}" --beta="${BETA:-0.2}" --top-m="${TOP_M:-8}" \
      --positive-gain="${POSITIVE_GAIN:-0.015}" --deployable-macro-ids="${DEPLOYABLE_MACRO_IDS:-2,3,5,6,7}" \
      --quality-mode=warn \
      2>&1 | tee "$log"
  fi
}
build_index "$BACKBONE_TRAIN_MIX" "$BACKBONE_GROUP_INDEX" "$BACKBONE_GROUP_SUMMARY" "$SOURCE_OUT/logs/build_backbone_train_index.log"
build_index "$BACKBONE_VAL_MIX" "$BACKBONE_VAL_GROUP_INDEX" "$BACKBONE_VAL_GROUP_SUMMARY" "$SOURCE_OUT/logs/build_backbone_dev_index.log"
build_index "$POLICY_TRAIN_MIX" "$POLICY_GROUP_INDEX" "$POLICY_GROUP_SUMMARY" "$SOURCE_OUT/logs/build_policy_train_index.log"
build_index "$POLICY_VAL_MIX" "$POLICY_VAL_GROUP_INDEX" "$POLICY_VAL_GROUP_SUMMARY" "$SOURCE_OUT/logs/build_policy_dev_index.log"

# Shared architecture contract.  These shapes are required later by the v48.45
# OCAF/SOWR strict-init contract.  No regime label/router is introduced by this
# rebuild; the legacy pair of delta adapters is retained only because the current
# downstream checkpoint geometry requires it.
common_arch=(
  SET_CONTEXT_ENABLED=false
  PREFERENCE_HEAD_ENABLED=false
  PREFERENCE_CONTEXT_ENABLED=false
  RELATIVE_INCLUDE_ABSOLUTE=false
  SET_TOURNAMENT_ENABLED=true
  SET_TOURNAMENT_HIDDEN=48
  SET_TOURNAMENT_HEADS=4
  SET_TOURNAMENT_DROPOUT=0.05
  SET_TOURNAMENT_REPLACE_BASE=true
  DELTA_HEAD_ENABLED=true
  DELTA_MODE=ordinal_evidence
  DELTA_HIDDEN=48
  DELTA_DROPOUT=0.02
  DELTA_REGIME_EXPERTS=true
  DELTA_POLICY_FEATURES=true
  EVIDENCE_CALIBRATOR_ENABLED=false
  SKIP_POST_TRAIN_CALIBRATION=1
  DETERMINISTIC_ALGORITHMS=true
  CUDNN_BENCHMARK=false
)

# S0: train the paper-matched recovery witness/backbone from scratch once.
# Direct proposal/evidence loss is disabled, so both source variants inherit the
# exact same root/observation/margin representation.
if [[ -f "$BACKBONE_CKPT" && ! -f "$BACKBONE_DONE" ]]; then
  echo "incomplete S0 detected (best.pt exists without TRAINING_COMPLETE.json); removing partial backbone" >&2
  rm -rf "$BACKBONE_RUN"
fi
if [[ ! -f "$BACKBONE_CKPT" || ! -f "$BACKBONE_DONE" ]]; then
  SOURCE_REBUILD_STAGE="S0_shared_recovery_backbone"
  echo "[source rebuild] S0 shared recovery backbone on GPU $GPU0" | tee "$SOURCE_OUT/logs/source_rebuild_status.log"
  env "${common_arch[@]}" \
    RUN="$BACKBONE_RUN" MODEL_DIR="$BACKBONE_RUN/model_v48_trac_sr" CAL_DIR="$BACKBONE_RUN/calibration" \
    ALLOW_SCRATCH_INIT=1 INIT_CKPT= VARIANT=balanced TRAIN_GPU="$GPU0" SEED="$SEED" \
    TRAIN_MIX="$BACKBONE_TRAIN_MIX" VAL_MIX="$BACKBONE_VAL_MIX" CAL_MIX="$BACKBONE_VAL_MIX" \
    GROUP_INDEX="$BACKBONE_GROUP_INDEX" VAL_GROUP_INDEX="$BACKBONE_VAL_GROUP_INDEX" \
    DIRECT_ONLY_FAST_PATH=false DIRECT_VALUE_WEIGHT=0 \
    LOSS_ASSIGN=1.0 LOSS_MARGIN=2.0 LOSS_SIG=0.5 LOSS_OBS=1.0 \
    LOSS_DEP=0.5 LOSS_ORC=0.5 LOSS_ANTI_ORACLE=1.0 LOSS_ARTIFACT_GAP=0.5 \
    LOSS_ADMISSION=0.2 LOSS_OPTION_Q=0.5 LOSS_OPTION_ADMISSION=0.4 \
    LOSS_OPTION_SUCCESS=0.5 LOSS_OPTION_SUCCESS_BCE=0.5 LOSS_OPTION_BEST=0.2 LOSS_UTILITY=0.2 \
    EPOCHS="${SOURCE_BACKBONE_EPOCHS:-24}" PATIENCE="${SOURCE_BACKBONE_PATIENCE:-6}" \
    LR="${SOURCE_BACKBONE_LR:-0.00015}" ENCODER_LR_SCALE=1.0 ENCODER_ANCHOR_WEIGHT=0 \
    BEST_METRIC=loss BEST_METRIC_MIN_DELTA=0.000001 \
    GROUP_BATCH_STRATIFIED=true GROUP_BATCHING_REPLACEMENT=true \
    NUM_WORKERS="${SOURCE_NUM_WORKERS:-4}" PREFETCH_FACTOR="${SOURCE_PREFETCH_FACTOR:-2}" BATCH_SIZE="${SOURCE_BACKBONE_BATCH_SIZE:-72}" \
    bash scripts/train_ocrap_v48_trac_sr.sh \
    >"$SOURCE_OUT/logs/train_shared_recovery_backbone.log" 2>&1
fi
[[ -f "$BACKBONE_CKPT" && -f "$BACKBONE_DONE" ]] || { echo "shared recovery backbone completion artifacts missing: $BACKBONE_CKPT / $BACKBONE_DONE" >&2; exit 30; }

train_source_variant() {
  # Do not reference a variable in the same `local` command that initializes it.
  # Under `set -u`, Bash expands all RHS expressions before these local assignments
  # become visible, so `local variant="$1" run=".../$variant"` aborts with an
  # unbound-variable error before S1 can even create its candidate directory.
  local variant gpu run
  variant="$1"
  gpu="$2"
  run="$SOURCE_OUT/candidates/$variant"
  rm -rf "$run"
  mkdir -p "$run/logs"
  echo "[source rebuild] S1 $variant source policy/evidence on GPU $gpu" | tee -a "$SOURCE_OUT/logs/source_rebuild_status.log"
  env "${common_arch[@]}" \
    RUN="$run" MODEL_DIR="$run/model_v48_trac_sr" CAL_DIR="$run/calibration" \
    INIT_CKPT="$BACKBONE_CKPT" VARIANT="$variant" TRAIN_GPU="$gpu" SEED="$SEED" \
    TRAIN_MIX="$POLICY_TRAIN_MIX" VAL_MIX="$POLICY_VAL_MIX" CAL_MIX="$POLICY_VAL_MIX" \
    GROUP_INDEX="$POLICY_GROUP_INDEX" VAL_GROUP_INDEX="$POLICY_VAL_GROUP_INDEX" \
    DIRECT_ONLY_FAST_PATH=true \
    TRAINABLE_PARAM_PREFIXES='direct_value_heads,direct_preference_set_ranker,direct_delta_adapters' \
    STRICT_INIT_PREFIXES='direct_preference_set_ranker,direct_delta_adapters' \
    ENCODER_ANCHOR_WEIGHT=0 DIRECT_VALUE_WEIGHT="${SOURCE_DIRECT_VALUE_WEIGHT:-10.0}" \
    PREFERENCE_WEIGHT=0 PREFERENCE_REGRET_WEIGHT=0 PREFERENCE_LISTWISE_WEIGHT=0 PREFERENCE_GAP_WEIGHT=0 \
    PREFERENCE_SET_WEIGHT=0 PREFERENCE_ALL_GROUP_SET_WEIGHT=0 PREFERENCE_CONDITIONAL_SET_WEIGHT=0 \
    PREFERENCE_PROPOSAL_TOPK_WEIGHT="${SOURCE_PROPOSAL_TOPK_WEIGHT:-1.0}" \
    PREFERENCE_PROPOSAL_TOPK="${SOURCE_PROPOSAL_TOP_K:-5}" PREFERENCE_PROPOSAL_MARGIN="${SOURCE_PROPOSAL_MARGIN:-0.02}" \
    DELTA_NLL_WEIGHT=0 DELTA_SIGN_WEIGHT=0 \
    ORDINAL_EVIDENCE_ORDERED_NLL_ALL_WEIGHT="${SOURCE_ORDINAL_ALL_WEIGHT:-0.20}" \
    ORDINAL_EVIDENCE_PROPOSAL_TOPK_WEIGHT="${SOURCE_ORDINAL_PROPOSAL_WEIGHT:-0.50}" \
    ORDINAL_EVIDENCE_PROPOSAL_TOPK="${SOURCE_PROPOSAL_TOP_K:-5}" ORDINAL_EVIDENCE_PROPOSAL_RANK_DECAY=0.85 \
    ORDINAL_EVIDENCE_INTRAGROUP_BENEFIT_WEIGHT="${SOURCE_INTRAGROUP_BENEFIT_WEIGHT:-0.15}" \
    ORDINAL_EVIDENCE_INTRAGROUP_HARM_WEIGHT="${SOURCE_INTRAGROUP_HARM_WEIGHT:-0.20}" \
    ORDINAL_EVIDENCE_HARM_CLASS_WEIGHT=1.5 ORDINAL_EVIDENCE_DEAD_CLASS_WEIGHT=0.5 ORDINAL_EVIDENCE_BENEFIT_CLASS_WEIGHT=1.5 \
    POLICY_METRIC_RISK_SOURCE=ordinal_evidence POLICY_METRIC_PROPOSAL_TOP_K="${SOURCE_PROPOSAL_TOP_K:-5}" \
    POLICY_METRIC_EVIDENCE_RERANK_TOP_K=true \
    EPOCHS="${SOURCE_POLICY_EPOCHS:-16}" PATIENCE="${SOURCE_POLICY_PATIENCE:-4}" \
    BEST_METRIC=direct_policy_risk_fold_worst BEST_METRIC_MIN_DELTA=0.000001 \
    GROUP_BATCH_STRATIFIED=true GROUP_BATCHING_REPLACEMENT=true \
    NUM_WORKERS="${SOURCE_NUM_WORKERS:-4}" PREFETCH_FACTOR="${SOURCE_PREFETCH_FACTOR:-2}" BATCH_SIZE="${SOURCE_POLICY_BATCH_SIZE:-72}" \
    bash scripts/train_ocrap_v48_trac_sr.sh \
    >"$run/logs/train_source_policy.log" 2>&1

  [[ -f "$run/model_v48_trac_sr/best.pt" ]] || { echo "$variant source checkpoint missing" >&2; return 30; }
  cat > "$run/POLICY_CONTRACT.env" <<POLICY
RISK_SOURCE=ordinal_evidence
CONDITIONAL_RECOVERY_RANKING=true
POLICY_FIRST_NO_FALLBACK=false
PROPOSAL_TOP_K=${SOURCE_PROPOSAL_TOP_K:-5}
EVIDENCE_RERANK_TOP_K=true
MACRO_CONSTRAINT_MODE=opportunity_normalized
MAX_MACRO_EXCESS_SHARE=0.15
SELECTION_SEMANTICS=rank_topk_then_filter_then_evidence_rerank
SOURCE_REBUILD_ID=v48.45-shared-backbone-source-rebuild
POLICY
}

SOURCE_REBUILD_STAGE="S1_source_policy_heads"
train_source_variant balanced "$GPU0" & p0=$!
train_source_variant precision "$GPU1" & p1=$!
set +e
wait "$p0"; s0=$?
wait "$p1"; s1=$?
set -e
printf 'source policy status: balanced=%s precision=%s\n' "$s0" "$s1" | tee -a "$SOURCE_OUT/logs/source_rebuild_status.log"
python - "$SOURCE_OUT" "$s0" "$s1" <<'PY_S1_STATUS'
import json,pathlib,sys,time
root=pathlib.Path(sys.argv[1])
doc={
  'event':'v48_45_source_policy_stage_status',
  'implementation_version':'v48.45.4-s1-nounset-hotfix',
  'created_unix':time.time(),
  'balanced_exit_code':int(sys.argv[2]),
  'precision_exit_code':int(sys.argv[3]),
  'both_succeeded':sys.argv[2]=='0' and sys.argv[3]=='0',
  'test_roots_read':False,
}
p=root/'S1_SOURCE_POLICY_STATUS.json'; tmp=p.with_name('.'+p.name+'.tmp')
tmp.write_text(json.dumps(doc,indent=2)+'\n',encoding='utf-8'); tmp.replace(p)
PY_S1_STATUS
[[ "$s0" == 0 && "$s1" == 0 ]] || exit 30

SOURCE_REBUILD_STAGE="seal_source_manifest"
python - "$SOURCE_OUT" "$BACKBONE_CKPT" "$BACKBONE_TRAIN_MIX" "$BACKBONE_VAL_MIX" "$POLICY_TRAIN_MIX" "$POLICY_VAL_MIX" "$BACKBONE_GROUP_INDEX" "$BACKBONE_VAL_GROUP_INDEX" "$POLICY_GROUP_INDEX" "$POLICY_VAL_GROUP_INDEX" "$SEED" <<'PY'
import hashlib,json,os,pathlib,sys,time
root=pathlib.Path(sys.argv[1]); backbone=pathlib.Path(sys.argv[2])
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
variants={}
for name in ('balanced','precision'):
    p=root/'candidates'/name/'model_v48_trac_sr'/'best.pt'
    if not p.is_file(): raise SystemExit(f'missing source checkpoint {p}')
    variants[name]={'checkpoint':str(p.resolve()),'sha256':sha(p),'size_bytes':p.stat().st_size}
doc={
 'event':'v48_45_source_rebuild_complete','implementation_version':'v48.45.4-s1-nounset-hotfix','created_unix':time.time(),
 'source_identity':'v48.45-shared-backbone-source-rebuild',
 'historical_v48_13_checkpoint_recovered':False,
 'attribution_scope':'within_this_source_rebuilt_A_B_C_D_round',
 'backbone':{'checkpoint':str(backbone.resolve()),'sha256':sha(backbone),'size_bytes':backbone.stat().st_size},
 'variants':variants,
 'backbone_train_mix':sys.argv[3],'backbone_development_mix':sys.argv[4],
 'policy_train_mix':sys.argv[5],'policy_development_mix':sys.argv[6],
 'backbone_train_group_index':str(pathlib.Path(sys.argv[7]).resolve()),
 'backbone_development_group_index':str(pathlib.Path(sys.argv[8]).resolve()),
 'policy_train_group_index':str(pathlib.Path(sys.argv[9]).resolve()),
 'policy_development_group_index':str(pathlib.Path(sys.argv[10]).resolve()),
 'seed':int(sys.argv[11]),
 'calibration_roots_read':False,'certificate_roots_read':False,'test_roots_read':False,
 'ablation_invariant':'all v48.45 A/B/C/D arms must consume these exact source hashes',
}
out=root/'SOURCE_REBUILD_COMPLETE.json'; tmp=out.with_name('.'+out.name+'.tmp')
tmp.write_text(json.dumps(doc,indent=2)+'\n',encoding='utf-8'); os.replace(tmp,out)
print(json.dumps(doc,indent=2))
PY

SOURCE_REBUILD_STAGE="final_source_contracts"
python tools/check_v48_36_source_checkpoint_contract.py \
  --source-run "$SOURCE_OUT" --output "$SOURCE_OUT/SOURCE_CHECKPOINT_CONTRACT.json" \
  2>&1 | tee "$SOURCE_OUT/logs/final_source_checkpoint_contract.log"
python tools/check_v48_45_rebuilt_source_quality.py \
  --source-run "$SOURCE_OUT" --output "$SOURCE_OUT/SOURCE_QUALITY_CONTRACT.json" \
  2>&1 | tee "$SOURCE_OUT/logs/final_source_quality_contract.log"

rm -f "$SOURCE_OUT/SOURCE_REBUILD_FAILED.json"
trap - EXIT
echo "source rebuild complete: $SOURCE_OUT"
echo "Use SOURCE_RUN=$SOURCE_OUT for every v48.45 A/B/C/D arm."
