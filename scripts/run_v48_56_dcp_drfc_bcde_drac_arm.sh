#!/usr/bin/env bash
set -Eeuo pipefail
ARM="${1:?usage: $0 A|B|C|D}"
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
BASE_OUT="${BASE_OUT:-runs}"
case "$ARM" in
  A) export OUTPUTDIR="${OUTPUTDIR:-$BASE_OUT/ocrap_v48_56_dcp_drfc_bcde_drac_ablation_A}" ;;
  B) export OUTPUTDIR="${OUTPUTDIR:-$BASE_OUT/ocrap_v48_56_dcp_drfc_bcde_drac_ablation_B}" ;;
  C) export OUTPUTDIR="${OUTPUTDIR:-$BASE_OUT/ocrap_v48_56_dcp_drfc_bcde_drac_ablation_C}" ;;
  D) export OUTPUTDIR="${OUTPUTDIR:-$BASE_OUT/ocrap_v48_56_dcp_drfc_bcde_drac_main}" ;;
  *) echo "unknown arm: $ARM" >&2; exit 2 ;;
esac
export RESUME_AFTER_ADAPTATION=0
unset V4836_FACTOR_CACHE_BALANCED V4836_FACTOR_CACHE_PRECISION || true

# Direct single-arm invocation is also self-contained. The 2x2 launcher prepares
# once and sets V4845_SKIP_PROTOCOL_PREPARE=1 so parallel arms never race here.
export OCRAP_ROOT="${OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}"
export PROTOCOL_ROOT="${PROTOCOL_ROOT:-$OCRAP_ROOT/calibration_v48_14_prism_4814}"
export CAL_NEAR="${CAL_NEAR:-$OCRAP_ROOT/calibration_near_contact}"
export CAL_CONTACT="${CAL_CONTACT:-$OCRAP_ROOT/calibration_contact}"
export CAL_SAFE="${CAL_SAFE:-$OCRAP_ROOT/calibration_safe}"
if [[ "${V4845_SKIP_PROTOCOL_PREPARE:-0}" != 1 ]]; then
  bash scripts/prepare_v48_45_protocol.sh
fi
[[ -s "$PROTOCOL_ROOT/V48_45_PROTOCOL_SEAL.json" ]] || {
  echo "v48.56 shared protocol seal missing: $PROTOCOL_ROOT/V48_45_PROTOCOL_SEAL.json" >&2
  exit 30
}
# Recompute the manifest/scene assignment contract at each arm boundary (sample
# existence was already checked by prepare). This prevents cross-arm input drift.
mkdir -p "$BASE_OUT"
protocol_probe="$BASE_OUT/.v48_45_protocol_recheck_${ARM}_$$.json"
set +e
python tools/check_v48_45_protocol_seal.py \
  --protocol-root "$PROTOCOL_ROOT" --near-source "$CAL_NEAR" --contact-source "$CAL_CONTACT" --safe-root "$CAL_SAFE" \
  --seed "${V4845_PROTOCOL_SEED:-4814}" --adapt-train-fraction "${V4845_ADAPT_TRAIN_FRACTION:-0.45}" \
  --adapt-dev-fraction "${V4845_ADAPT_DEV_FRACTION:-0.15}" --skip-sample-file-check --output "$protocol_probe" >/dev/null
protocol_probe_rc=$?
set -e
if [[ "$protocol_probe_rc" != 0 ]]; then
  echo "v48.47 per-arm protocol recheck failed for arm=$ARM" >&2
  [[ -s "$protocol_probe" ]] && cat "$protocol_probe" >&2
  rm -f "$protocol_probe"
  exit 30
fi
rm -f "$protocol_probe"
export V4845_PROTOCOL_SEAL_SHA256="$(sha256sum "$PROTOCOL_ROOT/V48_45_PROTOCOL_SEAL.json" | awk '{print $1}')"

# v48.45 source-run resolution must not depend on the versioned checkout cwd.
# The v48.13 reference checkpoints live under the persistent BASE_OUT used by
# the previous successful v48.44 run.  An explicit SOURCE_RUN still wins.
SOURCE_RUN_BASENAME="${V4845_SOURCE_RUN_BASENAME:-ocrap_v48_13_terra_proxy_4801}"
REBUILT_SOURCE_BASENAME="${V4845_REBUILT_SOURCE_BASENAME:-ocrap_v48_45_source_rebuild_s7}"
if [[ -z "${SOURCE_RUN:-}" ]]; then
  # After historical checkpoint loss, prefer the explicitly reconstructed,
  # hash-sealed common source. Historical v48.13 remains supported when present.
  if [[ -f "$BASE_OUT/$REBUILT_SOURCE_BASENAME/SOURCE_REBUILD_COMPLETE.json" && \
        -f "$BASE_OUT/$REBUILT_SOURCE_BASENAME/candidates/balanced/model_v48_trac_sr/best.pt" && \
        -f "$BASE_OUT/$REBUILT_SOURCE_BASENAME/candidates/precision/model_v48_trac_sr/best.pt" ]]; then
    SOURCE_RUN="$BASE_OUT/$REBUILT_SOURCE_BASENAME"
  elif [[ -f "$BASE_OUT/$SOURCE_RUN_BASENAME/candidates/balanced/model_v48_trac_sr/best.pt" && \
          -f "$BASE_OUT/$SOURCE_RUN_BASENAME/candidates/precision/model_v48_trac_sr/best.pt" ]]; then
    SOURCE_RUN="$BASE_OUT/$SOURCE_RUN_BASENAME"
  elif [[ -f "$REPO/runs/$REBUILT_SOURCE_BASENAME/SOURCE_REBUILD_COMPLETE.json" && \
          -f "$REPO/runs/$REBUILT_SOURCE_BASENAME/candidates/balanced/model_v48_trac_sr/best.pt" && \
          -f "$REPO/runs/$REBUILT_SOURCE_BASENAME/candidates/precision/model_v48_trac_sr/best.pt" ]]; then
    SOURCE_RUN="$REPO/runs/$REBUILT_SOURCE_BASENAME"
  elif [[ -f "$REPO/runs/$SOURCE_RUN_BASENAME/candidates/balanced/model_v48_trac_sr/best.pt" && \
          -f "$REPO/runs/$SOURCE_RUN_BASENAME/candidates/precision/model_v48_trac_sr/best.pt" ]]; then
    SOURCE_RUN="$REPO/runs/$SOURCE_RUN_BASENAME"
  else
    # Keep a deterministic canonical path so the dedicated source preflight
    # publishes an authoritative RC=30 instead of silently random-initializing.
    SOURCE_RUN="$BASE_OUT/$REBUILT_SOURCE_BASENAME"
  fi
fi
SOURCE_RUN="$(python - "$SOURCE_RUN" <<'PY_SOURCE_RUN'
from pathlib import Path
import sys
print(Path(sys.argv[1]).expanduser().resolve(strict=False))
PY_SOURCE_RUN
)"
export SOURCE_RUN

# v48.56 holds the best-supported q-hard BC-FC + smooth-NAP planner fixed and
# directly audits teacher/component semantic correctness.  No regime identifier,
# router, threshold relaxation, proposal expansion, root-logit recalibration,
# physical-margin distillation, or component-scale normalization is introduced.
export EVIDENCE_UNBOUNDED_BENEFIT_FACTOR=false
export EVIDENCE_UNBOUNDED_HARM_FACTORS=false
export EVIDENCE_BENEFIT_RESIDUAL_SCALE=1.0
export EVIDENCE_COMPONENT_SCALE=6.0
export EVIDENCE_COMPONENT_HEADS=true
export FACTOR_BENEFIT_MARGIN_REGRESSION_WEIGHT=1.00
export FACTOR_BENEFIT_MARGIN_TEMPERATURE=0.050
export FACTOR_COMPONENT_MARGIN_REGRESSION_WEIGHT=1.00
export FACTOR_COMPONENT_TAIL_WEIGHT=0.75
export FACTOR_COMPONENT_MARGIN_TARGET_MODE=raw
export FACTOR_COMPONENT_MARGIN_TARGET_SCALE=0.10
export FACTOR_COMPONENT_MARGIN_CANONICAL_SCALES=""
export FACTOR_COMPONENT_MARGIN_REGRESSION_RELIABILITY="1,1,1,0,0"
export EVIDENCE_DUAL_INTERACTION_BRIDGE=true
export EVIDENCE_FACTORIZED_HARM_INTERACTION=false
export EVIDENCE_PARTIAL_POOL_HARM_RESIDUAL=false
export EVIDENCE_RANK_BENEFIT_SKIP=false
export EVIDENCE_POSTPREFIX_OBS_TRANSPORT_BENEFIT=false
export EVIDENCE_POSTPREFIX_OBS_TRANSPORT_HARM=false
export EVIDENCE_ROCT_BENEFIT=true
export EVIDENCE_ROCT_DEPLOYABILITY=true
export EVIDENCE_ROCT_SCALE="${EVIDENCE_ROCT_SCALE:-3.0}"
export EVIDENCE_ROCT_ALPHA="${EVIDENCE_ROCT_ALPHA:-0.20}"
export EVIDENCE_ROCT_BETA="${EVIDENCE_ROCT_BETA:-0.20}"
export EVIDENCE_ROCT_TOP_M="${EVIDENCE_ROCT_TOP_M:-8}"
export EVIDENCE_ROCT_OPTION_TEMPERATURE="${EVIDENCE_ROCT_OPTION_TEMPERATURE:-0.35}"
export EVIDENCE_RESERVE_FACTOR_ALIGNMENT=true
export EVIDENCE_ADMISSION_PRIOR_MODE=joint_reserve
export EVIDENCE_JOINT_RESERVE_TEMPERATURE=0.050
export V4838_RFR_RESERVE_ONLY=1
export V4837_FACTOR_PRESERVING_IDENTITY=0
export V4836_IDENTITY_TRAIN_ALL=0
export V4836_COUPLE_ADMISSION_PRIOR=0
export V4836_ADAPTIVE_IDENTITY_MARGIN=0
export V4836_ENABLE_FINAL_CALIBRATION=0
export FACTOR_COMPONENT_UNDERESTIMATION_WEIGHT=0
export FACTOR_SAFE_POSITIVE_COMPONENT_OVERESTIMATION_WEIGHT=0
export FACTOR_JOINT_RESERVE_REGRESSION_WEIGHT=0
export FACTOR_JOINT_RESERVE_BOUNDARY_WEIGHT=0
export PROPOSAL_TOP_K=5
export EVIDENCE_CALIBRATOR_CONTEXT_SOURCE=physical_interaction

# Fixed paper/mainline mechanisms.
export V4845_SOWR_MARGIN_WITNESS=0
export V4845_SOWR_OBS_KERNEL=0
export V4846_SEQUENTIAL_WITNESS=0
export V4847_DECISION_OBS=0
export V4847_RECOVERY_FRONTIER=1
export V4848_NATIVE_CERTIFICATE=1
export EVIDENCE_NATIVE_CERTIFICATE_PRESERVATION=true
export EVIDENCE_NATIVE_MARGIN_COMPLETE_PRESERVATION=false
export EVIDENCE_NATIVE_ADVANTAGE_PRESERVATION=true
export EVIDENCE_NATIVE_EXACT_ADVANTAGE_PRESERVATION=false
export EVIDENCE_NATIVE_BOUNDARY_COMPLETE_ADVANTAGE_PRESERVATION=false
export V4850_DECISION_EQUIVALENT_FRONTIER=false
export V4851_BOUNDARY_COMPLETE_FRONTIER=true
export V4852_PHYSICAL_TEACHER_SIGN_ALIGNMENT=false
export V4853_PHYSICAL_STUDENT_SIGN_ALIGNMENT=false
export EVIDENCE_PHYSICAL_STUDENT_DRS=false
export V4854_INVARIANT_PHYSICAL_BOUNDARY_DISTILLATION=false
export V4852_REQUIRE_PSA_CONTRACT=0
export V4853_REQUIRE_CSE_CONTRACT=0
export V4854_REQUIRE_IPBD_CONTRACT=0
export V4855_REQUIRE_TCBC_CONTRACT=0
export V4856_REQUIRE_DRAC_CONTRACT=1
export EVIDENCE_DEP_BOUNDARY_ALIGNED=false
export EVIDENCE_GAP_ORDINAL_ONLY=false
export V4850_FRONTIER_GAP_TOLERANCE="${V4850_FRONTIER_GAP_TOLERANCE:-0.05}"
export V4850_FRONTIER_POSITIVE_GAIN="${V4850_FRONTIER_POSITIVE_GAIN:-0.015}"
export V4850_FRONTIER_PCD_WEIGHT="${V4850_FRONTIER_PCD_WEIGHT:-1.0}"
export EVIDENCE_NATIVE_DRS_TOLERANCE="${EVIDENCE_NATIVE_DRS_TOLERANCE:-0.05}"
export EVIDENCE_NATIVE_DEPLOYABILITY_TOLERANCE="${EVIDENCE_NATIVE_DEPLOYABILITY_TOLERANCE:-0.05}"
export EVIDENCE_NATIVE_GAP_TOLERANCE="${EVIDENCE_NATIVE_GAP_TOLERANCE:-0.05}"
export EVIDENCE_NATIVE_POSITIVE_GAIN="${EVIDENCE_NATIVE_POSITIVE_GAIN:-0.015}"
export TRAIN_OPTION_EXECUTION_SEMANTICS=observation_class
export EVAL_OPTION_EXECUTION_SEMANTICS=observation_class
export OPTION_EXECUTION_SEMANTICS=observation_class
export V4847_FRONTIER_EPOCHS="${V4847_FRONTIER_EPOCHS:-5}"
export V4847_WITNESS_PATIENCE="${V4847_WITNESS_PATIENCE:-2}"
export V4847_WITNESS_LR="${V4847_WITNESS_LR:-0.00004}"
export V4847_WITNESS_BATCH_SIZE="${V4847_WITNESS_BATCH_SIZE:-72}"
export V4847_FRONTIER_LOSS_WEIGHT="${V4847_FRONTIER_LOSS_WEIGHT:-2.00}"
export V4847_FRONTIER_MARGIN_ANCHOR_WEIGHT="${V4847_FRONTIER_MARGIN_ANCHOR_WEIGHT:-0.25}"
export V4847_FRONTIER_SIGN_TEMPERATURE="${V4847_FRONTIER_SIGN_TEMPERATURE:-0.08}"
export V4847_FRONTIER_SIGN_WEIGHT="${V4847_FRONTIER_SIGN_WEIGHT:-0.50}"

# Strict 2x2 semantic-role test:
#   A: current nominal-relative DEP + GAP hard-veto reference.
#   B: X only -- DEP uses the absolute teacher deployability boundary R_dep=0.
#   C: Y only -- GAP remains in PCD/order but is ordinal-only, not a hard veto.
#   D: X+Y -- Decision-Role Aligned Certificate (DRAC).
case "$ARM" in
  A)
    export OCRAP_ALGORITHM_VERSION="v48.56-DCP-DRFC-BCDE-DRAC-ablation-A"
    export OCRAP_IMPLEMENTATION_VERSION="v48.56-A-current-component-semantics-reference"
    export V4838_FACTOR_ALGORITHM_FAMILY="v48.56-A-current-component-semantics-reference"
    ;;
  B)
    export OCRAP_ALGORITHM_VERSION="v48.56-DCP-DRFC-BCDE-DRAC-ablation-B"
    export OCRAP_IMPLEMENTATION_VERSION="v48.56-B-deployability-zero-boundary"
    export V4838_FACTOR_ALGORITHM_FAMILY="v48.56-B-deployability-zero-boundary"
    export EVIDENCE_DEP_BOUNDARY_ALIGNED=true
    ;;
  C)
    export OCRAP_ALGORITHM_VERSION="v48.56-DCP-DRFC-BCDE-DRAC-ablation-C"
    export OCRAP_IMPLEMENTATION_VERSION="v48.56-C-gap-ordinal-only"
    export V4838_FACTOR_ALGORITHM_FAMILY="v48.56-C-gap-ordinal-only"
    export EVIDENCE_GAP_ORDINAL_ONLY=true
    export FACTOR_COMPONENT_MARGIN_REGRESSION_RELIABILITY="1,1,0,0,0"
    ;;
  D)
    export OCRAP_ALGORITHM_VERSION="v48.56-DCP-DRFC-BCDE-DRAC"
    export OCRAP_IMPLEMENTATION_VERSION="v48.56-D-decision-role-aligned-certificate"
    export V4838_FACTOR_ALGORITHM_FAMILY="v48.56-D-decision-role-aligned-certificate"
    export EVIDENCE_DEP_BOUNDARY_ALIGNED=true
    export EVIDENCE_GAP_ORDINAL_ONLY=true
    export FACTOR_COMPONENT_MARGIN_REGRESSION_RELIABILITY="1,1,0,0,0"
    ;;
  *) echo "unknown arm: $ARM" >&2; exit 2 ;;
esac

mkdir -p "$OUTPUTDIR"
python - "$OUTPUTDIR/V48_56_FACTOR_CONTRACT.json" "$ARM" "$EVIDENCE_DEP_BOUNDARY_ALIGNED" "$EVIDENCE_GAP_ORDINAL_ONLY" <<'PY_FACTOR'
import json,os,pathlib,sys,time
p=pathlib.Path(sys.argv[1]); arm=sys.argv[2]
x=sys.argv[3].lower() in {'1','true','yes'}; y=sys.argv[4].lower() in {'1','true','yes'}
d={
 'event':'v48_56_factor_contract','version':'v48.56-DCP-DRFC-BCDE-DRAC','arm':arm,
 'factor_x_deployability_zero_boundary':x,'factor_y_gap_ordinal_only':y,
 'reference_mechanism':'v48.53-A/v48.55-A q-hard BC-FC + smooth NAP',
 'deployability_teacher_coordinate':('0.5-sigmoid(R_dep_candidate)' if x else 'sigmoid(R_dep_nom)-sigmoid(R_dep_candidate)-0.05'),
 'deployability_material_boundary':'R_dep=0' if x else 'none_nominal_relative',
 'gap_role':('ordinal_PCD_and_order_only' if y else 'PCD_plus_noncompensatory_veto'),
 'gap_in_teacher_pcd':True,'gap_in_native_advantage':True,'gap_in_hard_component_veto':not y,
 'component_bce_reliability':('1,1,0,0,0' if y else '1,1,1,0,0'),
 'component_margin_regression_reliability':('1,1,0,0,0' if y else '1,1,1,0,0'),
 'component_margin_target_mode':'raw','component_margin_transform':'raw','component_margin_scale_source':'none',
 'native_certificate_preservation':True,'native_dep_boundary_aligned':x,
 'native_advantage_preservation':True,'native_exact_advantage_preservation':False,
 'native_boundary_complete_advantage_preservation':False,'native_margin_complete_preservation':False,
 'boundary_complete_frontier':True,'recovery_frontier_calibration':True,'frontier_order_coordinate':'smooth_boundary_drs_smooth_pcd',
 'student_sign_coordinate':'hard_qbest_ge_zero_root_mass_exact_pcd','teacher_sign_coordinate':'q_hard_proxy_drs_exact_pcd',
 'drs_coordinate':'hard_qbest_ge_zero_root_mass','teacher_benefit_coordinate':'exact_pcd_advantage',
 'physical_teacher_sign_alignment':False,'physical_student_sign_alignment':False,
 'invariant_physical_boundary_distillation':False,'root_logit_recalibration':False,
 'factor_x_drs_sign_only':False,'factor_y_continuous_component_canonicalization':False,
 'training_option_execution_semantics':'observation_class','evaluation_option_execution_semantics':'observation_class',
 'strategy_regime_conditioning':False,'new_tuned_thresholds':False,
 'proposal_top_k':int(os.environ.get('PROPOSAL_TOP_K','5')),'test_roots_read':False,
 'created_unix':time.time(),
}
p.write_text(json.dumps(d,indent=2,sort_keys=True)+'\n')
PY_FACTOR
exec bash scripts/run_v48_36_ocaf_dedicated.sh
