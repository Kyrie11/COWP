#!/usr/bin/env bash
set -Eeuo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"; cd "$REPO"
MAIN_RUN="${MAIN_RUN:-runs/ocrap_v48_56_dcp_drfc_bcde_drac_main}"
STATUS="$MAIN_RUN/AUTHORITATIVE_RUN_STATUS.json"; FACTOR="$MAIN_RUN/V48_56_FACTOR_CONTRACT.json"; NEXT="$MAIN_RUN/NEXT_COMMANDS.txt"
[[ -s "$STATUS" && -s "$FACTOR" ]] || { echo "Missing v48.56 authoritative/factor contract" >&2; exit 30; }
python - "$STATUS" "$FACTOR" "$NEXT" "$MAIN_RUN" <<'PY'
import json,pathlib,sys
s=json.load(open(sys.argv[1])); f=json.load(open(sys.argv[2])); n=pathlib.Path(sys.argv[3]); c=s.get('checks') or {}
if not (s.get('valid') and s.get('pipeline_valid') and s.get('authoritative_exit_code')==0): raise SystemExit('v48.56 post-gate NOT authorized: RC must be 0')
if not (c.get('certificate_executed',s.get('certificate_executed',True)) and c.get('gate_evaluated',s.get('gate_evaluated',True))): raise SystemExit('certificate/gate incomplete')
if f.get('version')!='v48.56-DCP-DRFC-BCDE-DRAC' or f.get('arm')!='D': raise SystemExit('requires v48.56 D/Main factor')
if f.get('factor_x_deployability_zero_boundary') is not True or f.get('factor_y_gap_ordinal_only') is not True: raise SystemExit('Main requires both DRAC decision-role factors')
if f.get('component_margin_target_mode')!='raw' or f.get('component_bce_reliability')!='1,1,0,0,0' or f.get('component_margin_regression_reliability')!='1,1,0,0,0': raise SystemExit('DRAC role/reliability contract violated')
if f.get('native_dep_boundary_aligned') is not True or f.get('gap_in_hard_component_veto') is not False: raise SystemExit('DRAC deployed component semantics violated')
if f.get('gap_in_teacher_pcd') is not True or f.get('gap_in_native_advantage') is not True: raise SystemExit('GAP ordinal evidence must remain in PCD/order')
if f.get('physical_teacher_sign_alignment') or f.get('physical_student_sign_alignment') or f.get('native_physical_student_drs') or f.get('invariant_physical_boundary_distillation'): raise SystemExit('physical-margin replacement/distillation must be off')
if f.get('student_sign_coordinate')!='hard_qbest_ge_zero_root_mass_exact_pcd' or f.get('teacher_sign_coordinate')!='q_hard_proxy_drs_exact_pcd': raise SystemExit('q-hard certificate invariant violated')
if f.get('frontier_order_coordinate')!='smooth_boundary_drs_smooth_pcd': raise SystemExit('smooth-order invariant violated')
if f.get('strategy_regime_conditioning') is not False or f.get('new_tuned_thresholds') is not False or f.get('test_roots_read') is not False: raise SystemExit('frozen protocol violated')
root=pathlib.Path(sys.argv[4])
# v48.56 explicitly STOPs TCBC/component normalization.  A stale scale artifact
# signals that this output directory was contaminated by an incompatible run.
if (root/'V48_55_COMPONENT_BOUNDARY_SCALES.json').exists():
    raise SystemExit('unexpected TCBC scale artifact in v48.56 Main')
for variant in ('balanced','precision'):
    cp=root/'candidates'/variant/'V48_56_DRAC_CONTRACT.json'
    if not cp.is_file() or not json.load(open(cp)).get('valid'): raise SystemExit(f'invalid DRAC preflight: {variant}')
if not n.is_file() or not n.read_text().strip(): raise SystemExit('Natural gate passed but NEXT_COMMANDS.txt missing')
print('AUTHORIZED: executing gate-generated Safe non-inferiority + stress/closed-loop commands')
PY
bash "$NEXT"
