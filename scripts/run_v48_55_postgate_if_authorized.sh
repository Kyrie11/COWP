#!/usr/bin/env bash
set -Eeuo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"; cd "$REPO"
MAIN_RUN="${MAIN_RUN:-runs/ocrap_v48_55_dcp_drfc_bcde_tcbc_main}"
STATUS="$MAIN_RUN/AUTHORITATIVE_RUN_STATUS.json"; FACTOR="$MAIN_RUN/V48_55_FACTOR_CONTRACT.json"; NEXT="$MAIN_RUN/NEXT_COMMANDS.txt"
[[ -s "$STATUS" && -s "$FACTOR" ]] || { echo "Missing v48.55 authoritative/factor contract" >&2; exit 30; }
python - "$STATUS" "$FACTOR" "$NEXT" "$MAIN_RUN" <<'PY'
import json,pathlib,sys
s=json.load(open(sys.argv[1])); f=json.load(open(sys.argv[2])); n=pathlib.Path(sys.argv[3]); c=s.get('checks') or {}
if not (s.get('valid') and s.get('pipeline_valid') and s.get('authoritative_exit_code')==0): raise SystemExit('v48.55 post-gate NOT authorized: RC must be 0')
if not (c.get('certificate_executed',s.get('certificate_executed',True)) and c.get('gate_evaluated',s.get('gate_evaluated',True))): raise SystemExit('certificate/gate incomplete')
if f.get('version')!='v48.55-DCP-DRFC-BCDE-TCBC' or f.get('arm')!='D': raise SystemExit('requires v48.55 D/Main factor')
if f.get('factor_x_drs_sign_only') is not True or f.get('factor_y_continuous_component_canonicalization') is not True: raise SystemExit('Main requires both TCBC factors')
if f.get('component_margin_target_mode')!='pooled_rms_linear' or f.get('component_margin_regression_reliability')!='0,1,1,0,0': raise SystemExit('TCBC coordinate contract violated')
if f.get('physical_teacher_sign_alignment') or f.get('physical_student_sign_alignment') or f.get('native_physical_student_drs') or f.get('invariant_physical_boundary_distillation'): raise SystemExit('physical-margin replacement/distillation must be off')
if f.get('student_sign_coordinate')!='hard_qbest_ge_zero_root_mass_exact_pcd' or f.get('teacher_sign_coordinate')!='q_hard_proxy_drs_exact_pcd': raise SystemExit('q-hard certificate invariant violated')
if f.get('frontier_order_coordinate')!='smooth_boundary_drs_smooth_pcd': raise SystemExit('smooth-order invariant violated')
if f.get('strategy_regime_conditioning') is not False or f.get('new_tuned_thresholds') is not False or f.get('test_roots_read') is not False: raise SystemExit('frozen protocol violated')
root=pathlib.Path(sys.argv[4])
scale=root/'V48_55_COMPONENT_BOUNDARY_SCALES.json'
if not scale.is_file(): raise SystemExit('TCBC scale artifact missing')
sd=json.load(open(scale))
if sd.get('strategy_regime_conditioning') is not False or sd.get('test_roots_read') is not False or sd.get('saturating_transform') is not False: raise SystemExit('invalid TCBC scale contract')
for variant in ('balanced','precision'):
    cp=root/'candidates'/variant/'V48_55_TCBC_CONTRACT.json'
    if not cp.is_file() or not json.load(open(cp)).get('valid'): raise SystemExit(f'invalid TCBC preflight: {variant}')
if not n.is_file() or not n.read_text().strip(): raise SystemExit('Natural gate passed but NEXT_COMMANDS.txt missing')
print('AUTHORIZED: executing gate-generated Safe non-inferiority + stress/closed-loop commands')
PY
bash "$NEXT"
