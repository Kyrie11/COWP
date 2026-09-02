#!/usr/bin/env bash
set -Eeuo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
MAIN_RUN="${MAIN_RUN:-runs/ocrap_v48_53_dcp_drfc_bcde_cse_main}"
STATUS="$MAIN_RUN/AUTHORITATIVE_RUN_STATUS.json"
FACTOR="$MAIN_RUN/V48_53_FACTOR_CONTRACT.json"
NEXT="$MAIN_RUN/NEXT_COMMANDS.txt"
[[ -s "$STATUS" ]] || { echo "Missing authoritative status: $STATUS" >&2; exit 30; }
[[ -s "$FACTOR" ]] || { echo "Missing v48.53 factor contract: $FACTOR" >&2; exit 30; }
python - "$STATUS" "$FACTOR" "$NEXT" <<'PY'
import json,pathlib,sys
status=json.load(open(sys.argv[1],encoding='utf-8')); factor=json.load(open(sys.argv[2],encoding='utf-8')); next_path=pathlib.Path(sys.argv[3])
checks=status.get('checks') or {}
if not (status.get('valid') and status.get('pipeline_valid') and status.get('authoritative_exit_code') == 0):
    raise SystemExit('v48.53 post-gate evaluation is NOT authorized: authoritative RC must be 0')
if not (checks.get('certificate_executed',status.get('certificate_executed',True)) and checks.get('gate_evaluated',status.get('gate_evaluated',True))):
    raise SystemExit('v48.53 post-gate evaluation is NOT authorized: certificate/gate contract incomplete')
if factor.get('version')!='v48.53-DCP-DRFC-BCDE-CSE' or factor.get('arm')!='D':
    raise SystemExit('v48.53 post-gate evaluation requires D/Main CSE factor contract')
if not (factor.get('native_certificate_preservation') and factor.get('recovery_frontier_calibration') and factor.get('native_advantage_preservation')):
    raise SystemExit('v48.53 requires NCP+BC-FC+smooth-NAP base')
if factor.get('native_margin_complete_preservation') or factor.get('native_exact_advantage_preservation') or factor.get('native_boundary_complete_advantage_preservation'):
    raise SystemExit('v48.53 forbids rejected MC-NCP, exact-only NAP, and BC-NAP')
if factor.get('boundary_complete_frontier') is not True or factor.get('decision_equivalent_frontier') is not False:
    raise SystemExit('v48.53 requires BC-FC without old DEFC')
if factor.get('physical_teacher_sign_alignment') is not True or factor.get('physical_student_sign_alignment') is not True:
    raise SystemExit('v48.53 D/Main requires symmetric physical teacher/student sign alignment')
if factor.get('native_physical_student_drs') is not True:
    raise SystemExit('v48.53 D/Main deployment/native DRS is not physical-student aligned')
if factor.get('teacher_sign_coordinate')!='q_selected_mstar_physical_drs_exact_pcd':
    raise SystemExit('v48.53 teacher physical sign coordinate mismatch')
if factor.get('student_sign_coordinate')!='q_selected_predicted_margin_physical_drs_exact_pcd':
    raise SystemExit('v48.53 student/deployment physical sign coordinate mismatch')
if factor.get('frontier_order_coordinate')!='smooth_boundary_drs_smooth_pcd':
    raise SystemExit('v48.53 smooth order-channel mismatch')
if factor.get('new_tuned_thresholds') is not False or factor.get('strategy_regime_conditioning') is not False or factor.get('test_roots_read') is not False:
    raise SystemExit('v48.53 factor contract violates frozen threshold/regime/test seal')
if not next_path.is_file() or not next_path.read_text(encoding='utf-8').strip():
    raise SystemExit('Natural gate passed but NEXT_COMMANDS.txt is missing')
print('AUTHORIZED: executing gate-generated Safe non-inferiority + stress/closed-loop commands')
PY
bash "$NEXT"
