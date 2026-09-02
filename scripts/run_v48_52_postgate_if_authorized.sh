#!/usr/bin/env bash
set -Eeuo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
MAIN_RUN="${MAIN_RUN:-runs/ocrap_v48_52_dcp_drfc_bcde_psa_main}"
STATUS="$MAIN_RUN/AUTHORITATIVE_RUN_STATUS.json"
FACTOR="$MAIN_RUN/V48_52_FACTOR_CONTRACT.json"
NEXT="$MAIN_RUN/NEXT_COMMANDS.txt"
[[ -s "$STATUS" ]] || { echo "Missing authoritative status: $STATUS" >&2; exit 30; }
[[ -s "$FACTOR" ]] || { echo "Missing v48.52 factor contract: $FACTOR" >&2; exit 30; }
python - "$STATUS" "$FACTOR" "$NEXT" <<'PY'
import json,pathlib,sys
status=json.load(open(sys.argv[1],encoding='utf-8'))
factor=json.load(open(sys.argv[2],encoding='utf-8'))
next_path=pathlib.Path(sys.argv[3])
checks=status.get('checks') or {}
if not (status.get('valid') and status.get('pipeline_valid') and status.get('authoritative_exit_code') == 0):
    raise SystemExit('v48.52 post-gate evaluation is NOT authorized: authoritative RC must be 0')
if not (checks.get('certificate_executed', status.get('certificate_executed', True)) and checks.get('gate_evaluated', status.get('gate_evaluated', True))):
    raise SystemExit('v48.52 post-gate evaluation is NOT authorized: certificate/gate contract incomplete')
if factor.get('version') != 'v48.52-DCP-DRFC-BCDE-PSA' or factor.get('arm') != 'B':
    raise SystemExit('v48.52 post-gate evaluation requires PSA B/Main factor contract')
if not factor.get('native_certificate_preservation') or not factor.get('recovery_frontier_calibration'):
    raise SystemExit('v48.52 post-gate evaluation requires NCP+DRFC base')
if factor.get('native_margin_complete_preservation'):
    raise SystemExit('v48.52 requires rejected MC-NCP to remain disabled')
if not factor.get('native_advantage_preservation'):
    raise SystemExit('v48.52 requires retained smooth NAP')
if factor.get('native_exact_advantage_preservation') or factor.get('native_boundary_complete_advantage_preservation'):
    raise SystemExit('v48.52 forbids rejected exact-only NAP and BC-NAP')
if not factor.get('boundary_complete_frontier') or factor.get('decision_equivalent_frontier'):
    raise SystemExit('v48.52 requires BC-FC without old v48.50 DEFC')
if factor.get('physical_teacher_sign_alignment') is not True:
    raise SystemExit('v48.52 Main requires Physical Sign Alignment')
if factor.get('teacher_sign_coordinate') != 'q_selected_mstar_physical_drs_exact_pcd':
    raise SystemExit('v48.52 physical teacher-sign coordinate mismatch')
if factor.get('frontier_order_coordinate') != 'smooth_boundary_drs_smooth_pcd':
    raise SystemExit('v48.52 smooth order-channel contract mismatch')
if factor.get('new_tuned_thresholds') is not False:
    raise SystemExit('v48.52 must not introduce tuned thresholds')
if factor.get('strategy_regime_conditioning') is not False or factor.get('test_roots_read') is not False:
    raise SystemExit('v48.52 factor contract violates regime-agnostic/test-seal requirements')
if not next_path.is_file() or not next_path.read_text(encoding='utf-8').strip():
    raise SystemExit('Natural gate passed but generated NEXT_COMMANDS.txt is missing')
print('AUTHORIZED: executing gate-generated Safe non-inferiority + stress/closed-loop commands')
PY
bash "$NEXT"
