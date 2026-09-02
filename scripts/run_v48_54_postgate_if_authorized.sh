#!/usr/bin/env bash
set -Eeuo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"; cd "$REPO"
MAIN_RUN="${MAIN_RUN:-runs/ocrap_v48_54_dcp_drfc_bcde_ipbd_main}"
STATUS="$MAIN_RUN/AUTHORITATIVE_RUN_STATUS.json"; FACTOR="$MAIN_RUN/V48_54_FACTOR_CONTRACT.json"; NEXT="$MAIN_RUN/NEXT_COMMANDS.txt"
[[ -s "$STATUS" && -s "$FACTOR" ]] || { echo "Missing v48.54 authoritative/factor contract" >&2; exit 30; }
python - "$STATUS" "$FACTOR" "$NEXT" <<'PY'
import json,pathlib,sys
s=json.load(open(sys.argv[1])); f=json.load(open(sys.argv[2])); n=pathlib.Path(sys.argv[3]); c=s.get('checks') or {}
if not (s.get('valid') and s.get('pipeline_valid') and s.get('authoritative_exit_code')==0): raise SystemExit('v48.54 post-gate NOT authorized: RC must be 0')
if not (c.get('certificate_executed',s.get('certificate_executed',True)) and c.get('gate_evaluated',s.get('gate_evaluated',True))): raise SystemExit('certificate/gate incomplete')
if f.get('version')!='v48.54-DCP-DRFC-BCDE-IPBD' or f.get('arm')!='B': raise SystemExit('requires v48.54 B/Main factor')
if f.get('invariant_physical_boundary_distillation') is not True: raise SystemExit('Main requires IPBD')
if f.get('physical_teacher_sign_alignment') or f.get('physical_student_sign_alignment') or f.get('native_physical_student_drs'): raise SystemExit('IPBD must not alter hard teacher/student/native DRS')
if f.get('student_sign_coordinate')!='hard_qbest_ge_zero_root_mass_exact_pcd' or f.get('teacher_sign_coordinate')!='q_hard_proxy_drs_exact_pcd': raise SystemExit('q-hard certificate invariant violated')
if f.get('frontier_order_coordinate')!='smooth_boundary_drs_smooth_pcd': raise SystemExit('smooth-order invariant violated')
if f.get('strategy_regime_conditioning') is not False or f.get('new_tuned_thresholds') is not False or f.get('test_roots_read') is not False: raise SystemExit('frozen protocol violated')
if not n.is_file() or not n.read_text().strip(): raise SystemExit('Natural gate passed but NEXT_COMMANDS.txt missing')
print('AUTHORIZED: executing gate-generated Safe non-inferiority + stress/closed-loop commands')
PY
bash "$NEXT"
