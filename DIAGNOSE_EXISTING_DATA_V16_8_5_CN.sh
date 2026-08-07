#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${PYTHON_BIN:-python}"
COWP_ROOT="${COWP_ROOT:-/data0/senzeyu2/dataset/COWP/formal}"
OLD_VAL_CACHE="${OLD_VAL_CACHE:-$COWP_ROOT/tensor_cache_val_waymax_transport_v16_8}"
OUT_DIR="${OUT_DIR:-outputs/data_decision_v16_8_5}"
mkdir -p "$OUT_DIR"

run_diag() {
  local tag="$1" mod="$2" rem="$3"
  "$PYTHON_BIN" -m cowp.scripts.45_diagnose_proposal_ceiling \
    --cache-dir "$OLD_VAL_CACHE" --output "$OUT_DIR/proposal_ceiling_${tag}.json" \
    --hard-count 0 --random-count 0 --control-count 0 \
    --subset-modulo "$mod" --subset-remainder "$rem" --seed 2026
}
run_diag full 1 0
run_diag calibration_mod2_r0 2 0
run_diag heldout_mod2_r1 2 1

"$PYTHON_BIN" - "$OUT_DIR" <<'PY'
import json, pathlib, sys
root=pathlib.Path(sys.argv[1])
rows={p.stem.replace('proposal_ceiling_',''):json.loads(p.read_text()) for p in root.glob('proposal_ceiling_*.json')}
print('\n===== existing proposal-bank feasibility =====')
need_fresh=False
for name in ('full','calibration_mod2_r0','heldout_mod2_r1'):
    r=rows[name]['scene_rates']
    fs=float(r['best_case_selected_false_safe_lower_bound'])
    pb=float(r['best_case_pbtr_lower_bound'])
    ncf=float(r['any_ncf'])
    feasible=(fs<=0.55 and pb<=0.45)
    need_fresh |= not feasible
    print(f'{name}: any_ncf={ncf:.5f}, FS_floor={fs:.5f}, PBTR_floor={pb:.5f}, current_gate_feasible={feasible}')
print('need_new_proposal_bank_for_current_mechanism_gate=', need_fresh)
print('IMPORTANT: this does NOT by itself justify a 4-day rebuild; only the paired fresh probe can do that.')
PY
