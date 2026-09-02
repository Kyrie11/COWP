#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTEST_DISABLE_PLUGIN_AUTOLOAD="${PYTEST_DISABLE_PLUGIN_AUTOLOAD:-1}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"

# The uploaded repository retains historical tests whose referenced v48.12-v48.32
# launchers are not present. Those files are documented separately and are not
# part of the v48.35.2 release contract. Run the supported matrix in isolated
# processes to prevent cross-suite global-runtime contamination.
batch_1=(
  tests/test_calibrated_selector_v5.py
  tests/test_cli_overrides_and_metrics.py
  tests/test_closed_loop_resume_optimization.py
  tests/test_core.py
  tests/test_external_baseline_cuda_runtime.py
  tests/test_manifest_repair_v48.py
  tests/test_ocrap_closed_loop_hotpath.py
  tests/test_v39_recovery_advantage.py
  tests/test_v40_direct_value.py
  tests/test_v42_ocsava.py
)
batch_2=(
  tests/test_v43_rsc.py
  tests/test_v44_rava.py
  tests/test_v45_rave.py
  tests/test_v46_race.py
  tests/test_v47_trac.py
  tests/test_v48_10_cope.py
  tests/test_v48_11_caster.py
  tests/test_v48_15_prism_cc.py
  tests/test_v48_16_anchor.py
  tests/test_v48_17_bridge.py
)
batch_3=(
  tests/test_v48_18_duet_bridge.py
  tests/test_v48_19_facet_bridge.py
  tests/test_v48_33_eligible_set_policy.py
  tests/test_v48_34_1_rc30_model_contract_hotfix.py
  tests/test_v48_34_barrier_crossfit.py
  tests/test_v48_35_1_rc30_training_contract_hotfix.py
  tests/test_v48_35_2_engineering_integrity.py
  tests/test_v48_35_continuous_frontier.py
  tests/test_v48_5_ecpr.py
  tests/test_v48_6_rpgc.py
)
batch_4=(
  tests/test_v48_7_spire.py
  tests/test_v48_8_scope.py
  tests/test_v48_9_pacer.py
  tests/test_v48_trac_sr.py
)
# v50 tests are intentionally outside the v48.35.2 release contract. They are
# retained in the mixed research repository, but are not allowed to influence
# v48.35.2 release status.

for name in batch_1 batch_2 batch_3 batch_4; do
  declare -n batch="$name"
  for test_file in "${batch[@]}"; do
    [[ -f "$test_file" ]] || { echo "missing release test: $test_file" >&2; exit 4; }
  done
  echo "[v48.35.2] running $name (${#batch[@]} files)"
  python -m pytest -q "${batch[@]}"
done

echo "[v48.35.2] supported release matrix passed"
