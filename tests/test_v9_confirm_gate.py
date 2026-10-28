from pathlib import Path

def test_confirmation_refuses_failed_m0():
    script=(Path(__file__).resolve().parents[1]/'scripts/24_v8_confirm.sh').read_text()
    assert 'RIDE_DEV_M0_REPORT' in script
    assert "M0_CROSS_FAMILY_PAIRS_AVAILABLE_NOT_RQ1_OR_RQ2_PROOF" in script
    assert 'STOP_CONFIRM' in script
