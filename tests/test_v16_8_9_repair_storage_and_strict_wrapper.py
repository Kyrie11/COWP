from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np


def test_contract_repair_reads_slash_and_encoded_npz_keys():
    mod = importlib.import_module("cowp.scripts.61_repair_v16_8_9_audit_transport_contract")
    arr = np.asarray([1, 2, 3])
    slash = {"cowp/audit/root_affected": arr}
    encoded = {"cowp__audit__root_affected": arr}
    assert np.array_equal(mod._get(slash, "cowp/audit/root_affected"), arr)
    assert np.array_equal(mod._get(encoded, "cowp/audit/root_affected"), arr)
    mod._put(slash, "cowp/audit/root_budget_crossed", arr)
    mod._put(encoded, "cowp/audit/root_budget_crossed", arr)
    assert "cowp/audit/root_budget_crossed" in slash
    assert "cowp__audit__root_budget_crossed" in encoded


def test_strict_wrapper_does_not_use_bash_special_random_variable():
    text = Path("NEXT_RUN_COMMANDS_V16_8_9_STRICT_PROPOSAL_PROBE_CN.sh").read_text(encoding="utf-8")
    assert 'RANDOM="$PROBE_ROOT/' not in text
    assert '"$RANDOM"' not in text
    assert "RANDOM_IDS_FILE=" in text
    assert "v16_8_9_strict_verdict.json" in text
    assert "emit_early_verdict" in text
