from __future__ import annotations

from cowp.waymax_eval.baselines import ablation_for_method


def test_ablation_switches_exist():
    assert ablation_for_method('cowp_wo_option_preservation')['use_option_preservation'] is False
    assert ablation_for_method('cowp_wo_witness_rejection')['use_hard_witness_rejection'] is False
