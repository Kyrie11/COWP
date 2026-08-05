from __future__ import annotations

import inspect
import json
import subprocess
import sys
from pathlib import Path

from cowp.waymax_eval.rollout import (
    learned_offline_candidate_eval,
    learned_offline_candidate_eval_budget_sweep,
    learned_offline_candidate_eval_methods,
    learned_offline_candidate_eval_sweep,
)


def test_learned_offline_public_apis_expose_deterministic_subset_controls() -> None:
    for fn in (
        learned_offline_candidate_eval,
        learned_offline_candidate_eval_sweep,
        learned_offline_candidate_eval_budget_sweep,
        learned_offline_candidate_eval_methods,
    ):
        params = inspect.signature(fn).parameters
        assert "subset_modulo" in params
        assert "subset_remainder" in params
        assert "max_scenes" in params


def _metrics(remainder: int, false_safe: float = 0.10) -> dict:
    return {
        "bcot_risk_budget": 0.35,
        "CertificateSemantics/Version": "v16_8_2_decoupled",
        "FallbackSemantics/ExplicitAccounting": True,
        "EvaluationSubset/Modulo": 2,
        "EvaluationSubset/Remainder": remainder,
        "EvaluationSubset/Scenes": 20,
        "EvaluationSubset/IndexSHA256": f"hash-{remainder}",
        "EP": 1.0,
        "FallbackRate": 0.10,
        "SelectedFalseSafeRate": false_safe,
        "LearnedAcceptedCandidateRate": 0.20,
        "LearnedAcceptNCFRecall": 0.50,
        "WitnessQuality/AUPRC": 0.80,
        "BCOT/FalseSafe_AUPRC": 0.80,
        "RootTransport/ConflictConditioned_AUPRC": 0.80,
    }


def test_mechanism_verifier_uses_disjoint_heldout_metrics(tmp_path: Path) -> None:
    heldout = {
        "cowp": _metrics(1, false_safe=0.10),
        "conventional_safety": _metrics(1, false_safe=0.30),
    }
    sweep_rows = []
    for budget, ep in [(0.2, 0.8), (0.35, 1.0), (0.5, 1.1)]:
        row = _metrics(0, false_safe=0.15)
        row["bcot_risk_budget"] = budget
        row["EP"] = ep
        sweep_rows.append(row)
    sweep = {"bcot_risk_budget_sweep": sweep_rows}
    calibration = {
        "bcot_risk_budget": 0.35,
        "status": "constraints_satisfied",
        # Deliberately terrible calibration metrics: the verifier must not report
        # them as held-out performance.
        "selection_metrics": {**_metrics(0, false_safe=0.99), "LearnedAcceptNCFRecall": 0.0},
    }
    hp, sp, cp, op = [tmp_path / n for n in ("heldout.json", "sweep.json", "cal.json", "out.json")]
    hp.write_text(json.dumps(heldout), encoding="utf-8")
    sp.write_text(json.dumps(sweep), encoding="utf-8")
    cp.write_text(json.dumps(calibration), encoding="utf-8")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "cowp.scripts.25_verify_mechanism_effect",
            "--input",
            str(hp),
            "--sweep-input",
            str(sp),
            "--calibration-json",
            str(cp),
            "--output",
            str(op),
            "--min-unique-selection-points",
            "2",
            "--min-ncf-recall",
            "0.3",
            "--min-witness-auprc",
            "0.6",
            "--min-bcot-auprc",
            "0.6",
            "--min-root-transport-auprc",
            "0.6",
            "--min-accepted-rate",
            "0.1",
            "--max-fallback",
            "0.25",
            "--min-false-safe-improvement",
            "0.08",
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    report = json.loads(op.read_text(encoding="utf-8"))
    assert report["pass"] is True
    assert report["metrics_source"] == "heldout_input_only"
    assert report["learned_accept_ncf_recall"] == 0.50
    assert report["calibration_heldout_disjoint"] is True


def test_certificate_metrics_are_not_overwritten_by_shortlist_and_valid_fallback_is_counted() -> None:
    import numpy as np

    from cowp.waymax_eval.rollout import _LearnedMetricsAccumulator

    acc = _LearnedMetricsAccumulator()
    label = {
        "cowp/candidates/valid": np.asarray([True, True, True, True]),
        "cowp/candidates/conventional_safe": np.asarray([True, True, True, True]),
        "cowp/candidates/noncoercive_feasible": np.asarray([True, True, False, False]),
        "cowp/candidates/false_safe": np.asarray([False, False, True, False]),
        "cowp/candidates/trajectory": np.zeros((4, 3, 3), dtype=np.float32),
        "cowp/critical/valid": np.asarray([], dtype=bool),
    }
    acc.add_selection(
        selected_idx=0,
        accepted_mask=np.asarray([True, True, True, False]),
        shortlist_mask=np.asarray([True, False, False, False]),
        fallback_used=True,
        label=label,
    )
    metrics = acc.finish(auprc=0.0, rank_good=0, rank_total=0, witness_threshold=0.7)
    assert metrics["CertificateSemantics/Version"] == "v16_8_2_decoupled"
    assert metrics["LearnedAcceptedCandidateRate"] == 0.75
    assert metrics["SelectionShortlist/CandidateRate"] == 0.25
    assert metrics["FallbackRate"] == 1.0
    assert metrics["FallbackSelection/SelectedCandidateRate"] == 1.0
