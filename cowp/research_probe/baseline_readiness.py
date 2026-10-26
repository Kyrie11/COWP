"""Evidence-safe readiness report for user-supplied baseline repositories.

Does not invoke external methods, invent checkpoints, or equate prediction with control.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def readiness(cohort: Path, gameformer: Path, ra_cbf: Path, scenario: Path | None, checkpoint: Path | None):
    manifest = cohort / 'cohort_manifest.json'
    if not manifest.exists():
        return {'status': 'BLOCKED_NO_COHORT', 'cohort_manifest': str(manifest),
                'how_to_fix': 'Run sample_raw first.'}
    data = json.loads(manifest.read_text())
    n = len(data.get('entries', []))
    g_source = gameformer / 'open_loop_planning' / 'open_loop_test.py'
    r_source = ra_cbf / 'tbsim' / 'safety_funcs' / 'cbfs.py'
    scenario_shards = []
    if scenario and scenario.exists():
        scenario_shards = list(scenario.glob('*tfrecord*'))[:4]
        if not scenario_shards:
            scenario_shards = list(scenario.glob('**/*tfrecord*'))[:4]
    gameformer_ready = bool(g_source.exists() and scenario_shards and checkpoint and checkpoint.is_file())
    return {
        'scientific_claim_allowed': False,
        'cohort': {'manifest': str(manifest), 'entries': n, 'source_format': data.get('source_format'),
                   'observational_only': True},
        'gameformer': {
            'source_present': g_source.exists(), 'source_file': str(g_source),
            'needs_scenario_proto_maps': True,
            'scenario_files_found': [str(x) for x in scenario_shards],
            'checkpoint_present': bool(checkpoint and checkpoint.is_file()),
            'ready_for_native_open_loop': gameformer_ready,
            'native_open_loop_not_closed_loop': True,
            'instructions': 'Once Scenario proto shards and a compatible trained checkpoint exist, '
            'run prepare_gameformer_subset; then run authors open_loop_test.py with --candidate_csv. '
            'This evaluates open-loop GameFormer, NOT failure under decentralized closed-loop execution.'
        },
        'jfp': {
            'source_provided': False, 'paper_only': True,
            'status': 'NO_MODEL_NO_CHECKPOINT',
            'paper_scope': 'Joint/conditional multi-agent prediction, not an executable safety controller.'
        },
        'ra_cbf': {
            'source_present': r_source.exists(), 'source_file': str(r_source),
            'checkpoint_present_in_distribution': False,
            'full_closed_loop_source_present': False,
            'status': 'REQUIRES_TRAJDATA_NUSCENES_ADAPTATION_AND_POLICY_IMPLEMENTATION',
            'paper_scope': 'Learned responsibility allocation; author repository says complete closed-loop simulator unavailable.'
        },
        'studt': {
            'paper_available': True, 'source_in_supplied_archives': False,
            'status': 'NOT_REPRODUCED',
            'warning': 'Same consensual safe-set rule for all agents; original assumptions must be preserved for native comparison.'
        },
        'r3r': {
            'status': 'INPUT_EXPORT_ONLY',
            'warning': 'R3R is communication-based; exported WOMD initial/goal states do not constitute running R3R.'
        }
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cohort', type=Path, required=True)
    p.add_argument('--gameformer-root', type=Path, required=True)
    p.add_argument('--ra-cbf-root', type=Path, required=True)
    p.add_argument('--scenario-root', type=Path)
    p.add_argument('--gameformer-checkpoint', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a=p.parse_args()
    result=readiness(a.cohort,a.gameformer_root,a.ra_cbf_root,a.scenario_root,a.gameformer_checkpoint)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__=='__main__':main()
