"""Explicitly fail CARLA smoke runs that never simulated valid physics frames."""
from __future__ import annotations
import argparse,json
from pathlib import Path


def validate(rows, min_valid=1, require_all=False, require_complete=False):
    manifest=next((r for r in rows if r.get('type')=='manifest'),None)
    trials=[r for r in rows if r.get('type')=='trial']
    end=next((r for r in rows if r.get('type')=='run_end'),None)
    valid=[r for r in trials if r.get('metrics',{}).get('valid') and r.get('trace')]
    reasons={}
    valid_ids={id(r) for r in valid}
    for trial in trials:
        if id(trial) not in valid_ids:
            for reason in (trial.get('metrics',{}).get('reasons') or
                           [trial.get('reason') or 'missing_metrics']):
                reasons[reason]=reasons.get(reason,0)+1
    summary={'finished_batch':bool(end and end.get('completed')),
             'total_trials':len(trials),'valid_trajectories':len(valid),
             'invalid_reasons':reasons,'minimum_required':min_valid}
    passed=bool(manifest and summary['finished_batch'] and len(valid)>=min_valid)
    if require_all:
        passed = passed and len(trials) > 0 and len(valid) == len(trials)
        summary['require_all'] = True
        summary['all_trials_valid'] = len(trials) > 0 and len(valid) == len(trials)
    else:
        summary['require_all'] = False
    if require_complete:
        complete = len(trials) > 0 and all(bool(t.get('trace')) and
                                           not t.get('simulation_exception')
                                           for t in trials)
        passed = passed and complete
        summary['require_complete'] = True
        summary['all_trials_have_trace_without_exception'] = complete
    else:
        summary['require_complete'] = False
    return passed,summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jsonl',required=True)
    parser.add_argument('--min-valid',type=int,default=1)
    parser.add_argument('--require-all',action='store_true',
                        help='Fail unless every recorded trial has a non-empty valid trace')
    parser.add_argument('--require-complete',action='store_true',
                        help='Fail unless every trial emitted a trace and no simulation exception')
    a=parser.parse_args(argv)
    if a.min_valid<1:parser.error('min-valid must be positive')
    rows=[json.loads(x) for x in Path(a.jsonl).read_text().splitlines() if x.strip()]
    passed,summary=validate(rows,a.min_valid,a.require_all,a.require_complete)
    print('RIDE_SMOKE_'+('PASS' if passed else 'FAIL')+' '+json.dumps(summary))
    return 0 if passed else 4


if __name__=='__main__':raise SystemExit(main())
