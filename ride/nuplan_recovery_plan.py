"""Read-only nuPlan/ReactSim missing-log recovery manifest.

This tool does NOT fetch data or rewrite official scenario lists. It distinguishes
filename-level absence from original-token SQL failure and creates a precisely
scoped investigation list. Official archive membership must be verified separately.
"""
from __future__ import annotations
import argparse,collections,hashlib,json
from pathlib import Path


def make_plan(coverage: dict, extra_roots=()):
    groups=coverage.get('groups',[])
    if not groups:
        raise ValueError('Coverage JSON has no groups')
    missing=[g for g in groups if g.get('status')=='MISSING_IN_ALL_LOCAL_SPLITS']
    extras={}
    missing_names={g['db_name'] for g in missing}
    for root in map(Path,extra_roots):
        if not root.exists():
            continue
        # Exactly one traversal per additional data root; thousands of repeated
        # recursive scans for one missing filename each are prohibitively slow.
        for candidate in root.rglob('*.db'):
            if candidate.name in missing_names and candidate.is_file():
                extras.setdefault(candidate.name,[]).append(str(candidate))
    truly_missing=[g for g in missing if g['db_name'] not in extras]
    by_month=collections.Counter(g['log_name'][:7] for g in truly_missing)
    present=[g for g in groups if g.get('status')!='MISSING_IN_ALL_LOCAL_SPLITS']
    report={
      'coverage_version':'ride_nuplan_recovery_v1',
      'source_coverage_file':coverage.get('ids_file'),
      'source_local_db_counts':coverage.get('db_counts_by_split',{}),
      'required_scenarios':sum(int(g['scenario_count']) for g in groups),
      'found_scenarios_filename_only':sum(int(g['scenario_count']) for g in present),
      'missing_scenarios_filename_only':sum(int(g['scenario_count']) for g in missing),
      'missing_log_filenames':len(missing),
      'missing_by_month':dict(sorted(by_month.items())),
      'recovered_in_extra_roots':extras,
      'remaining_missing_log_filenames':len(truly_missing),
      'remaining_missing_scenarios':sum(int(g['scenario_count']) for g in truly_missing),
      'missing_db_filenames':sorted(g['db_name'] for g in truly_missing),
      'data_integrity': 'UNVERIFIED: filename presence does not imply valid token or eligible nuPlan scenario',
      'download_advice': 'Match DB against official nuPlan v1.1 archive inventory and versions. Incrementally recover archives covering truly missing logs; never redownload all or insert test into trainval.'
    }
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--coverage',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--missing-list',type=Path,required=True)
    p.add_argument('--extra-root',action='append',type=Path,default=[])
    a=p.parse_args(argv)
    report=make_plan(json.loads(a.coverage.read_text(encoding='utf8')),a.extra_root)
    for dest in (a.output,a.missing_list):dest.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    a.missing_list.write_text('\n'.join(report['missing_db_filenames'])+'\n',encoding='utf8')
    print(json.dumps({k:report[k] for k in ('required_scenarios','missing_log_filenames','missing_scenarios_filename_only','remaining_missing_log_filenames','missing_by_month')},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
