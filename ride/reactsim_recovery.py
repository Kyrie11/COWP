"""Investigate missing nuPlan DB log names across local directories and ZIPs.

No downloads, no copy, no train/test mixing. S3 archives must be verified with
`aws s3 ls` and the nuPlan license before choosing a download.
"""
from __future__ import annotations
import argparse,collections,json,zipfile
from pathlib import Path


def plan(coverage_path: Path,extra_roots=(),archives=()):
    coverage=json.loads(Path(coverage_path).read_text())
    missing={g['db_name']:g for g in coverage['groups']
             if g['status']=='MISSING_IN_ALL_LOCAL_SPLITS'}
    found={}; archive_hits={}; counts=collections.Counter()
    for root in extra_roots:
        root=Path(root)
        if not root.exists():continue
        for p in root.rglob('*.db'):
            if p.name in missing:
                found.setdefault(p.name,[]).append(str(p))
    for archive in archives:
        path=Path(archive)
        with zipfile.ZipFile(path) as z:
            for item in z.infolist():
                name=Path(item.filename).name
                if name in missing:
                    archive_hits.setdefault(name,[]).append({'archive':str(path),'member':item.filename,'bytes':item.file_size})
    for name in missing:
        if name in found:status='FOUND_IN_EXTRA_ROOT'
        elif name in archive_hits:status='FOUND_IN_PROVIDED_ZIP'
        else:status='NOT_FOUND_IN_SCANNED_SOURCES'
        counts[status]+=1
    return {'coverage_source':str(coverage_path),'missing_distinct_logs':len(missing),
            'status_counts':dict(counts),'missing_db_names':sorted(missing),
            'found_local':found,'found_in_archives':archive_hits,
            'remaining_after_extra_search':sorted(n for n in missing if n not in found and n not in archive_hits),
            'note':'A missing filename does not prove an upstream file never existed; full source archive catalog and ID commit need verification. Never mix nuPlan test into trainval.'}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--coverage',type=Path,required=True)
    p.add_argument('--extra-root',type=Path,action='append',default=[])
    p.add_argument('--archive',type=Path,action='append',default=[])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--missing-list',type=Path)
    a=p.parse_args(argv)
    out=plan(a.coverage,a.extra_root,a.archive)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    if a.missing_list:
        a.missing_list.parent.mkdir(parents=True,exist_ok=True)
        a.missing_list.write_text('\n'.join(out['remaining_after_extra_search'])+'\n')
    print(json.dumps({'missing_logs':out['missing_distinct_logs'],'status_counts':out['status_counts']},indent=2))
if __name__=='__main__':main()
