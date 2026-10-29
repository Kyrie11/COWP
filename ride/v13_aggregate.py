"""Aggregate RIDE v13 diagnostic reports by PHYSICAL corridors, not rows/seeds.

Proximity deduplication is a conservative heuristic for overlapping local roads.
Do not interpret its rate as prevalence of real-world failures: CARLA stress
locations were sampled nonrandomly and multiple configs can share corridors.
"""
from __future__ import annotations
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from .v13_model_transfer import VERSION,write_new


def aggregate_reports(reports, *, radius_m=90.):
    if radius_m <= 0 or not math.isfinite(radius_m):
        raise ValueError('Radius must be finite and positive')
    if not reports: raise ValueError('No reports')
    nodes=[]
    modes={r.get('mode','PROSPECTIVE_SOURCE_SIMULATOR_LOOKAHEAD') for r in reports}
    if len(modes)!=1:
        raise ValueError('Do not pool retrospective and prospective evidence')
    orientations={(r['source_family'],r['target_family']) for r in reports}
    if len(orientations)!=1:
        raise ValueError('Do not mix different source-to-target orientations')
    settings={json.dumps(r['locked_params'],sort_keys=True) for r in reports}
    if len(settings)!=1:
        raise ValueError('Inconsistent preregistered endpoint parameters')
    for i,report in enumerate(reports):
        if report.get('protocol')!=VERSION:
            raise ValueError('Protocol mismatch')
        ctx=report['frozen_context']
        anchors={str(a['anchor']):a for a in ctx['anchors']}
        contexts=defaultdict(list)
        for row in report['results']:
            contexts[(str(row['anchor']),str(row['seed']))].append(row)
        for (anchor,seed),items in contexts.items():
            a=anchors.get(anchor)
            if not a or not isinstance(a.get('key'),list) or len(a['key'])<2:
                raise ValueError('Missing physical coordinate in frozen anchor: '+str(anchor))
            xy=(float(a['key'][0]),float(a['key'][1]))
            if not all(math.isfinite(v) for v in xy):
                raise ValueError('Bad physical coordinate')
            nodes.append({'map':ctx['map'],'xy':xy,'report_index':i,'anchor':anchor,
                          'seed':seed,'rows':items})
    parent=list(range(len(nodes)))
    def root(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]]
            i=parent[i]
        return i
    def merge(i,j):
        x,y=root(i),root(j)
        if x!=y:parent[y]=x
    for i,a in enumerate(nodes):
        for j in range(i):
            b=nodes[j]
            if a['map']==b['map'] and math.dist(a['xy'],b['xy'])<=radius_m:
                merge(i,j)
    clusters=defaultdict(list)
    for i,node in enumerate(nodes):clusters[root(i)].append(node)
    records=[]
    for k,members in clusters.items():
        rows=[x for m in members for x in m['rows']]
        valid=[x for x in rows if x.get('scientific_status')!='UNKNOWN']
        errors=[x for x in valid if x.get('paired_model_induced_selection_error')]
        records.append({'map':members[0]['map'],
                        'anchor_seed_occurrences':len(members),
                        'members':[{'report_index':m['report_index'],'anchor':m['anchor'],'seed':m['seed']} for m in members],
                        'valid_comparisons':len(valid),'diagnostic_error_rows':len(errors),
                        'any_diagnostic_model_error':bool(errors)})
    records.sort(key=lambda x:(x['map'],x['members'][0]['report_index'],x['members'][0]['anchor']))
    return {'protocol':VERSION,'mode':next(iter(modes)),'radius_m':radius_m,
            'summary':{'reports':len(reports),'anchor_seed_occurrences':len(nodes),
                       'physical_corridor_clusters':len(records),
                       'corridors_with_valid_comparisons':sum(x['valid_comparisons']>0 for x in records),
                       'corridors_with_diagnostic_model_error':sum(x['any_diagnostic_model_error'] for x in records),
                       'caveat':'Corridor grouping is a spatial heuristic, NOT an IID sampling frame or human-effect prevalence.'},
            'clusters':records}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reports',nargs='+',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--radius-m',type=float,default=90.)
    a=p.parse_args(argv)
    res=aggregate_reports([json.loads(Path(f).read_text(encoding='utf8')) for f in a.reports],radius_m=a.radius_m)
    write_new(a.output,res)
    print(json.dumps(res['summary'],indent=2,ensure_ascii=False))
    return 0

if __name__=='__main__':raise SystemExit(main())
