"""Merge completed CARLA experiments without losing map/seed provenance."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from .evidence import read_jsonl


def combine(files):
    all_rows=[];provenance=[];seen=set()
    for fname in files:
        rows=read_jsonl(fname)
        manifest=[r for r in rows if r.get('type')=='manifest']
        if len(manifest)!=1 or manifest[0].get('simulator')!='CARLA':
            raise ValueError('Expected one CARLA experiment manifest: '+str(fname))
        meta=manifest[0]
        trials=[r for r in rows if r.get('type')=='trial']
        endings=[r for r in rows if r.get('type')=='run_end']
        if len(endings)!=1 or endings[0].get('completed') is not True or endings[0].get('trial_count')!=len(trials):
            raise ValueError('Incomplete or interrupted CARLA run; not a valid completed panel: '+str(fname))
        kind=meta.get('archetype')
        if not kind or kind=='MULTI' or kind in seen:
            raise ValueError('Duplicate or missing archetype in CARLA merge: '+str(kind))
        seen.add(kind)
        provenance.append({'file':str(fname),'archetype':kind,
            'server':meta.get('carla_server_version'),'client':meta.get('carla_client_version'),
            'map':meta.get('map'),'seed':meta.get('seed'),
            'count':len(trials),'anchors':len(meta.get('anchors',[])),
            'profiles':meta.get('profiles'),'families':meta.get('families')})
        for trial in trials:
            if trial.get('archetype')!=kind:
                raise ValueError('CARLA trial archetype inconsistent with manifest')
            all_rows.append(trial)
    servers={v['server'] for v in provenance}
    clients={v['client'] for v in provenance}
    if len(servers)!=1 or len(clients)!=1 or None in servers or None in clients:
        raise ValueError('Mixed/missing CARLA server or client versions prohibited')
    return [{'type':'manifest','simulator':'CARLA','archetype':'MULTI',
             'carla_server_version':next(iter(servers)),
             'carla_client_version':next(iter(clients)),
             'source_manifests':provenance}]+all_rows


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('inputs',nargs='+');p.add_argument('--output',required=True)
    a=p.parse_args(argv);rows=combine(a.inputs)
    dst=Path(a.output);dst.parent.mkdir(parents=True,exist_ok=True)
    dst.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows),encoding='utf8')
    print('Combined',len(rows)-1,'verified completed trial rows into',dst)


if __name__=='__main__':main()
