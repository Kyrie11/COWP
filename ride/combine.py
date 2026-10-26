"""Combine case-specific CARLA trial logs while preserving their provenance."""
import argparse,json
from pathlib import Path
from .evidence import read_jsonl


def combine(files):
    all_rows=[]; provenance=[]
    for fname in files:
        rows=read_jsonl(fname)
        meta=next((r for r in rows if r.get('type')=='manifest'),None)
        if not meta or meta.get('simulator')!='CARLA':raise ValueError('Missing CARLA manifest: '+str(fname))
        archetype=meta.get('archetype','convoy')
        provenance.append({'file':str(fname),'archetype':archetype,'server':meta.get('carla_server_version'),
                           'map':meta.get('map'),'count':sum(r.get('type')=='trial' for r in rows)})
        for r in rows:
            if r.get('type')=='trial':
                x=dict(r);x['archetype']=archetype;all_rows.append(x)
    servers={x['server'] for x in provenance}
    if len(servers)!=1:raise ValueError('Mixed server versions prohibited: '+repr(servers))
    return [{'type':'manifest','simulator':'CARLA','archetype':'MULTI',
             'carla_server_version':next(iter(servers)),'source_manifests':provenance}]+all_rows


def main():
    p=argparse.ArgumentParser();p.add_argument('inputs',nargs='+');p.add_argument('--output',required=True)
    a=p.parse_args();rows=combine(a.inputs)
    dest=Path(a.output);dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf8')
    print('combined',len(rows)-1,'trials to',dest)
if __name__=='__main__':main()
