"""Build a *separately labelled*, small ReactSim cache smoke-list from local DB.

This does not revise or filter the authoritative 2636-ID benchmark list.
It cannot support full-benchmark score or human behavior claims.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from ride.reactsim_tokens import inspect_token


def subset(ids:Path,db_dir:Path,limit:int):
    if limit<1:raise ValueError('limit must be positive')
    raw=ids.read_bytes();data=json.loads(raw)
    selected=[];n=0
    for group in data['groups']:
        db=db_dir/(group['log_name']+'.db')
        if not db.is_file():continue
        found=[]
        for token in group['scenario_tokens']:
            if n>=limit:break
            if inspect_token(db,token)['status']=='ELIGIBLE_BY_SQL_JOINS':
                found.append(token);n+=1
        if found:selected.append({'log_name':group['log_name'],'scenario_tokens':found})
        if n>=limit:break
    if not selected:raise ValueError('No SQL eligible local scenarios')
    return {'version':1,'groups':selected}, {'scientific_status':'NON_BENCHMARK_EXPLORATORY_SUBSET',
      'authoritative_ids_sha256':hashlib.sha256(raw).hexdigest(),
      'selected_scenarios':n,'selected_logs':len(selected),
      'full_reactsim_benchmark':False,
      'warning':'SQL joins do not guarantee sufficient 91-frame horizon. Do not claim full 2636 evaluation.'}


def main(argv=None):
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--ids',type=Path,required=True)
    a.add_argument('--db-dir',type=Path,required=True)
    a.add_argument('--limit',type=int,default=16)
    a.add_argument('--output',type=Path,required=True)
    arg=a.parse_args(argv)
    out,meta=subset(arg.ids,arg.db_dir,arg.limit)
    if arg.output.exists():raise SystemExit('REFUSE_OVERWRITE: '+str(arg.output))
    arg.output.parent.mkdir(parents=True,exist_ok=True)
    arg.output.write_text(json.dumps(out,indent=2)+'\n')
    arg.output.with_suffix('.provenance.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(meta,indent=2))
if __name__=='__main__':main()
