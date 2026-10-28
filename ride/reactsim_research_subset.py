"""Build a separate, explicitly NON-OFFICIAL diagnostic ReactSim scenario subset.

Uses the SAME public get_scenarios_from_db query as ReactSim's scenario_cache,
not mere sqlite lidar_pc existence. Writes a new ID list, does not mutate the
publisher's 2636 IDs; the diagnostic subset MUST NOT be called a full benchmark.
"""
from __future__ import annotations
import argparse,collections,hashlib,json
from pathlib import Path


def select(ids:Path, db_dir:Path, target:int=32, max_groups:int=200):
    from nuplan.database.nuplan_db.nuplan_scenario_queries import get_scenarios_from_db
    if target<1 or max_groups<1:raise ValueError('target and max_groups must be positive')
    raw=ids.read_bytes();groups=json.loads(raw)['groups']
    chosen=[];counts=collections.Counter();processed_groups=0
    for grp in groups:
        if processed_groups>=max_groups or sum(len(q['scenario_tokens']) for q in chosen)>=target:break
        processed_groups+=1
        db=db_dir/(grp['log_name']+'.db')
        if not db.is_file():counts['db_missing']+=len(grp['scenario_tokens']);continue
        tokens=grp['scenario_tokens'];obtained=set()
        try:
            for i in range(0,len(tokens),500):
                for row in get_scenarios_from_db(
                    log_file=str(db),filter_tokens=tokens[i:i+500],filter_types=None,
                    filter_map_names=None,include_invalid_mission_goals=True,
                    include_cameras=False):
                    obtained.add(row['token'].hex())
        except Exception as ex:
            counts['query_errors']+=1
            print(f'SKIP query error for {db.name}: {type(ex).__name__}: {ex}')
            continue
        keep=[tok for tok in tokens if tok in obtained][:target-sum(len(q['scenario_tokens']) for q in chosen)]
        counts['token_filtered_or_missing_in_db']+=len(tokens)-len(keep)
        if keep:chosen.append({'log_name':grp['log_name'],'scenario_tokens':keep})
    subset={'version':1,'groups':chosen}
    audit={'mode':'EXPLORATORY_SUBSET_NOT_FULL_BENCHMARK','official_original_ids_sha256':hashlib.sha256(raw).hexdigest(),
           'source_ids':str(ids),'source_db_dir':str(db_dir),'requested_subset_tokens':target,
           'selected_scenarios':sum(len(g['scenario_tokens']) for g in chosen),
           'selected_unique_logs':len(chosen),'groups_scanned':processed_groups,
           'official_ids_total':sum(len(g['scenario_tokens']) for g in groups),
           'ineligible_counts':dict(counts),
           'warning':'Biased convenience sample with a shared source simulator. DO NOT report as complete ReactSim-Bench or real counterfactual ground truth.'}
    return subset,audit


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ids',type=Path,required=True)
    p.add_argument('--db-dir',type=Path,required=True)
    p.add_argument('--max-groups',type=int,default=200)
    p.add_argument('--target',type=int,default=32)
    p.add_argument('--output-ids',type=Path,required=True)
    p.add_argument('--output-audit',type=Path,required=True)
    a=p.parse_args(argv)
    sub,audit=select(a.ids,a.db_dir,a.target,a.max_groups)
    for dest in (a.output_ids,a.output_audit):dest.parent.mkdir(exist_ok=True,parents=True)
    a.output_ids.write_text(json.dumps(sub,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    a.output_audit.write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(audit,ensure_ascii=False,indent=2))
    if audit['selected_scenarios']==0:raise SystemExit('No valid subset token found')
if __name__=='__main__':main()
