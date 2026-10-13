"""Build a fair same-ID GameFormer CSV by scanning *actual* Scenario proto records.

Never infer Scenario shard membership from TFExample shard number.  Data from
both formats may be sharded differently.  No ego counterfactuals are produced.
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path


def csv_shard_name(path: Path) -> str:
    name=path.name
    prefix='uncompressed_scenario_validation_'
    if name.startswith(prefix):return name[len(prefix):]
    return name


def make_subset(cohort, scenario_dir, output_csv, report_path, limit_files=0):
    import tensorflow as tf
    from waymo_open_dataset.protos import scenario_pb2
    cohort=Path(cohort); scenario_dir=Path(scenario_dir)
    manifest=json.loads((cohort/'cohort_manifest.json').read_text())
    requested={e['scenario_id'] for e in manifest['entries']}
    if not requested:raise ValueError('Empty cohort')
    shards=sorted(p for p in scenario_dir.glob('**/*tfrecord*') if p.is_file())
    if limit_files: shards=shards[:limit_files]
    if not shards:raise FileNotFoundError(f'No Scenario TFRecords beneath {scenario_dir}')
    if not tf.executing_eagerly():raise RuntimeError('TensorFlow eager execution required')
    matched={};scanned=0
    for shard in shards:
        for raw in tf.data.TFRecordDataset(str(shard)).as_numpy_iterator():
            sc=scenario_pb2.Scenario.FromString(bytes(raw))
            scanned+=1
            if sc.scenario_id in requested:
                if sc.scenario_id in matched:
                    raise ValueError(f'Duplicate Scenario ID in Scenario source: {sc.scenario_id}')
                matched[sc.scenario_id]=csv_shard_name(shard)
        if len(matched)==len(requested):break
    output_csv=Path(output_csv);report_path=Path(report_path)
    output_csv.parent.mkdir(parents=True,exist_ok=True)
    report_path.parent.mkdir(parents=True,exist_ok=True)
    # Mirror author's CSV columns used by the official open_loop_test.py.
    with output_csv.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['File','Scenario ID','tags','tags.1','tags.2'])
        w.writeheader()
        for sid in sorted(matched):w.writerow({'File':matched[sid],'Scenario ID':sid})
    r={'status':'PREPARED_MATCHED_OPEN_LOOP_INPUT_NOT_SCIENTIFIC_RESULT',
       'source_scenario_dir':str(scenario_dir),'scanned_records':scanned,
       'requested_scenarios':len(requested),'matched_scenarios':len(matched),
       'missing_scenario_ids':sorted(requested-matched.keys()),
       'csv':str(output_csv), 'note':'GameFormer original code has no WOMD closed-loop planner; compare matched open-loop metrics only.'}
    report_path.write_text(json.dumps(r,indent=2))
    return r


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cohort',type=Path,required=True)
    p.add_argument('--scenario-dir',type=Path,required=True)
    p.add_argument('--output-csv',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--limit-files',type=int,default=0)
    a=p.parse_args()
    print(json.dumps(make_subset(a.cohort,a.scenario_dir,a.output_csv,a.report,a.limit_files),indent=2))
if __name__=='__main__':main()
