import json
from pathlib import Path
from research_probe.baseline_readiness import readiness
from research_probe.gameformer_subset import csv_shard_name
from research_probe.native_womd import load_exclusion_file


def test_shard_stripping_no_fake_cross_format_mapping():
    p=Path('uncompressed_scenario_validation_validation.tfrecord-00000-of-00150')
    assert csv_shard_name(p)=='validation.tfrecord-00000-of-00150'
    assert csv_shard_name(Path('validation.tfrecord-00000-of-00150'))=='validation.tfrecord-00000-of-00150'


def test_missing_baseline_capabilities_not_reported_as_executed(tmp_path):
    cohort=tmp_path/'cohort';cohort.mkdir()
    (cohort/'cohort_manifest.json').write_text(json.dumps({'entries':[{'scenario_id':'a'}],'source_format':'tfexample'}))
    result=readiness(cohort,tmp_path/'missingGameformer',tmp_path/'missingRA',None,None)
    assert not result['scientific_claim_allowed']
    assert result['gameformer']['ready_for_native_open_loop'] is False
    assert result['ra_cbf']['full_closed_loop_source_present'] is False
    assert result['jfp']['paper_only'] is True


def test_antileakage_id_input(tmp_path):
    p=tmp_path/'ids.txt';p.write_text('# a comment\nabc\nxyz\n')
    assert load_exclusion_file(p)=={'abc','xyz'}
    p=tmp_path/'cohort.json';p.write_text(json.dumps({'entries':[{'scenario_id':'abc'}]}))
    assert load_exclusion_file(p)=={'abc'}
