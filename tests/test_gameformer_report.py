import csv,json
from research_probe.gameformer_report import score

def test_matched_gameformer_report_is_openloop_only(tmp_path):
    c=tmp_path/'cohort';c.mkdir()
    (c/'cohort_manifest.json').write_text(json.dumps({'entries':[{'scenario_id':'id01'}]}))
    p=tmp_path/'log.csv'
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['scenarios','collision','miss']);w.writeheader()
        w.writerow({'scenarios':'id01_10','collision':1,'miss':0})
        w.writerow({'scenarios':'other_10','collision':0,'miss':0})
    r=score(c,p)
    assert r['matched_rows']==1 and r['unmatched_rows']==1
    assert r['statistics']['collision']['mean']==1
    assert not r['scientific_claim_allowed']
