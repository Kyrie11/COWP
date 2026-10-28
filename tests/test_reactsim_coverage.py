from pathlib import Path
import pytest
from ride.reactsim_coverage import compare


def test_reactsim_coverage_readonly(tmp_path):
    custom=tmp_path/'custom'; processed=tmp_path/'processed'
    custom.mkdir();processed.mkdir()
    (custom/'singapore_1_a.pkl').write_bytes(b'anything')
    (custom/'singapore_2_b.pkl').write_bytes(b'anything')
    (processed/'singapore_2_b.pkl').write_bytes(b'anything')
    (processed/'irrelevant.pkl').write_bytes(b'anything')
    out=compare(custom,processed)
    assert out['matched_names']==1
    assert out['custom_coverage_fraction']==0.5
    assert out['missing_sample']==['singapore_1_a']


def test_reactsim_coverage_empty_or_missing(tmp_path):
    a=tmp_path/'a';b=tmp_path/'b';a.mkdir();b.mkdir()
    assert compare(a,b)['custom_coverage_fraction'] is None
    with pytest.raises(FileNotFoundError):
        compare(tmp_path/'missing', b)
