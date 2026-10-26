"""Regression for TFExample format mutation after the first record.

A fake TensorFlow module deliberately tests three records in a single shard,
without depending on a local TensorFlow installation.
"""
import json
import sys
from types import SimpleNamespace
import numpy as np
from research_probe.native_womd import build_cohort, decode_example


class ListFeature:
    def __init__(self, floats=(), ints=(), byte_values=()):
        self.float_list = SimpleNamespace(value=list(floats))
        self.int64_list = SimpleNamespace(value=list(ints))
        self.bytes_list = SimpleNamespace(value=list(byte_values))


def make_fake_example(index):
    n = 3
    feats = {
        'scenario/id': ListFeature(byte_values=[f'example_{index:03d}'.encode()]),
        'state/is_sdc': ListFeature(ints=[1, 0, 0]),
        'state/type': ListFeature(ints=[1, 1, 1]),
        'state/id': ListFeature(ints=[100 + index, 200 + index, 300 + index]),
    }
    for phase, steps in [('past', 10), ('current', 1), ('future', 80)]:
        for name in ('x', 'y', 'velocity_x', 'velocity_y', 'bbox_yaw', 'length', 'width', 'valid'):
            if name == 'valid':
                feats[f'state/{phase}/{name}'] = ListFeature(ints=[1] * (n * steps))
            else:
                vals = []
                for agent in range(n):
                    for k in range(steps):
                        vals.append({'x': agent * 8.0 + (k / 10), 'y': 0, 'velocity_x': 1,
                                     'velocity_y': 0, 'bbox_yaw': 0, 'length': 4.5, 'width': 1.9}[name])
                feats[f'state/{phase}/{name}'] = ListFeature(floats=vals)
    return SimpleNamespace(features=SimpleNamespace(feature=feats))


def test_three_consecutive_tfexamples_do_not_switch_to_scenario(monkeypatch, tmp_path):
    raw = tmp_path / 'tf_example'
    raw.mkdir()
    (raw / 'validation_tfexample.tfrecord-00000-of-00150').write_bytes(b'fake-records')

    class TFRecordDataset:
        def __init__(self, *_args, **_kwargs):
            pass

        def as_numpy_iterator(self):
            return iter([b'0', b'1', b'2'])

    fake_tf = SimpleNamespace(
        data=SimpleNamespace(TFRecordDataset=TFRecordDataset),
        executing_eagerly=lambda: True,
        train=SimpleNamespace(Example=SimpleNamespace(FromString=lambda record: make_fake_example(int(record)))),
    )
    monkeypatch.setitem(sys.modules, 'tensorflow', fake_tf)
    result = build_cohort(raw, tmp_path / 'output', max_shards=1,
                          max_records_per_shard=3, keep_random=3, keep_interactive=3)
    assert result['counts']['examined'] == 3
    assert result['counts']['bad_records'] == 0
    assert result['counts']['usable'] == 3
    assert result['source_format'] == 'tfexample'
    assert len(result['entries']) == 3
    assert all((tmp_path / 'output' / entry['file']).exists() for entry in result['entries'])
    data = json.loads((tmp_path / 'output' / 'cohort_manifest.json').read_text())
    assert data['source_format'] == 'tfexample'


def test_decode_realistic_tfexample_fields():
    parsed = decode_example(make_fake_example(1))
    assert parsed['current_x'].shape == (3, 1)
    assert parsed['past_x'].shape == (3, 10)
    assert parsed['future_valid'].shape == (3, 80)
    assert parsed['scenario_id'] == 'example_001'
