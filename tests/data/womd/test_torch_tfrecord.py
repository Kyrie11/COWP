from pathlib import Path

import pytest

from ocrap.data.womd.torch_tfrecord import TFRecordCRCError, TFRecordReader, write_tfrecord


def test_tfrecord_reads_two_records(tmp_path: Path):
    p = tmp_path / "fake.tfrecord"
    write_tfrecord(p, [b"a", b"bc"])
    got = list(TFRecordReader([p]))
    assert [g.data for g in got] == [b"a", b"bc"]
    assert got[0].offset == 0


def test_tfrecord_crc_error_fails(tmp_path: Path):
    p = tmp_path / "bad.tfrecord"
    write_tfrecord(p, [b"abc"])
    data = bytearray(p.read_bytes())
    data[-1] ^= 0xFF
    p.write_bytes(data)
    with pytest.raises(TFRecordCRCError):
        list(TFRecordReader([p], verify_crc=True))


def test_tfrecord_max_records_and_shard_resume(tmp_path: Path):
    p1 = tmp_path / "a.tfrecord"
    p2 = tmp_path / "b.tfrecord"
    write_tfrecord(p1, [b"a1", b"a2"])
    write_tfrecord(p2, [b"b1"])
    assert [r.data for r in TFRecordReader([p1, p2], max_records=2)] == [b"a1", b"a2"]
    assert [r.data for r in TFRecordReader([p1, p2], start_shard=1, shard_stride=2)] == [b"b1"]
