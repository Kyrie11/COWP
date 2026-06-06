from __future__ import annotations

import glob
from pathlib import Path
from typing import Iterator

import numpy as np


def _import_tensorflow():
    try:
        import tensorflow as tf  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError("TensorFlow is required for WOMD tf.Example parsing. Install tensorflow>=2.11.") from exc
    return tf


def resolve_glob_patterns(patterns: str | list[str]) -> list[str]:
    if isinstance(patterns, str):
        patterns = [patterns]
    files: list[str] = []
    for pat in patterns:
        files.extend(sorted(glob.glob(pat)))
    if not files:
        raise FileNotFoundError(f"No tf.Example TFRecord files matched: {patterns}")
    return files


def iter_tfexample_records(patterns: str | list[str]) -> Iterator[bytes]:
    tf = _import_tensorflow()
    files = resolve_glob_patterns(patterns)
    dataset = tf.data.TFRecordDataset(files, num_parallel_reads=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    for rec in dataset:
        yield bytes(rec.numpy())


def decode_tfexample(serialized: bytes) -> dict[str, np.ndarray | bytes]:
    tf = _import_tensorflow()
    example = tf.train.Example()
    example.ParseFromString(serialized)
    out: dict[str, np.ndarray | bytes] = {}
    for key, feat in example.features.feature.items():
        kind = feat.WhichOneof("kind")
        if kind == "float_list":
            out[key] = np.asarray(feat.float_list.value, dtype=np.float32)
        elif kind == "int64_list":
            out[key] = np.asarray(feat.int64_list.value, dtype=np.int64)
        elif kind == "bytes_list":
            vals = list(feat.bytes_list.value)
            out[key] = vals[0] if len(vals) == 1 else np.asarray(vals, dtype=object)
    return out


def iter_tfexamples(patterns: str | list[str]) -> Iterator[dict[str, np.ndarray | bytes]]:
    for raw in iter_tfexample_records(patterns):
        yield decode_tfexample(raw)


def scenario_id_from_tfexample(example: dict[str, np.ndarray | bytes]) -> str:
    for key in ("scenario/id", "scenario_id", "scenario/id_bytes"):
        if key in example:
            val = example[key]
            if isinstance(val, bytes):
                return val.decode("utf-8")
            if isinstance(val, np.ndarray) and val.dtype == object and len(val):
                first = val.flat[0]
                return first.decode("utf-8") if isinstance(first, bytes) else str(first)
            if isinstance(val, np.ndarray):
                return str(val.flat[0])
    raise KeyError("Could not find scenario id in tf.Example. Expected one of scenario/id or scenario_id.")


def save_tfexample_npz(example: dict[str, np.ndarray | bytes], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for k, v in example.items():
        safe = k.replace("/", "__")
        arrays[safe] = np.frombuffer(v, dtype=np.uint8) if isinstance(v, bytes) else v
    np.savez_compressed(path, **arrays)
