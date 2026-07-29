from __future__ import annotations

import gzip
import glob
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator


@dataclass(frozen=True)
class TFRecordExample:
    path: str
    offset: int
    data: bytes


class TFRecordCRCError(IOError):
    pass


_CRC32C_TABLE: list[int] | None = None


def _make_crc32c_table() -> list[int]:
    table = []
    poly = 0x82F63B78
    for i in range(256):
        crc = i
        for _ in range(8):
            crc = (crc >> 1) ^ poly if (crc & 1) else crc >> 1
        table.append(crc & 0xFFFFFFFF)
    return table


def crc32c(data: bytes) -> int:
    try:
        import crc32c as _crc32c  # type: ignore

        return int(_crc32c.crc32c(data)) & 0xFFFFFFFF
    except Exception:
        global _CRC32C_TABLE
        if _CRC32C_TABLE is None:
            _CRC32C_TABLE = _make_crc32c_table()
        crc = 0xFFFFFFFF
        for b in data:
            crc = _CRC32C_TABLE[(crc ^ b) & 0xFF] ^ (crc >> 8)
        return (~crc) & 0xFFFFFFFF


def masked_crc32c(data: bytes) -> int:
    x = crc32c(data)
    return (((x >> 15) | ((x << 17) & 0xFFFFFFFF)) + 0xA282EAD8) & 0xFFFFFFFF


def expand_paths(paths: str | os.PathLike | Iterable[str | os.PathLike]) -> list[str]:
    if isinstance(paths, (str, os.PathLike)):
        items = [paths]
    else:
        items = list(paths)
    out: list[str] = []
    for item in items:
        s = str(item)
        matches = sorted(glob.glob(s)) if any(ch in s for ch in "*?[") else [s]
        out.extend(matches)
    return sorted(dict.fromkeys(out))


class TFRecordReader:
    def __init__(self, paths, verify_crc: bool = True, compression: str | None = None, start_shard: int = 0, shard_stride: int = 1, max_records: int | None = None):
        self.paths = expand_paths(paths)
        self.verify_crc = bool(verify_crc)
        self.compression = compression
        self.start_shard = int(start_shard)
        self.shard_stride = max(1, int(shard_stride))
        self.max_records = max_records

    def _open(self, path: str):
        comp = self.compression
        if comp is None and path.endswith(".gz"):
            comp = "gz"
        if comp == "gz":
            return gzip.open(path, "rb")
        if comp not in (None, ""):
            raise ValueError(f"Unsupported TFRecord compression {comp}")
        return open(path, "rb")

    def __iter__(self) -> Iterator[TFRecordExample]:
        emitted = 0
        selected = self.paths[self.start_shard :: self.shard_stride]
        for path in selected:
            with self._open(path) as f:
                while True:
                    offset = f.tell()
                    length_bytes = f.read(8)
                    if not length_bytes:
                        break
                    if len(length_bytes) != 8:
                        raise EOFError(f"Truncated TFRecord length at {path}:{offset}")
                    length_crc_bytes = f.read(4)
                    if len(length_crc_bytes) != 4:
                        raise EOFError(f"Truncated TFRecord length CRC at {path}:{offset}")
                    length = struct.unpack("<Q", length_bytes)[0]
                    expected_len_crc = struct.unpack("<I", length_crc_bytes)[0]
                    if self.verify_crc and masked_crc32c(length_bytes) != expected_len_crc:
                        raise TFRecordCRCError(f"CRC mismatch for length at path={path} offset={offset}")
                    data = f.read(length)
                    if len(data) != length:
                        raise EOFError(f"Truncated TFRecord data at {path}:{offset}")
                    data_crc_bytes = f.read(4)
                    if len(data_crc_bytes) != 4:
                        raise EOFError(f"Truncated TFRecord data CRC at {path}:{offset}")
                    expected_data_crc = struct.unpack("<I", data_crc_bytes)[0]
                    if self.verify_crc and masked_crc32c(data) != expected_data_crc:
                        raise TFRecordCRCError(f"CRC mismatch for data at path={path} offset={offset}")
                    yield TFRecordExample(path=str(path), offset=int(offset), data=data)
                    emitted += 1
                    if self.max_records is not None and emitted >= self.max_records:
                        return


def write_tfrecord(path: str | os.PathLike, records: Iterable[bytes]) -> None:
    with open(path, "wb") as f:
        for data in records:
            length_bytes = struct.pack("<Q", len(data))
            f.write(length_bytes)
            f.write(struct.pack("<I", masked_crc32c(length_bytes)))
            f.write(data)
            f.write(struct.pack("<I", masked_crc32c(data)))
