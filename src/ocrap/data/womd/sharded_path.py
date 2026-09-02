"""Utilities for WOMD/Waymax sharded TFRecord path specifications.

Waymax accepts TensorFlow-style sharded specifications such as
``/path/validation_tfexample.tfrecord@150``.  The suffix declares the number
of files in the shard set; it is *not* a scenario/record limit.  Files on disk
are normally named ``<prefix>-00000-of-00150`` ...
``<prefix>-00149-of-00150``.
"""
from __future__ import annotations

from dataclasses import dataclass
import glob
import re
from pathlib import Path
from typing import Iterable

_SHARD_SPEC_RE = re.compile(r"^(?P<prefix>.+)@(?P<count>[1-9][0-9]*)$")
_SHARD_FILE_RE = re.compile(r"^(?P<prefix>.*)-(?P<index>[0-9]+)-of-(?P<count>[0-9]+)(?P<suffix>.*)$")
_PLACEHOLDER_PARTS = (
    "/absolute/path",
    "<path>",
    "<womd",
    "your/path",
    "path/to/",
    "todo/",
)


@dataclass(frozen=True)
class WomdPathPart:
    raw: str
    prefix: str
    shard_count: int | None

    @property
    def is_sharded(self) -> bool:
        return self.shard_count is not None


@dataclass(frozen=True)
class WomdPathResolution:
    spec: str
    parts: tuple[WomdPathPart, ...]
    files: tuple[str, ...]
    missing_files: tuple[str, ...]
    placeholder_parts: tuple[str, ...]
    errors: tuple[str, ...]

    @property
    def declared_shard_count(self) -> int | None:
        counts = {p.shard_count for p in self.parts if p.shard_count is not None}
        return next(iter(counts)) if len(counts) == 1 else None

    @property
    def valid(self) -> bool:
        return bool(self.parts and self.files and not self.missing_files and not self.placeholder_parts and not self.errors)

    def as_dict(self, *, max_files: int = 20, max_missing: int = 20) -> dict:
        return {
            "spec": self.spec,
            "parts": [
                {"raw": p.raw, "prefix": p.prefix, "shard_count": p.shard_count}
                for p in self.parts
            ],
            "declared_shard_count": self.declared_shard_count,
            "num_resolved_files": len(self.files),
            "resolved_files": list(self.files[:max_files]),
            "num_missing_files": len(self.missing_files),
            "missing_files": list(self.missing_files[:max_missing]),
            "placeholder_parts": list(self.placeholder_parts),
            "errors": list(self.errors),
            "valid": self.valid,
        }


def split_womd_spec(spec: str) -> tuple[WomdPathPart, ...]:
    parts: list[WomdPathPart] = []
    for raw in (x.strip() for x in str(spec or "").split(",")):
        if not raw:
            continue
        match = _SHARD_SPEC_RE.match(raw)
        if match:
            parts.append(WomdPathPart(raw=raw, prefix=match.group("prefix"), shard_count=int(match.group("count"))))
        else:
            parts.append(WomdPathPart(raw=raw, prefix=raw, shard_count=None))
    return tuple(parts)


def is_placeholder_path(text: str) -> bool:
    low = str(text or "").strip().lower()
    return not low or low.startswith("@") or any(token in low for token in _PLACEHOLDER_PARTS)


def sharded_filename(prefix: str, index: int, count: int) -> str:
    width = max(5, len(str(max(count - 1, 0))), len(str(count)))
    return f"{prefix}-{index:0{width}d}-of-{count:0{width}d}"


def _resolve_one(part: WomdPathPart) -> tuple[list[str], list[str], list[str]]:
    files: list[str] = []
    missing: list[str] = []
    errors: list[str] = []
    if part.shard_count is not None:
        for i in range(part.shard_count):
            expected = sharded_filename(part.prefix, i, part.shard_count)
            if Path(expected).is_file():
                files.append(expected)
                continue
            # Accommodate compressed/side-suffixed TFRecord shards while keeping
            # deterministic ordering.  Index sidecars are deliberately excluded.
            matches = [
                p for p in sorted(glob.glob(expected + "*"))
                if Path(p).is_file() and not p.endswith((".index", ".crc"))
            ]
            if len(matches) == 1:
                files.append(matches[0])
            elif len(matches) > 1:
                errors.append(f"ambiguous shard {expected!r}: {matches[:5]!r}")
            else:
                missing.append(expected)
        return files, missing, errors

    if any(c in part.prefix for c in "*?["):
        files.extend(p for p in sorted(glob.glob(part.prefix)) if Path(p).is_file())
    elif Path(part.prefix).is_file():
        files.append(part.prefix)
    else:
        # A bare prefix is not silently treated as a valid shard set.  Report
        # nearby shard files to make the missing @N suffix obvious.
        nearby = [p for p in sorted(glob.glob(part.prefix + "-*-of-*")) if Path(p).is_file()]
        if nearby:
            m = _SHARD_FILE_RE.match(nearby[0])
            count = int(m.group("count")) if m else None
            errors.append(
                f"bare shard prefix {part.prefix!r}; use {part.prefix!r}@{count or '<num_shards>'}"
            )
            files.extend(nearby)
        else:
            missing.append(part.prefix)
    return files, missing, errors


def resolve_womd_spec(spec: str) -> WomdPathResolution:
    parts = split_womd_spec(spec)
    files: list[str] = []
    missing: list[str] = []
    errors: list[str] = []
    placeholders = [p.raw for p in parts if is_placeholder_path(p.prefix)]
    if not parts:
        errors.append("empty WOMD path specification")
    for part in parts:
        if part.raw in placeholders:
            continue
        f, m, e = _resolve_one(part)
        files.extend(f); missing.extend(m); errors.extend(e)
    # Detect duplicate paths because duplicated shards cause duplicated scenes.
    seen: set[str] = set(); duplicates: list[str] = []
    unique: list[str] = []
    for p in files:
        if p in seen:
            duplicates.append(p)
        else:
            seen.add(p); unique.append(p)
    if duplicates:
        errors.append(f"duplicate WOMD files in specification: {duplicates[:5]!r}")
    return WomdPathResolution(
        spec=str(spec or ""),
        parts=parts,
        files=tuple(unique),
        missing_files=tuple(missing),
        placeholder_parts=tuple(placeholders),
        errors=tuple(errors),
    )


def ensure_sharded_spec(spec: str, shard_count: int = 150) -> str:
    """Append ``@shard_count`` only to a single bare shard prefix.

    Explicit sharded specs, comma lists, glob patterns and concrete files are
    returned unchanged.  This helper exists for backwards-compatible shell
    inputs; new commands should pass ``...tfrecord@150`` explicitly.
    """
    text = str(spec or "").strip()
    if not text or "," in text or _SHARD_SPEC_RE.match(text) or any(c in text for c in "*?["):
        return text
    if Path(text).is_file() or _SHARD_FILE_RE.match(text):
        return text
    return f"{text}@{int(shard_count)}"


def infer_shard_count(paths: Iterable[str]) -> int | None:
    counts: set[int] = set()
    for path in paths:
        match = _SHARD_FILE_RE.match(str(path))
        if match:
            counts.add(int(match.group("count")))
    return next(iter(counts)) if len(counts) == 1 else None
