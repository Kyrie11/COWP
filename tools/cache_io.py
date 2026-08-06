#!/usr/bin/env python3
"""Robust readers for COWP NPZ label/tensor caches.

The project has evolved through several schemas, so this module first resolves
candidate-level arrays using aliases and conservative lexical matching.  Use
explicit --*-key overrides whenever the schema report identifies ambiguity.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping
import gzip
import io
import re

import numpy as np


FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "scenario_id": (
        "scenario_id", "scenario/id", "scenario/id_bytes", "scene_id", "id",
    ),
    "valid": (
        "candidate_valid", "candidate_valid_mask", "candidates/valid",
        "cowp/candidates/valid", "planner/candidate_valid",
        "ego/candidates/valid", "ego_candidate_valid",
    ),
    "conventional_safe": (
        "candidate_conventional_safe", "candidates/conventional_safe",
        "cowp/candidates/conventional_safe", "candidate_physical_safe",
        "candidate_safe_physical", "candidate_label_conventional_safe",
    ),
    "conventional_unsafe": (
        "candidate_conventional_unsafe", "candidates/conventional_unsafe",
        "cowp/candidates/conventional_unsafe", "candidate_physical_unsafe",
        "offline_conventional_unsafe", "candidate_unsafe_physical",
    ),
    "ncf": (
        "candidate_ncf", "candidate_noncoercive_feasible",
        "candidate_non_coercive_feasible", "candidates/noncoercive_feasible",
        "cowp/candidates/noncoercive_feasible", "candidate_label_ncf",
        "candidate_priority_aware_ncf",
    ),
    "priority_eligible": (
        "candidate_priority_eligible", "candidates/priority_eligible",
        "cowp/candidates/priority_eligible", "candidate_has_protected_relation",
        "candidate_protected_eligible",
    ),
    "priority_ncf": (
        "candidate_priority_ncf", "candidate_protected_ncf",
        "candidate_priority_noncoercive_feasible",
        "candidates/priority_noncoercive_feasible",
        "cowp/candidates/priority_noncoercive_feasible",
    ),
    "source": (
        "candidate_source", "candidate_proposal_source", "proposal_source",
        "candidates/source", "cowp/candidates/source",
        "candidate_provenance_source", "candidate_generator_source",
    ),
}

POSITIVE_TOKENS: dict[str, tuple[str, ...]] = {
    "valid": ("candidate", "valid"),
    "conventional_safe": ("candidate", "conventional", "safe"),
    "conventional_unsafe": ("candidate", "conventional", "unsafe"),
    "ncf": ("candidate", "ncf"),
    "priority_eligible": ("candidate", "priority", "eligible"),
    "priority_ncf": ("candidate", "priority", "ncf"),
    "source": ("candidate", "source"),
}
NEGATIVE_TOKENS = (
    "pred", "prob", "logit", "score", "loss", "metric", "selected",
    "accepted", "shortlist", "waymax", "replay",
)


class CacheSchemaError(RuntimeError):
    pass


def normalize_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")


def discover_npz_files(cache_dir: Path) -> list[Path]:
    files = sorted(cache_dir.rglob("*.npz"))
    files += sorted(cache_dir.rglob("*.npz.gz"))
    # Deduplicate if a broad glob ever overlaps.
    return list(dict.fromkeys(files))


def load_npz(path: Path) -> dict[str, np.ndarray]:
    try:
        if path.name.endswith(".npz.gz"):
            with gzip.open(path, "rb") as f:
                payload = io.BytesIO(f.read())
            obj = np.load(payload, allow_pickle=True)
        else:
            obj = np.load(path, allow_pickle=True)
        with obj:
            return {k: obj[k] for k in obj.files}
    except Exception as exc:
        raise CacheSchemaError(f"failed to read {path}: {exc}") from exc


def _lexical_score(key: str, field: str) -> float:
    norm = normalize_key(key)
    tokens = set(norm.split("_"))
    required = POSITIVE_TOKENS[field]
    score = 0.0
    for tok in required:
        if tok in tokens:
            score += 3.0
        elif tok in norm:
            score += 1.0
    if field == "ncf" and ("noncoercive" in norm or "non_coercive" in norm):
        score += 8.0
    if field == "priority_ncf" and ("noncoercive" in norm or "non_coercive" in norm):
        score += 5.0
    if field == "conventional_safe" and "unsafe" in tokens:
        score -= 12.0
    if field == "conventional_unsafe" and "unsafe" in tokens:
        score += 7.0
    if field == "source" and ("provenance" in tokens or "generator" in tokens):
        score += 2.0
    for tok in NEGATIVE_TOKENS:
        if tok in tokens:
            score -= 2.5
    if "label" in tokens or "target" in tokens or "gt" in tokens:
        score += 1.0
    return score


def resolve_key(
    data: Mapping[str, np.ndarray],
    field: str,
    override: str | None = None,
    *,
    required: bool = False,
) -> str | None:
    if override:
        if override not in data:
            raise CacheSchemaError(
                f"explicit key {override!r} for {field} is absent; "
                f"available keys: {sorted(data)}"
            )
        return override

    norm_to_key = {normalize_key(k): k for k in data}
    for alias in FIELD_ALIASES[field]:
        if alias in data:
            return alias
        norm = normalize_key(alias)
        if norm in norm_to_key:
            return norm_to_key[norm]

    if field == "scenario_id":
        return None

    scored = sorted(
        ((_lexical_score(k, field), k) for k in data),
        reverse=True,
    )
    if scored and scored[0][0] >= 6.0:
        if len(scored) > 1 and abs(scored[0][0] - scored[1][0]) < 1e-9:
            if required:
                raise CacheSchemaError(
                    f"ambiguous {field} keys: {scored[:4]}; use an explicit override"
                )
            return None
        return scored[0][1]

    if required:
        raise CacheSchemaError(
            f"could not resolve required field {field}; available keys: {sorted(data)}"
        )
    return None


def _decode_scalar(value: Any) -> str:
    arr = np.asarray(value)
    if arr.size != 1:
        raise CacheSchemaError(f"scenario id is not scalar: shape={arr.shape}")
    item = arr.reshape(-1)[0]
    if isinstance(item, bytes):
        return item.decode("utf-8", errors="replace")
    return str(item)


def scenario_id_from(path: Path, data: Mapping[str, np.ndarray], key: str | None) -> str:
    if key is not None:
        return _decode_scalar(data[key])
    stem = path.name
    for suffix in (".npz.gz", ".npz"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem


def _candidate_vector(
    value: np.ndarray,
    field: str,
    *,
    expected_k: int | None = None,
) -> np.ndarray:
    arr = np.asarray(value)
    arr = np.squeeze(arr)
    if arr.ndim == 0:
        return np.asarray([bool(arr.item())], dtype=bool)
    if arr.ndim == 1:
        return np.asarray(arr != 0, dtype=bool)

    if expected_k is not None:
        axes = [i for i, n in enumerate(arr.shape) if n == expected_k]
        if len(axes) == 1:
            arr = np.moveaxis(arr, axes[0], 0)
            trailing = arr.shape[1:]
            if int(np.prod(trailing)) == 1:
                return np.asarray(arr.reshape(expected_k) != 0, dtype=bool)
            if field == "priority_eligible":
                return np.asarray(np.any(arr != 0, axis=tuple(range(1, arr.ndim))), dtype=bool)

    raise CacheSchemaError(
        f"{field} must be a candidate-level vector; got shape={arr.shape}. "
        "Do not silently reduce pair/root tensors. Provide the candidate-level label key."
    )


def _candidate_source_vector(value: np.ndarray, expected_k: int) -> np.ndarray:
    arr = np.asarray(value)
    arr = np.squeeze(arr)
    if arr.ndim != 1 or arr.shape[0] != expected_k:
        raise CacheSchemaError(
            f"source must be a length-K vector; got {arr.shape}, expected K={expected_k}"
        )
    out: list[str] = []
    for x in arr:
        if isinstance(x, bytes):
            out.append(x.decode("utf-8", errors="replace"))
        else:
            out.append(str(x))
    return np.asarray(out, dtype=object)


@dataclass
class SceneRecord:
    scenario_id: str
    valid: np.ndarray
    conventional_safe: np.ndarray
    ncf: np.ndarray
    priority_eligible: np.ndarray | None
    priority_ncf: np.ndarray | None
    source: np.ndarray | None
    resolved_keys: dict[str, str | None]

    def any_valid(self) -> bool:
        return bool(np.any(self.valid))

    def any_conventional_safe(self) -> bool:
        return bool(np.any(self.valid & self.conventional_safe))

    def any_ncf(self) -> bool:
        return bool(np.any(self.valid & self.ncf))

    def any_priority_eligible(self) -> bool | None:
        if self.priority_eligible is None:
            return None
        return bool(np.any(self.valid & self.priority_eligible))

    def any_priority_ncf(self) -> bool | None:
        if self.priority_ncf is None:
            return None
        return bool(np.any(self.valid & self.priority_ncf))

    def ncf_sources(self) -> set[str]:
        if self.source is None:
            return set()
        mask = self.valid & self.ncf
        return {str(x) for x in self.source[mask]}


def infer_scene_record(
    path: Path,
    data: Mapping[str, np.ndarray],
    overrides: Mapping[str, str | None] | None = None,
) -> SceneRecord:
    overrides = dict(overrides or {})
    keys: dict[str, str | None] = {}
    keys["scenario_id"] = resolve_key(data, "scenario_id", overrides.get("scenario_id"))
    keys["valid"] = resolve_key(data, "valid", overrides.get("valid"), required=True)
    valid = _candidate_vector(data[keys["valid"]], "valid")
    k = int(valid.shape[0])

    keys["conventional_safe"] = resolve_key(
        data, "conventional_safe", overrides.get("conventional_safe")
    )
    keys["conventional_unsafe"] = resolve_key(
        data, "conventional_unsafe", overrides.get("conventional_unsafe")
    )
    if keys["conventional_safe"] is not None:
        conventional_safe = _candidate_vector(
            data[keys["conventional_safe"]], "conventional_safe", expected_k=k
        )
    elif keys["conventional_unsafe"] is not None:
        conventional_unsafe = _candidate_vector(
            data[keys["conventional_unsafe"]], "conventional_unsafe", expected_k=k
        )
        conventional_safe = ~conventional_unsafe
    else:
        raise CacheSchemaError(
            "neither a candidate-level conventional-safe nor conventional-unsafe label was found"
        )

    keys["ncf"] = resolve_key(data, "ncf", overrides.get("ncf"), required=True)
    ncf = _candidate_vector(data[keys["ncf"]], "ncf", expected_k=k)

    keys["priority_eligible"] = resolve_key(
        data, "priority_eligible", overrides.get("priority_eligible")
    )
    priority_eligible = (
        _candidate_vector(
            data[keys["priority_eligible"]], "priority_eligible", expected_k=k
        )
        if keys["priority_eligible"] is not None
        else None
    )

    keys["priority_ncf"] = resolve_key(
        data, "priority_ncf", overrides.get("priority_ncf")
    )
    priority_ncf = (
        _candidate_vector(data[keys["priority_ncf"]], "priority_ncf", expected_k=k)
        if keys["priority_ncf"] is not None
        else None
    )

    keys["source"] = resolve_key(data, "source", overrides.get("source"))
    source = (
        _candidate_source_vector(data[keys["source"]], expected_k=k)
        if keys["source"] is not None
        else None
    )

    lengths = {
        "valid": valid.shape[0],
        "conventional_safe": conventional_safe.shape[0],
        "ncf": ncf.shape[0],
    }
    if len(set(lengths.values())) != 1:
        raise CacheSchemaError(f"candidate vector length mismatch: {lengths}")

    return SceneRecord(
        scenario_id=scenario_id_from(path, data, keys["scenario_id"]),
        valid=valid,
        conventional_safe=conventional_safe,
        ncf=ncf,
        priority_eligible=priority_eligible,
        priority_ncf=priority_ncf,
        source=source,
        resolved_keys=keys,
    )


def parse_key_overrides(args: Any) -> dict[str, str | None]:
    return {
        "scenario_id": getattr(args, "scenario_id_key", None),
        "valid": getattr(args, "valid_key", None),
        "conventional_safe": getattr(args, "conventional_safe_key", None),
        "conventional_unsafe": getattr(args, "conventional_unsafe_key", None),
        "ncf": getattr(args, "ncf_key", None),
        "priority_eligible": getattr(args, "priority_eligible_key", None),
        "priority_ncf": getattr(args, "priority_ncf_key", None),
        "source": getattr(args, "source_key", None),
    }


def add_key_override_arguments(parser: Any) -> None:
    parser.add_argument("--scenario-id-key")
    parser.add_argument("--valid-key")
    parser.add_argument("--conventional-safe-key")
    parser.add_argument("--conventional-unsafe-key")
    parser.add_argument("--ncf-key")
    parser.add_argument("--priority-eligible-key")
    parser.add_argument("--priority-ncf-key")
    parser.add_argument("--source-key")
