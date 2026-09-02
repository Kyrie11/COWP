from __future__ import annotations

import re
from typing import Iterable


_VERSION_RE = re.compile(r"_v\d+$")
_PREFIXES = (
    "evidence_adapt_dev_",
    "evidence_adapt_train_",
    "certificate_pool_",
    "calibration_",
    "test_",
    "val_",
    "validation_",
    "train_",
    "training_",
)


def normalize_bucket_name(value: str | None) -> str:
    """Normalize a dataset/bucket label without discarding its provenance.

    Dataset folders such as ``evidence_adapt_dev_near_contact`` carry both a
    provenance prefix and a semantic regime.  Runtime selector overrides and
    contact physics must use the semantic regime while logs retain the original
    bucket string.
    """
    text = str(value or "").strip().lower().replace("-", "_")
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def canonical_regime_name(value: str | None) -> str | None:
    """Return ``safe``, ``near_contact`` or ``contact`` when unambiguous.

    ``near_contact`` is checked before ``contact`` so it can never be classified
    as a post-contact target merely because its name contains ``contact``.
    """
    norm = normalize_bucket_name(value)
    if not norm:
        return None
    if "near_contact" in norm or "near_collision" in norm:
        return "near_contact"
    if (
        "post_contact" in norm
        or "post_collision" in norm
        or norm == "contact"
        or norm.endswith("_contact")
        or norm.endswith("_collision")
    ):
        return "contact"
    if (
        norm == "safe"
        or norm == "normal"
        or norm.endswith("_safe")
        or norm.endswith("_normal")
    ):
        return "safe"
    return None


def bucket_aliases(value: str | None) -> list[str]:
    """Return stable aliases for selector/gamma lookup.

    The most specific spelling is kept first, followed by versionless and
    provenance-stripped forms, then the canonical regime.  This preserves any
    explicitly configured dataset override while guaranteeing that generic
    ``near_contact``/``contact`` entries work for adaptation-dev folders.
    """
    norm = normalize_bucket_name(value)
    if not norm:
        return []

    candidates: list[str] = [norm, _VERSION_RE.sub("", norm)]
    queue = list(candidates)
    for item in queue:
        for prefix in _PREFIXES:
            if item.startswith(prefix) and len(item) > len(prefix):
                candidates.append(item[len(prefix) :])

    regime = canonical_regime_name(norm)
    if regime:
        candidates.append(regime)
        if regime == "contact":
            candidates.extend(["post_contact", "post_collision"])
        elif regime == "safe":
            candidates.append("normal")

    # Existing configs sometimes use hyphens. Keep both forms at the end so an
    # exact underscore key remains preferred.
    candidates.extend(x.replace("_", "-") for x in list(candidates))

    out: list[str] = []
    for item in candidates:
        if item and item not in out:
            out.append(item)
    return out


def is_post_contact_bucket(value: str | None) -> bool:
    return canonical_regime_name(value) == "contact"


def first_mapping_value(mapping: dict, aliases: Iterable[str]):
    """Return the first non-empty bucket override, or ``None`` when absent."""
    for alias in aliases:
        if alias in mapping and mapping[alias] not in {None, ""}:
            return mapping[alias]
    return None
