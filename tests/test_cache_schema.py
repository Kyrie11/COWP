from __future__ import annotations

from cowp.label.label_engine import build_labels_for_scene
from cowp.data.cache_schema import validate_schema, validate_numeric_invariants


def test_schema_for_toy_scene(toy_scene, cfg):
    label = build_labels_for_scene(toy_scene, cfg)
    errors = validate_schema(label, cfg) + validate_numeric_invariants(label, cfg)
    assert errors == []
