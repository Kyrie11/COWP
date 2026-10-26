from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from cowp.core.types import Lane, MapData
from cowp.geometry.boxes import normalize_angle
from cowp.geometry.map_projection import polyline_arc_length


@dataclass
class ConflictRegion:
    conflict_id: int
    conflict_type: str
    center_xy: np.ndarray
    radius: float
    involved_lane_ids: tuple[int, int]
    priority_rule_hint: str = "UNKNOWN"


def _ccw(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
    return bool((c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0]))


def segment_intersection(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray | None:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    c = np.asarray(c, dtype=np.float32)
    d = np.asarray(d, dtype=np.float32)
    if _ccw(a, c, d) == _ccw(b, c, d) or _ccw(a, b, c) == _ccw(a, b, d):
        return None
    r = b - a
    s = d - c
    denom = r[0] * s[1] - r[1] * s[0]
    if abs(float(denom)) < 1e-8:
        return None
    t = ((c[0] - a[0]) * s[1] - (c[1] - a[1]) * s[0]) / denom
    return a + t * r




def _bbox_xy(xy: np.ndarray) -> tuple[float, float, float, float]:
    return (float(np.min(xy[:, 0])), float(np.min(xy[:, 1])), float(np.max(xy[:, 0])), float(np.max(xy[:, 1])))


def _bbox_might_overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float], margin: float) -> bool:
    return not (a[2] + margin < b[0] or b[2] + margin < a[0] or a[3] + margin < b[1] or b[3] + margin < a[1])


def _segment_bboxes(xy: np.ndarray) -> list[tuple[float, float, float, float]]:
    return [
        (
            float(min(xy[i, 0], xy[i + 1, 0])),
            float(min(xy[i, 1], xy[i + 1, 1])),
            float(max(xy[i, 0], xy[i + 1, 0])),
            float(max(xy[i, 1], xy[i + 1, 1])),
        )
        for i in range(max(0, len(xy) - 1))
    ]


def lane_heading_at(lane: Lane, s_query: float | None = None) -> float:
    xy = lane.xy
    if len(xy) < 2:
        return 0.0
    if s_query is None:
        j = 0
    else:
        arcs = polyline_arc_length(xy)
        j = int(np.clip(np.searchsorted(arcs, s_query) - 1, 0, len(xy) - 2))
    d = xy[j + 1] - xy[j]
    return float(np.arctan2(d[1], d[0]))


def _lane_min_distance_to_point(lane: Lane, point_xy: np.ndarray) -> float:
    xy = np.asarray(lane.xy, dtype=np.float32)
    if xy.ndim != 2 or len(xy) == 0:
        return float("inf")
    p = np.asarray(point_xy, dtype=np.float32).reshape(-1)[:2]
    return float(np.min(np.linalg.norm(xy[:, :2] - p[None, :], axis=-1)))


def _region_reference_rank(
    region: ConflictRegion,
    reference_xy: np.ndarray | None,
    reference_heading: float | None,
    *,
    behind_tolerance_m: float,
) -> tuple[float, ...]:
    """Causal ego-centric rank for a map conflict region.

    The old implementation returned as soon as the raw map-feature traversal hit
    ``max_conflict_regions``.  WOMD does not promise that map-feature insertion
    order is a meaningful planning order, so that behavior can make a fixed 64
    slot bank mostly unrelated to the SDC's reachable conflicts.  Ranking uses
    only the current SDC pose (no logged future / sdc_paths), and is therefore
    valid both for offline construction and online planning.
    """
    if reference_xy is None:
        return (0.0, float(region.conflict_id))
    ref = np.asarray(reference_xy, dtype=np.float32).reshape(-1)[:2]
    d = np.asarray(region.center_xy, dtype=np.float32)[:2] - ref
    dist = float(np.linalg.norm(d))
    behind = 0.0
    forward = 0.0
    lateral = dist
    if reference_heading is not None and np.isfinite(reference_heading):
        h = float(reference_heading)
        fwd = np.asarray([np.cos(h), np.sin(h)], dtype=np.float32)
        left = np.asarray([-np.sin(h), np.cos(h)], dtype=np.float32)
        forward = float(np.dot(d, fwd))
        lateral = abs(float(np.dot(d, left)))
        behind = 1.0 if forward < -float(behind_tolerance_m) else 0.0
    type_rank = {"MERGE": 0.0, "LANE_INTERSECTION": 1.0, "SWEPT_OVERLAP": 2.0}.get(
        str(region.conflict_type), 3.0
    )
    # Euclidean distance is the dominant causal relevance signal.  Forward/lateral
    # terms make ties stable without assuming a logged route.
    return (behind, dist, max(-forward, 0.0), lateral, type_rank, float(region.conflict_id))


def _dedupe_conflict_regions(regions: list[ConflictRegion], tol_m: float) -> list[ConflictRegion]:
    if tol_m <= 0.0:
        return list(regions)
    kept: list[ConflictRegion] = []
    by_key: dict[tuple[str, tuple[int, int]], list[np.ndarray]] = {}
    for r in regions:
        lanes = tuple(sorted((int(r.involved_lane_ids[0]), int(r.involved_lane_ids[1]))))
        key = (str(r.conflict_type), lanes)
        centers = by_key.setdefault(key, [])
        c = np.asarray(r.center_xy, dtype=np.float32)[:2]
        if any(float(np.linalg.norm(c - old)) <= tol_m for old in centers):
            continue
        centers.append(c)
        kept.append(r)
    return kept


def build_conflict_regions(
    map_data: MapData,
    cfg: dict,
    *,
    reference_xy: np.ndarray | None = None,
    reference_heading: float | None = None,
    diagnostics: dict[str, object] | None = None,
) -> list[ConflictRegion]:
    """Build a bounded, ego-relevant conflict-region bank.

    ``reference_xy/reference_heading`` should be the *current* SDC pose.  When
    present, lanes are considered in ego-centric order and the final fixed-size
    bank is ranked by current-pose relevance.  This eliminates dependence on raw
    WOMD map-feature ordering while preserving causality.  Callers that omit a
    reference keep deterministic legacy ordering, but no longer early-return at
    the first ``max_conflict_regions`` entries.
    """
    conflict_cfg = cfg.get("conflict", cfg)
    heading_thresh = np.deg2rad(float(conflict_cfg.get("lane_intersection_heading_threshold_deg", 30.0)))
    intersection_radius = float(conflict_cfg.get("intersection_radius_m", 5.0))
    merge_radius = float(conflict_cfg.get("merge_radius_m", 8.0))
    max_regions = int(cfg.get("limits", {}).get("max_conflict_regions", 64))
    pool_limit = max(
        max_regions,
        int(conflict_cfg.get("candidate_pool_max_regions", max(384, 6 * max_regions))),
    )
    dedup_m = float(conflict_cfg.get("dedup_center_distance_m", 1.5))
    behind_tol = float(conflict_cfg.get("reference_behind_tolerance_m", 12.0))
    max_pair_intersections = max(1, int(conflict_cfg.get("max_intersections_per_lane_pair", 4)))

    lanes = [lane for lane in map_data.lanes.values() if len(np.asarray(lane.xy)) >= 2]
    total_lane_count = len(lanes)
    if reference_xy is not None and lanes:
        lane_dist = {int(lane.lane_id): _lane_min_distance_to_point(lane, reference_xy) for lane in lanes}
        lanes.sort(key=lambda lane: (lane_dist[int(lane.lane_id)], int(lane.lane_id)))
        radius = float(conflict_cfg.get("reference_lane_radius_m", 180.0))
        min_lanes = max(2, int(conflict_cfg.get("reference_min_lanes_considered", 48)))
        max_lanes = max(min_lanes, int(conflict_cfg.get("reference_max_lanes_considered", 192)))
        near = [lane for lane in lanes if lane_dist[int(lane.lane_id)] <= radius]
        keep_n = min(len(lanes), max(min_lanes, min(max_lanes, len(near))))
        lanes = lanes[:keep_n]
    else:
        lane_dist = {}

    regions: list[ConflictRegion] = []
    lane_bboxes = {lane.lane_id: _bbox_xy(lane.xy) for lane in lanes}
    lane_segment_bboxes = {lane.lane_id: _segment_bboxes(lane.xy) for lane in lanes}
    bbox_margin = max(intersection_radius, merge_radius) + 2.0
    raw_intersections = 0
    pool_saturated = False

    for i, lane_a in enumerate(lanes):
        if len(regions) >= pool_limit:
            pool_saturated = True
            break
        xy_a = np.asarray(lane_a.xy, dtype=np.float32)
        for lane_b in lanes[i + 1 :]:
            if len(regions) >= pool_limit:
                pool_saturated = True
                break
            xy_b = np.asarray(lane_b.xy, dtype=np.float32)
            common_exit = set(lane_a.exit_lanes).intersection(lane_b.exit_lanes)
            endpoint_dist = float(np.linalg.norm(xy_a[-1] - xy_b[-1]))
            if not common_exit and endpoint_dist >= 4.0:
                if not _bbox_might_overlap(
                    lane_bboxes[lane_a.lane_id], lane_bboxes[lane_b.lane_id], bbox_margin
                ):
                    continue
            if common_exit or endpoint_dist < 4.0:
                center = 0.5 * (xy_a[-1] + xy_b[-1])
                regions.append(
                    ConflictRegion(
                        len(regions), "MERGE", center.astype(np.float32), merge_radius,
                        (lane_a.lane_id, lane_b.lane_id), "MAINLINE_OR_ARRIVAL",
                    )
                )
                continue

            seg_boxes_a = lane_segment_bboxes.get(lane_a.lane_id, [])
            seg_boxes_b = lane_segment_bboxes.get(lane_b.lane_id, [])
            pair_hits = 0
            for ia in range(len(xy_a) - 1):
                if pair_hits >= max_pair_intersections or len(regions) >= pool_limit:
                    break
                for ib in range(len(xy_b) - 1):
                    if pair_hits >= max_pair_intersections or len(regions) >= pool_limit:
                        break
                    if seg_boxes_a and seg_boxes_b and not _bbox_might_overlap(
                        seg_boxes_a[ia], seg_boxes_b[ib], 0.0
                    ):
                        continue
                    p = segment_intersection(xy_a[ia], xy_a[ia + 1], xy_b[ib], xy_b[ib + 1])
                    if p is None:
                        continue
                    ha = float(np.arctan2(*(xy_a[ia + 1] - xy_a[ia])[::-1]))
                    hb = float(np.arctan2(*(xy_b[ib + 1] - xy_b[ib])[::-1]))
                    hdiff = abs(float(normalize_angle(ha - hb)))
                    ctype = "LANE_INTERSECTION" if hdiff > heading_thresh else "SWEPT_OVERLAP"
                    regions.append(
                        ConflictRegion(
                            len(regions), ctype, p.astype(np.float32),
                            intersection_radius if ctype == "LANE_INTERSECTION" else merge_radius,
                            (lane_a.lane_id, lane_b.lane_id), "TOPOLOGY_OR_ARRIVAL",
                        )
                    )
                    pair_hits += 1
                    raw_intersections += 1

    raw_count = len(regions)
    regions = _dedupe_conflict_regions(regions, dedup_m)
    dedup_count = len(regions)
    if reference_xy is not None:
        regions.sort(
            key=lambda r: _region_reference_rank(
                r, reference_xy, reference_heading, behind_tolerance_m=behind_tol
            )
        )
    selected = regions[:max_regions]
    # Conflict ids are cache-local array indices.  Reassign after ranking so every
    # downstream proposal/label refers to the exact selected bank.
    for new_id, region in enumerate(selected):
        region.conflict_id = int(new_id)

    if diagnostics is not None:
        diagnostics.update({
            "total_map_lanes": int(total_lane_count),
            "lanes_considered": int(len(lanes)),
            "raw_region_count": int(raw_count),
            "deduplicated_region_count": int(dedup_count),
            "selected_region_count": int(len(selected)),
            "selected_cap_saturated": bool(len(selected) >= max_regions),
            "candidate_pool_saturated": bool(pool_saturated),
            "raw_segment_intersections": int(raw_intersections),
            "ego_reference_used": bool(reference_xy is not None),
        })
    return selected


def build_scene_conflict_regions(scene, cfg: dict, *, diagnostics: dict[str, object] | None = None) -> list[ConflictRegion]:
    """Scene wrapper that always uses the inference-visible current SDC pose."""
    cur = int(scene.current_time_index)
    ego = np.asarray(scene.states[int(scene.sdc_track_index), cur], dtype=np.float32)
    heading = float(ego[6]) if ego.size > 6 and np.isfinite(ego[6]) else None
    return build_conflict_regions(
        scene.map_data, cfg, reference_xy=ego[:2], reference_heading=heading, diagnostics=diagnostics
    )

def trajectory_entry_to_region(
    traj: np.ndarray,
    region: ConflictRegion,
    *,
    current_state: np.ndarray | None = None,
    radius: float | None = None,
    dt: float = 0.1,
) -> tuple[float, float]:
    """Continuous first-entry time and arc length to a circular conflict region.

    ``tta_to_region`` intentionally keeps its historical sample-index semantics.
    Interaction-timing proposal generation needs a stronger geometric contract:
    the distance used by the arrival solver must refer to the *same vehicle
    envelope / region boundary* used to declare a conflict.  This helper linearly
    interpolates the first boundary crossing between trajectory samples and
    returns both crossing time and travelled arc length from the observed current
    state.

    When ``current_state`` is omitted, the first trajectory sample is treated as
    time/distance zero.  Passing the current state is preferred for generated
    ego primitives, whose first row is one integration step in the future.
    """
    x = np.asarray(traj, dtype=np.float32)
    if x.ndim != 2 or len(x) == 0 or x.shape[1] < 7:
        return float("inf"), float("inf")
    dt = max(float(dt), 1.0e-6)
    r = float(radius if radius is not None else region.radius)

    if current_state is not None:
        cur = np.asarray(current_state, dtype=np.float32).reshape(-1)
        if cur.size < 2:
            return float("inf"), float("inf")
        start_xy = cur[:2]
        length0 = float(cur[7]) if cur.size > 7 and cur[7] > 0 else float(x[0, 5])
        width0 = float(cur[8]) if cur.size > 8 and cur[8] > 0 else float(x[0, 6])
        xy = np.concatenate([start_xy[None, :], x[:, :2]], axis=0)
        half_diag = np.concatenate(
            [
                np.asarray([0.5 * np.sqrt(max(length0, 0.1) ** 2 + max(width0, 0.1) ** 2)], dtype=np.float32),
                0.5 * np.sqrt(np.maximum(x[:, 5], 0.1) ** 2 + np.maximum(x[:, 6], 0.1) ** 2),
            ]
        )
        times = np.arange(len(xy), dtype=np.float32) * dt
    else:
        xy = x[:, :2]
        half_diag = 0.5 * np.sqrt(np.maximum(x[:, 5], 0.1) ** 2 + np.maximum(x[:, 6], 0.1) ** 2)
        times = np.arange(len(xy), dtype=np.float32) * dt

    dist = np.linalg.norm(xy - np.asarray(region.center_xy, dtype=np.float32)[None, :2], axis=-1)
    clearance = dist - (r + half_diag)
    if not np.all(np.isfinite(clearance)):
        return float("inf"), float("inf")

    seg = np.linalg.norm(np.diff(xy, axis=0), axis=-1) if len(xy) > 1 else np.zeros(0, dtype=np.float32)
    cumulative = np.concatenate([np.zeros(1, dtype=np.float32), np.cumsum(seg, dtype=np.float32)])
    if clearance[0] <= 0.0:
        return 0.0, 0.0
    hits = np.where(clearance <= 0.0)[0]
    if len(hits) == 0:
        return float("inf"), float("inf")
    i = int(hits[0])
    if i <= 0:
        return 0.0, 0.0
    c0 = float(clearance[i - 1])
    c1 = float(clearance[i])
    denom = c0 - c1
    frac = float(np.clip(c0 / denom, 0.0, 1.0)) if abs(denom) > 1.0e-9 else 1.0
    entry_t = float(times[i - 1] + frac * (times[i] - times[i - 1]))
    entry_s = float(cumulative[i - 1] + frac * seg[i - 1])
    return entry_t, entry_s


def tta_to_region(traj: np.ndarray, region: ConflictRegion, radius: float | None = None, dt: float = 0.1) -> float:
    if len(traj) == 0:
        return float("inf")
    r = float(radius if radius is not None else region.radius)
    half_diag = 0.5 * np.sqrt(np.maximum(traj[:, 5], 0.1) ** 2 + np.maximum(traj[:, 6], 0.1) ** 2)
    dist = np.linalg.norm(traj[:, :2] - region.center_xy[None, :], axis=-1)
    hit = np.where(dist < r + half_diag)[0]
    if len(hit) == 0:
        return float("inf")
    return float(hit[0] * dt)


def closest_conflict_for_pair(traj_a: np.ndarray, traj_b: np.ndarray, regions: Iterable[ConflictRegion], dt: float = 0.1) -> tuple[ConflictRegion | None, float, float, float]:
    best = (None, float("inf"), float("inf"), float("inf"))
    for region in regions:
        ta = tta_to_region(traj_a, region, dt=dt)
        tb = tta_to_region(traj_b, region, dt=dt)
        if np.isinf(ta) and np.isinf(tb):
            continue
        delta = abs(ta - tb)
        score = delta + 0.01 * min(ta, tb)
        if score < best[3]:
            best = (region, ta, tb, score)
    return best
