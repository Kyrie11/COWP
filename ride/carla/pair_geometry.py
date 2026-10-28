"""Strict geometry for longitudinal TTC, including OpenDRIVE road boundaries.

Both agents must lie close to the *same frozen physical lane centerline*. Merely
having equal lane IDs on different roads is never sufficient. Used only for
longitudinal, locally aligned follower pairs; junctions remain excluded.
"""
from __future__ import annotations
import math


def longitudinal_pair_eligible(front_wp, rear_wp, front_xyz, rear_xyz, *,
                               target_corridor_xy=None, max_lateral_m=1.25):
    if front_wp is None or rear_wp is None:
        return False
    yaw1=float(front_wp.transform.rotation.yaw)
    yaw2=float(rear_wp.transform.rotation.yaw)
    yawdiff=abs((yaw1-yaw2+180.0)%360.0-180.0)
    if yawdiff>=18.0 or front_wp.lane_id!=rear_wp.lane_id:
        return False
    if bool(front_wp.is_junction) or bool(rear_wp.is_junction):
        return False
    if target_corridor_xy is None:
        return front_wp.road_id==rear_wp.road_id
    from .lane_geometry import distance_to_polyline
    return all(math.isfinite(v) and v <= max_lateral_m for v in (
        distance_to_polyline(float(front_xyz.x),float(front_xyz.y),target_corridor_xy),
        distance_to_polyline(float(rear_xyz.x),float(rear_xyz.y),target_corridor_xy)))
