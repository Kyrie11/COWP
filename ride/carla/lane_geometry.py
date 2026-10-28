"""Map-corridor geometry for verifying a physical lane change.

Unlike CARLA's OpenDRIVE road_id, a frozen polyline can cross road boundaries.
This checks vehicle center proximity only: vehicle footprint road legality remains
an independent, stricter check for a final scientific claim.
"""
from __future__ import annotations
import math


def distance_to_polyline(x: float, y: float, points) -> float:
    """Shortest distance to consecutive XY segments; fail closed for bad data."""
    if len(points) < 2:
        raise ValueError('Need at least two centerline points')
    best = float('inf')
    for a, b in zip(points[:-1], points[1:]):
        ax,ay=float(a[0]),float(a[1]);bx,by=float(b[0]),float(b[1])
        dx,dy=bx-ax,by-ay
        length2=dx*dx+dy*dy
        if length2 <= 1e-10:
            raise ValueError('Degenerate frozen lane segment')
        fraction=max(0.,min(1.,((x-ax)*dx+(y-ay)*dy)/length2))
        best=min(best,math.hypot(x-(ax+fraction*dx),y-(ay+fraction*dy)))
    return best


def on_centerline(x, y, points, *, tolerance_m=1.5):
    if not 0.0 < tolerance_m <= 2.5:
        raise ValueError('Unsafe lane-center tolerance')
    return distance_to_polyline(float(x),float(y),points) <= tolerance_m


def map_lane_center_valid(location, road_map, carla, tolerance_m=0.12):
    """Physical center lies within the nearest drivable lane ribbon.

    Does not assert that ALL vehicle corners lie on-road, nor give any special
    right-of-way allowance at intersections.  Both strict/tolerant checks are
    retained as separate observations.
    """
    wp=road_map.get_waypoint(location,project_to_road=True,lane_type=carla.LaneType.Driving)
    if wp is None or wp.lane_type!=carla.LaneType.Driving or wp.is_junction:
        return False, None
    pos=wp.transform.location
    delta=math.hypot(float(location.x-pos.x),float(location.y-pos.y))
    width=float(wp.lane_width)
    if width<=0 or not math.isfinite(width):
        return False, delta
    return delta <= width*0.5 + tolerance_m, delta


def project_to_polyline(x: float, y: float, points):
    """Return (station_m, offset_m, tangent_x, tangent_y) on ordered road centerline.

    Can cross OpenDRIVE road-ID changes. For an adjacent ego lane the station
    is a longitudinal progress proxy, not a certificate of being inside target
    lane. Ties use earliest station, so equal-distance projections are stable.
    """
    if points is None or len(points) < 2:
        raise ValueError('Need at least two centerline points')
    x,y=float(x),float(y)
    if not all(math.isfinite(v) for v in (x,y)):
        raise ValueError('Nonfinite projection query')
    best=None; acc=0.
    for a,b in zip(points[:-1],points[1:]):
        ax,ay=float(a[0]),float(a[1]);bx,by=float(b[0]),float(b[1])
        dx,dy=bx-ax,by-ay
        length=math.hypot(dx,dy)
        if length <= 1e-6 or not all(math.isfinite(z) for z in (ax,ay,bx,by,length)):
            raise ValueError('Invalid centerline segment')
        u=max(0.,min(1.,((x-ax)*dx+(y-ay)*dy)/(length*length)))
        d=math.hypot(x-(ax+u*dx),y-(ay+u*dy))
        station=acc+u*length
        candidate=(d,station,dx/length,dy/length)
        if best is None or candidate[:2]<best[:2]:best=candidate
        acc+=length
    return best[1],best[0],best[2],best[3]
