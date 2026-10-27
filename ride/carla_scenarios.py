"""CARLA 0.9.16 map-grounded scenario *screening* constructors.

The constructors use CARLA's actual lane waypoints. A setup is not a real-world
failure, and the investigator must inspect per-scenario validity and trajectories.
Unsupported maps are explicitly skipped, not 'negative' scientific evidence.
"""
from __future__ import annotations
import math, random

# The frozen anchor protocol changed: old spawn-only manifests cannot be
# silently reused in a revised map-search experiment.
# v3 separates support_gap from the historical convoy alias and includes the
# CARLA 0.9.16 yaw fix.  Old frozen anchors must be regenerated.
ANCHOR_PROTOCOL_VERSION = "ride_lane_corridor_v3_20261027"


def yaw_diff(a, b): return abs((float(a)-float(b)+180.)%360.-180.)


def _one(items):return items[0] if len(items)==1 else None


def _driving(wp,carla):return (wp is not None and wp.lane_type==carla.LaneType.Driving and not wp.is_junction)


def _same_dir(wp,other):return yaw_diff(wp.transform.rotation.yaw,other.transform.rotation.yaw)<18.


def _xy(a,b):return math.hypot(a.x-b.x,a.y-b.y)


def cut_in_candidate(wp,carla):
    """Ego in an adjacent lane, other vehicles in target lane. No initial overlap."""
    if not _driving(wp,carla):return None
    follower_b=_one(wp.previous(18.))
    future=_one(wp.next(75.))
    front=_one(wp.next(12.))
    if not all(_driving(v,carla) for v in [follower_b,future,front]):return None
    for neighbor in (front.get_left_lane(),front.get_right_lane()):
        if not _driving(neighbor,carla):continue
        if neighbor.road_id!=wp.road_id or not _same_dir(neighbor,wp):continue
        # Normal neighboring driving lanes are only ~3.0-4.0m apart.
        # The previous 5m minimum silently rejected ordinary cut-in anchors.
        lateral=_xy(neighbor.transform.location,front.transform.location)
        if not 2.4<=lateral<=6.5:continue
        return {'ego':neighbor,'follower_a':wp,'follower_b':follower_b,'end':future,
                '_target_lane_id':wp.lane_id}


def support_gap_candidate(wp, carla):
    """Return a real adjacent-lane gap/merge anchor.

    The finite response-bank experiment needs the ego to enter a target lane
    between a front vehicle and a following pair.  It must not reuse the
    straight convoy geometry, because that would make a ``support_gap`` label
    with no gap or merge mechanism.  The same local lane construction is used
    as ``cut_in_candidate`` so that the two scenarios differ only in the
    downstream finite-response protocol.
    """
    return cut_in_candidate(wp, carla)
    return None


def _unique_lane_step(wp, distance, carla, *, backward=False):
    """Advance only if there is exactly one continuous, drivable lane path.

    Large `next(100)` jumps can cross intersections/branching and omit the
    intermediate lane geometry. Small steps explicitly reject such routes.
    """
    candidates = wp.previous(distance) if backward else wp.next(distance)
    if len(candidates) != 1:
        return None, 'lane_end_or_branch'
    nxt = candidates[0]
    if not _driving(nxt, carla):
        return None, 'junction_or_non_driving'
    if nxt.lane_id != wp.lane_id:
        return None, 'lane_identity_changed'
    if yaw_diff(nxt.transform.rotation.yaw, wp.transform.rotation.yaw) > 8.:
        return None, 'sharp_local_bend'
    return nxt, None


def convoy_candidate(wp, carla, rejection=None):
    """Find an actual 118m lane corridor for S2/S4, including a 100m goal.

    Relative roles (meters along the driving lane) stay identical to the
    original protocol: B=-18, A=0, ego=+20, goal=+100. We do *not* shorten
    the goal, since that would change responder behavior and the experiment.
    OpenDRIVE may change road_id on a continuous lane; the initial trailing
    pair must share a road for interpretable same-lane TTC, while the future
    goal is checked by physical lane continuity rather than one road_id.
    """
    def reject(reason):
        if rejection is not None:
            rejection[reason] = rejection.get(reason, 0) + 1
        return None

    if not _driving(wp, carla):
        return reject('not_nonjunction_driving_lane')
    # Check behind A too: a single `previous(18)` could jump over a junction.
    back = wp
    for _ in range(6):
        back, why = _unique_lane_step(back, 3., carla, backward=True)
        if why:
            return reject('rear_' + why)
        if back.road_id != wp.road_id:
            return reject('rear_road_id_changed')
    # Step all the way to the goal, so no branch, junction, or hidden
    # 90-degree turn can exist between the three vehicles and the goal.
    cur = wp
    ego = None
    for i in range(1, 21):
        cur, why = _unique_lane_step(cur, 5., carla)
        if why:
            return reject('forward_' + why)
        # The interacting pair and ego must begin on one unambiguous road.
        # A future map road boundary alone should not invalidate the route.
        if i <= 8 and cur.road_id != wp.road_id:
            return reject('initial_corridor_road_id_changed')
        if yaw_diff(cur.transform.rotation.yaw, wp.transform.rotation.yaw) > 25.:
            return reject('corridor_curvature_over_25deg')
        if i == 4:
            ego = cur
    # Projection-based progress/TTC assumes the corridor is not a U-turn.
    if ego is None or _xy(ego.transform.location, wp.transform.location) < 18.:
        return reject('ego_position_degenerate')
    if _xy(cur.transform.location, ego.transform.location) < 70.:
        return reject('goal_position_degenerate')
    return {'ego': ego, 'follower_a': wp, 'follower_b': back, 'end': cur}


def junction_candidate(wp,world,carla,incoming=None):
    """Two crossing approach lanes leading to the same junction neighborhood.

    This is explicitly exploratory: it does not assume crossing maneuvers are
    legal, or that the generated route reaches the crossing region. These are
    *trial validity* checks, not a self-validating failure dataset.
    """
    if not _driving(wp,carla):return None
    ahead=wp.next(14.)
    if not ahead or not any(x.is_junction for x in ahead):return None
    ja=min(ahead,key=lambda x:_xy(x.transform.location,wp.transform.location))
    candidates=incoming if incoming is not None else world.get_map().generate_waypoints(5.)
    for x in candidates:
        if not _driving(x,carla):continue
        xs=x.next(14.)
        if not xs or not any(y.is_junction for y in xs):continue
        a=next(y for y in xs if y.is_junction)
        if _xy(a.transform.location,ja.transform.location)>14.:continue
        if not 65.<yaw_diff(x.transform.rotation.yaw,wp.transform.rotation.yaw)<115.:continue
        b=_one(x.previous(17.))
        ego_options=wp.next(85.)
        follower_options=x.next(85.)
        ego_end=min(ego_options,key=lambda z:yaw_diff(z.transform.rotation.yaw,wp.transform.rotation.yaw)) if ego_options else None
        follower_end=min(follower_options,key=lambda z:yaw_diff(z.transform.rotation.yaw,x.transform.rotation.yaw)) if follower_options else None
        if b is None or ego_end is None or follower_end is None:continue
        if _xy(wp.transform.location,x.transform.location)<10.:continue
        return {'ego':wp,'follower_a':x,'follower_b':b,'end':ego_end,'end_follower':follower_end}
    return None


def build_anchors(world, archetype, count, seed, *, with_diagnostics=False):
    """Select map-grounded anchors, checking all candidate road geometry.

    First check recommended spawn points; if insufficient, search the map's
    dense driving-lane waypoints. Spawn points are NOT an exhaustive road map.
    The result and rejection counts are deterministic for a fixed map and seed.
    """
    if archetype not in ('cut_in', 'convoy', 'intersection', 'support_gap'):
        raise ValueError('Unknown case archetype: ' + archetype)
    import carla
    road_map = world.get_map()
    kind = archetype
    find = {'cut_in':cut_in_candidate, 'support_gap':support_gap_candidate,
            'convoy':convoy_candidate, 'intersection':None}[kind]
    rng = random.Random(seed)
    diagnostics = {'map':road_map.name, 'archetype':archetype,
                   'spawn_points_scanned':0, 'map_waypoints_scanned':0,
                   'rejections':{}, 'anchors_found':0}
    rows = []; used = []

    def try_candidate(wp):
        if kind == 'intersection':
            return junction_candidate(wp,world,carla,incoming)
        if kind == 'convoy':
            return convoy_candidate(wp,carla,diagnostics['rejections'])
        return find(wp,carla)

    def add(case):
        center = case['follower_a'].transform.location
        key = [float(center.x),float(center.y),float(center.z)]
        if any(math.dist(key,x) < 65. for x in used):
            return
        used.append(key)
        coords = {k:[round(getattr(v.transform.location,axis),3)
                     for axis in ('x','y','z')]
                  for k,v in case.items() if not k.startswith('_')}
        identities = {k:{'road_id':int(v.road_id),'lane_id':int(v.lane_id),
                          'yaw':round(float(v.transform.rotation.yaw),3)}
                      for k,v in case.items() if not k.startswith('_')}
        rows.append({'anchor':len(rows),'key':key,'coords':coords,
                     'waypoint_identities':identities,
                     'archetype':archetype,'target_lane_id':case.get('_target_lane_id'),
                     'map':road_map.name})

    if kind == 'intersection':
        incoming=[wp for wp in road_map.generate_waypoints(4.)
                  if _driving(wp,carla) and any(x.is_junction for x in wp.next(14.))]
        candidates = list(incoming)
        rng.shuffle(candidates)
        for wp in candidates:
            diagnostics['map_waypoints_scanned'] += 1
            case = try_candidate(wp)
            if case is not None: add(case)
            if len(rows) >= count:break
    else:
        incoming=None
        spawn_points=list(road_map.get_spawn_points())
        rng.shuffle(spawn_points)
        for tr in spawn_points:
            diagnostics['spawn_points_scanned'] += 1
            wp = road_map.get_waypoint(tr.location,project_to_road=True,
                                       lane_type=carla.LaneType.Driving)
            case = try_candidate(wp)
            if case is not None:add(case)
            if len(rows) >= count:break
        if len(rows) < count:
            # CARLA's recommended spawn points are a *subset* of lanes; scan
            # physical road centerlines without changing scene geometry.
            road_waypoints=list(road_map.generate_waypoints(10.))
            road_waypoints.sort(key=lambda x:(x.road_id,x.lane_id,
                round(float(getattr(x,'s',0.)),2),
                round(float(x.transform.location.x),2),
                round(float(x.transform.location.y),2)))
            rng.shuffle(road_waypoints)
            for wp in road_waypoints:
                diagnostics['map_waypoints_scanned'] += 1
                case = try_candidate(wp)
                if case is not None:add(case)
                if len(rows) >= count:break
    diagnostics['anchors_found'] = len(rows)
    return (rows, diagnostics) if with_diagnostics else rows


def restore(world,anchor,carla):
    """Resolves stored physical waypoints after world reload; fails on mismatch."""
    scene={}
    for k,xyz in anchor['coords'].items():
        loc=carla.Location(x=xyz[0],y=xyz[1],z=xyz[2])
        wp=world.get_map().get_waypoint(loc,project_to_road=True,lane_type=carla.LaneType.Driving)
        if wp is None or _xy(wp.transform.location,loc)>2.:
            raise RuntimeError('Anchor waypoint failed reproducibility check: '+k)
        expected=anchor.get('waypoint_identities',{}).get(k)
        if not expected:
            raise RuntimeError('Frozen anchor missing waypoint road/lane ID: '+k)
        if (wp.road_id!=expected['road_id'] or wp.lane_id!=expected['lane_id'] or
                yaw_diff(wp.transform.rotation.yaw,expected['yaw'])>5.):
            raise RuntimeError('Frozen anchor road/lane or heading changed: '+k)
        scene[k]=wp
    return scene


def ego_target_waypoint(world,vehicle,scene,archetype,profile,t,carla):
    wp=world.get_map().get_waypoint(vehicle.get_location(),project_to_road=True,
                                    lane_type=carla.LaneType.Driving)
    if wp is None:raise RuntimeError('Ego exited supported drivable lane')
    if archetype in ('cut_in', 'support_gap'):
        switch={'early_slow':1.5,'late_slow':4.5,'mild_early':2.5,'mild_late':3.5,'cruise':9.0}[profile]
        target_lane_id=scene['follower_a'].lane_id
        if t>=switch and wp.lane_id!=target_lane_id:
            choices=[x for x in (wp.get_left_lane(),wp.get_right_lane()) if x and
                     x.lane_id==target_lane_id and _driving(x,carla) and _same_dir(x,wp)]
            if choices:
                nxt=_one(choices[0].next(9.))
                if nxt is not None:return nxt
    ahead=wp.next(8.)
    if not ahead:return wp
    # Choose the continuation with minimum direction change, explicitly not
    # claiming traffic-light compliance or a human-like navigation plan.
    # CARLA's Rotation exposes the heading as ``yaw``.  ``Rotation.y`` is
    # not a valid 0.9.16 attribute and caused every non-convoy scene to abort
    # before its first trace step (see evidence/carla/*_audit.json).
    return min(ahead,key=lambda v:yaw_diff(v.transform.rotation.yaw,wp.transform.rotation.yaw))
