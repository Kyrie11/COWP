"""CARLA 0.9.16 map-grounded scenario *screening* constructors.

The constructors use CARLA's actual lane waypoints. A setup is not a real-world
failure, and the investigator must inspect per-scenario validity and trajectories.
Unsupported maps are explicitly skipped, not 'negative' scientific evidence.
"""
from __future__ import annotations
import math, random


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
    return None


def convoy_candidate(wp,carla):
    """Three vehicles in one uninterrupted, nearly straight driving corridor."""
    if not _driving(wp,carla):return None
    b=_one(wp.previous(18.));e=_one(wp.next(20.));end=_one(wp.next(100.))
    if not all(_driving(x,carla) for x in (b,e,end)):return None
    if not (wp.road_id==e.road_id==end.road_id==b.road_id and
            wp.lane_id==e.lane_id==end.lane_id==b.lane_id):return None
    if not all(yaw_diff(wp.transform.rotation.yaw,x.transform.rotation.yaw)<12.
               for x in (e,end,b)):return None
    # Check intermediate lane path for side streets/junctions rather than only
    # comparing the first and final waypoint heading.
    cur=wp
    for _ in range(20):
        cur=_one(cur.next(5.)) if cur else None
        if not _driving(cur,carla) or cur.road_id!=wp.road_id or cur.lane_id!=wp.lane_id:
            return None
    return {'ego':e,'follower_a':wp,'follower_b':b,'end':end}


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


def build_anchors(world, archetype, count, seed):
    """Use WAYPOINTS, not arbitrary world coordinates; no forced collision geometry."""
    if archetype not in ('cut_in','convoy','intersection','support_gap'):
        raise ValueError('Unknown case archetype: '+archetype)
    import carla
    kind='convoy' if archetype=='support_gap' else archetype
    find={'cut_in':cut_in_candidate,'convoy':convoy_candidate,'intersection':None}[kind]
    if kind=='intersection':
        incoming=[wp for wp in world.get_map().generate_waypoints(4.) if _driving(wp,carla)
                  and any(x.is_junction for x in wp.next(14.))]
        pts=[wp.transform for wp in incoming]
    else:
        incoming=None
        pts=world.get_map().get_spawn_points()
    random.Random(seed).shuffle(pts)
    rows=[]; used=[]
    for tr in pts:
        wp=world.get_map().get_waypoint(tr.location,project_to_road=True,lane_type=carla.LaneType.Driving)
        case=junction_candidate(wp,world,carla,incoming) if kind=='intersection' else find(wp,carla)
        if case is None:continue
        c=case['follower_a'].transform.location
        key=[float(c.x),float(c.y),float(c.z)]
        if any(math.dist(key,x)<65. for x in used):continue
        used.append(key)
        coords={k:[round(getattr(v.transform.location,axis),3) for axis in ('x','y','z')]
                for k,v in case.items() if not k.startswith('_')}
        identities={k:{'road_id':int(v.road_id),'lane_id':int(v.lane_id),
                        'yaw':round(float(v.transform.rotation.yaw),3)}
                    for k,v in case.items() if not k.startswith('_')}
        rows.append({'anchor':len(rows),'key':key,'coords':coords,'waypoint_identities':identities,
                     'archetype':archetype,'target_lane_id':case.get('_target_lane_id'),
                     'map':world.get_map().name})
        if len(rows)>=count:break
    return rows


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
    if archetype=='cut_in':
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
    return min(ahead,key=lambda v:yaw_diff(v.transform.rotation.y,wp.transform.rotation.yaw))
