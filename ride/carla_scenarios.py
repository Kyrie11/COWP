"""CARLA 0.9.16 map-grounded scenario *screening* constructors.

The constructors use CARLA's actual lane waypoints. A setup is not a real-world
failure, and the investigator must inspect per-scenario validity and trajectories.
Unsupported maps are explicitly skipped, not 'negative' scientific evidence.
"""
from __future__ import annotations
import math, random

# The frozen anchor protocol changed: old spawn-only manifests cannot be
# silently reused in a revised map-search experiment.
# v5 uses a longer insertion gap, a continuous target-lane centerline and
# a common pre-policy warm-up. It is a NEW protocol, not a re-interpretation
# of old v4 outputs. All v4 anchors MUST be invalidated.
ANCHOR_PROTOCOL_VERSION = "ride_lane_corridor_v12_20261028"


def yaw_diff(a, b): return abs((float(a)-float(b)+180.)%360.-180.)


def _one(items):return items[0] if len(items)==1 else None


def _driving(wp,carla):return (wp is not None and wp.lane_type==carla.LaneType.Driving and not wp.is_junction)


def _same_dir(wp,other):return yaw_diff(wp.transform.rotation.yaw,other.transform.rotation.yaw)<18.


def _xy(a,b):return math.hypot(a.x-b.x,a.y-b.y)


def cut_in_candidate(wp, carla, ego_gap_m=20., rear_gap_m=45.):
    """Parametric adjacent-lane cut-in with physically verified gap geometry.

    Ego is -ego_gap_m behind A in an adjacent lane; B is -rear_gap_m
    behind A in the target lane. These are precommitted stress settings,
    never fit from post-intervention outcomes. Default reproduces v5 geometry.
    """
    if (ego_gap_m < 20. or rear_gap_m < ego_gap_m+15. or
        rear_gap_m > 70. or abs(ego_gap_m/5-round(ego_gap_m/5))>1e-8 or
        abs(rear_gap_m/5-round(rear_gap_m/5))>1e-8):
        raise ValueError('Cut-in gaps must be 5m multiples, ego>=20m, rear>=ego+15m, rear<=70m')
    if not _driving(wp, carla):
        return None
    n_ego=int(round(ego_gap_m/5))
    n_rear=int(round(rear_gap_m/5))
    backwards=[wp]
    for _ in range(n_rear):
        nxt=_one(backwards[-1].previous(5.0))
        if (not _driving(nxt,carla) or nxt.lane_id!=wp.lane_id or
                not _same_dir(nxt,wp)):
            return None
        backwards.append(nxt)
    middle,behind=backwards[n_ego],backwards[n_rear]
    # The 8s rollout at 9m/s can travel ~72m. Requiring only 35m of
    # nonjunction road in v3 allowed 2513/3233 discovery trials to touch a
    # junction, and BehaviorAgent could exhaust a dead-end local route.
    # Verify a full 90m continuous target corridor, including the endpoint.
    cur = wp
    forwards=[wp]
    for _ in range(18):
        nxt = _one(cur.next(5.0))
        if (not _driving(nxt, carla) or not _same_dir(nxt, wp) or
                nxt.lane_id != wp.lane_id):
            return None
        cur = nxt
        forwards.append(cur)
    end = cur
    # The neighbor must exist at the prospective ego insertion position.
    for neighbor in (middle.get_left_lane(), middle.get_right_lane()):
        if not _driving(neighbor, carla):
            continue
        if neighbor.road_id != wp.road_id or not _same_dir(neighbor, wp):
            continue
        lateral = _xy(neighbor.transform.location, middle.transform.location)
        if not 2.4 <= lateral <= 6.5:
            continue
        # Check the ego starting lane through the merge, not merely at one
        # map waypoint. Otherwise the lane can end mid-manoeuvre.
        ego_route=neighbor
        valid_ego_route=True
        for _ in range(12):
            future=_one(ego_route.next(5.0))
            if not _driving(future,carla) or not _same_dir(future,wp):
                valid_ego_route=False;break
            ego_route=future
        if not valid_ego_route:
            continue
        # Verify physical forward order with same-heading projection.
        angle = math.radians(float(wp.transform.rotation.yaw))
        hx, hy = math.cos(angle), math.sin(angle)
        def forward_gap(a, b):
            pa, pb = a.transform.location, b.transform.location
            return float((pa.x-pb.x)*hx + (pa.y-pb.y)*hy)
        if not (abs(forward_gap(wp, neighbor)-ego_gap_m)<=3. and
                abs(forward_gap(neighbor, behind)-(rear_gap_m-ego_gap_m))<=3.):
            continue
        # The target-lane centerline is much more stable than comparing the
        # frozen follower road_id with the ego's *current* OpenDRIVE road_id.
        centerline=list(reversed(backwards))+forwards[1:]
        corridor=[[float(v.transform.location.x),float(v.transform.location.y)]
                  for v in centerline]
        return {'ego': neighbor, 'follower_a': wp, 'follower_b': behind,
                'end': end, '_target_lane_id': wp.lane_id,
                '_target_corridor_xy':corridor,
                '_target_lane_width_m':float(getattr(wp,'lane_width',3.5))}
    return None


def chain_cut_in_candidate(wp, carla, ego_gap_m=20., rear_gap_m=45., chain_bc_gap_m=25.):
    """Four-car chain: L(A) -- B -- C share target lane, ego merges ahead of B.

    C is behind B at a pre-registered along-lane distance. Require a smooth,
    nonjunction CARLA lane to spawn C; never synthesize coordinates.
    This creates an *opportunity to test* secondary effects, not a positive
    witness. RQ2 primary pair is B-C, independent of ego collision handling.
    """
    if not 20. <= chain_bc_gap_m <= 45. or abs(chain_bc_gap_m / 5. - round(chain_bc_gap_m / 5.)) > 1e-8:
        raise ValueError('chain_bc_gap_m must be a 5m multiple in [20,45]')
    case = cut_in_candidate(wp, carla, ego_gap_m, rear_gap_m)
    if case is None:
        return None
    behind = case['follower_b']
    traversed=[]
    for _ in range(int(round(chain_bc_gap_m / 5.))):
        nxt = _one(behind.previous(5.))
        if (not _driving(nxt,carla) or nxt.lane_id != wp.lane_id or
            not _same_dir(nxt,wp)):
            return None
        behind = nxt
        traversed.append(behind)
    angle = math.radians(float(wp.transform.rotation.yaw))
    hx,hy = math.cos(angle),math.sin(angle)
    b = case['follower_b'].transform.location
    c = behind.transform.location
    projected = (b.x-c.x)*hx + (b.y-c.y)*hy
    if abs(projected - chain_bc_gap_m) > 3.:
        return None
    case['follower_c'] = behind
    # v12 route-topology preflight. The BehaviorAgent normal route on some
    # Town04 anchors contains only a few waypoints, then throws None.is_junction.
    # These are NOT scientific negatives. Require a unique, nonjunction,
    # >=75m follower corridor *for all three roles* before freezing.
    def has_route(role, meters=75.):
        cur=case[role] if role != 'follower_c' else behind
        d=0.
        while d < meters:
            xs=cur.next(5.)
            if len(xs)!=1:return False
            nxt=xs[0]
            if (not _driving(nxt,carla) or nxt.lane_id!=cur.lane_id or
                    yaw_diff(nxt.transform.rotation.yaw,cur.transform.rotation.yaw)>12.):
                return False
            v=cur.transform.location;u=nxt.transform.location
            delta=math.hypot(v.x-u.x,v.y-u.y)
            if not 1.<delta<8.:return False
            d+=delta
            cur=nxt
        return True
    if not all(has_route(role) for role in ('follower_a','follower_b','follower_c')):
        return None
    # The CARLA TTC checker must have a frozen target lane reaching C.
    # The original cut-in corridor stopped at B, incorrectly making B-C
    # 'non-collinear' whenever C was 20-45m further back.
    extension = [[float(v.transform.location.x),float(v.transform.location.y)]
                 for v in reversed(traversed)]
    case['_target_corridor_xy']=extension+case['_target_corridor_xy']
    case['_chain_bc_gap_m'] = float(projected)
    return case


def support_gap_candidate(wp, carla, ego_gap_m=20., rear_gap_m=45.):
    """Same *physical* gap class as cut_in, with finite-library diagnostics.

    This is not an independent fourth spatial archetype; S4 is a different
    scientific experiment (nested recourse coverage) on the cut-in geometry.
    """
    return cut_in_candidate(wp, carla, ego_gap_m, rear_gap_m)


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
    """Find an actual 118m lane corridor for S2 convoy, including a 100m goal.

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


def _anchor_roundtrip_valid(road_map, carla, coords, identities):
    """Reject a frozen candidate if the same saved XYZ will not resolve to
    the same OpenDRIVE identity. No outcome-dependent retry or silent remapping.
    v10 exhibited 10/60 confirmation failures when frozen follower_c
    road identities were restored; precision and seam ambiguity are suspects.
    """
    for role, xyz in coords.items():
        loc=carla.Location(x=float(xyz[0]),y=float(xyz[1]),z=float(xyz[2]))
        wp=road_map.get_waypoint(loc,project_to_road=True,
                                 lane_type=carla.LaneType.Driving)
        exp=identities[role]
        if wp is None or _xy(wp.transform.location,loc)>2.0:
            return False, role+':unresolved_waypoint'
        if (wp.road_id!=exp['road_id'] or wp.lane_id!=exp['lane_id'] or
            yaw_diff(wp.transform.rotation.yaw,exp['yaw'])>5.):
            return False, role+':identity_ambiguous_after_freeze'
    return True,''


def build_anchors(world, archetype, count, seed, *, with_diagnostics=False,
                  cutin_ego_gap_m=20., cutin_rear_gap_m=45., chain_bc_gap_m=25.,
                  forbidden_anchor_keys=(), exclusion_radius_m=90.):
    """Select map-grounded anchors, checking all candidate road geometry.

    First check recommended spawn points; if insufficient, search the map's
    dense driving-lane waypoints. Spawn points are NOT an exhaustive road map.
    The result and rejection counts are deterministic for a fixed map and seed.
    """
    if archetype not in ('cut_in', 'convoy', 'intersection', 'support_gap', 'chain_cut_in'):
        raise ValueError('Unknown case archetype: ' + archetype)
    import carla
    road_map = world.get_map()
    kind = archetype
    find = {'cut_in':cut_in_candidate, 'support_gap':support_gap_candidate,
            'chain_cut_in':chain_cut_in_candidate,
            'convoy':convoy_candidate, 'intersection':None}[kind]
    rng = random.Random(seed)
    diagnostics = {'map':road_map.name, 'archetype':archetype,
                   'spawn_points_scanned':0, 'map_waypoints_scanned':0,
                   'rejections':{}, 'anchors_found':0, 'excluded_for_holdout':0}
    rows = []; used = []

    def try_candidate(wp):
        if kind == 'intersection':
            return junction_candidate(wp,world,carla,incoming)
        if kind == 'convoy':
            return convoy_candidate(wp,carla,diagnostics['rejections'])
        return (find(wp,carla,cutin_ego_gap_m,cutin_rear_gap_m,chain_bc_gap_m)
                if kind == 'chain_cut_in' else
                find(wp,carla,cutin_ego_gap_m,cutin_rear_gap_m))

    def add(case):
        center = case['follower_a'].transform.location
        key = [float(center.x),float(center.y),float(center.z)]
        if any(math.dist(key,x) < max(90.0,exclusion_radius_m) for x in used):
            return
        if any(math.dist(key,x) < exclusion_radius_m for x in forbidden_anchor_keys):
            diagnostics["excluded_for_holdout"] += 1
            return
        # Keep enough floating-point precision that a CARLA road boundary
        # does not move to an adjacent road_id when frozen and restored.
        coords = {k:[round(float(getattr(v.transform.location,axis)),7)
                     for axis in ('x','y','z')]
                  for k,v in case.items() if not k.startswith('_')}
        identities = {k:{'road_id':int(v.road_id),'lane_id':int(v.lane_id),
                          'yaw':round(float(v.transform.rotation.yaw),6)}
                      for k,v in case.items() if not k.startswith('_')}
        ok,reason=_anchor_roundtrip_valid(road_map,carla,coords,identities)
        if not ok:
            rejects=diagnostics['rejections']
            rejects['frozen_roundtrip:'+reason]=rejects.get('frozen_roundtrip:'+reason,0)+1
            return
        used.append(key)
        geo=None
        if archetype in ('cut_in','support_gap','chain_cut_in'):
            forward=case['follower_a'].transform.get_forward_vector()
            lead=case['follower_a'].transform.location
            ego=case['ego'].transform.location
            rear=case['follower_b'].transform.location
            geo={'front_to_ego_m':round((lead.x-ego.x)*forward.x+(lead.y-ego.y)*forward.y,3),
                 'front_to_rear_m':round((lead.x-rear.x)*forward.x+(lead.y-rear.y)*forward.y,3)}
            if archetype == 'chain_cut_in':
                geo['rear_b_to_c_m']=round(float(case['_chain_bc_gap_m']),3)
        rows.append({'anchor':len(rows),'key':key,'coords':coords,'gap_geometry_m':geo,
                     'waypoint_identities':identities,
                     'archetype':archetype,'target_lane_id':case.get('_target_lane_id'),
                     'target_corridor_xy':case.get('_target_corridor_xy'),
                     'target_lane_width_m':case.get('_target_lane_width_m'),
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
    if anchor.get('archetype') in ('cut_in','support_gap','chain_cut_in'):
        from ride.carla.lane_geometry import distance_to_polyline
        corridor=anchor.get('target_corridor_xy')
        if not corridor or len(corridor)<20:
            raise RuntimeError('Missing v5 frozen target-lane corridor')
        # Verify the physical target lane still matches the frozen corridor.
        center=scene['follower_a'].transform.location
        if distance_to_polyline(center.x,center.y,corridor) > 0.5:
            raise RuntimeError('Frozen target lane centerline no longer matches map')
        scene['_target_corridor_xy']=corridor
        scene['_target_lane_width_m']=float(anchor['target_lane_width_m'])
    return scene


def ego_target_waypoint(world,vehicle,scene,archetype,profile,t,carla,*,late_switch_s=4.5,lateral_lookahead_m=15.0):
    wp=world.get_map().get_waypoint(vehicle.get_location(),project_to_road=True,
                                    lane_type=carla.LaneType.Driving)
    if wp is None:raise RuntimeError('Ego exited supported drivable lane')
    if archetype in ('cut_in', 'support_gap', 'chain_cut_in'):
        switch={'early_slow':1.5,'late_slow':late_switch_s,'mild_early':2.5,
                'mild_late':3.5,'merge_early':1.5,'merge_late':late_switch_s,'cruise':9.0,
                'speed_early':2.5,'speed_late':2.5}[profile]
        target_lane_id=scene['follower_a'].lane_id
        if t>=switch and wp.lane_id!=target_lane_id:
            choices=[x for x in (wp.get_left_lane(),wp.get_right_lane()) if x and
                     x.lane_id==target_lane_id and _driving(x,carla) and _same_dir(x,wp)]
            if choices:
                # A longer waypoint lookahead avoids the sharp, short-radius
                # lateral lunge in the old 7m-target cut-in controller.
                nxt=_one(choices[0].next(float(lateral_lookahead_m)))
                if nxt is not None:return nxt
    ahead=wp.next(8.)
    if not ahead:return wp
    # Choose the continuation with minimum direction change, explicitly not
    # claiming traffic-light compliance or a human-like navigation plan.
    # CARLA's Rotation exposes the heading as ``yaw``.  ``Rotation.y`` is
    # not a valid 0.9.16 attribute and caused every non-convoy scene to abort
    # before its first trace step (see evidence/carla/*_audit.json).
    return min(ahead,key=lambda v:yaw_diff(v.transform.rotation.yaw,wp.transform.rotation.yaw))
