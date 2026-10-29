"""Route-coverage gate and frozen map-local plan for CARLA BehaviorAgent.

CARLA BehaviorAgent may dereference a None incoming waypoint when its
LocalPlanner queue becomes empty.  Do not monkeypatch upstream or silently
substitute a new policy. This module supplies a *new, explicitly recorded*
map-grounded LANEFOLLOW route to BehaviorAgent using the existing public-ish
set_global_plan interface, or fails BEFORE the science rollout.

This changes route evidence relative to v11, so panels must not be pooled.
"""
from __future__ import annotations
import math


def _heading_diff(a, b):
    return abs((float(a)-float(b)+180.0)%360.0-180.0)


def forward_lane_plan(first, *, carla, horizon_m, step_m=4.0):
    """Follow a unique road waypoint successor; never select a branch silently."""
    if not (math.isfinite(horizon_m) and 30.0 <= horizon_m <= 200.0):
        raise ValueError('ROUTE_HORIZON_UNSUPPORTED')
    if not (1.0 <= step_m <= 6.0):
        raise ValueError('ROUTE_STEP_UNSUPPORTED')
    if first is None:
        raise RuntimeError('BEHAVIOR_ROUTE_START_MISSING')
    route=[first]
    current=first
    d=0.0
    while d < horizon_m:
        if current.lane_type != carla.LaneType.Driving or bool(current.is_junction):
            raise RuntimeError('BEHAVIOR_ROUTE_NOT_NONJUNCTION_DRIVING')
        ahead=current.next(step_m)
        if len(ahead)!=1:
            raise RuntimeError('BEHAVIOR_ROUTE_END_OR_BRANCH')
        nxt=ahead[0]
        if (nxt.lane_type!=carla.LaneType.Driving or bool(nxt.is_junction) or
                int(nxt.lane_id)!=int(current.lane_id) or
                _heading_diff(current.transform.rotation.yaw,
                              nxt.transform.rotation.yaw)>12.0):
            raise RuntimeError('BEHAVIOR_ROUTE_DISCONTINUITY')
        a=current.transform.location;b=nxt.transform.location
        ds=math.hypot(float(a.x-b.x),float(a.y-b.y))
        if not 1.0<ds<1.6*step_m:
            raise RuntimeError('BEHAVIOR_ROUTE_ZERO_OR_JUMP_SEGMENT')
        d+=ds
        current=nxt
        route.append(nxt)
        if len(route)>120:
            raise RuntimeError('BEHAVIOR_ROUTE_EXCESSIVE_LENGTH')
    return route,d


def install_chain_lane_route(agent, vehicle, road_map, carla, *, cruise_mps, duration_s):
    """Install a frozen local corridor, not a new car-following strategy.

    The plan is generated from the shared physical map and live initial vehicle
    location, before intervention. A 15m lookahead reserve prevents empty local
    queues during 8 seconds of driving at nominal speed. All errors fail closed.
    """
    if not math.isfinite(float(cruise_mps)) or float(cruise_mps)<=0:
        raise ValueError('BEHAVIOR_ROUTE_SPEED_INVALID')
    wp=road_map.get_waypoint(vehicle.get_location(),project_to_road=True,
                             lane_type=carla.LaneType.Driving)
    horizon=float(cruise_mps)*float(duration_s)+18.0
    points,length=forward_lane_plan(wp,carla=carla,horizon_m=horizon)
    try:
        from agents.navigation.local_planner import RoadOption
    except ImportError as exc:
        raise RuntimeError('BEHAVIOR_ROUTE_ROADOPTION_MODULE_MISSING') from exc
    plan=[(w,RoadOption.LANEFOLLOW) for w in points]
    set_plan=getattr(agent,'set_global_plan',None)
    interface='BehaviorAgent.set_global_plan'
    if not callable(set_plan):
        set_plan=getattr(getattr(agent,'_local_planner',None),'set_global_plan',None)
        interface='LocalPlanner.set_global_plan'
    if not callable(set_plan):
        raise RuntimeError('BEHAVIOR_ROUTE_PLAN_API_MISSING')
    set_plan(plan,stop_waypoint_creation=True,clean_queue=True)
    local=getattr(agent,'_local_planner',None)
    get_inc=getattr(local,'get_incoming_waypoint_and_direction',None)
    if not callable(get_inc):
        raise RuntimeError('BEHAVIOR_ROUTE_LOOKAHEAD_API_MISSING')
    incoming,_=get_inc(steps=10)
    if incoming is None:
        raise RuntimeError('BEHAVIOR_ROUTE_QUEUE_EMPTY_AFTER_INSTALL')
    return {'protocol':'explicit_frozen_map_corridor_v12',
            'road_option':'LANEFOLLOW', 'interface':interface,
            'route_waypoints':len(plan), 'route_length_m':round(length,3),
            'horizon_requested_m':round(horizon,3),
            'policy':'CARLA BehaviorAgent; only its nominal navigation route is fixed'}
