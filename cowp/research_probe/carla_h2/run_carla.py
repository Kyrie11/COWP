"""CARLA 0.9.16 pre-registered paired physical-response experiment.

Runs REAL CARLA physics on one map. This does NOT replay WOMD or identify human
counterfactuals. Per branch the world is reloaded and TM seeded; geometry,
roles, scripted ego target-speed commands are identical. Responders are either
CARLA Traffic Manager or CARLA BehaviorAgent (normal).

An optional exploratory JOINT-WITNESS search executes separately controlled
responder policies using SAME bad ego action. It proves only an EXISTING
response in a bounded CARLA controlled-search family when it passes; a failure
never proves global impossibility. No planner baseline claim is made here.

To run from your CARLA root, ensure `import carla` and
`from agents.navigation... import ...` both work. See README.
"""
from __future__ import annotations
import argparse, json, math, os, random, sys, traceback, time
from pathlib import Path
from .protocol import PROFILES, DT, DURATION_S, target_speed_mps, closing_ttc, summarise_trial


def xy_distance(a,b):return math.hypot(a.x-b.x,a.y-b.y)


def vehicle_speed(actor):
    v=actor.get_velocity()
    return math.sqrt(v.x*v.x+v.y*v.y+v.z*v.z)


def _unique_step(wp, meters, backward=False):
    cand=wp.previous(meters) if backward else wp.next(meters)
    if len(cand)!=1:raise ValueError('Nonunique lane predecessor/successor')
    w=cand[0]
    if w.is_junction or w.lane_id!=wp.lane_id or w.lane_type!=_carla().LaneType.Driving:
        raise ValueError('Junction or lane change in controlled corridor')
    return w


def corridor(wp):
    try:
        if wp.is_junction or wp.lane_type!=_carla().LaneType.Driving:return None
        rear=_unique_step(wp,18,backward=True)
        ego=_unique_step(wp,20)
        w=ego
        for _ in range(20):w=_unique_step(w,4)
        # Along-initial-lane progress is a meaningful task metric only on
        # nearly straight segments (else use map arc-length / Frenet progress).
        delta=((w.transform.rotation.yaw-ego.transform.rotation.y+180.)%360.)-180.
        if abs(delta)>12.:return None
        return {'ego':ego,'follower_a':wp,'follower_b':rear,'end':w}
    except (ValueError,IndexError,AttributeError):return None


def list_corridors(world, max_anchors=6, seed=7):
    wmap=world.get_map();pts=wmap.get_spawn_points()
    random.Random(seed).shuffle(pts)
    found=[];coordinates=[]
    for tr in pts:
        wp=wmap.get_waypoint(tr.location,project_to_road=True)
        c=corridor(wp)
        if c is None:continue
        key=(round(c['follower_a'].transform.location.x,1),round(c['follower_a'].transform.location.y,1),
             round(c['follower_a'].transform.location.z,1))
        if any(math.dist(key,pr)<80 for pr in coordinates):continue
        coordinates.append(key)
        found.append({'anchor':len(found),'key':key,'coords':{
            role:[round(getattr(w.transform.location,dim),3) for dim in ('x','y','z')]
            for role,w in c.items()}})
        if len(found)==max_anchors:break
    return found


def find_matching_corridor(world,key):
    wp=world.get_map().get_waypoint(
        _carla().Location(x=key[0],y=key[1],z=key[2]), project_to_road=True)
    res=corridor(wp)
    if res is None or xy_distance(res['follower_a'].transform.location,
                                     _carla().Location(x=key[0],y=key[1],z=key[2]))>2.:
        raise RuntimeError('Saved corridor no longer maps to same lane')
    return res


def _carla():
    import carla
    return carla


def _spawn(world,wp,blueprint):
    carla=_carla()
    t=wp.transform
    # Copy the transform. Mutating the returned waypoint transform is unsafe.
    tr=carla.Transform(
        carla.Location(x=t.location.x,y=t.location.y,z=t.location.z+0.25),
        carla.Rotation(pitch=t.rotation.pitch,yaw=t.rotation.yaw,roll=t.rotation.roll))
    actor=world.try_spawn_actor(blueprint,tr)
    if actor is None:raise RuntimeError('Actor spawn failed; no collision-free spawn in selected lane')
    return actor


def _blueprint(world):
    lib=world.get_blueprint_library()
    bps=list(lib.filter('vehicle.tesla.model3'))
    if not bps:
        bps=[b for b in lib.filter('vehicle.*') if b.has_attribute('number_of_wheels') and
             int(b.get_attribute('number_of_wheels').as_int())==4]
    if not bps:raise RuntimeError('No four-wheel vehicle blueprint')
    return bps[0]


def _target_waypoint(world,vehicle):
    carla=_carla()
    wp=world.get_map().get_waypoint(vehicle.get_location(),project_to_road=True,
                                    lane_type=carla.LaneType.Driving)
    if wp is None:raise RuntimeError('Vehicle not on a drivable lane')
    a=wp.next(7.)
    return a[0] if len(a)==1 else wp


def _pid(vehicle):
    from agents.navigation.controller import VehiclePIDController
    return VehiclePIDController(vehicle,
        args_lateral={'K_P':1.8,'K_I':0.02,'K_D':0.20,'dt':DT},
        args_longitudinal={'K_P':1.2,'K_I':0.05,'K_D':0.0,'dt':DT},
        max_throttle=0.75,max_brake=0.9,max_steering=0.7)


def run_trial(args,anchor,family,profile,seed,*,joint_targets=None,recovery=None):
    """One branch; own map restart for initial-state parity and TM state reset."""
    carla=_carla();client=carla.Client(args.host,args.port)
    client.set_timeout(args.timeout)
    # Reload EVERY trial, not just vehicle respawn: TM has hidden policy state.
    world=client.load_world(args.town) if args.town else client.reload_world()
    old=world.get_settings()
    settings=world.get_settings();settings.synchronous_mode=True;settings.fixed_delta_seconds=DT
    settings.no_rendering_mode=bool(args.no_rendering)
    settings.substepping=True;settings.max_substep_delta_time=0.01;settings.max_substeps=10
    world.apply_settings(settings)
    tm=client.get_trafficmanager(args.tm_port);tm.set_synchronous_mode(True)
    tm.set_random_device_seed(seed)
    random.seed(seed)
    actors=[];sensors=[];collisions=[]
    result={'anchor':anchor['anchor'],'family':family,'profile':profile,'seed':seed,
            'joint_targets':joint_targets,'recovery':recovery,'recovery_time_s':args.recovery_time_s,'trace':[],'collision_events':[],
            'archetype':getattr(args,'archetype','convoy'),
            'source':'CARLA_0_9_16_GENERATED_SCENE_NOT_WOMD',
            'causal_scope':'simulation_intervention_only'}
    try:
        if getattr(args,'archetype','convoy')=='convoy':
            c=find_matching_corridor(world,anchor['key'])
        else:
            from ride.carla_scenarios import restore
            c=restore(world,anchor,carla)
        bp=_blueprint(world)
        vehicles={}
        for role in ('ego','follower_a','follower_b'):
            # Pass fresh CARLA Transform: do not mutate live map waypoint position
            vehicles[role]=_spawn(world,c[role],bp)
            actors.append(vehicles[role])
        actor_to_role={v.id:k for k,v in vehicles.items()}
        # Listen for physical collision events; pair IDs are scenario-local role
        # names, not nonpersistent actor numeric IDs.
        for role,v in vehicles.items():
            b=world.get_blueprint_library().find('sensor.other.collision')
            sensor=world.spawn_actor(b,carla.Transform(),attach_to=v)
            sensors.append(sensor)
            def on_collision(evt,r=role):
                other=actor_to_role.get(evt.other_actor.id,'external')
                collisions.append({'frame':int(evt.frame),'role_a':r,'role_b':other})
            sensor.listen(on_collision)
        # Initialize same longitudinal velocities in lane-forward direction.
        for role,v in vehicles.items():
            d=c[role].transform.get_forward_vector()
            v.set_target_velocity(carla.Vector3D(d.x*args.cruise_mps,d.y*args.cruise_mps,0))
        ego_ctrl=_pid(vehicles['ego'])
        responders={}
        if family=='behavior':
            from agents.navigation.behavior_agent import BehaviorAgent
            for role in ('follower_a','follower_b'):
                a=BehaviorAgent(vehicles[role],behavior='normal')
                a.set_destination(c.get('end_follower',c['end']).transform.location)
                responders[role]=a
        elif family=='tm':
            for role in ('follower_a','follower_b'):
                a=vehicles[role]
                a.set_autopilot(True,args.tm_port)
                tm.auto_lane_change(a,False)
                tm.distance_to_leading_vehicle(a,2.5)
                tm.vehicle_percentage_speed_difference(a,0.)
        elif family=='joint_witness':
            if joint_targets is None or len(joint_targets)!=2:
                raise ValueError('joint_witness must specify two target speeds')
            responders={k:_pid(vehicles[k]) for k in ('follower_a','follower_b')}
        else:raise ValueError(family)
        # Settling time is part of every branch; record actual initial state.
        for j in range(args.warmup_ticks):
            ctrl=ego_ctrl.run_step(args.cruise_mps*3.6,_target_waypoint(world,vehicles['ego']))
            vehicles['ego'].apply_control(ctrl)
            if family=='behavior':
                for role,a in responders.items():vehicles[role].apply_control(a.run_step())
            if family=='joint_witness':
                for role,a in responders.items():vehicles[role].apply_control(
                    a.run_step(args.cruise_mps*3.6,_target_waypoint(world,vehicles[role])))
            world.tick()
        speeds={k:vehicle_speed(v) for k,v in vehicles.items()}
        init_xy={k:v.get_location() for k,v in vehicles.items()}
        forward=c['ego'].transform.get_forward_vector()
        xy0=init_xy['ego']
        prev_xy=dict(init_xy)
        travel={k:0. for k in vehicles}
        out_init={k:{'xyz':[v.x,v.y,v.z],'speed_mps':speeds[k]} for k,v in init_xy.items()}
        for k in out_init:out_init[k]['speed_mps']=speeds[k]
        result['initial_states']=out_init
        # No initial collisions or impossible follower overlaps.
        from ride.geometry import carla_actor_overlap
        initial_pairs=[(x,y) for x,y in
                       [('ego','follower_a'),('follower_a','follower_b'),('ego','follower_b')]
                       if carla_actor_overlap(vehicles[x],vehicles[y])]
        result['initial_overlap_pairs']=initial_pairs
        result['initial_overlap']=bool(initial_pairs)
        if result['initial_overlap']:
            result['reason']='overlap_after_warmup';return result
        for step in range(round(args.duration_s/DT)):
            t=step*DT
            target=(args.cruise_mps if getattr(args,'archetype','convoy')=='cut_in'
                    else target_speed_mps(profile,t,args.cruise_mps))
            if recovery is not None and t>=args.recovery_time_s:
                target=0. if recovery=='stop' else args.cruise_mps
            if getattr(args,'archetype','convoy')=='convoy':
                ego_wp=_target_waypoint(world,vehicles['ego'])
            else:
                from ride.carla_scenarios import ego_target_waypoint
                ego_wp=ego_target_waypoint(world,vehicles['ego'],c,args.archetype,profile,t,carla)
            vehicles['ego'].apply_control(ego_ctrl.run_step(target*3.6,ego_wp))
            if family=='behavior':
                for role,a in responders.items():vehicles[role].apply_control(a.run_step())
            if family=='joint_witness':
                for role,a in responders.items():
                    target_r=joint_targets[0 if role=='follower_a' else 1]
                    vehicles[role].apply_control(a.run_step(float(target_r)*3.6,_target_waypoint(world,vehicles[role])))
            world.tick()
            now={k:v.get_location() for k,v in vehicles.items()}
            new_speeds={k:vehicle_speed(v) for k,v in vehicles.items()}
            # Arc-length approximation for each ACTUALLY EXECUTED motion; the
            # original shared tangent projection miscomputed intersection progress.
            for k in now:
                travel[k]+=xy_distance(now[k],prev_xy[k])
            prev_xy=dict(now)
            s=dict(travel)
            # TM and BehaviorAgent may brake or route unexpectedly: explicit
            # offroad/lane-ID checks to prohibit invalid positive witnesses.
            is_road=[]
            for role,v in vehicles.items():
                wp=world.get_map().get_waypoint(v.get_location(),project_to_road=False,
                                               lane_type=carla.LaneType.Driving)
                is_road.append(wp is not None and (getattr(args,'archetype','convoy')=='intersection'
                                       or wp.lane_id==c[role].lane_id or
                                       (getattr(args,'archetype','convoy')=='cut_in' and role=='ego')))
            if getattr(args,'archetype','convoy')=='intersection':
                gap=9999.
                ttc=9999.  # Longitudinal same-lane TTC is not defined here.
            else:
                # Report relative CARLA road distances only for same-lane pair.
                # follower A initially 18m ahead of follower B.
                gap=(18.+s['follower_a']-s['follower_b'])-4.5
                ttc=closing_ttc(s['follower_b'],18.+s['follower_a'],
                                new_speeds['follower_b'],new_speeds['follower_a'])
            # CARLA traffic-light observations: report traffic signal legality.
            # A trial that moves at a red signal is not a valid 'safe' witness.
            red_motion=[]
            for role,v in vehicles.items():
                if v.is_at_traffic_light():
                    light=v.get_traffic_light()
                    if light is not None and light.get_state()==carla.TrafficLightState.Red and new_speeds[role]>0.5:
                        red_motion.append(role)
            frame={'t':round((step+1)*DT,3),
                'red_light_motion_roles':red_motion,'ego_target_speed_mps':float(target),
                'ego_speed_mps':new_speeds['ego'],'ego_progress_m':s['ego'],
                'follower_a_accel':(new_speeds['follower_a']-speeds['follower_a'])/DT,
                'follower_b_accel':(new_speeds['follower_b']-speeds['follower_b'])/DT,
                'follower_a_speed_mps':new_speeds['follower_a'],
                'follower_b_speed_mps':new_speeds['follower_b'],
                'rear_pair_gap_m':gap,'rear_pair_ttc_s':min(ttc,9999.),
                'road_valid_all':all(is_road),
                'rear_pair_ttc_applicable':getattr(args,'archetype','convoy')!='intersection',
                'follower_a_progress_m':s['follower_a'],'follower_b_progress_m':s['follower_b'],
                'xy_by_role':{k:[float(v.x),float(v.y)] for k,v in now.items()}}
            result['trace'].append(frame)
            speeds=new_speeds
        # Sensor callbacks are asynchronous relative to synchronous physics
        # frames. This bounded receive grace drains last-frame events without
        # taking another simulation tick (which would change the action).
        time.sleep(0.12)
        result['collision_events']=collisions[:]
        pairs={tuple(sorted((x['role_a'],x['role_b']))) for x in collisions}
        result['ego_collision']=any('ego' in p for p in pairs)
        result['non_ego_collision']=any('ego' not in p for p in pairs)
        result['non_ego_pair_collision']=('follower_a','follower_b') in pairs
        result['unexpected_actor']=any('external' in p for p in pairs)
        result['metrics']=summarise_trial(result)
        return result
    except Exception as ex:
        result['simulation_exception']=type(ex).__name__+': '+str(ex)
        result['traceback']=traceback.format_exc(limit=4)
        result['metrics']=summarise_trial(result) if result['trace'] else {'valid':False,'reasons':['exception']}
        return result
    finally:
        for a in sensors:
            try:a.stop()
            except Exception:pass
        for a in sensors+actors:
            try:a.destroy()
            except Exception:pass
        try:tm.set_synchronous_mode(False)
        except Exception:pass
        try:world.apply_settings(old)
        except Exception:pass


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=2000)
    p.add_argument('--tm-port',type=int,default=8000)
    p.add_argument('--town',default='Town04')
    p.add_argument('--timeout',type=float,default=40)
    p.add_argument('--anchors',type=int,default=3)
    p.add_argument('--seed',type=int,default=7)
    p.add_argument('--repetitions',type=int,default=2)
    p.add_argument('--profiles',nargs='+',choices=PROFILES,default=list(PROFILES))
    p.add_argument('--families',nargs='+',choices=('tm','behavior'),default=['tm','behavior'])
    p.add_argument('--duration-s',type=float,default=DURATION_S)
    p.add_argument('--cruise-mps',type=float,default=9.)
    p.add_argument('--warmup-ticks',type=int,default=0)
    p.add_argument('--no-rendering',action='store_true')
    p.add_argument('--joint-profile',choices=PROFILES)
    p.add_argument('--joint-target-grid',type=float,nargs='+',default=[7.,10.,13.])
    p.add_argument('--joint-limit',type=int,default=0,help='0 means no joint witness search')
    p.add_argument('--recovery-profile',choices=PROFILES,
                   help='Also execute recovery trials sharing an exact original ego prefix')
    p.add_argument('--recovery-time-s',type=float,default=3.5)
    p.add_argument('--recovery-options',nargs='+',choices=('stop','cruise'),
                   default=['stop','cruise'])
    p.add_argument('--archetype',choices=('convoy','cut_in','intersection','support_gap'),default='convoy')
    p.add_argument('--manifest-only',action='store_true',help='Freeze scenario identities without executing outcomes')
    p.add_argument('--anchors-from',help='Frozen manifest-only JSONL containing exact scenario anchors')
    p.add_argument('--output',required=True)
    a=p.parse_args()
    if a.duration_s<6 or a.cruise_mps<=0:raise ValueError('Invalid trial length/speed')
    client=_carla().Client(a.host,a.port);client.set_timeout(a.timeout)
    server_version=str(client.get_server_version())
    client_version=str(client.get_client_version())
    if not server_version.startswith('0.9.16'):
        raise RuntimeError(f'Expected CARLA 0.9.16 server, got {server_version}')
    if not client_version.startswith('0.9.16'):
        raise RuntimeError(f'Expected CARLA 0.9.16 Python client, got {client_version}')
    if a.town:world=client.load_world(a.town)
    else:world=client.get_world()
    if a.anchors_from:
        with open(a.anchors_from,encoding='utf8') as f:
            frozen=json.loads(f.readline())
        if (frozen.get('type')!='manifest' or frozen.get('archetype')!=a.archetype or
            frozen.get('carla_server_version')!=server_version or
            frozen.get('map')!=world.get_map().name):
            raise RuntimeError('Frozen manifest map/version/archetype mismatch')
        anchors=frozen.get('anchors',[])
        if not anchors:raise RuntimeError('Frozen manifest has no anchors')
    elif a.archetype=='convoy':anchors=list_corridors(world,a.anchors,a.seed)
    else:
        from ride.carla_scenarios import build_anchors
        anchors=build_anchors(world,a.archetype,a.anchors,a.seed)
    if not anchors:raise RuntimeError('NO_ELIGIBLE_MAP_ANCHOR: map lacks supported '+a.archetype+' scenario; this is a setup skip, not an observed negative result')
    output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True)
    rows=[];trial_id=0
    with output.open('w',encoding='utf8') as dst:
        meta={'type':'manifest','simulator':'CARLA','archetype':a.archetype,'version_requested':'0.9.16',
              'map':world.get_map().name,'anchors':anchors,'dt':DT,
              'carla_server_version':server_version,'carla_client_version':client_version,'cruise_mps':a.cruise_mps,
              'profiles':a.profiles,'families':a.families,'repetitions':a.repetitions,
              'disclosure':'Not calibrated human response; no WOMD replay; exploratory'}
        dst.write(json.dumps(meta)+'\n');dst.flush()
        if a.manifest_only:
            print(json.dumps({'manifest_only':True,'anchors':len(anchors),
                              'output':str(output)},indent=2))
            return
        for anchor in anchors:
            for rep in range(a.repetitions):
                for family in a.families:
                    for profile in a.profiles:
                        seed=a.seed+rep
                        row=run_trial(a,anchor,family,profile,seed)
                        row['type']='trial';row['trial_id']=trial_id;trial_id+=1
                        dst.write(json.dumps(row,allow_nan=False)+'\n');dst.flush()
                        print(f'[{trial_id}] {anchor["anchor"]} {family} {profile} '
                              f'{row.get("metrics",{}).get("valid")} {row.get("simulation_exception","")}',flush=True)
            if a.recovery_profile:
                for family in a.families:
                    for recovery in a.recovery_options:
                        row=run_trial(a,anchor,family,a.recovery_profile,a.seed,
                                      recovery=recovery)
                        row['type']='trial';row['trial_id']=trial_id;trial_id+=1
                        dst.write(json.dumps(row,allow_nan=False)+'\n');dst.flush()
            if a.joint_limit>0 and a.joint_profile:
                count=0
                for va in a.joint_target_grid:
                    for vb in a.joint_target_grid:
                        if count>=a.joint_limit:break
                        row=run_trial(a,anchor,'joint_witness',a.joint_profile,a.seed,
                                      joint_targets=(va,vb))
                        row['type']='trial';row['trial_id']=trial_id;trial_id+=1
                        dst.write(json.dumps(row,allow_nan=False)+'\n');dst.flush();count+=1
                    if count>=a.joint_limit:break
    print(json.dumps({'trials':trial_id,'anchors':len(anchors),'output':str(output)},indent=2))
if __name__=='__main__':main()
