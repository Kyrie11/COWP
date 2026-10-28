"""CARLA 0.9.16 pre-registered paired physical-response experiment.

Runs REAL CARLA physics on one map. This does NOT replay WOMD or identify human
counterfactuals. By default per branch experiment actors are destroyed/recreated in one world
(to avoid an Unreal reload fault); TM is seeded and traffic lights are reset.
Strict per-trial reload remains opt-in and should only be used once server health
is established. Actor reset does NOT promise complete Unreal-world determinism. Responders are either
CARLA Traffic Manager or CARLA BehaviorAgent (normal).

An optional exploratory JOINT-WITNESS search executes separately controlled
responder policies using SAME bad ego action. It proves only an EXISTING
response in a bounded CARLA controlled-search family when it passes; a failure
never proves global impossibility. No planner baseline claim is made here.

To run from your CARLA root, ensure `import carla` and
`from agents.navigation... import ...` both work. See README.
"""
from __future__ import annotations
import argparse, json, math, os, random, sys, traceback, time, uuid
from pathlib import Path
from .protocol import PROFILES, DT, DURATION_S, target_speed_mps, closing_ttc, summarise_trial
from .actor_lifecycle import check_spawn_state, destroy_verified, inventory


def xy_distance(a,b):return math.hypot(a.x-b.x,a.y-b.y)


def vehicle_speed(actor):
    v=actor.get_velocity()
    return math.sqrt(v.x*v.x+v.y*v.y+v.z*v.z)


def _carla():
    import carla
    return carla


def _spawn(world,wp,blueprint,role_name=None):
    carla=_carla()
    t=wp.transform
    # Copy the transform. Mutating the returned waypoint transform is unsafe.
    tr=carla.Transform(
        carla.Location(x=t.location.x,y=t.location.y,z=t.location.z+0.25),
        carla.Rotation(pitch=t.rotation.pitch,yaw=t.rotation.yaw,roll=t.rotation.roll))
    if role_name:
        if not blueprint.has_attribute('role_name'):
            raise RuntimeError('RIDE_ACTOR_OWNERSHIP_TAG_UNAVAILABLE: blueprint has no role_name')
        blueprint.set_attribute('role_name', role_name)
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


def _prepare_trial_world(client, args, family, profile):
    """Choose an explicitly recorded reset protocol; never mask reload failures."""
    current_world = client.get_world()
    current_name = str(current_world.get_map().name)
    target = str(args.town or current_name)
    same_map = current_name == target or current_name.endswith('/' + target)
    mode = args.world_reset
    print(f'RIDE_TRIAL_RESET_BEGIN family={family} profile={profile} '
          f'target={target} mode={mode}', flush=True)
    if mode == 'actors':
        if not same_map:
            raise RuntimeError('ACTOR_RESET_MAP_MISMATCH: currently '+current_name+
                               ', expected '+target+'. Use a separate map setup before trials.')
        # This is a dedicated-server experiment. Never delete someone else's actors.
        occupied = [r for r in inventory(current_world)
                    if r['type_id'].startswith('vehicle.')]
        if occupied:
            raise RuntimeError('ACTOR_RESET_WORLD_NOT_EMPTY: '+str(len(occupied))+
                               ' existing vehicle(s) '+json.dumps(occupied,ensure_ascii=False)+
                               '. First inspect with python -m ride.carla.actor_lifecycle; '
                               'only explicitly delete confirmed owned IDs.')
        world = current_world
        print(f'RIDE_TRIAL_WORLD_REUSE family={family} profile={profile} '
              f'map={current_name}', flush=True)
    elif mode == 'reload':
        try:
            world = client.reload_world() if same_map else client.load_world(target)
        except RuntimeError as exc:
            raise RuntimeError('CARLA_WORLD_RELOAD_FAILED: server did not complete the '
                               'world reset. Check the Unreal server log/Signal 11; '
                               'do not automatically retry a possibly dead instance. '
                               'For a separately labelled exploratory run, set '
                               'RIDE_WORLD_RESET=actors. Original error: '+str(exc)) from exc
        print(f'RIDE_TRIAL_WORLD_RESET family={family} profile={profile} '
              f'map={world.get_map().name}', flush=True)
    else:
        raise ValueError('Unknown world reset mode: '+str(mode))
    return world


def run_trial(args,anchor,family,profile,seed,*,joint_targets=None,recovery=None):
    """One trial; full map reset is opt-in, actor-only reset is default."""
    carla=_carla();client=carla.Client(args.host,args.port)
    client.set_timeout(args.timeout)
    world=_prepare_trial_world(client,args,family,profile)
    old=world.get_settings()
    settings=world.get_settings();settings.synchronous_mode=True;settings.fixed_delta_seconds=DT
    settings.no_rendering_mode=bool(args.no_rendering)
    settings.substepping=True;settings.max_substep_delta_time=0.01;settings.max_substeps=10
    tm=None
    actors=[];sensors=[];collisions=[];overlap_proxy=set();vehicles={}
    # Unique identifier for every trial, enabling safe cleanup of future orphans.
    trial_uid=uuid.uuid4().hex[:12]
    result={'anchor':anchor['anchor'],'family':family,'profile':profile,'seed':seed,
            'joint_targets':joint_targets,'recovery':recovery,'recovery_time_s':args.recovery_time_s,'trace':[],'collision_events':[],
            'archetype':getattr(args,'archetype','convoy'),
            'source':'CARLA_0_9_16_GENERATED_SCENE_NOT_WOMD',
            'causal_scope':'simulation_intervention_only',
            'world_reset_protocol':args.world_reset, 'trial_uid':trial_uid,
            'traffic_lights_reset':args.world_reset == 'actors',
            'traffic_light_legality_verified':False, 'warmup_ticks':args.warmup_ticks}
    try:
        world.apply_settings(settings)
        print(f'RIDE_WORLD_SETTINGS_APPLIED family={family} profile={profile}',flush=True)
        if args.world_reset == 'actors':
            # Traffic-signal cycles must start from the same phase in each trial.
            world.reset_all_traffic_lights()
            print('RIDE_TRAFFIC_LIGHTS_RESET', flush=True)
        random.seed(seed)
        from ride.carla_scenarios import restore
        c=restore(world,anchor,carla)
        bp=_blueprint(world)
        vehicles={}
        for role in ('ego','follower_a','follower_b'):
            # Pass fresh CARLA Transform: do not mutate live map waypoint position
            vehicles[role]=_spawn(world,c[role],bp,role_name=f'ride.{trial_uid}.{role}')
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
                collisions.append({'frame':int(evt.frame),'role_a':r,'role_b':other,
                                   'other_actor_type_id':str(evt.other_actor.type_id),
                                   'other_actor_id':int(evt.other_actor.id)})
            sensor.listen(on_collision)
        # Initialize same longitudinal velocities in lane-forward direction.
        for role,v in vehicles.items():
            d=c[role].transform.get_forward_vector()
            initial_speed=(args.cruise_mps + args.rear_speed_delta_mps
                           if role=='follower_b' and args.archetype in ('cut_in','support_gap')
                           else args.cruise_mps)
            v.set_target_velocity(carla.Vector3D(d.x*initial_speed,d.y*initial_speed,0))
        # CARLA synchronous spawning and velocity updates must be committed by a
        # simulator tick. A zero-state read before this frame invalidates all
        # collision/overlap conclusions (as happened in the uploaded smoke A).
        print('RIDE_SPAWN_PRIME_TICK_BEGIN '+trial_uid, flush=True)
        world.tick()
        spawn_checks=check_spawn_state(vehicles,c)
        result['spawn_checks']=spawn_checks
        result['actor_ids']={role:int(actor.id) for role,actor in vehicles.items()}
        print('RIDE_SPAWN_PRIME_TICK_DONE '+json.dumps(spawn_checks),flush=True)
        ego_ctrl=_pid(vehicles['ego'])
        responders={}
        # POLICY-INDEPENDENT PRE-INTERVENTION STATE. V4 installed TM or
        # BehaviorAgent before the warm-up. Different response models then
        # *changed the starting state*, invalidating the paired intervention.
        # V5 uses the same PID settle policy for every role across ALL arms.
        # The first TM/BehaviorAgent control is applied only after the
        # intervention start frame. This still requires recorded state parity.
        settle_controllers={role:_pid(vehicles[role])
                            for role in ('follower_a','follower_b')}
        for j in range(args.warmup_ticks):
            vehicles['ego'].apply_control(ego_ctrl.run_step(
                args.cruise_mps*3.6,_target_waypoint(world,vehicles['ego'])))
            for role,ctrl in settle_controllers.items():
                speed=(args.cruise_mps+args.rear_speed_delta_mps
                       if role=='follower_b' and args.archetype in ('cut_in','support_gap')
                       else args.cruise_mps)
                vehicles[role].apply_control(ctrl.run_step(
                    speed*3.6,_target_waypoint(world,vehicles[role])))
            world.tick()
        print('RIDE_COMMON_WARMUP_DONE '+trial_uid,flush=True)
        if family=='behavior':
            from agents.navigation.behavior_agent import BehaviorAgent
            for role in ('follower_a','follower_b'):
                a=BehaviorAgent(vehicles[role],behavior='normal')
                a.set_destination(c.get('end_follower',c['end']).transform.location)
                responders[role]=a
        elif family=='tm':
            print(f'RIDE_TM_CREATE_BEGIN port={args.tm_port} seed={seed}',flush=True)
            tm=client.get_trafficmanager(args.tm_port)
            tm.set_synchronous_mode(True)
            tm.set_random_device_seed(seed)
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
        speeds={k:vehicle_speed(v) for k,v in vehicles.items()}
        init_xy={k:v.get_location() for k,v in vehicles.items()}
        xy0=init_xy['ego']
        goal=c['end'].transform.location
        start_to_goal=xy_distance(xy0,goal)
        if start_to_goal<8.:
            raise RuntimeError('Task target too close for matched progress assessment')
        route_direction=((goal.x-xy0.x)/start_to_goal,(goal.y-xy0.y)/start_to_goal)
        from .lane_geometry import project_to_polyline
        frozen_corridor=c.get('_target_corridor_xy')
        ego_station0=(project_to_polyline(xy0.x,xy0.y,frozen_corridor)[0]
                      if frozen_corridor is not None else None)
        # For curved junctions the goal-distance reduction is only a proxy.
        # For lane corridors signed projection is NOT equivalent to odometer.
        use_goal_distance=(args.archetype=='intersection')
        prev_xy=dict(init_xy)
        travel={k:0. for k in vehicles}
        out_init={k:{'xyz':[v.x,v.y,v.z],'speed_mps':speeds[k]} for k,v in init_xy.items()}
        for k in out_init:
            out_init[k]['speed_mps']=speeds[k]
            out_init[k]['yaw_deg']=float(vehicles[k].get_transform().rotation.yaw)
        result['initial_states']=out_init
        # A collision sensor is armed BEFORE the prime tick and before the
        # common warm-up. Those contacts are pre-intervention confounders,
        # never ego-action-induced outcomes. Use CARLA frame IDs to separate
        # late-arriving asynchronous collision callbacks reliably.
        intervention_start_frame=int(world.get_snapshot().frame)
        result['intervention_start_frame']=intervention_start_frame
        # No initial collisions or impossible follower overlaps.
        from ride.geometry import carla_actor_overlap
        initial_pairs=[(x,y) for x,y in
                       [('ego','follower_a'),('follower_a','follower_b'),('ego','follower_b')]
                       if carla_actor_overlap(vehicles[x],vehicles[y])]
        result['initial_overlap_pairs']=initial_pairs
        result['initial_overlap']=bool(initial_pairs)
        if result['initial_overlap']:
            result['reason']='overlap_after_warmup'
            result['metrics']={'valid':False,'reasons':['overlap_after_warmup']}
            return result
        for step in range(round(args.duration_s/DT)):
            t=step*DT
            target=(args.cruise_mps if getattr(args,'archetype','convoy') in ('cut_in','support_gap')
                    else target_speed_mps(profile,t,args.cruise_mps))
            if recovery is not None and t>=args.recovery_time_s:
                target=0. if recovery=='stop' else args.cruise_mps
            if getattr(args,'archetype','convoy')=='convoy':
                ego_wp=_target_waypoint(world,vehicles['ego'])
            else:
                from ride.carla_scenarios import ego_target_waypoint
                ego_wp=ego_target_waypoint(world,vehicles['ego'],c,args.archetype,profile,t,carla,
                                     late_switch_s=args.cutin_late_switch_s,
                                     lateral_lookahead_m=args.cutin_lateral_lookahead_m)
            ego_command=ego_ctrl.run_step(target*3.6,ego_wp)
            vehicles['ego'].apply_control(ego_command)
            if family=='behavior':
                for role,a in responders.items():vehicles[role].apply_control(a.run_step())
            if family=='joint_witness':
                for role,a in responders.items():
                    target_r=joint_targets[0 if role=='follower_a' else 1]
                    vehicles[role].apply_control(a.run_step(float(target_r)*3.6,_target_waypoint(world,vehicles[role])))
            world.tick()
            # OBB is a sensor-health diagnostic only; NEVER substitute an OBB
            # overlap for a physically reported collision event.
            for x,y in [('ego','follower_a'),('ego','follower_b'),
                        ('follower_a','follower_b')]:
                if carla_actor_overlap(vehicles[x],vehicles[y]):
                    overlap_proxy.add(tuple(sorted((x,y))))
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
            from .lane_geometry import map_lane_center_valid, on_centerline, distance_to_polyline
            is_road=[];road_role_valid={};road_role_strict={};road_lateral_offset={}
            for role,v in vehicles.items():
                strict_wp=world.get_map().get_waypoint(v.get_location(),project_to_road=False,
                                                      lane_type=carla.LaneType.Driving)
                strict_ok=(strict_wp is not None and strict_wp.lane_type==carla.LaneType.Driving)
                center_ok, offset=map_lane_center_valid(v.get_location(),world.get_map(),carla)
                # For road validity, a planned merge can cross a lane boundary;
                # road_id/initial lane_id equality is NOT a legal-road test.
                road_role_valid[role]=center_ok
                road_role_strict[role]=strict_ok
                road_lateral_offset[role]=offset
                is_road.append(center_ok)
            # TTC applies only to a verified same-road/same-lane following pair.
            # Using 18m plus individual odometers gives incorrect gap on bends,
            # lane changes and non-constant CARLA vehicle trajectories.
            awp=world.get_map().get_waypoint(now['follower_a'],project_to_road=False,
                                              lane_type=carla.LaneType.Driving)
            bwp=world.get_map().get_waypoint(now['follower_b'],project_to_road=False,
                                              lane_type=carla.LaneType.Driving)
            pair_same_lane=(awp is not None and bwp is not None and
                            awp.road_id==bwp.road_id and awp.lane_id==bwp.lane_id and
                            abs((awp.transform.rotation.yaw-bwp.transform.rotation.yaw+180.)%360.-180.)<18.)
            pair_stations={}
            # Use the continuous frozen target-lane polyline across road-ID boundaries.
            if frozen_corridor is not None:
                pair_stations={role:project_to_polyline(loc.x,loc.y,frozen_corridor)
                               for role,loc in now.items()}
                lane_tol=min(1.5,0.42*float(c['_target_lane_width_m']))
                pair_same_lane=(pair_stations['follower_a'][1] <= lane_tol and
                                pair_stations['follower_b'][1] <= lane_tol and
                                pair_stations['follower_a'][0] > pair_stations['follower_b'][0])
            if pair_same_lane and args.archetype!='intersection':
                bumper_length=(vehicles['follower_a'].bounding_box.extent.x+
                               vehicles['follower_b'].bounding_box.extent.x)
                va=vehicles['follower_a'].get_velocity();vb=vehicles['follower_b'].get_velocity()
                if pair_stations:
                    front=pair_stations['follower_a'];rear=pair_stations['follower_b']
                    along=front[0]-rear[0]
                    front_speed=float(va.x*front[2]+va.y*front[3])
                    rear_speed=float(vb.x*rear[2]+vb.y*rear[3])
                else:
                    h=awp.transform.get_forward_vector()
                    diff=now['follower_a']-now['follower_b']
                    along=float(diff.x*h.x+diff.y*h.y)
                    front_speed=float(va.x*h.x+va.y*h.y)
                    rear_speed=float(vb.x*h.x+vb.y*h.y)
                gap=along-bumper_length
                ttc=closing_ttc(0.,along,rear_speed,front_speed,
                                vehicle_length=bumper_length)
            else:
                gap=None;ttc=None
            goal_distance=xy_distance(now['ego'],goal)
            route_progress=(
                pair_stations['ego'][0]-ego_station0 if ego_station0 is not None else
                (start_to_goal-goal_distance if use_goal_distance else
                 (now['ego'].x-xy0.x)*route_direction[0]+(now['ego'].y-xy0.y)*route_direction[1]))
            ego_wp_now=world.get_map().get_waypoint(now['ego'],project_to_road=False,
                                                    lane_type=carla.LaneType.Driving)
            a_wp_now=world.get_map().get_waypoint(now['follower_a'],project_to_road=False,
                                                  lane_type=carla.LaneType.Driving)
            corridor=c.get('_target_corridor_xy')
            target_centerline_error=(distance_to_polyline(now['ego'].x,now['ego'].y,corridor)
                                     if corridor is not None else None)
            # A crossing from road_id 45 to road_id 6 is not a lane failure.
            # The frozen physical corridor is checked regardless of road IDs.
            ego_in_target=bool(corridor is not None and on_centerline(
                now['ego'].x,now['ego'].y,corridor,
                tolerance_m=min(1.5,0.42*c['_target_lane_width_m'])))
            # CARLA traffic-light observations: report traffic signal legality.
            # A trial that moves at a red signal is not a valid 'safe' witness.
            red_motion=[]
            traffic_signal_states={}
            controls={}
            physics_long_accel={}
            for role,v in vehicles.items():
                ctl=v.get_control()
                controls[role]={'throttle':float(ctl.throttle),
                                'brake':float(ctl.brake),
                                'steer':float(ctl.steer)}
                raw_accel=v.get_acceleration()
                heading=v.get_transform().get_forward_vector()
                physics_long_accel[role]=float(raw_accel.x*heading.x + raw_accel.y*heading.y)
                if v.is_at_traffic_light():
                    light=v.get_traffic_light()
                    if light is not None:
                        state=str(light.get_state())
                        traffic_signal_states[role]={'traffic_light_id':int(light.id),
                                                     'state':state,
                                                     'is_trigger_overlap_only':True}
                        if light.get_state()==carla.TrafficLightState.Red and new_speeds[role]>0.5:
                            red_motion.append(role)
            frame={'t':round((step+1)*DT,3),
                'controls_by_role':controls,
                'physics_forward_accel_by_role':physics_long_accel,
                'traffic_signal_states_by_role':traffic_signal_states,
                # CARLA trigger-volume motion does not prove crossing a stop line.
                'red_light_motion_roles':red_motion,
                'ego_target_speed_mps':float(target),
                'ego_controller_steer':float(ego_command.steer),
                'ego_controller_brake':float(ego_command.brake),
                'ego_controller_throttle':float(ego_command.throttle),
                'ego_target_waypoint_lane_id':int(ego_wp.lane_id),
                'ego_target_waypoint_road_id':int(ego_wp.road_id),
                'ego_speed_mps':new_speeds['ego'],'ego_progress_m':float(route_progress),
                'ego_path_length_m':s['ego'],
                'progress_measure':('frozen_target_corridor_station' if ego_station0 is not None
                                    else ('goal_distance_reduction_proxy' if use_goal_distance
                                          else 'goal_axis_projection')),
                'target_corridor_station_m_by_role':{role:val[0] for role,val in pair_stations.items()},
                'target_corridor_offset_m_by_role':{role:val[1] for role,val in pair_stations.items()},
                'ego_lane_id':(int(ego_wp_now.lane_id) if ego_wp_now is not None else None),
                'ego_road_id':(int(ego_wp_now.road_id) if ego_wp_now is not None else None),
                'target_lane_id':(int(c['follower_a'].lane_id)
                                  if getattr(args,'archetype','convoy') in ('cut_in','support_gap') else None),
                'ego_in_target_lane':ego_in_target,
                'ego_target_centerline_error_m':target_centerline_error,
                'ego_in_junction':bool(ego_wp_now and ego_wp_now.is_junction),
                'follower_a_in_junction':bool(a_wp_now and a_wp_now.is_junction),
                'follower_a_accel':(new_speeds['follower_a']-speeds['follower_a'])/DT,
                'follower_b_accel':(new_speeds['follower_b']-speeds['follower_b'])/DT,
                'follower_a_lane_id':(int(awp.lane_id) if awp else None),
                'follower_a_road_id':(int(awp.road_id) if awp else None),
                'follower_b_lane_id':(int(bwp.lane_id) if bwp else None),
                'follower_b_road_id':(int(bwp.road_id) if bwp else None),
                'follower_a_speed_mps':new_speeds['follower_a'],
                'follower_b_speed_mps':new_speeds['follower_b'],
                'rear_pair_gap_m':gap,'rear_pair_ttc_s':min(ttc,9999.) if ttc is not None else None,
                'road_valid_all':all(is_road),
                'road_center_valid_by_role':road_role_valid,
                'road_strict_valid_by_role':road_role_strict,
                'road_center_offset_m_by_role':road_lateral_offset,
                'rear_pair_ttc_applicable':bool(pair_same_lane and args.archetype!='intersection'),
                'follower_a_progress_m':s['follower_a'],'follower_b_progress_m':s['follower_b'],
                'ego_follower_a_distance_m':xy_distance(now['ego'],now['follower_a']),
                'xy_by_role':{k:[float(v.x),float(v.y)] for k,v in now.items()}}
            result['trace'].append(frame)
            speeds=new_speeds
        # Sensor callbacks are asynchronous relative to synchronous physics
        # frames. This bounded receive grace drains last-frame events without
        # taking another simulation tick (which would change the action).
        time.sleep(0.12)
        pre_action=[event for event in collisions if event['frame']<=intervention_start_frame]
        post_action=[event for event in collisions if event['frame']>intervention_start_frame]
        result['pre_action_collision_events']=pre_action
        result['collision_events']=post_action
        pairs={tuple(sorted((x['role_a'],x['role_b']))) for x in post_action}
        result['ego_collision']=any('ego' in p for p in pairs)
        result['non_ego_collision']=any('ego' not in p for p in pairs)
        result['non_ego_pair_collision']=('follower_a','follower_b') in pairs
        result['unexpected_actor']=any('external' in p for p in pairs)
        result['kinematic_overlap_pairs']=[list(x) for x in sorted(overlap_proxy)]
        result['sensor_overlap_disagreement_pairs']=[list(x) for x in sorted(overlap_proxy-pairs)]
        result['metrics']=summarise_trial(result)
        return result
    except Exception as ex:
        # A dead/unresponsive CARLA server must abort the batch rather than
        # generating dozens of misleading per-trial invalid scientific rows.
        message = str(ex).lower()
        if ('time-out' in message or 'connection refused' in message or
                'failed to connect' in message or 'rpc error' in message):
            print('RIDE_FATAL_SIMULATOR_CONNECTION '+repr(ex),flush=True)
            raise
        result['simulation_exception']=type(ex).__name__+': '+str(ex)
        result['traceback']=traceback.format_exc(limit=4)
        result['metrics']=summarise_trial(result) if result['trace'] else {'valid':False,'reasons':['exception']}
        return result
    finally:
        # Server-verified batch removal replaces unreliable consecutive
        # actor.destroy calls. Keep synchronous world tick authority until the
        # deletion acknowledgement; otherwise a later trial can see orphans.
        if tm is not None:
            for role in ('follower_a', 'follower_b'):
                if role in vehicles:
                    try:vehicles[role].set_autopilot(False, args.tm_port)
                    except Exception as exc:
                        print('RIDE_TM_DISABLE_WARNING '+repr(exc),flush=True)
        for sensor in sensors:
            try:sensor.stop()
            except Exception:pass
        cleanup_exc=None
        try:
            destroy_verified(client,world,[a.id for a in sensors+actors],
                             carla_api=carla,label=trial_uid)
        except Exception as exc:
            cleanup_exc=exc
            print('RIDE_ACTOR_CLEANUP_FAILED '+repr(exc),flush=True)
        if tm is not None:
            try:tm.set_synchronous_mode(False)
            except Exception as exc:print('RIDE_TM_RESTORE_WARNING '+repr(exc),flush=True)
        try:world.apply_settings(old)
        except Exception as exc:
            print('RIDE_WORLD_RESTORE_WARNING '+repr(exc),flush=True)
        if cleanup_exc is not None:
            raise RuntimeError('RIDE_TRIAL_NOT_CLEAN: '+str(cleanup_exc)) from cleanup_exc


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host',default=os.getenv('CARLA_HOST','127.0.0.1'));p.add_argument('--port',type=int,default=int(os.getenv('CARLA_RPC_PORT','2000')))
    p.add_argument('--tm-port',type=int,default=int(os.getenv('CARLA_TM_PORT','8100')))
    p.add_argument('--town',default='Town04')
    p.add_argument('--timeout',type=float,default=120,
                   help='CARLA RPC timeout in seconds (map load can take >40 s)')
    p.add_argument('--world-reset',choices=('actors','reload'),default='actors',
                   help='actors: reuse Town, re-spawn RIDE vehicles (default). '
                        'reload: strict world reset per trial; may trigger Unreal crashes.')
    p.add_argument('--anchors',type=int,default=3)
    p.add_argument('--seed',type=int,default=7)
    p.add_argument('--anchor-selection-seed',type=int,default=None,
                   help='Decouple physical corridor selection from responder random seed')
    p.add_argument('--repetitions',type=int,default=2)
    p.add_argument('--profiles',nargs='+',choices=PROFILES,default=list(PROFILES))
    p.add_argument('--families',nargs='+',choices=('tm','behavior'),default=['tm','behavior'])
    p.add_argument('--duration-s',type=float,default=DURATION_S)
    p.add_argument('--cruise-mps',type=float,default=9.)
    p.add_argument('--cutin-ego-gap-m',type=float,default=20.,
                   help='Positive along-lane distance from lead A to adjacent ego (5m grid)')
    p.add_argument('--cutin-rear-gap-m',type=float,default=45.,
                   help='Positive along-lane distance from lead A to rear B (5m grid)')
    p.add_argument('--cutin-late-switch-s',type=float,default=4.5,
                   help='Lateral switch time for late_slow/merge_late; freeze per protocol')
    p.add_argument('--cutin-lateral-lookahead-m',type=float,default=15.,
                   help='Target-lane waypoint lookahead for cut-in controller')
    p.add_argument('--rear-speed-delta-mps',type=float,default=0.,
                   help='Extra initial and shared warm-up speed of rear follower on cut-in; '
                        'prespecified stress factor, NOT a fitted behavioral truth')
    p.add_argument('--warmup-ticks',type=int,default=20,
                   help='common pre-action settle frames; action time t=0 starts after warmup')
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
    p.add_argument('--inspect-anchors',action='store_true',
                   help='Scan map and print eligible anchors and rejection counts; no result files or trials')
    p.add_argument('--anchors-from',help='Frozen manifest-only JSONL containing exact scenario anchors')
    p.add_argument('--output',help='Required unless --inspect-anchors is used')
    a=p.parse_args()
    if not a.inspect_anchors and not a.output:
        p.error('--output is required for freezing/executing experiments')
    if a.timeout <= 0:
        raise ValueError('--timeout must be positive')
    if a.warmup_ticks < 0: raise ValueError('Negative warmup ticks')
    if a.duration_s<6 or a.cruise_mps<=0 or a.anchors<1 or a.repetitions<1 or a.joint_limit<0 or not 0<=a.rear_speed_delta_mps<=4.:
        raise ValueError('Invalid trial duration, speed, anchor count, repetition or search limit')
    if a.recovery_time_s<=0 or a.recovery_time_s>=a.duration_s:
        raise ValueError('Recovery cutoff outside horizon')
    if not 2.0 <= a.cutin_late_switch_s <= 6.0 or not 5.0 <= a.cutin_lateral_lookahead_m <= 30.0:
        raise ValueError('Cut-in control parameters outside pre-registered bounds')
    if any(v<0 for v in a.joint_target_grid):raise ValueError('Negative responder speed not allowed')
    if (a.cutin_ego_gap_m < 20. or a.cutin_rear_gap_m < a.cutin_ego_gap_m+15. or
        a.cutin_rear_gap_m > 70. or
        abs(a.cutin_ego_gap_m/5-round(a.cutin_ego_gap_m/5))>1e-8 or
        abs(a.cutin_rear_gap_m/5-round(a.cutin_rear_gap_m/5))>1e-8):
        raise ValueError('Invalid precommitted 5m-grid cut-in gap: ego>=20, rear>=ego+15, rear<=70')
    client=_carla().Client(a.host,a.port);client.set_timeout(a.timeout)
    server_version=str(client.get_server_version())
    client_version=str(client.get_client_version())
    if not server_version.startswith('0.9.16'):
        raise RuntimeError(f'Expected CARLA 0.9.16 server, got {server_version}')
    if not client_version.startswith('0.9.16'):
        raise RuntimeError(f'Expected CARLA 0.9.16 Python client, got {client_version}')
    world=client.get_world()
    if a.town and not (str(world.get_map().name)==a.town or
                        str(world.get_map().name).endswith('/'+a.town)):
        print(f'RIDE_WORLD_SWITCH to {a.town}',flush=True)
        world=client.load_world(a.town)
    else:
        print(f'RIDE_WORLD_REUSE {world.get_map().name}',flush=True)
    if a.inspect_anchors and a.anchors_from:
        raise ValueError('--inspect-anchors cannot be combined with --anchors-from')
    from ride.carla_scenarios import ANCHOR_PROTOCOL_VERSION
    if a.anchors_from:
        with open(a.anchors_from,encoding='utf8') as f:
            frozen=json.loads(f.readline())
        if (frozen.get('type')!='manifest' or frozen.get('archetype')!=a.archetype or
            frozen.get('anchor_protocol_version')!=ANCHOR_PROTOCOL_VERSION or
            frozen.get('carla_server_version')!=server_version or
            frozen.get('map')!=world.get_map().name or
            frozen.get('anchor_selection_seed', frozen.get('seed')) != (
                a.anchor_selection_seed if a.anchor_selection_seed is not None else a.seed) or
            frozen.get('cruise_mps')!=a.cruise_mps or
            frozen.get('duration_s')!=a.duration_s or
            frozen.get('warmup_ticks',0)!=a.warmup_ticks or
            frozen.get('rear_speed_delta_mps')!=a.rear_speed_delta_mps or
            frozen.get('cutin_ego_gap_m')!=a.cutin_ego_gap_m or
            frozen.get('cutin_rear_gap_m')!=a.cutin_rear_gap_m or
            frozen.get('cutin_late_switch_s',4.5)!=a.cutin_late_switch_s or
            frozen.get('cutin_lateral_lookahead_m',15.)!=a.cutin_lateral_lookahead_m):
            raise RuntimeError('Frozen manifest map/version/archetype mismatch')
        anchors=frozen.get('anchors',[])
        if not anchors:raise RuntimeError('Frozen manifest has no anchors')
    else:
        from ride.carla_scenarios import build_anchors
        anchors, anchor_stats=build_anchors(world,a.archetype,a.anchors,(a.anchor_selection_seed
                                           if a.anchor_selection_seed is not None else a.seed),
                                           with_diagnostics=True,
                                           cutin_ego_gap_m=a.cutin_ego_gap_m,
                                           cutin_rear_gap_m=a.cutin_rear_gap_m)
        print('CARLA_MAP_ANCHOR_DIAGNOSTICS '+json.dumps(anchor_stats,sort_keys=True),flush=True)
    if a.inspect_anchors:
        print(json.dumps({'town':a.town,'eligible_anchors':len(anchors),
                          'anchors':anchors},indent=2,ensure_ascii=False))
        return
    if not anchors:
        raise RuntimeError('NO_ELIGIBLE_MAP_ANCHOR: no eligible ' + a.archetype +
            ' corridor found after spawn-point AND map-waypoint scan on ' +
            str(world.get_map().name) + '. Inspect diagnostic rejection counts; '
            'try a different town with a fresh RIDE_OUTPUT_ROOT. This is a setup '
            'skip, NOT a negative scientific result.')
    output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True)
    rows=[];trial_id=0
    with output.open('x',encoding='utf8') as dst:
        meta={'type':'manifest','simulator':'CARLA','archetype':a.archetype,'version_requested':'0.9.16',
              'map':world.get_map().name,'anchors':anchors,'dt':DT,
              'rpc_host':a.host,'rpc_port':a.port,'tm_port':a.tm_port,
              'world_reset_protocol':('actors_only_per_trial_with_traffic_light_reset'
                                      if a.world_reset=='actors' else 'reload_world_per_trial'),
              'world_reset_policy':a.world_reset,
              'actor_cleanup_protocol':'server_verified_batch_destroy_with_sync_tick',
              'traffic_light_legality_protocol':'unverified_red_trigger_motion_proxy_only',
              'spawn_init_protocol':'mandatory_post_spawn_sync_tick_and_anchor_validation',
              'scientific_reset_limitation':('actor reset does not restore all hidden Unreal '
                  'state; inspect initial_states, traffic lights, and run strict reload '
                  'on a stable server before making deterministic causal claims'
                  if a.world_reset=='actors' else 'Unreal full world reload'),
              'anchor_protocol_version':ANCHOR_PROTOCOL_VERSION,
              'carla_server_version':server_version,'carla_client_version':client_version,'cruise_mps':a.cruise_mps,
              'physics_ticks_per_recorded_step':1,'physics_dt_s':DT,
              'profiles':a.profiles,'families':a.families,'repetitions':a.repetitions,
              'seed':a.seed,'anchor_selection_seed':(a.anchor_selection_seed
                        if a.anchor_selection_seed is not None else a.seed),
              'duration_s':a.duration_s,
              'rear_speed_delta_mps':a.rear_speed_delta_mps,
              'cutin_ego_gap_m':a.cutin_ego_gap_m,
              'cutin_rear_gap_m':a.cutin_rear_gap_m,
              'cutin_late_switch_s':a.cutin_late_switch_s,
              'cutin_lateral_lookahead_m':a.cutin_lateral_lookahead_m,
              'pre_intervention_policy':'shared_pid_all_roles',
              'cut_in_profiles_use_lateral_switch_time_only':True,
              'action_protocol_v8':{'archetype':'cut_in', 'main_variable':'lateral_merge_command_start',
                  'speed_mps_across_cut_in_arms':a.cruise_mps,
                  'early_slow_switch_s':1.5, 'mild_early_switch_s':2.5,
                  'late_slow_switch_s':a.cutin_late_switch_s,
                  'mild_late_switch_s':a.cutin_late_switch_s,
                  'caveat':'historical profile names refer to longitudinal braking, but cut-in is lateral timing only'},
              'joint_profile':a.joint_profile,'joint_limit':a.joint_limit,
              'joint_target_grid':a.joint_target_grid,
              'recovery_profile':a.recovery_profile,'warmup_ticks':a.warmup_ticks,
              'no_rendering':a.no_rendering,
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
                        try:
                            row=run_trial(a,anchor,family,profile,seed)
                        except Exception as exc:
                            dst.write(json.dumps({'type':'run_error','completed':False,
                                'trial_id':trial_id,'family':family,'profile':profile,
                                'error':str(exc)})+'\n');dst.flush()
                            raise
                        row['type']='trial';row['trial_id']=trial_id;trial_id+=1
                        dst.write(json.dumps(row,allow_nan=False)+'\n');dst.flush()
                        print(f'[{trial_id}] {anchor["anchor"]} {family} {profile} '
                              f'{row.get("metrics",{}).get("valid")} {row.get("simulation_exception","")}',flush=True)
            if a.recovery_profile:
                for rep in range(a.repetitions):
                    for family in a.families:
                        for recovery in a.recovery_options:
                            row=run_trial(a,anchor,family,a.recovery_profile,a.seed+rep,
                                          recovery=recovery)
                            row['type']='trial';row['trial_id']=trial_id;trial_id+=1
                            dst.write(json.dumps(row,allow_nan=False)+'\n');dst.flush()
            if a.joint_limit>0 and a.joint_profile:
                for rep in range(a.repetitions):
                    count=0
                    for va in a.joint_target_grid:
                        for vb in a.joint_target_grid:
                            if count>=a.joint_limit:break
                            row=run_trial(a,anchor,'joint_witness',a.joint_profile,a.seed+rep,
                                          joint_targets=(va,vb))
                            row['type']='trial';row['trial_id']=trial_id;trial_id+=1
                            dst.write(json.dumps(row,allow_nan=False)+'\n');dst.flush();count+=1
                        if count>=a.joint_limit:break
        dst.write(json.dumps({'type':'run_end','completed':True,'trial_count':trial_id})+'\n')
        dst.flush()
    print(json.dumps({'trials':trial_id,'anchors':len(anchors),'output':str(output)},indent=2))
if __name__=='__main__':main()
