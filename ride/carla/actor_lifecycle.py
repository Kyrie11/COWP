"""Inspect and safely clean CARLA actors without reloading Unreal world.

Use only on a dedicated simulator whose tick authority you control. By default
this utility only lists actors. Unknown actors are NEVER automatically removed.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re

ROLE_PREFIX = 'ride.'
CONFIRM = 'DELETE_THE_LISTED_IDS'


def actor_record(actor):
    attrs = getattr(actor, 'attributes', {}) or {}
    loc = actor.get_location()
    return {'id': int(actor.id), 'type_id': str(actor.type_id),
            'role_name': str(attrs.get('role_name', '')),
            'xyz': [round(float(loc.x), 3), round(float(loc.y), 3), round(float(loc.z), 3)]}


def inventory(world):
    actors = world.get_actors()
    return sorted((actor_record(a) for a in actors if
                   str(a.type_id).startswith(('vehicle.', 'sensor.'))),
                  key=lambda a: a['id'])


def check_spawn_state(vehicles, expected, max_distance=3.0):
    """Fail closed on CARLA actors which have not received a physics snapshot."""
    observed = {}
    for role, actor in vehicles.items():
        loc = actor.get_location()
        got = (float(loc.x), float(loc.y), float(loc.z))
        want_loc = expected[role].transform.location
        want = (float(want_loc.x), float(want_loc.y), float(want_loc.z))
        distance = math.dist(got, want)
        observed[role] = {'xyz': list(got), 'anchor_xyz': list(want),
                          'spawn_position_error_m': distance}
        if (not all(math.isfinite(x) for x in got) or distance > max_distance):
            raise RuntimeError(
                'CARLA_SPAWN_NOT_INITIALIZED: role=' + role +
                ' actual=' + str(got) + ' expected=' + str(want) +
                ' distance_m=' + str(round(distance, 3)) +
                '; require a successful world.tick() before querying state')
    return observed


def destroy_verified(client, world, ids, *, carla_api, label='RIDE'):
    """Destroy *exact IDs*; verify the server's actor registry after one tick.

    Does not alter world sync settings. Caller must own the tick on this server.
    Missing IDs are benign for crashed sensor wrappers. Verification fails closed.
    """
    selected = sorted(set(int(x) for x in ids))
    if not selected:
        return {'requested_ids': [], 'errors': [], 'remaining_ids': []}
    live = {int(a.id) for a in world.get_actors()}
    present = [i for i in selected if i in live]
    errors = []
    if present:
        replies = client.apply_batch_sync(
            [carla_api.command.DestroyActor(i) for i in present], False)
        errors = [str(getattr(r, 'error', '')) for r in replies
                  if str(getattr(r, 'error', ''))]
        if world.get_settings().synchronous_mode:
            world.tick()
        else:
            world.wait_for_tick(seconds=10)
    remaining = sorted({int(a.id) for a in world.get_actors()}.intersection(selected))
    output = {'requested_ids': selected, 'errors': errors, 'remaining_ids': remaining}
    print('RIDE_ACTOR_CLEANUP ' + json.dumps({'label':label, **output}), flush=True)
    if remaining:
        raise RuntimeError('RIDE_ACTOR_CLEANUP_INCOMPLETE: '+str(output))
    return output


def cli(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host', default=os.environ.get('CARLA_HOST', '127.0.0.1'))
    p.add_argument('--port',type=int,default=int(os.environ.get('CARLA_RPC_PORT','2000')))
    g=p.add_mutually_exclusive_group()
    g.add_argument('--cleanup-owned',action='store_true',
                   help='Delete only explicitly role_name=ride.* actors')
    g.add_argument('--cleanup-ids',type=int,nargs='+',
                   help='Explicit legacy actor IDs, after inspecting --list output')
    p.add_argument('--confirm',default='',
                   help='Required exactly: '+CONFIRM+' (destructive)')
    a=p.parse_args(argv)
    import carla
    client=carla.Client(a.host,a.port)
    client.set_timeout(30)
    world=client.get_world()
    rows=inventory(world)
    print('RIDE_ACTOR_INVENTORY '+json.dumps({
        'map':world.get_map().name,'rpc_port':a.port,
        'synchronous_mode':bool(world.get_settings().synchronous_mode),
        'actors':rows},ensure_ascii=False,indent=2),flush=True)
    if not a.cleanup_owned and not a.cleanup_ids:
        return 0
    if a.confirm != CONFIRM:
        p.error('Destructive operation requires --confirm '+CONFIRM)
    if a.cleanup_owned:
        chosen={x['id'] for x in rows if x['role_name'].startswith(ROLE_PREFIX)
                and x['type_id'].startswith('vehicle.')}
    else:
        chosen=set(a.cleanup_ids)
        known={x['id'] for x in rows}
        if chosen - known:
            p.error('Unknown or already absent actor IDs: '+str(sorted(chosen-known)))
        if any(not x['type_id'].startswith(('vehicle.','sensor.'))
               for x in rows if x['id'] in chosen):
            p.error('Refusing non-vehicle/sensor actor IDs')
    # Attached collision sensors are owned by the selected vehicles only; CARLA
    # Actor.parent (if available) records the attachment relationship.
    actor_by_id={int(actor.id):actor for actor in world.get_actors()}
    for actor in actor_by_id.values():
        if str(actor.type_id).startswith('sensor.'):
            parent=getattr(actor, 'parent', None)
            if parent is not None and int(parent.id) in chosen:
                chosen.add(int(actor.id))
    if not chosen:
        print('RIDE_ACTOR_CLEANUP nothing selected')
        return 0
    print('RIDE_ACTOR_DELETE_EXPLICIT_IDS '+str(sorted(chosen)),flush=True)
    destroy_verified(client,world,chosen,carla_api=carla,label='operator-approved')
    return 0


if __name__=='__main__':
    raise SystemExit(cli())
