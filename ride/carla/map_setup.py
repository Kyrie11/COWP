"""Explicit, one-time CARLA map preparation before RIDE preflight/trials.

This is an OPERATOR action, never called by preflight, experiment freeze or trials.
CARLA load_world is destructive even when no vehicles were found. Only use on a
known dedicated/idle server. There are intentionally no retries on load errors.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys


# A map may be reported as 'Carla/Maps/Town04' or '/Game/Carla/Maps/Town04'.
def map_matches(loaded: str, requested: str) -> bool:
    return loaded == requested or loaded.endswith('/' + requested)


def _map_asset_name(asset: str) -> str:
    # CARLA get_available_maps() commonly returns /Game/Carla/Maps/Town04.
    # Some builds report Unreal object path Town04.Town04.
    return str(asset).rsplit('/', 1)[-1].rsplit('.', 1)[-1]


def _active_actor_types(world):
    # Static map objects (traffic lights/signs, static props) are expected.
    # Unknown types are treated as active, to avoid destroying others' work.
    allowed_prefixes = ('traffic.', 'static.')
    allowed_names = {'spectator'}
    return [str(actor.type_id) for actor in world.get_actors()
            if str(actor.type_id) not in allowed_names and
            not str(actor.type_id).startswith(allowed_prefixes)]


def prepare_map(client, requested: str, *, allow_world_load: bool = False) -> dict:
    """Validate and optionally perform ONE explicit, guarded map switch.

    No map mutation happens unless allow_world_load is true. On a matching map,
    no mutation happens regardless of the flag. Caller should run preflight next.
    """
    if not re.fullmatch(r'Town\d{2}(?:_Opt)?', requested):
        raise ValueError('INVALID_CARLA_TOWN: ' + repr(requested))
    sv = str(client.get_server_version())
    cv = str(client.get_client_version())
    if not (sv.startswith('0.9.16') and cv.startswith('0.9.16')):
        raise RuntimeError('CARLA_VERSION_MISMATCH: server=' + sv + ' client=' + cv +
                           ' (requires matching 0.9.16)')
    world = client.get_world()
    before = str(world.get_map().name)
    result = {'server_version': sv, 'client_version': cv,
              'before': before, 'requested': requested}
    if map_matches(before, requested):
        result.update(action='already_loaded', after=before)
        return result
    if not allow_world_load:
        raise RuntimeError('RIDE_MAP_SWITCH_REQUIRES_EXPLICIT_OPT_IN: loaded=' + before +
                           ' requested=' + requested +
                           '; only run on a dedicated idle server with --allow-world-load')
    settings = world.get_settings()
    if bool(settings.synchronous_mode):
        raise RuntimeError('RIDE_MAP_SETUP_REFUSED_SYNCHRONOUS_WORLD: another client may be running')
    actors = _active_actor_types(world)
    if actors:
        summary = ', '.join(sorted(set(actors))[:12])
        raise RuntimeError(f'RIDE_MAP_SETUP_REFUSED_ACTIVE_ACTORS: {len(actors)} present ({summary}); '
                           'stop other clients and use a dedicated empty server')
    available = [str(a) for a in client.get_available_maps()]
    if requested not in {_map_asset_name(a) for a in available}:
        raise RuntimeError('RIDE_MAP_NOT_INSTALLED: ' + requested +
                           '; available=' + ', '.join(available))
    print('RIDE_EXPLICIT_ONE_TIME_WORLD_LOAD from=' + before +
          ' to=' + requested + ' (world will be replaced)', flush=True)
    # Deliberately no retry: on a timeout/Signal 11, the Unreal server may be dead.
    try:
        client.load_world(requested)
    except Exception as exc:
        raise RuntimeError('RIDE_ONE_TIME_MAP_LOAD_FAILED: ' + str(exc) +
                           '; do NOT auto-retry; inspect CARLA server logs/Signal 11') from exc
    # Re-fetch after the episode changes. A returned World may be stale.
    after = str(client.get_world().get_map().name)
    if not map_matches(after, requested):
        raise RuntimeError('RIDE_MAP_LOAD_VERIFICATION_FAILED: after=' + after +
                           ' expected=' + requested)
    result.update(action='loaded_once', after=after)
    return result


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host', default=os.getenv('CARLA_HOST', '127.0.0.1'))
    p.add_argument('--port', type=int, default=int(os.getenv('CARLA_RPC_PORT', '2000')))
    p.add_argument('--timeout', type=float, default=180.0)
    p.add_argument('--town', required=True)
    p.add_argument('--allow-world-load', action='store_true',
                   help='Explicit consent to destroy/replace an EMPTY dedicated CARLA world')
    a = p.parse_args(argv)
    try:
        import carla
        client = carla.Client(a.host, a.port)
        client.set_timeout(a.timeout)
        result = prepare_map(client, a.town, allow_world_load=a.allow_world_load)
        print('RIDE_MAP_SETUP_RESULT ' + json.dumps(result, ensure_ascii=False), flush=True)
        return 0
    except Exception as exc:
        print(f'RIDE_MAP_SETUP_FAILED {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
