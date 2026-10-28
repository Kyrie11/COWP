"""Separate CARLA server-health probe; no map mutations by default.

This module is deliberately separate from RIDE trial execution so a reload
failure can be tested in isolation and never treated as scientific evidence.
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def inspect(client):
    world = client.get_world()
    snapshot = world.get_snapshot()
    settings = world.get_settings()
    return {
        'server_version':str(client.get_server_version()),
        'client_version':str(client.get_client_version()),
        'map':str(world.get_map().name),
        'frame':int(snapshot.frame),
        'synchronous_mode':bool(settings.synchronous_mode),
        'fixed_delta_seconds':float(settings.fixed_delta_seconds or 0.0),
        'no_rendering_mode':bool(settings.no_rendering_mode),
        'vehicles':len(list(world.get_actors().filter('vehicle.*'))),
    }


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host',default=os.getenv('CARLA_HOST','127.0.0.1'))
    parser.add_argument('--port',type=int,default=int(os.getenv('CARLA_RPC_PORT','2000')))
    parser.add_argument('--timeout',type=float,default=30)
    parser.add_argument('--mode',choices=('inspect','reload'),default='inspect')
    parser.add_argument('--expected-town',default='',help='Fail before any scene freeze when loaded map differs')
    parser.add_argument('--require-empty',action='store_true',help='Refuse batch while actors from other experiments exist')
    parser.add_argument('--allow-destructive',action='store_true',
                        help='Required for reload; destroys all actors in the current world')
    args=parser.parse_args(argv)
    if args.mode == 'reload' and not args.allow_destructive:
        parser.error('reload is destructive; pass --allow-destructive on a dedicated server')
    import carla
    try:
        client=carla.Client(args.host,args.port)
        client.set_timeout(args.timeout)
        before=inspect(client)
        print('RIDE_CARLA_HEALTH_BEFORE '+json.dumps(before,ensure_ascii=False),flush=True)
        if args.expected_town and not (before['map']==args.expected_town or
                                      before['map'].endswith('/'+args.expected_town)):
            raise RuntimeError('RIDE_MAP_MISMATCH_REFUSE_AUTO_LOAD: '+before['map']+
                               ' != '+args.expected_town+'; start separate server already on requested Town')
        if args.require_empty and before['vehicles']:
            raise RuntimeError('RIDE_WORLD_NOT_EMPTY: '+str(before['vehicles'])+' existing vehicle(s)')
        if args.mode == 'reload':
            if before['vehicles']:
                raise RuntimeError('Refusing reload with pre-existing vehicles. '
                                   'Only use on an empty, dedicated CARLA server.')
            print('RIDE_CARLA_HEALTH_RELOAD_BEGIN',flush=True)
            client.reload_world()
            print('RIDE_CARLA_HEALTH_RELOAD_DONE',flush=True)
            print('RIDE_CARLA_HEALTH_AFTER '+json.dumps(inspect(client),ensure_ascii=False),flush=True)
    except Exception as exc:
        print(f'RIDE_CARLA_HEALTH_FAILED {type(exc).__name__}: {exc}',file=sys.stderr,flush=True)
        return 2
    return 0


if __name__=='__main__':
    sys.exit(main())
