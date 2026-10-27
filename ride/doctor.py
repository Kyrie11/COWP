"""Inspect RIDE, CARLA 0.9.16 and native WOMD/Waymax Python environments."""
from __future__ import annotations
import argparse, importlib.util, json, os, sys
from pathlib import Path


def inspect(carla_root, probe_server=False, host="127.0.0.1", port=2000, tm_port=8100):
    root=Path(carla_root).expanduser().resolve()
    api=root/'PythonAPI'/'carla'
    report={'python':sys.executable,'python_version':sys.version,'carla_root':str(root),
            'server_script_exists':(root/'CarlaUE4.sh').exists() or (root/'CarlaUnreal.sh').exists(),
            'agents_file_exists':(api/'agents'/'navigation'/'behavior_agent.py').exists(),
            'pythonapi_parent':str(api),'pythonpath':os.environ.get('PYTHONPATH',''),
            'server_host':host,'rpc_port':port,'tm_port':tm_port,
            'modules':{},'server':None}
    for name in ('ride','ride.carla.run','ride.womd.native','ride.waymax.paired_idm',
                 'carla','agents','waymax','jax','tensorflow'):
        try:
            module=importlib.util.find_spec(name)
            report['modules'][name]={'found':module is not None,
                         'origin':None if module is None else module.origin}
        except Exception as ex:
            report['modules'][name]={'found':False,'error':repr(ex)}
    if probe_server:
        if not report['modules']['carla']['found']:
            report['server']={'error':'CARLA Python client not installed/importable'}
        else:
            try:
                import carla
                client=carla.Client(host,port);client.set_timeout(5)
                report['server']={'server_version':client.get_server_version(),
                                  'client_version':client.get_client_version()}
            except Exception as ex:
                report['server']={'error':str(ex)}
    return report


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--carla-root',default=os.environ.get('CARLA_ROOT','/data0/senzeyu2/dataset/CARLA_0.9.16'))
    p.add_argument('--server',action='store_true')
    p.add_argument('--host',default=os.environ.get('CARLA_HOST','127.0.0.1'))
    p.add_argument('--port',type=int,default=int(os.environ.get('CARLA_RPC_PORT','2000')))
    p.add_argument('--tm-port',type=int,default=int(os.environ.get('CARLA_TM_PORT','8100')))
    a=p.parse_args(argv)
    result=inspect(a.carla_root,a.server,a.host,a.port,a.tm_port)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if not result['agents_file_exists']:
        print('WARN: BehaviorAgent source absent; the carla wheel alone is insufficient.')
    elif not result['modules']['agents']['found']:
        print('WARN: Add $CARLA_ROOT/PythonAPI/carla to this interpreter PYTHONPATH.')
    if a.server:
        state=result['server'] or {}
        if ('error' in state or not str(state.get('server_version','')).startswith('0.9.16')
            or not str(state.get('client_version','')).startswith('0.9.16')):
            raise SystemExit('ERROR: CARLA 0.9.16 server/client validation failed')


if __name__=='__main__':main()
