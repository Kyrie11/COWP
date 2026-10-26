"""CARLA / Waymax environment inspection. No undocumented implicit fixes."""
import argparse,importlib.util,json,os,sys
from pathlib import Path


def inspect(carla_root,probe_server=False):
    root=Path(carla_root);ap=root/'PythonAPI'/'carla'
    report={'python':sys.executable,'python_version':sys.version,'carla_root':str(root),
            'server_script_exists':(root/'CarlaUE4.sh').exists() or (root/'CarlaUnreal.sh').exists(),
            'agents_file_exists':(ap/'agents'/'navigation'/'behavior_agent.py').exists(),
            'pythonapi_parent':str(ap),'pythonpath':os.environ.get('PYTHONPATH',''),
            'modules':{},'server':None}
    for x in ('carla','agents','waymax','jax','research_probe'):
        try:
            s=importlib.util.find_spec(x)
            report['modules'][x]={'found':s is not None,'origin':None if s is None else s.origin}
        except Exception as ex:report['modules'][x]={'found':False,'error':repr(ex)}
    if probe_server and report['modules']['carla']['found']:
        try:
            import carla
            client=carla.Client('127.0.0.1',2000);client.set_timeout(5)
            report['server']={'server_version':client.get_server_version(),
                              'client_version':client.get_client_version()}
        except Exception as ex:report['server']={'error':str(ex)}
    return report


def main():
    a=argparse.ArgumentParser();a.add_argument('--carla-root',default='/data0/senzeyu2/dataset/CARLA_0.9.16')
    a.add_argument('--server',action='store_true');q=a.parse_args()
    r=inspect(q.carla_root,q.server)
    print(json.dumps(r,ensure_ascii=False,indent=2))
    if not r['agents_file_exists']:
        print('WARN: CARLA BehaviorAgent source is absent. Python package carla alone does not ship agents/navigation.')
    if not r['modules']['agents']['found']:
        print('WARN: Add <CARLA_ROOT>/PythonAPI/carla to PYTHONPATH for the SAME Python interpreter.')
if __name__=='__main__':main()
