"""Non-destructive TCP diagnostics for a CARLA 0.9.16 instance.

This cannot prove who owns a process in restricted /proc namespaces; it never
kills other users' servers or takes over a running instance.
"""
from __future__ import annotations

import argparse
import re
import socket
import subprocess


def _ss_listeners():
    try:
        p = subprocess.run(['ss', '-H', '-ltnp'], capture_output=True, text=True,
                           timeout=3, check=False)
        return p.stdout.splitlines() if p.returncode == 0 else []
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []


def _socket_listening(port: int, host: str = '127.0.0.1') -> bool:
    for addr in dict.fromkeys((host, '127.0.0.1')):
        try:
            with socket.create_connection((addr, port), timeout=0.25):
                return True
        except (OSError, ValueError):
            pass
    return False


def inspect_ports(rpc_port: int, tm_port: int = 8100, host: str = '127.0.0.1'):
    if not (1024 <= rpc_port <= 65534 and 1024 <= tm_port <= 65535):
        raise ValueError('RPC port must be 1024..65534 and TM port 1024..65535')
    if tm_port in (rpc_port, rpc_port + 1):
        raise ValueError('TM port cannot overlap RPC or streaming port')
    lines = _ss_listeners()
    result = {}
    for label, port in (('rpc', rpc_port), ('streaming', rpc_port + 1), ('tm', tm_port)):
        pattern = re.compile(rf':{port}\s')
        owners = [line.strip() for line in lines if pattern.search(line)]
        result[label] = {'port': port,
                         'listening': bool(owners) or _socket_listening(port, host),
                         'ss_details': owners}
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('status', 'launch', 'client'), default='status')
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--rpc-port', type=int, default=2000)
    p.add_argument('--tm-port', type=int, default=8100)
    a = p.parse_args(argv)
    ports = inspect_ports(a.rpc_port, a.tm_port, a.host)
    for name, data in ports.items():
        status = 'LISTENING' if data['listening'] else 'free / not reachable'
        print(f"{name}: TCP {data['port']} {status}", flush=True)
        for line in data['ss_details']:
            print('    '+line, flush=True)
    if a.mode == 'launch' and (ports['rpc']['listening'] or ports['streaming']['listening']):
        p.error('Refusing to start a second CARLA on busy RPC/streaming ports. '
                'Inspect ss -ltnp and ps -ef, reuse the existing server if appropriate, '
                'or choose a different CARLA_RPC_PORT.')
    if a.mode == 'client' and not ports['rpc']['listening']:
        p.error('No CARLA RPC listener detected. Start a server first, then use '
                'python -m ride.doctor --server to confirm its version.')
    if a.mode == 'client' and ports['tm']['listening']:
        print('WARNING: TM port already listening; check whether another CARLA '
              'client owns this Traffic Manager. Prefer a dedicated free TM port.', flush=True)


if __name__ == '__main__':
    main()
