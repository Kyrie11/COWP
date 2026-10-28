"""Infrastructure regression tests; no real CARLA server is used."""
from __future__ import annotations

import os
from pathlib import Path
import socket
import subprocess
import sys

import pytest

from ride.carla.ports import inspect_ports

ROOT = Path(__file__).resolve().parents[1]


def source_env(extra=None):
    env = os.environ.copy()
    env.pop('RIDE_CARLA_ROOT', None)
    env.pop('RIDE_ALLOW_CUSTOM_CARLA_DIR', None)
    env.update(extra or {})
    p = subprocess.run(['bash', '-c',
                        'source scripts/00_env.sh || exit $?; '
                        'printf "CARLA=%s\\nPYTHONPATH=%s\\nTM=%s\\n" '
                        '"$CARLA_ROOT" "$PYTHONPATH" "$CARLA_TM_PORT"'],
                       env=env, cwd=ROOT, capture_output=True, text=True)
    return p


def test_inherited_legacy_root_and_pythonpath_are_corrected():
    p = source_env({'CARLA_ROOT': '/data0/senzeyu2/dataset/CRADrive/CARLA_0.9.15',
                    'PYTHONPATH': '/some/CRADrive/CARLA_0.9.15/PythonAPI/carla:/other'})
    assert p.returncode == 0, p.stderr
    assert 'CARLA=/data0/senzeyu2/dataset/CARLA_0.9.16' in p.stdout
    assert 'CARLA_0.9.15' not in p.stdout
    assert 'RIDE WARNING' in p.stderr
    assert '/other' in p.stdout


def test_explicit_correct_root_is_respected():
    p = source_env({'CARLA_ROOT': '/unused/CARLA_0.9.16',
                    'RIDE_CARLA_ROOT': '/explicit/CARLA_0.9.16'})
    assert p.returncode == 0, p.stderr
    assert 'CARLA=/explicit/CARLA_0.9.16' in p.stdout
    assert 'TM=8100' in p.stdout


def test_invalid_version_fails_fast():
    p = source_env({'RIDE_CARLA_ROOT': '/somewhere/CARLA_0.9.14'})
    assert p.returncode != 0
    assert 'expected CARLA 0.9.16' in p.stderr


def test_port_checker_recognises_listener_and_prevents_client_tm_overlap():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        s.listen(1)
        port = s.getsockname()[1]
        # Port cannot equal TM port; select a second unrelated free port.
        tm_port = port - 2 if port >= 1030 else port + 3
        data = inspect_ports(port, tm_port)
        assert data['rpc']['listening']
        with pytest.raises(ValueError, match='cannot overlap'):
            inspect_ports(port, port + 1)


def test_start_script_refuses_busy_rpc_without_launching_fake_server(tmp_path):
    root = tmp_path / 'CARLA_0.9.16'
    root.mkdir()
    marker = tmp_path / 'would_have_started'
    exe = root / 'CarlaUE4.sh'
    exe.write_text('#!/usr/bin/env bash\nprintf started > "'+str(marker)+'"\n')
    exe.chmod(0o755)
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        s.listen(1)
        port = s.getsockname()[1]
        tm_port = port - 2 if port >= 1030 else port + 3
        env = os.environ.copy()
        env.update(RIDE_CARLA_ROOT=str(root), CARLA_RPC_PORT=str(port),
                   CARLA_TM_PORT=str(tm_port), CARLA_HOST='127.0.0.1')
        p = subprocess.run(['bash', 'scripts/02_start_carla_headless.sh'],
                           env=env, cwd=ROOT, capture_output=True, text=True, timeout=10)
        assert p.returncode != 0
        assert 'Refusing to start a second CARLA' in p.stderr
        assert not marker.exists()


def test_start_script_forwards_custom_port_and_adapter(tmp_path):
    root = tmp_path / 'CARLA_0.9.16'
    root.mkdir()
    exe = root / 'CarlaUE4.sh'
    exe.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@"\n')
    exe.chmod(0o755)
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        port = s.getsockname()[1]
    tm_port = port - 2 if port >= 1030 else port + 3
    env = os.environ.copy()
    env.update(RIDE_CARLA_ROOT=str(root), CARLA_RPC_PORT=str(port),
               CARLA_TM_PORT=str(tm_port), CARLA_GPU_ADAPTER='1')
    p = subprocess.run(['bash', 'scripts/02_start_carla_headless.sh'],
                       env=env, cwd=ROOT, capture_output=True, text=True, timeout=10)
    assert p.returncode == 0, p.stderr
    assert '-carla-rpc-port='+str(port) in p.stdout
    assert '-graphicsadapter=1' in p.stdout
    assert '-RenderOffScreen' in p.stdout


def test_one_time_map_shell_requires_explicit_consent_before_any_client_action():
    env = os.environ.copy()
    env.pop('RIDE_ALLOW_INITIAL_MAP_LOAD', None)
    p = subprocess.run(['bash', 'scripts/02c_prepare_carla_town.sh', 'Town04'],
                       env=env, cwd=ROOT, capture_output=True, text=True, timeout=10)
    assert p.returncode == 2
    assert 'RIDE_MAP_SETUP_REFUSED' in p.stderr
    assert 'RIDE_ALLOW_INITIAL_MAP_LOAD=1' in p.stderr
