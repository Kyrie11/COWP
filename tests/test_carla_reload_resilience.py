"""CARLA reset regression tests with fakes; requires no CARLA server."""
from types import SimpleNamespace
import pytest

from ride.carla.run import _prepare_trial_world
from ride.carla.health import inspect


class FakeWorld:
    def __init__(self, name='Carla/Maps/Town04', count=0):
        self.name=name
        self.count=count
    def get_map(self):
        return SimpleNamespace(name=self.name)
    def get_actors(self):
        return SimpleNamespace(filter=lambda pattern: [object()] * self.count)
    def get_snapshot(self):
        return SimpleNamespace(frame=123)
    def get_settings(self):
        return SimpleNamespace(synchronous_mode=False, fixed_delta_seconds=None,
                               no_rendering_mode=False)


class FakeClient:
    def __init__(self, world, error=None):
        self.world=world
        self.error=error
        self.reload_calls=0
        self.load_calls=0
    def get_world(self):
        return self.world
    def reload_world(self):
        self.reload_calls+=1
        if self.error:raise self.error
        return self.world
    def load_world(self, name):
        self.load_calls+=1
        return self.world
    def get_server_version(self):
        return '0.9.16'
    def get_client_version(self):
        return '0.9.16'


def args(mode='actors', town='Town04'):
    return SimpleNamespace(world_reset=mode,town=town)


def test_default_actor_reset_never_reload_even_for_same_town():
    client=FakeClient(FakeWorld())
    assert _prepare_trial_world(client,args(),'behavior','early_slow') is client.world
    assert client.reload_calls==0 and client.load_calls==0


def test_actor_reset_refuses_foreign_vehicles():
    client=FakeClient(FakeWorld(count=1))
    with pytest.raises(RuntimeError,match='WORLD_NOT_EMPTY'):
        _prepare_trial_world(client,args(),'tm','early_slow')
    assert client.reload_calls==0


def test_actor_reset_never_switches_map():
    client=FakeClient(FakeWorld(name='Town03'))
    with pytest.raises(RuntimeError,match='MAP_MISMATCH'):
        _prepare_trial_world(client,args(),'tm','early_slow')
    assert client.load_calls==0 and client.reload_calls==0


def test_opt_in_strict_reload_calls_reload_once():
    client=FakeClient(FakeWorld())
    assert _prepare_trial_world(client,args('reload'),'tm','early_slow') is client.world
    assert client.reload_calls==1


def test_reload_timeout_explains_server_fault_instead_of_reporting_trial_success():
    client=FakeClient(FakeWorld(),RuntimeError('time-out of 40000ms'))
    with pytest.raises(RuntimeError,match='CARLA_WORLD_RELOAD_FAILED'):
        _prepare_trial_world(client,args('reload'),'tm','early_slow')


def test_read_only_health_inspection_does_not_reload():
    client=FakeClient(FakeWorld())
    data=inspect(client)
    assert data['map']=='Carla/Maps/Town04'
    assert data['server_version']=='0.9.16'
    assert data['vehicles']==0
    assert client.reload_calls==0
