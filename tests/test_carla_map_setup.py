"""Guarded map bootstrap unit tests. All fakes; no CARLA engine required."""
from types import SimpleNamespace
import pytest

from ride.carla.map_setup import prepare_map, _map_asset_name


class FakeWorld:
    def __init__(self, name='Carla/Maps/Town10HD_Opt', actors=(), synchronous=False):
        self.name = name
        self.actors = actors
        self.synchronous = synchronous

    def get_map(self):
        return SimpleNamespace(name=self.name)

    def get_settings(self):
        return SimpleNamespace(synchronous_mode=self.synchronous)

    def get_actors(self):
        return [SimpleNamespace(type_id=t) for t in self.actors]


class FakeClient:
    def __init__(self, world=None, available=None, error=None):
        self.world = world or FakeWorld()
        self.available = available if available is not None else [
            '/Game/Carla/Maps/Town04', '/Game/Carla/Maps/Town10HD_Opt']
        self.error = error
        self.load_calls = []

    def get_client_version(self):
        return '0.9.16'

    def get_server_version(self):
        return '0.9.16'

    def get_world(self):
        return self.world

    def get_available_maps(self):
        return self.available

    def load_world(self, name):
        self.load_calls.append(name)
        if self.error:
            raise self.error
        self.world.name = 'Carla/Maps/' + name


def test_map_asset_syntax():
    assert _map_asset_name('/Game/Carla/Maps/Town04') == 'Town04'
    assert _map_asset_name('/Game/Carla/Maps/Town04.Town04') == 'Town04'


def test_matching_map_is_noop():
    c = FakeClient(FakeWorld(name='Carla/Maps/Town04', actors=['vehicle.x']))
    assert prepare_map(c, 'Town04')['action'] == 'already_loaded'
    assert not c.load_calls


def test_mismatch_requires_explicit_approval():
    c = FakeClient()
    with pytest.raises(RuntimeError, match='EXPLICIT_OPT_IN'):
        prepare_map(c, 'Town04')
    assert not c.load_calls


@pytest.mark.parametrize('actors', [
    ('vehicle.tesla.model3',), ('walker.pedestrian.001',),
    ('sensor.camera.rgb',), ('controller.ai.walker',), ('unknown.actor',)
])
def test_active_actors_stop_switch(actors):
    c = FakeClient(FakeWorld(actors=('traffic.traffic_light',) + actors))
    with pytest.raises(RuntimeError, match='ACTIVE_ACTORS'):
        prepare_map(c, 'Town04', allow_world_load=True)
    assert not c.load_calls


def test_synchronous_world_stops_switch():
    c = FakeClient(FakeWorld(synchronous=True))
    with pytest.raises(RuntimeError, match='SYNCHRONOUS_WORLD'):
        prepare_map(c, 'Town04', allow_world_load=True)
    assert not c.load_calls


def test_missing_map_stops_switch():
    c = FakeClient(available=['/Game/Carla/Maps/Town05'])
    with pytest.raises(RuntimeError, match='MAP_NOT_INSTALLED'):
        prepare_map(c, 'Town04', allow_world_load=True)
    assert not c.load_calls


def test_one_time_switch_succeeds_and_verifies():
    c = FakeClient(FakeWorld(actors=['static.prop.tree', 'traffic.traffic_light']))
    result = prepare_map(c, 'Town04', allow_world_load=True)
    assert c.load_calls == ['Town04']
    assert result['before'] == 'Carla/Maps/Town10HD_Opt'
    assert result['after'] == 'Carla/Maps/Town04'
    assert result['action'] == 'loaded_once'


def test_switch_exception_no_retry():
    c = FakeClient(error=RuntimeError('Signal 11 / timeout'))
    with pytest.raises(RuntimeError, match='do NOT auto-retry'):
        prepare_map(c, 'Town04', allow_world_load=True)
    assert c.load_calls == ['Town04']


def test_invalid_map_name_stops_switch():
    c = FakeClient()
    with pytest.raises(ValueError, match='INVALID_CARLA_TOWN'):
        prepare_map(c, '../Town04', allow_world_load=True)
    assert not c.load_calls


def test_map_verification_failure_is_not_success():
    class WrongMapClient(FakeClient):
        def load_world(self, name):
            self.load_calls.append(name)
            # Simulate a build ignoring a requested map.

    c = WrongMapClient()
    with pytest.raises(RuntimeError, match='LOAD_VERIFICATION_FAILED'):
        prepare_map(c, 'Town04', allow_world_load=True)
    assert c.load_calls == ['Town04']


def test_static_objects_do_not_block_switch():
    c = FakeClient(FakeWorld(actors=['traffic.stop', 'traffic.traffic_light', 'static.prop.bench']))
    assert prepare_map(c, 'Town04', allow_world_load=True)['action'] == 'loaded_once'
