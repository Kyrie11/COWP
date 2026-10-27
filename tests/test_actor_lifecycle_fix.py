"""CARLA actor ownership/initialization/cleanup regressions, using mocked API."""
from types import SimpleNamespace as NS
import pytest
from ride.carla.actor_lifecycle import (check_spawn_state, inventory,
                                        destroy_verified, ROLE_PREFIX)
from ride.carla.smoke_validate import validate

class Loc:
    def __init__(self,x,y,z=0):self.x,self.y,self.z=x,y,z

class Actor:
    def __init__(self,id,x=0,y=0,role='',kind='vehicle.tesla.model3'):
        self.id=id;self.type_id=kind;self.attributes={'role_name':role}
        self._loc=Loc(x,y)
    def get_location(self):return self._loc

class World:
    def __init__(self,actors=None,sync=True):
        self.actors=list(actors or []);self.sync=sync;self.ticks=0
    def get_actors(self):return list(self.actors)
    def get_settings(self):return NS(synchronous_mode=self.sync)
    def tick(self):self.ticks+=1
    def wait_for_tick(self,seconds=10):self.ticks+=1

class Client:
    def __init__(self,world,fail=False):self.world=world;self.fail=fail;self.destroyed=[]
    def apply_batch_sync(self,commands,flag):
        assert flag is False
        if not self.fail:
            ids=[cmd.value for cmd in commands]
            self.destroyed+=ids
            self.world.actors=[a for a in self.world.actors if a.id not in ids]
        return [NS(error='') for _ in commands]

class Carla:
    class command:
        class DestroyActor:
            def __init__(self,id):self.value=id

def test_zero_location_before_first_frame_is_rejected():
    actors={'ego':Actor(1),'follower_a':Actor(2),'follower_b':Actor(3)}
    anchor={role:NS(transform=NS(location=Loc(x,y))) for role,x,y in
            [('ego',-511,220),('follower_a',-511,200),('follower_b',-511,182)]}
    with pytest.raises(RuntimeError,match='CARLA_SPAWN_NOT_INITIALIZED'):
        check_spawn_state(actors,anchor)
    actors['ego']._loc=Loc(-511,220)
    actors['follower_a']._loc=Loc(-511,200)
    actors['follower_b']._loc=Loc(-511,182)
    assert len(check_spawn_state(actors,anchor))==3

def test_actor_inventory_marks_ours_but_never_claims_foreign():
    world=World([Actor(1,2,3,'ride.abc.ego'),Actor(2,4,6,'hero')])
    rows=inventory(world)
    assert rows[0]['role_name'].startswith(ROLE_PREFIX)
    assert not rows[1]['role_name'].startswith(ROLE_PREFIX)

def test_batch_destroy_verifies_actor_disappearance_in_sync_world():
    world=World([Actor(1),Actor(2),Actor(3)])
    client=Client(world)
    result=destroy_verified(client,world,[1,2],carla_api=Carla)
    assert result['remaining_ids']==[]
    assert client.destroyed==[1,2]
    assert world.ticks==1
    assert [a.id for a in world.actors]==[3]

def test_failed_cleanup_does_not_silently_pass():
    world=World([Actor(1)])
    with pytest.raises(RuntimeError,match='CLEANUP_INCOMPLETE'):
        destroy_verified(Client(world,fail=True),world,[1],carla_api=Carla)

def test_read_only_inventory_does_not_tick_or_delete():
    world=World([Actor(1)])
    inventory(world)
    assert world.ticks==0

def test_incomplete_smoke_with_empty_trace_is_not_success():
    rows=[{'type':'manifest'}, {'type':'trial','trace':[],
           'metrics':{'valid':False,'reasons':['overlap_after_warmup']}},
          {'type':'run_end','completed':True,'trial_count':1}]
    ok,report=validate(rows)
    assert not ok and report['valid_trajectories']==0

def test_completed_valid_smoke_passes():
    rows=[{'type':'manifest'}, {'type':'trial','trace':[{'t':0.05}],
           'metrics':{'valid':True,'reasons':[]}},
          {'type':'run_end','completed':True,'trial_count':1}]
    assert validate(rows)[0]
