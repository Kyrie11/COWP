from __future__ import annotations

import numpy as np

from cowp.core.constants import PriorityRelation
from cowp.label.burden import compute_burden
from cowp.label.trajectory_primitives import constant_accel_trajectory
from cowp.label.witness import certify_witnesses


def test_witness_positive_for_cut_in_hard_brake(toy_scene, cfg):
    H = int(cfg["time"]["future_steps"])
    K = int(cfg["limits"]["max_candidates"])
    A = int(cfg["limits"]["max_critical_agents"])
    M = int(cfg["limits"]["max_natural_alternatives"])
    R = int(cfg["limits"]["max_safe_responses"])
    ego_cur = toy_scene.states[0, 10]
    ag_cur = toy_scene.states[1, 10]
    # Ego cuts into agent lane and slows in front, creating unsafe natural path.
    ego = constant_accel_trajectory(ego_cur, H, 0.1, accel=0.0, lateral_offset=3.5)
    natural_tr = constant_accel_trajectory(ag_cur, H, 0.1, accel=0.0)
    emergency = constant_accel_trajectory(ag_cur, H, 0.1, accel=-6.0)
    candidates = {"trajectory": np.zeros((K,H,7), dtype=np.float32), "valid": np.zeros(K, dtype=bool), "ego_utility_prior": np.zeros(K)}
    candidates["trajectory"][0] = ego
    candidates["valid"][0] = True
    critical = {"track_index": np.array([1]+[-1]*(A-1), dtype=np.int32), "valid": np.array([1]+[0]*(A-1), dtype=bool), "base_priority": np.array([PriorityRelation.AGENT_PRIORITY]+[0]*(A-1), dtype=np.int32)}
    natural = {"traj": np.zeros((A,M,H,7), dtype=np.float32), "valid": np.zeros((A,M), dtype=bool), "weight": np.zeros((A,M), dtype=np.float32), "burden_neutral": np.zeros((A,M), dtype=np.float32), "beta": np.ones(A, dtype=np.float32)*0.2}
    natural["traj"][0,0] = natural_tr
    natural["valid"][0,0] = True
    natural["weight"][0,0] = 1.0
    response = {"traj": np.zeros((K,A,R,H,7), dtype=np.float32), "valid": np.zeros((K,A,R), dtype=bool), "is_safe": np.zeros((K,A,R), dtype=bool), "burden_total": np.zeros((K,A,R), dtype=np.float32), "burden_components": np.zeros((K,A,R,6), dtype=np.float32)}
    response["traj"][0,0,0] = emergency
    response["valid"][0,0,0] = True
    response["is_safe"][0,0,0] = True
    b, comps = compute_burden(emergency, ego, cfg, object_type=1, natural_ref=natural_tr)
    response["burden_total"][0,0,0] = max(b, 1.0)
    response["burden_components"][0,0,0] = comps
    wit = certify_witnesses(toy_scene, candidates, critical, natural, response, cfg)
    assert wit["exists"][0,0]


def test_collision_candidate_not_false_safe(toy_scene, cfg):
    H = int(cfg["time"]["future_steps"])
    K = int(cfg["limits"]["max_candidates"])
    A = int(cfg["limits"]["max_critical_agents"])
    M = int(cfg["limits"]["max_natural_alternatives"])
    R = int(cfg["limits"]["max_safe_responses"])
    ego_cur = toy_scene.states[0, 10]
    ego = constant_accel_trajectory(ego_cur, H, 0.1, accel=0.0)
    candidates = {"trajectory": np.zeros((K,H,7), dtype=np.float32), "valid": np.zeros(K, dtype=bool), "ego_utility_prior": np.zeros(K)}
    candidates["trajectory"][0] = ego
    candidates["valid"][0] = True
    critical = {"track_index": np.array([-1]*A, dtype=np.int32), "valid": np.zeros(A, dtype=bool), "base_priority": np.zeros(A, dtype=np.int32)}
    natural = {"traj": np.zeros((A,M,H,7), dtype=np.float32), "valid": np.zeros((A,M), dtype=bool), "weight": np.zeros((A,M), dtype=np.float32), "burden_neutral": np.zeros((A,M), dtype=np.float32), "beta": np.ones(A, dtype=np.float32)*0.65}
    response = {"traj": np.zeros((K,A,R,H,7), dtype=np.float32), "valid": np.zeros((K,A,R), dtype=bool), "is_safe": np.zeros((K,A,R), dtype=bool), "burden_total": np.zeros((K,A,R), dtype=np.float32), "burden_components": np.zeros((K,A,R,6), dtype=np.float32)}
    wit = certify_witnesses(toy_scene, candidates, critical, natural, response, cfg)
    assert not bool(wit["candidate_false_safe"][0])
