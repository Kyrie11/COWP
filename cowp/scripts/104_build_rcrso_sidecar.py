from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from cowp.core.config import load_config
from cowp.data.dataset import COWPNpzDataset
from cowp.label.safe_responses import expand_root_control_knots
from cowp.label.audit_relevance import canonical_root_weights
from cowp.models.recourse_set_operator import RCRSOConfig, build_rcrso_features_np
from cowp.waymax_eval.policy_wrapper import (
    _constant_velocity_trajectory_from_state_np,
    _root_conditioned_control_reachable_response_profiles_np,
    _shift_append_terminal_reference_np,
    _verified_root_conditioned_recourse_set_profiles_np,
)


def _scenario_id(path: Path, data: dict[str, np.ndarray]) -> str:
    for key in ("scenario/id", "womd/scenario/id", "scenario_id"):
        if key in data:
            x = np.asarray(data[key]).reshape(-1)
            if x.size:
                v = x[0]
                if isinstance(v, (bytes, np.bytes_)):
                    return v.decode("utf-8")
                return str(v)
    return path.stem


def _load_ids(paths: list[str]) -> set[str]:
    out: set[str] = set()
    for raw in paths:
        p = Path(raw)
        if not p.is_file():
            continue
        if p.suffix.lower() == ".json":
            obj = json.loads(p.read_text(encoding="utf-8"))
            vals = obj.get("scenario_ids", obj) if isinstance(obj, dict) else obj
            if isinstance(vals, list):
                out.update(str(x) for x in vals)
        else:
            out.update(x.strip() for x in p.read_text(encoding="utf-8").splitlines() if x.strip())
    return out


def _np_state11(data: dict[str, np.ndarray]) -> tuple[np.ndarray, int, np.ndarray]:
    hist = data.get("state/history")
    if hist is not None:
        h = np.asarray(hist, dtype=np.float32)
        if h.ndim == 4:
            h = h[0]
        cur = h[:, -1]
        state = np.zeros((cur.shape[0], 11), dtype=np.float32)
        state[:, 0:2] = cur[:, 0:2]
        state[:, 3:5] = cur[:, 7:9]
        state[:, 5] = np.linalg.norm(state[:, 3:5], axis=-1)
        state[:, 6] = cur[:, 6]
        state[:, 7:10] = cur[:, 3:6]
        state[:, 10] = cur[:, 10] if cur.shape[1] > 10 else 1.0
    else:
        def arr(name: str, default: float = 0.0) -> np.ndarray:
            x = data.get(f"state/current/{name}")
            if x is None:
                x = data.get(f"womd/state/current/{name}")
            if x is None:
                n = len(np.asarray(data.get("state/current/x", data.get("womd/state/current/x"))).reshape(-1))
                return np.full(n, default, dtype=np.float32)
            return np.asarray(x, dtype=np.float32).reshape(-1)
        x, y = arr("x"), arr("y")
        n = min(x.size, y.size)
        state = np.zeros((n, 11), dtype=np.float32)
        state[:, 0], state[:, 1] = x[:n], y[:n]
        state[:, 3], state[:, 4] = arr("velocity_x")[:n], arr("velocity_y")[:n]
        state[:, 5] = np.linalg.norm(state[:, 3:5], axis=-1)
        yaw = data.get("state/current/bbox_yaw", data.get("state/current/heading"))
        if yaw is not None: state[:, 6] = np.asarray(yaw, dtype=np.float32).reshape(-1)[:n]
        state[:, 7], state[:, 8], state[:, 9] = arr("length",4.8)[:n], arr("width",1.9)[:n], arr("height",1.6)[:n]
        state[:, 10] = arr("valid",1.0)[:n]
    is_sdc = data.get("state/is_sdc", data.get("womd/state/is_sdc"))
    if is_sdc is not None:
        mask = np.asarray(is_sdc).reshape(-1)[: state.shape[0]].astype(bool)
        sdc = int(np.flatnonzero(mask)[0]) if mask.any() else 0
    else:
        sdc = 0
    typ = data.get("state/type", data.get("womd/state/type"))
    object_types = np.asarray(typ, dtype=np.int32).reshape(-1)[: state.shape[0]] if typ is not None else np.zeros(state.shape[0], np.int32)
    if object_types.size < state.shape[0]:
        object_types = np.pad(object_types, (0, state.shape[0]-object_types.size))
    return state, sdc, object_types


def _reshape_roadgraph_vector_field(
    value: np.ndarray,
    *,
    field_name: str,
    expected_points: int | None = None,
    preferred_width: int = 3,
) -> np.ndarray:
    """Restore raw WOMD roadgraph vector fields to ``[P,C]``.

    Formal compact tensor caches intentionally preserve raw tf.Example arrays.
    In that representation ``roadgraph_samples/{xyz,dir}`` may be flat ``[P*3]``
    rather than already shaped ``[P,3]``.  Waymax's dataloader restores this
    shape before simulator use; the V45 sidecar builder must do the same instead
    of indexing a flat raw feature as if it were 2-D.
    """
    a = np.asarray(value, dtype=np.float32)
    if a.ndim == 0:
        raise ValueError(f"{field_name} is scalar; expected a roadgraph vector array")
    # Remove only leading singleton batch dimensions.  Do not squeeze the final
    # component dimension because [P,1] is malformed for xyz/dir.
    while a.ndim > 2 and a.shape[0] == 1:
        a = a[0]
    if a.ndim == 1:
        if expected_points is not None and expected_points > 0 and a.size % expected_points == 0:
            width = int(a.size // expected_points)
            if width >= 2:
                return a.reshape(expected_points, width)
        if a.size % int(preferred_width) == 0:
            return a.reshape(-1, int(preferred_width))
        raise ValueError(
            f"{field_name} flat size {a.size} is incompatible with expected_points={expected_points} "
            f"and preferred_width={preferred_width}"
        )
    if a.ndim == 2:
        if a.shape[-1] >= 2:
            if expected_points is not None and expected_points > 0 and a.shape[0] != expected_points:
                # Raw/batched exports occasionally retain more than one leading
                # block.  A reshape is valid only when the total element count
                # preserves a point-major vector width.
                if a.size % expected_points == 0 and int(a.size // expected_points) >= 2:
                    return a.reshape(expected_points, int(a.size // expected_points))
                raise ValueError(
                    f"{field_name} has {a.shape[0]} points, expected {expected_points}; shape={a.shape}"
                )
            return a
        raise ValueError(f"{field_name} trailing width {a.shape[-1]} < 2; shape={a.shape}")
    # Non-singleton leading dimensions are not part of the formal cache contract.
    if a.size % int(preferred_width) == 0:
        flat = a.reshape(-1, int(preferred_width))
        if expected_points is None or expected_points <= 0 or flat.shape[0] == expected_points:
            return flat
    raise ValueError(f"Cannot restore {field_name} roadgraph shape {a.shape}")


def _roadgraph(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    xyz = data.get("roadgraph_samples/xyz", data.get("womd/roadgraph_samples/xyz"))
    if xyz is None:
        x = data.get("roadgraph_samples/x", data.get("womd/roadgraph_samples/x"))
        y = data.get("roadgraph_samples/y", data.get("womd/roadgraph_samples/y"))
        if x is None or y is None:
            return {"xy": np.zeros((0,2),np.float32), "heading": np.zeros(0,np.float32), "valid": np.zeros(0,bool), "types": np.zeros(0,np.int32)}
        xx = np.asarray(x, dtype=np.float32).reshape(-1)
        yy = np.asarray(y, dtype=np.float32).reshape(-1)
        if xx.size != yy.size:
            raise ValueError(f"roadgraph_samples/x,y size mismatch: {xx.size} vs {yy.size}")
        xy = np.stack([xx, yy], axis=-1).astype(np.float32)
    else:
        xyz2 = _reshape_roadgraph_vector_field(
            np.asarray(xyz), field_name="roadgraph_samples/xyz", preferred_width=3,
        )
        xy = np.asarray(xyz2[:, :2], dtype=np.float32)
    P = int(len(xy))
    valid=data.get("roadgraph_samples/valid",data.get("womd/roadgraph_samples/valid"))
    if valid is not None:
        vv = np.asarray(valid).reshape(-1)
        if vv.size < P:
            raise ValueError(f"roadgraph_samples/valid has {vv.size} points, expected at least {P}")
        v = vv[:P].astype(bool)
    else:
        v=np.ones(P,bool)
    typ=data.get("roadgraph_samples/type",data.get("womd/roadgraph_samples/type"))
    if typ is not None:
        tt = np.asarray(typ, dtype=np.int32).reshape(-1)
        if tt.size < P:
            raise ValueError(f"roadgraph_samples/type has {tt.size} points, expected at least {P}")
        t = tt[:P]
    else:
        t=np.zeros(P,np.int32)
    d=data.get("roadgraph_samples/dir",data.get("womd/roadgraph_samples/dir"))
    if d is not None:
        dd = _reshape_roadgraph_vector_field(
            np.asarray(d), field_name="roadgraph_samples/dir", expected_points=P, preferred_width=3,
        )
        heading=np.arctan2(dd[:P,1],dd[:P,0]).astype(np.float32)
    else:
        # Compatibility fallback only for old/debug caches without the official
        # direction field. Formal compact-5k caches are expected to carry dir.
        diff=np.gradient(xy,axis=0) if P>1 else np.zeros_like(xy)
        heading=np.arctan2(diff[:,1],diff[:,0]).astype(np.float32)
    finite = np.isfinite(xy).all(axis=-1) & np.isfinite(heading)
    v = v & finite
    return {"xy":xy,"heading":heading,"valid":v,"types":t}


def _critical_input_indices(data: dict[str, np.ndarray]) -> np.ndarray:
    """Return model/WOMD input rows for critical slots.

    ``track_index`` is a Scenario-proto track index and is not guaranteed to be
    the row in raw WOMD state tensors.  ``COWPNpzDataset.load`` creates the
    aligned ``input_index`` when object ids are available; all model-facing code
    uses it preferentially, and the sidecar must follow the same contract.
    """
    raw = data.get("cowp/critical/input_index", data.get("cowp/critical/track_index"))
    if raw is None:
        return np.zeros((0,), dtype=np.int64)
    return np.asarray(raw, dtype=np.int64).reshape(-1)


def _one_step_cv_successor_for_environment(state: np.ndarray, sdc_index: int, cfg: dict) -> np.ndarray:
    """Match the non-ego part of the online causal successor surrogate."""
    nxt = np.asarray(state, dtype=np.float32).copy()
    if nxt.ndim != 2:
        return nxt
    dt = max(float(cfg.get("time", {}).get("dt", 0.1)), 1.0e-6)
    valid = nxt[:, 10] > 0.5 if nxt.shape[1] > 10 else np.ones(nxt.shape[0], dtype=bool)
    if 0 <= int(sdc_index) < nxt.shape[0]:
        valid[int(sdc_index)] = False
    if nxt.shape[1] >= 5:
        nxt[valid, 0:2] = nxt[valid, 0:2] + nxt[valid, 3:5] * dt
    return nxt


def _sidecar_roadgraph_subset(
    road: dict[str, np.ndarray],
    root: np.ndarray,
    ego_current: np.ndarray,
    ego_shifted: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Store a compact roadgraph subset without changing drivable-mask truth.

    Same-root longitudinal responses stay on the supplied root geometry.  Keeping
    all roadgraph points within a generous 70 m box around root/current/shifted
    ego therefore contains every point that the frozen 60 m fallback can use. If
    that region has no valid vehicle-lane center point, include the nearest global
    lane point so the downstream predicate fails the same way as the full map
    instead of treating an empty sidecar roadgraph as vacuously drivable.
    """
    xy=np.asarray(road.get("xy",np.zeros((0,2),np.float32)),np.float32).reshape(-1,2)
    heading=np.asarray(road.get("heading",np.zeros(len(xy),np.float32)),np.float32).reshape(-1)
    types=np.asarray(road.get("types",np.zeros(len(xy),np.int32)),np.int32).reshape(-1)
    valid=np.asarray(road.get("valid",np.zeros(len(xy),bool)),bool).reshape(-1)
    P=min(len(xy),len(heading),len(types),len(valid))
    xy,heading,types,valid=xy[:P],heading[:P],types[:P],valid[:P]
    if P<=0:
        return xy,heading,types,valid
    contexts=[np.asarray(root,np.float32)[:,:2],np.asarray(ego_current,np.float32)[:,:2],np.asarray(ego_shifted,np.float32)[:,:2]]
    allxy=np.concatenate([q for q in contexts if q.size],axis=0)
    finite_ctx=np.isfinite(allxy).all(axis=-1)
    if not np.any(finite_ctx):
        return xy,heading,types,valid
    allxy=allxy[finite_ctx]
    lo=np.nanmin(allxy,axis=0)-70.0; hi=np.nanmax(allxy,axis=0)+70.0
    keep=(xy[:,0]>=lo[0])&(xy[:,0]<=hi[0])&(xy[:,1]>=lo[1])&(xy[:,1]<=hi[1])
    # Vehicle lane centers are WOMD roadgraph types 1 (freeway) and 2 (surface).
    lane=valid & np.isin(types,np.asarray([1,2],np.int32)) & np.isfinite(xy).all(axis=-1)
    if not np.any(lane & keep) and np.any(lane):
        d=np.linalg.norm(xy-allxy[0][None,:],axis=-1)
        nearest=int(np.argmin(np.where(lane,d,np.inf)))
        keep[nearest]=True
    inds=np.flatnonzero(keep)
    return xy[inds],heading[inds],types[inds],valid[inds]


def _response_to_normalized_knots(root: np.ndarray, response: np.ndarray, cfg: dict, knot_count: int) -> np.ndarray:
    dt=max(float(cfg.get("time",{}).get("dt",0.1)),1e-6)
    rs=np.linalg.norm(np.asarray(root)[:,3:5],axis=-1)
    qs=np.linalg.norm(np.asarray(response)[:,3:5],axis=-1)
    h=min(len(rs),len(qs))
    dv=(qs[:h]-rs[:h]).astype(np.float32)
    accel=np.diff(np.concatenate([[0.0],dv])).astype(np.float32)/dt
    idx=np.linspace(0,max(h-1,0),knot_count).round().astype(int)
    knots=accel[idx]
    c=cfg.get("candidate",{})
    dec=max(float(c.get("max_decel_mps2",6.0)),1e-6); acc=max(float(c.get("max_accel_mps2",4.0)),1e-6)
    return np.clip(np.where(knots<0,knots/dec,knots/acc),-1,1).astype(np.float32)


def _retained_roots_from_canonical(
    canonical_weight: np.ndarray,
    valid: np.ndarray,
    required_mass: float,
    min_roots: int,
    max_roots: int,
) -> list[int]:
    """Select the offline retained-root set from the canonical probability measure.

    The compact cache stores raw natural weights as well as the canonical
    floor-smoothed audit/transport weights.  V45 must not re-apply only the
    p_min threshold and omit probability-floor smoothing: that would train the
    recourse operator on a different universal-root set than the frozen
    certificate.  Offline natural roots are already geometrically de-duplicated
    by the label generator, so only the frozen max-root cap and cumulative-mass
    rule remain here.
    """
    w=np.where(np.asarray(valid,bool),np.maximum(np.asarray(canonical_weight,float),0.0),0.0)
    order=np.argsort(-w,kind="stable")
    eligible=[int(j) for j in order if float(w[j])>0.0]
    if int(max_roots)>0:
        eligible=eligible[:int(max_roots)]
    if len(eligible)<int(min_roots):
        return []
    total=float(sum(float(w[j]) for j in eligible))
    if total+1.0e-9<float(required_mass):
        return []
    out=[]; mass=0.0
    for j in eligible:
        out.append(int(j)); mass+=float(w[j])
        if len(out)>=int(min_roots) and mass+1.0e-9>=float(required_mass):
            break
    return out


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--cache-dir",required=True)
    ap.add_argument("--output-root",required=True)
    ap.add_argument("--split",choices=["train","val","heldout"],required=True)
    ap.add_argument("--label-config",default="configs/label_cowp_v16_8.yaml")
    ap.add_argument("--data-config",default="configs/data.yaml")
    ap.add_argument("--eval-config",default="configs/eval_cowp_v16_8.yaml")
    ap.add_argument("--max-scenes",type=int,default=None)
    ap.add_argument("--max-examples-per-scene",type=int,default=64)
    ap.add_argument("--max-positive-controls",type=int,default=32)
    ap.add_argument("--max-negative-controls",type=int,default=32)
    ap.add_argument("--rich-sobol-proposals",type=int,default=32)
    ap.add_argument("--control-knots",type=int,default=8)
    ap.add_argument("--environment-cap",type=int,default=24)
    ap.add_argument("--forbidden-id-file",action="append",default=[])
    ap.add_argument("--num-shards",type=int,default=1)
    ap.add_argument("--shard-index",type=int,default=0)
    args=ap.parse_args()
    if args.num_shards < 1 or not (0 <= args.shard_index < args.num_shards):
        raise ValueError(f"invalid shard {args.shard_index}/{args.num_shards}")
    cfg=load_config(args.label_config,args.data_config,args.eval_config)
    pcfg=cfg.get("planning",{}); ncfg=cfg.get("natural",{}); rcfg=cfg.get("response",{}).get("root_conditioned_transport",{})
    required_root_mass=float(np.clip(1.0-float(pcfg.get("set_transport_cvar_tail_mass",cfg.get("ncf",{}).get("cvar_tail_mass",0.25))),0.0,1.0))
    minimum_root_count=max(int(ncfg.get("certificate_min_low_burden_roots",2)),1)
    max_roots_per_agent=max(int(rcfg.get("max_roots_per_agent",24)),0)
    forbidden=_load_ids(list(args.forbidden_id_file))
    ds=COWPNpzDataset(args.cache_dir)
    outdir=Path(args.output_root)/args.split; outdir.mkdir(parents=True,exist_ok=True)
    rcrso_cfg=RCRSOConfig(control_knots=int(args.control_knots))
    manifest=[]; counts={"scenes":0,"examples":0,"forbidden_skipped":0,"positive_examples":0,"analytic_nonempty":0,"rich_verified":0}
    try:
        import torch
        sobol=torch.quasirandom.SobolEngine(dimension=int(args.control_knots),scramble=False)
        sobol_bank=(sobol.draw(int(args.rich_sobol_proposals)).cpu().numpy().astype(np.float32)*2.0-1.0) if args.rich_sobol_proposals>0 else np.zeros((0,args.control_knots),np.float32)
    except Exception:
        rng=np.random.default_rng(0); sobol_bank=rng.uniform(-1,1,size=(max(args.rich_sobol_proposals,0),args.control_knots)).astype(np.float32)

    for scene_i,path in enumerate(ds.paths):
        if scene_i % int(args.num_shards) != int(args.shard_index):
            continue
        if args.max_scenes is not None and counts["scenes"]>=args.max_scenes: break
        data=ds.load(scene_i,None); sid=_scenario_id(path,data)
        if sid in forbidden:
            counts["forbidden_skipped"]+=1; continue
        required=("cowp/candidates/trajectory","cowp/candidates/valid","cowp/critical/track_index","cowp/critical/valid","cowp/natural/traj","cowp/natural/weight","cowp/natural/source","cowp/natural/valid","cowp/natural/beta","cowp/response/traj","cowp/response/valid","cowp/response/is_safe","cowp/response/is_low_burden")
        if any(k not in data for k in required): continue
        state,sdc,obj=_np_state11(data); road=_roadgraph(data)
        cand=np.asarray(data["cowp/candidates/trajectory"],np.float32); cvalid=np.asarray(data["cowp/candidates/valid"],bool)
        crit=_critical_input_indices(data); critv=np.asarray(data["cowp/critical/valid"],bool).reshape(-1)
        nat=np.asarray(data["cowp/natural/traj"],np.float32); nw=np.asarray(data["cowp/natural/weight"],np.float32); ns=np.asarray(data["cowp/natural/source"],int); nv=np.asarray(data["cowp/natural/valid"],bool); beta=np.asarray(data["cowp/natural/beta"],float)
        cached_canonical=data.get("cowp/audit/canonical_root_weight",data.get("cowp/transport/canonical_root_weight"))
        if cached_canonical is not None and np.asarray(cached_canonical).shape==nw.shape:
            canonical_w=np.asarray(cached_canonical,np.float32)
        else:
            canonical_w=canonical_root_weights({"valid":nv,"weight":nw},cfg).astype(np.float32)
        rt=np.asarray(data["cowp/response/traj"],np.float32); rv=np.asarray(data["cowp/response/valid"],bool); rsafe=np.asarray(data["cowp/response/is_safe"],bool); rlow=np.asarray(data["cowp/response/is_low_burden"],bool)
        rroot=np.asarray(data.get("cowp/response/root_index", data.get("cowp/transport/response_root_index", np.full(rv.shape,-1))),int)
        rbur=np.asarray(data.get("cowp/response/burden_total",np.zeros(rv.shape)),float)
        pair_rel=np.asarray(data.get("cowp/audit/pair_relevant",np.ones((cand.shape[0],crit.shape[0]),bool)),bool)
        counts["scenes"]+=1; made=0
        for k in range(min(cand.shape[0],cvalid.size)):
            if not cvalid[k]: continue
            ego_cur=cand[k]; ego_shift=_shift_append_terminal_reference_np(ego_cur,float(cfg.get("time",{}).get("dt",0.1)))
            # Stage-0 FullHypothesisRootCoverage is meaningful only when a
            # candidate hypothesis contains *all* retained roots of every included
            # audit-relevant actor.  Never truncate a hypothesis midway merely to
            # hit an example-count budget.  The budget is therefore applied only
            # between complete candidate groups; the first group may exceed it.
            group_contexts=[]
            for a in range(min(crit.size,critv.size,nat.shape[0])):
                j=int(crit[a])
                if not critv[a] or not (0<=j<state.shape[0]) or j==sdc or not pair_rel[k,a]:
                    continue
                retained=_retained_roots_from_canonical(
                    canonical_w[a],nv[a],required_root_mass,minimum_root_count,max_roots_per_agent
                )
                if not retained:
                    # Online interaction support cannot be ready if the frozen
                    # canonical root-count/mass contract itself fails.  Do not
                    # silently drop that actor and make FullHypothesisRootCoverage
                    # easier than the online certificate.
                    group_contexts=[]
                    break
                for m in retained:
                    group_contexts.append((int(a),int(j),int(m)))
            if not group_contexts:
                continue
            if made>0 and made+len(group_contexts)>int(args.max_examples_per_scene):
                break
            counts["hypothesis_groups"] = int(counts.get("hypothesis_groups",0)) + 1
            for a,j,m in group_contexts:
                    root=nat[a,m]
                    env=[]
                    successor_for_env=_one_step_cv_successor_for_environment(state,sdc,cfg)
                    dists=np.linalg.norm(state[:,:2]-state[j,:2][None],axis=-1); order=np.argsort(dists)
                    for e in order:
                        e=int(e)
                        if e in (sdc,j) or state[e,10]<=0.5: continue
                        curcv=_constant_velocity_trajectory_from_state_np(state,e,len(root),cfg)
                        if curcv is None: continue
                        shiftedcv=_constant_velocity_trajectory_from_state_np(successor_for_env,e,len(root),cfg)
                        if shiftedcv is None: continue
                        env.append({"agent_index":e,"object_type":int(obj[e]) if e<obj.size else 0,"trajectory":curcv,"shifted_trajectory":shiftedcv})
                        if len(env)>=args.environment_cap: break
                    # Build a teacher *proposal* pool first.  Stored response labels
                    # and V44 analytic trajectories were generated under related but
                    # not identical contracts, and inverse-mapping a trajectory to 8
                    # knots is only approximate.  Therefore no teacher control is
                    # marked positive until the knot-generated trajectory is replayed
                    # through the exact frozen V42--V44 hard verifier used online.
                    teacher_knots=[]; teacher_source=[]
                    if k<rv.shape[0] and a<rv.shape[1]:
                        for r in range(rv.shape[2]):
                            if not (rv[k,a,r] and rsafe[k,a,r] and rlow[k,a,r]): continue
                            if rroot.shape==rv.shape and int(rroot[k,a,r]) not in (-1,m): continue
                            teacher_knots.append(_response_to_normalized_knots(root,rt[k,a,r],cfg,args.control_knots)); teacher_source.append(0)
                    analytic,ad=_root_conditioned_control_reachable_response_profiles_np(state,j,int(obj[j]) if j<obj.size else 0,root,float(beta[a]),ego_cur,ego_shift,env,road,cfg,root_ordinal=m,compatibility_cache={})
                    if analytic: counts["analytic_nonempty"]+=1
                    for rec in analytic:
                        teacher_knots.append(_response_to_normalized_knots(root,rec["trajectory"],cfg,args.control_knots)); teacher_source.append(1)
                    for q in sobol_bank:
                        teacher_knots.append(np.asarray(q,np.float32)); teacher_source.append(2)

                    # Exact float32 dedup before verification. Prefer the lowest
                    # source id only as provenance when two sources generate the
                    # identical control; this does not change hard-set membership.
                    proposal_map={}
                    for q,src in zip(teacher_knots,teacher_source):
                        q=np.clip(np.asarray(q,np.float32).reshape(-1),-1.0,1.0)
                        key=np.ascontiguousarray(q).tobytes()
                        old=proposal_map.get(key)
                        if old is None or int(src)<int(old[1]): proposal_map[key]=(q,int(src))
                    proposal_entries=list(proposal_map.values())
                    proposal_arr=np.stack([x[0] for x in proposal_entries],axis=0).astype(np.float32) if proposal_entries else np.zeros((0,args.control_knots),np.float32)
                    verified,vd=_verified_root_conditioned_recourse_set_profiles_np(
                        state,j,int(obj[j]) if j<obj.size else 0,root,float(beta[a]),ego_cur,ego_shift,env,road,cfg,proposal_arr,
                        profile_index_base=30000,root_ordinal=m,compatibility_cache={}
                    ) if len(proposal_arr) else ([],{"proposal_outcomes":[]})
                    outcomes=list(vd.get("proposal_outcomes", []))
                    verified_by_index={int(rec.get("profile_index",30000))-30000:rec for rec in verified}
                    pos_knots=[]; pos_burden=[]; pos_source=[]; rejected_controls=[]
                    for qi,(q,src) in enumerate(proposal_entries):
                        rec=verified_by_index.get(qi)
                        reason=outcomes[qi] if qi<len(outcomes) else "unknown"
                        if rec is not None:
                            pos_knots.append(np.asarray(rec.get("rcrso_control_knots",q),np.float32)); pos_burden.append(float(rec["burden"])); pos_source.append(int(src))
                            if int(src)==2: counts["rich_verified"]+=1
                        else:
                            reason_code={
                                "no_low_burden_static_control":0,
                                "roadgraph_or_waymax_kinematic_reject":1,
                                "ego_current_reject":2,
                                "ego_shift_reject":3,
                                "environment_current_or_shift_reject":4,
                            }
                            rejected_controls.append((np.asarray(q,np.float32),int(reason_code.get(reason,5))))
                    counts["teacher_proposals"] = int(counts.get("teacher_proposals",0)) + int(len(proposal_entries))
                    counts["teacher_verified"] = int(counts.get("teacher_verified",0)) + int(len(verified))
                    # Exact dedup in knot space, keep lower burden representative.
                    unique={}
                    for q,b,src in zip(pos_knots,pos_burden,pos_source):
                        key=np.ascontiguousarray(np.round(q,6).astype(np.float32)).tobytes(); old=unique.get(key)
                        if old is None or b<old[1]: unique[key]=(q,b,src)
                    vals=sorted(unique.values(),key=lambda x:(x[1],x[2]))[:args.max_positive_controls]
                    P=args.max_positive_controls; targets=np.zeros((P,args.control_knots),np.float32); tvalid=np.zeros(P,bool); tb=np.zeros(P,np.float32); ts=np.full(P,-1,np.int64)
                    for qi,(q,b,src) in enumerate(vals): targets[qi]=q; tvalid[qi]=True; tb[qi]=b; ts[qi]=src
                    # Keep verifier-rejected Sobol controls as hard negatives.  To
                    # focus supervision near the feasible-set boundary without an
                    # outcome-tuned margin, deterministically retain rejected knots
                    # closest to any verified positive (or smallest norm when this
                    # root has no positive).  These labels train proposal ordering
                    # only and never alter the online hard verifier.
                    N=args.max_negative_controls; neg=np.zeros((N,args.control_knots),np.float32); nvalid=np.zeros(N,bool); nreason=np.full(N,-1,np.int64)
                    if rejected_controls and N>0:
                        pos_arr=np.stack([x[0] for x in vals],axis=0).astype(np.float32) if vals else np.zeros((0,args.control_knots),np.float32)
                        scored=[]
                        for q,code in rejected_controls:
                            if len(pos_arr): score=float(np.min(np.mean((pos_arr-q[None,:])**2,axis=1)))
                            else: score=float(np.mean(q*q))
                            scored.append((score,np.ascontiguousarray(np.round(q,6).astype(np.float32)).tobytes(),q,code))
                        seen_neg=set(); kept=[]
                        for _,key,q,code in sorted(scored,key=lambda x:(x[0],x[1])):
                            if key in seen_neg: continue
                            seen_neg.add(key); kept.append((q,code))
                            if len(kept)>=N: break
                        for ni,(q,code) in enumerate(kept): neg[ni]=q; nvalid[ni]=True; nreason[ni]=code
                        counts["hard_negative_examples"] = int(counts.get("hard_negative_examples",0)) + int(nvalid.sum())
                    features=build_rcrso_features_np(root=root,root_mass=float(canonical_w[a,m]),root_source=int(ns[a,m]),blocker_state=state[j],current_ego_trajectory=ego_cur,shifted_ego_trajectory=ego_shift,environment=env,cfg=rcrso_cfg,verifier_cfg=cfg,blocker_object_type=int(obj[j]) if j<obj.size else 0)
                    E=args.environment_cap*2; envtok=np.zeros((E,rcrso_cfg.environment_feature_dim),np.float32); envvalid=np.zeros(E,bool); rawenv=features["environment_tokens"]; nenv=min(E,len(rawenv)); envtok[:nenv]=rawenv[:nenv]; envvalid[:nenv]=True
                    sh=int.from_bytes(hashlib.sha256(sid.encode()).digest()[:8],"little",signed=False) & ((1<<63)-1)
                    hyp=(k*1000000)+(j*1000)+m; hyp_group=k
                    name=f"{sh:016x}_{k:03d}_{j:03d}_{m:02d}.npz"
                    # Keep the exact causal verifier context in the sidecar so Stage-0 can
                    # re-run hard admission for RCRSO proposals instead of using a distance/AUC proxy.
                    env_cur=np.zeros((args.environment_cap,len(root),7),np.float32); env_shift=np.zeros_like(env_cur); env_obj=np.zeros(args.environment_cap,np.int64); env_idx=np.full(args.environment_cap,-1,np.int64)
                    for ei,actor in enumerate(env[:args.environment_cap]):
                        env_cur[ei]=np.asarray(actor["trajectory"],np.float32); env_shift[ei]=np.asarray(actor["shifted_trajectory"],np.float32); env_obj[ei]=int(actor["object_type"]); env_idx[ei]=int(actor["agent_index"])
                    # Stage-0 replays the frozen roadgraph predicate. Store a
                    # compact but semantics-preserving local subset; do not cap by
                    # arbitrary file order or let an empty subset become vacuously
                    # drivable.
                    rxy,rh,rtpe,rvld=_sidecar_roadgraph_subset(road,root,ego_cur,ego_shift)
                    np.savez_compressed(outdir/name,root_tokens=features["root_tokens"],ego_tokens=features["ego_tokens"],environment_tokens=envtok,environment_valid=envvalid,blocker_state=features["blocker_state"],conflict_features=features["conflict_features"],target_control_knots=targets,target_valid=tvalid,target_burden=tb,target_source=ts,negative_control_knots=neg,negative_valid=nvalid,negative_reason=nreason,root_mass=np.float32(canonical_w[a,m]),root_source=np.int64(ns[a,m]),fixed_verified_nonempty=np.bool_(any(x==0 for x in pos_source)),analytic_verified_nonempty=np.bool_(bool(analytic)),scenario_hash=np.int64(sh),hypothesis_id=np.int64(hyp),hypothesis_group_id=np.int64(hyp_group),candidate_index=np.int64(k),agent_index=np.int64(j),root_index=np.int64(m),root_trajectory=np.asarray(root,np.float32),blocker_state_global=np.asarray(state[j],np.float32),blocker_object_type=np.int64(obj[j] if j<obj.size else 0),beta=np.float32(beta[a]),ego_current=np.asarray(ego_cur,np.float32),ego_shifted=np.asarray(ego_shift,np.float32),environment_current=env_cur,environment_shifted=env_shift,environment_object_type=env_obj,environment_agent_index=env_idx,roadgraph_xy=rxy,roadgraph_heading=rh,roadgraph_types=rtpe,roadgraph_valid=rvld)
                    manifest.append({"file":name,"scenario_id":sid,"candidate_index":k,"agent_index":j,"root_index":m,"verified_targets":int(tvalid.sum()),"fixed_nonempty":bool(any(x==0 for x in pos_source)),"analytic_nonempty":bool(analytic)})
                    counts["examples"]+=1; counts["positive_examples"]+=int(bool(tvalid.any())); made+=1
    suffix = "" if int(args.num_shards) == 1 else f"_s{int(args.shard_index)}of{int(args.num_shards)}"
    (Path(args.output_root)/f"manifest_{args.split}{suffix}.jsonl").write_text("\n".join(json.dumps(x,sort_keys=True) for x in manifest)+("\n" if manifest else ""),encoding="utf-8")
    summary={"version":"V16.8.45R1","split":args.split,"cache_dir":str(args.cache_dir),"forbidden_id_count":len(forbidden),"num_shards":int(args.num_shards),"shard_index":int(args.shard_index),"counts":counts,"rcrso_config":rcrso_cfg.to_dict(),"contract":{"base_compact5k_modified":False,"lost7_or_counterfactual48_allowed":False,"hard_verifier_semantics":"V42-V44 frozen predicates","rich_proposal_source":"deterministic Sobol knots; proposals admitted only after hard verifier","hard_negative_source":"frozen-verifier-rejected teacher/Sobol controls retained nearest to verified support","canonical_root_weight_semantics":"cached audit/transport canonical weights or shared canonical_root_weights fallback","required_root_mass":float(required_root_mass),"minimum_root_count":int(minimum_root_count),"max_roots_per_agent":int(max_roots_per_agent)}}
    (Path(args.output_root)/f"summary_{args.split}{suffix}.json").write_text(json.dumps(summary,indent=2,sort_keys=True),encoding="utf-8")
    print(json.dumps(summary,indent=2,sort_keys=True))

if __name__=="__main__": main()
