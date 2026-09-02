from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from ocrap.data.serialization import load_npz, load_npz_selected
from ocrap.models.data import MODEL_SAMPLE_NPZ_KEYS, OCRAPSampleDataset, sample_to_feature, fix_sample_geometry


REPO = Path(__file__).resolve().parents[1]


def _sample() -> dict[str, np.ndarray]:
    k, l = 3, 4
    return {
        "agent_history": np.zeros((2, 5, 16), np.float32),
        "agent_valid": np.ones((2, 5), bool),
        "ego_state": np.arange(9, dtype=np.float32),
        "bev_occ": np.zeros((7, 4, 4), np.float32),
        "route": np.arange(20, dtype=np.float32).reshape(10, 2),
        "map_polylines": np.arange(48, dtype=np.float32).reshape(3, 8, 2),
        "dynamic_map": np.arange(12, dtype=np.float32).reshape(3, 4),
        "prefix_param": np.arange(5, dtype=np.float32),
        "prefix_states": np.arange(40, dtype=np.float32).reshape(8, 5),
        "prefix_controls": np.arange(16, dtype=np.float32).reshape(8, 2),
        "prefix_macro_type_id": np.asarray(2, np.int64),
        "utility": np.asarray(1.5, np.float32),
        "hard_violation": np.asarray(0.0, np.float32),
        "harm_proxy": np.asarray(0.2, np.float32),
        "feasible": np.asarray(1.0, np.float32),
        "is_nominal": np.asarray(0.0, np.float32),
        "scene_id": np.asarray("s0"),
        "time_index": np.asarray(7, np.int64),
        "candidate_index": np.asarray(1, np.int64),
        "root_probs": np.asarray([0.2, 0.3, 0.5], np.float32),
        "m_star": np.arange(k*l, dtype=np.float32).reshape(k,l)/10,
        "c_star": np.eye(k, dtype=np.float32),
        "y_obs": np.eye(k, dtype=np.float32),
        "root_valid": np.ones(k, bool),
        "option_valid": np.ones(l, bool),
        "root_signature": np.zeros((k,2), np.float32),
        "root_future_signature": np.zeros((k,2), np.float32),
        "recovery_params": np.zeros((l,3), np.float32),
        "recovery_modes": np.asarray(["stop","brake_lane","lateral_escape","yield_rejoin"]),
        "r_dep_star": np.asarray(0.1, np.float32),
        "r_orc_star": np.asarray(0.2, np.float32),
        "i_art_star": np.asarray(0.0, np.float32),
        "unused_large_debug_tensor": np.ones((16,16), np.float32),
    }


def test_selective_npz_loader_is_model_semantics_identical(tmp_path: Path):
    p = tmp_path / "sample.npz"
    np.savez_compressed(p, **_sample())
    full = load_npz(p)
    selected = load_npz_selected(p, MODEL_SAMPLE_NPZ_KEYS)
    assert "unused_large_debug_tensor" in full
    assert "unused_large_debug_tensor" not in selected
    np.testing.assert_array_equal(sample_to_feature(full), sample_to_feature(selected))
    f1 = fix_sample_geometry(full, num_roots=3, num_options=4, d_signature=2, d_future_signature=2)
    f2 = fix_sample_geometry(selected, num_roots=3, num_options=4, d_signature=2, d_future_signature=2)
    assert f1.keys() == f2.keys()
    for key in f1:
        np.testing.assert_array_equal(f1[key], f2[key])



def test_in_memory_dataset_cache_returns_bit_identical_items(tmp_path: Path):
    sample_dir = tmp_path / "near_contact" / "samples"
    sample_dir.mkdir(parents=True)
    paths=[]
    for i in range(3):
        d=_sample(); d["candidate_index"]=np.asarray(i,np.int64); d["scene_id"]=np.asarray("s0")
        p=sample_dir/f"s{i}.npz"; np.savez_compressed(p, **d); paths.append(p)
    cfg0={"num_roots":3,"num_recovery_options":4,"model":{"d_signature":2,"d_future_signature":2},"training":{"cache_samples_in_memory":False}}
    cfg1={"num_roots":3,"num_recovery_options":4,"model":{"d_signature":2,"d_future_signature":2},"training":{"cache_samples_in_memory":True}}
    cold=OCRAPSampleDataset(paths,cfg0); cached=OCRAPSampleDataset(paths,cfg1)
    assert cached._item_cache is not None
    for i in range(len(paths)):
        a,b=cold[i],cached[i]
        assert a.keys()==b.keys()
        for k in a:
            assert torch.equal(a[k],b[k]), k

def test_sowr_stage_explicitly_disables_outer_downstream_roct():
    s = (REPO / "scripts/adapt_ocrap_v48_45_sowr_stage.sh").read_text()
    assert "EVIDENCE_ROCT_BENEFIT=false" in s
    assert "EVIDENCE_ROCT_DEPLOYABILITY=false" in s
    assert "EVIDENCE_COMPONENT_HEADS=false" in s
    assert "EVIDENCE_DUAL_INTERACTION_BRIDGE=false" in s
    # Match the immutable v48.45 rebuilt-source architecture rather than falling
    # back to generic train-script defaults for preference/tournament/delta heads.
    assert "PREFERENCE_HEAD_ENABLED=false" in s
    assert "PREFERENCE_CONTEXT_ENABLED=false" in s
    assert "RELATIVE_INCLUDE_ABSOLUTE=false" in s
    assert "SET_TOURNAMENT_ENABLED=true" in s
    assert "DELTA_MODE=ordinal_evidence" in s
    assert "DELTA_REGIME_EXPERTS=true" in s
    assert "DELTA_POLICY_FEATURES=true" in s
    arm = (REPO / "scripts/run_v48_45_sowr_ablation_arm.sh").read_text()
    assert "export EVIDENCE_ROCT_BENEFIT=true" in arm
    assert "export EVIDENCE_ROCT_DEPLOYABILITY=true" in arm
    assert "export EVIDENCE_COMPONENT_HEADS=true" in arm


def test_stage_isolation_contract_accepts_identity_epoch_zero(tmp_path: Path):
    source = tmp_path / "source.pt"
    dest = tmp_path / "dest.pt"
    out = tmp_path / "contract.json"
    state = {
        "encoder.weight": torch.tensor([1.0]),
        "root_logit_head.weight": torch.tensor([2.0]),
    }
    base = {"model_state": state}
    torch.save(base, source)
    dst = {
        "model_state": {k: v.clone() for k,v in state.items()},
        "trainable_param_prefixes": ["root_logit_head."],
    }
    torch.save(dst, dest)
    cp = subprocess.run([
        sys.executable, str(REPO / "tools/check_v48_45_sowr_stage_isolation.py"),
        "--source", str(source), "--checkpoint", str(dest),
        "--allowed-prefixes", "root_logit_head.", "--output", str(out),
    ], capture_output=True, text=True)
    assert cp.returncode == 0, cp.stderr + cp.stdout
    doc = json.loads(out.read_text())
    assert doc["valid"] is True
    assert doc["changed_key_count"] == 0
    assert doc["checks"]["downstream_evidence_disabled"] is True


def test_launcher_uses_io_parallelism_without_parallel_gpu_arms():
    s = (REPO / "scripts/run_v48_45_sowr_2x2_parallel.sh").read_text()
    assert 'ABLATION_NUM_WORKERS:-3' in s
    assert 'ABLATION_PREFETCH_FACTOR:-3' in s
    assert 'ABLATION_CACHE_SAMPLES_IN_MEMORY:-true' in s
    assert 'MAX_PARALLEL_ARMS="${MAX_PARALLEL_ARMS:-1}"' in s
    assert "reusable_arm_rc" in s


def test_resume_tool_preserves_only_hash_matching_authoritative_arm(tmp_path: Path):
    base=tmp_path/'runs'; source=base/'source'; seal=tmp_path/'seal.json'
    seal.write_text('{"valid":true}\n')
    for v in ('balanced','precision'):
        p=source/'candidates'/v/'model_v48_trac_sr'/'best.pt'; p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes((v+'-ckpt').encode())
    import hashlib
    sh={v:hashlib.sha256((source/'candidates'/v/'model_v48_trac_sr'/'best.pt').read_bytes()).hexdigest() for v in ('balanced','precision')}
    seal_sha=hashlib.sha256(seal.read_bytes()).hexdigest()
    a=base/'ocrap_v48_45_sowr_ablation_A'; a.mkdir(parents=True)
    (a/'AUTHORITATIVE_RUN_STATUS.json').write_text(json.dumps({'authoritative_exit_code':20,'pipeline_valid':True,'test_roots_read':False}))
    (a/'V48_36_COMPLETE.json').write_text(json.dumps({'pipeline_exit_code':20,'pipeline_valid':True,'test_roots_read':False}))
    (a/'ATTEMPT_STARTED.json').write_text(json.dumps({'protocol_seal_sha256':seal_sha,'test_roots_read':False}))
    (a/'SOURCE_CHECKPOINT_CONTRACT.json').write_text(json.dumps({'checks':{v:{'sha256':sh[v]} for v in sh},'test_roots_read':False}))
    b=base/'ocrap_v48_45_sowr_ablation_B'; b.mkdir(parents=True); (b/'PIPELINE_FAILED.json').write_text('{}')
    plan=tmp_path/'plan.json'
    cp=subprocess.run([sys.executable,str(REPO/'tools/prepare_v48_45_6_resume.py'),'--base-out',str(base),'--protocol-seal',str(seal),'--source-run',str(source),'--apply','--output',str(plan)],capture_output=True,text=True)
    assert cp.returncode==0,cp.stderr+cp.stdout
    doc=json.loads(plan.read_text())
    assert doc['arms']['A']['action']=='preserve_authoritative' and a.is_dir()
    assert doc['arms']['B']['action']=='remove_for_clean_retry' and not b.exists()


def test_source_architecture_contract_matches_rebuilt_source_geometry(tmp_path: Path):
    from importlib.util import spec_from_file_location, module_from_spec
    mod_path=REPO/'tools/check_v48_45_sowr_source_architecture.py'
    spec=spec_from_file_location('v48456_arch',mod_path); mod=module_from_spec(spec); assert spec and spec.loader; spec.loader.exec_module(mod)
    state={
      'encoder.x':torch.tensor([1.]), 'root_queries':torch.tensor([1.]),
      'root_cross_attn.in_proj_weight':torch.tensor([1.]), 'root_self_attn.in_proj_weight':torch.tensor([1.]),
      'root_ffn.0.weight':torch.tensor([1.]), 'root_logit_head.weight':torch.tensor([1.]),
      'obs_embed_head.0.weight':torch.tensor([1.]), 'margin_head.0.weight':torch.tensor([1.]),
    }
    d={'model_state':state,**mod.EXPECTED}
    ck=tmp_path/'src.pt'; out=tmp_path/'arch.json'; torch.save(d,ck)
    cp=subprocess.run([sys.executable,str(mod_path),'--checkpoint',str(ck),'--output',str(out)],capture_output=True,text=True)
    assert cp.returncode==0,cp.stderr+cp.stdout
    assert json.loads(out.read_text())['valid'] is True
