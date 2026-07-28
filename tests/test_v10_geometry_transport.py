import torch

from cowp.models.set_transport_head import SetTransportCertificateHead
from cowp.models.response_decoder import ResponseDecoder


def _inputs():
    B,K,N,A,M,R,T,D=2,3,5,2,4,6,12,16
    z_agent=torch.randn(B,N,D)
    z_candidate=torch.randn(B,K,D)
    z_graph=torch.randn(B,D)
    critical=torch.tensor([[1,2],[2,3]])
    natural={
        "mode_latent": torch.randn(B,A,M,D),
        "logits": torch.randn(B,A,M),
        "source_logits": torch.randn(B,A,M,4),
        "priority_logits": torch.randn(B,A,M),
        "traj": torch.randn(B,A,M,T,7),
    }
    response={
        "safe_logits": torch.randn(B,K,A,R),
        "low_logits": torch.randn(B,K,A,R),
        "valid_logits": torch.randn(B,K,A,R),
        "mode_logits": torch.randn(B,K,A,R),
        "root_logits": torch.randn(B,K,A,R,M),
        "burden_total": torch.rand(B,K,A,R),
    }
    return z_agent,z_candidate,z_graph,critical,natural,response,torch.rand(B,A),torch.randn(B,K,T,7)


def test_geometry_conditioned_transport_shapes_and_gradients():
    head=SetTransportCertificateHead(d_model=16, hidden=12, geometry_steps=6)
    za,zc,zg,ci,nat,res,beta,cand=_inputs()
    out=head(z_agent=za,z_candidate=zc,z_graph=zg,critical_indices=ci,natural=nat,response=res,beta=beta,candidate_traj=cand,natural_traj=nat["traj"])
    assert out["mode_conflict_logits"].shape==(2,3,2,4)
    assert out["root_recovery_mass"].shape==(2,3,2)
    assert torch.isfinite(out["witness_prob"]).all()
    out["witness_prob"].mean().backward()
    assert head.geometry[0].weight.grad is not None


def test_weighted_root_mass_does_not_explode_for_uniform_roots():
    head=SetTransportCertificateHead(d_model=16, hidden=12)
    za,zc,zg,ci,nat,res,beta,cand=_inputs()
    res["root_logits"].zero_()
    res["safe_logits"].fill_(8.0); res["low_logits"].fill_(8.0); res["valid_logits"].fill_(8.0)
    res["mode_logits"].zero_()
    out=head(z_agent=za,z_candidate=zc,z_graph=zg,critical_indices=ci,natural=nat,response=res,beta=beta,candidate_traj=cand,natural_traj=nat["traj"])
    # Uniform roots allocate about 1/M mass, not near-one existence per root.
    assert float(out["root_response_exist"].max()) < 0.35


def test_response_root_refinement_is_zero_initialized():
    dec=ResponseDecoder(d_model=16,responses=6,future_steps=12,natural_modes=4)
    assert torch.count_nonzero(dec.root_refine.weight)==0
    out=dec(torch.randn(2,5,16),torch.randn(2,3,16),torch.randn(2,16),torch.tensor([[1,2],[2,3]]),decode_traj=False)
    assert out["root_logits"].shape==(2,3,2,6,4)
