from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from cowp.data.recourse_sidecar import RecourseSidecarDataset, collate_recourse_sidecar
from cowp.models.recourse_set_operator import RCRSOConfig, RootConditionedRecourseSetTransformer


def _move(batch: dict, device: torch.device) -> dict:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k,v in batch.items()}


def _loss(model, batch: dict, query_count: int) -> tuple[torch.Tensor, dict[str,float]]:
    out=model(root_tokens=batch["root_tokens"],ego_tokens=batch["ego_tokens"],environment_tokens=batch["environment_tokens"],blocker_state=batch["blocker_state"],conflict_features=batch["conflict_features"],environment_valid=batch.get("environment_valid"),query_count=query_count)
    pred=out["control_knots"] # B,K,D
    target=batch["target_control_knots"]
    valid=batch["target_valid"].bool()
    burden_t=batch["target_burden"]
    B,K,D=pred.shape; P=target.shape[1]
    dist=((pred[:,:,None,:]-target[:,None,:,:])**2).mean(dim=-1) # B,K,P
    large=torch.full_like(dist,1e4)
    valid_dist=torch.where(valid[:,None,:],dist,large)
    has=valid.any(dim=1)
    # target -> best query is the completeness term; query -> target keeps the set near verified support.
    target_best=valid_dist.min(dim=1).values
    # Low-burden verified targets receive slightly larger matching weight.  This
    # is training-only proposal ordering; beta and hard burden admission are not
    # changed online.
    target_weight=valid.float()/(1.0+burden_t.clamp_min(0.0))
    coverage=(target_best*target_weight).sum(dim=1)/target_weight.sum(dim=1).clamp_min(1.0)
    query_best,match_idx=valid_dist.min(dim=2)
    proposal=(query_best.mean(dim=1))
    mass=batch.get("root_mass",torch.ones(B,device=pred.device)).float().reshape(B).clamp_min(0.01)
    pos_weight=has.float()*mass
    set_loss=((coverage+0.35*proposal)*pos_weight).sum()/pos_weight.sum().clamp_min(1.0)

    # Verifier-derived hard negatives come from the sidecar.  The auxiliary
    # feasibility head is only an ordering signal: query slots are labelled by
    # whether their nearest frozen teacher control is verified or rejected.
    neg=batch.get("negative_control_knots")
    neg_valid=batch.get("negative_valid")
    if neg is not None and neg_valid is not None and int(neg.shape[1])>0:
        ndist=((pred[:,:,None,:]-neg[:,None,:,:])**2).mean(dim=-1)
        ndist=torch.where(neg_valid[:,None,:].bool(),ndist,torch.full_like(ndist,1e4))
        neg_best=ndist.min(dim=2).values
        has_neg=neg_valid.bool().any(dim=1)
        teacher_positive=(query_best.detach()<=neg_best.detach()) & has[:,None]
        supervised=(has[:,None]|has_neg[:,None])
        feas_raw=torch.nn.functional.binary_cross_entropy_with_logits(out["feasible_logits"],teacher_positive.float(),reduction="none")
        feas_loss=(feas_raw*supervised.float()*mass[:,None]).sum()/(supervised.float()*mass[:,None]).sum().clamp_min(1.0)
        # Keep proposal slots away from verifier-rejected controls when a positive
        # teacher set exists, without turning this into online scalar feasibility.
        contrast=(torch.relu(0.03+query_best-neg_best)*has[:,None]*has_neg[:,None]).sum()/(has[:,None]*has_neg[:,None]).float().sum().clamp_min(1.0)
    else:
        close=(query_best.detach()<0.04)&has[:,None]
        feas_raw=torch.nn.functional.binary_cross_entropy_with_logits(out["feasible_logits"],close.float(),reduction="none")
        feas_loss=(feas_raw*mass[:,None]).mean()
        contrast=pred.new_tensor(0.0)
    matched_burden=torch.gather(burden_t,1,match_idx.clamp_max(P-1))
    burden_mask=has[:,None]
    burden_loss=(torch.abs(out["burden_prediction"]-matched_burden)*burden_mask.float()).sum()/burden_mask.float().sum().clamp_min(1.0)

    if K>1:
        pair=torch.cdist(pred,pred,p=2)/math.sqrt(max(D,1))
        eye=torch.eye(K,device=pred.device,dtype=torch.bool)[None]
        diversity=torch.relu(0.12-pair.masked_fill(eye,1e6)).sum()/(B*K*max(K-1,1))
    else: diversity=pred.new_tensor(0.0)
    smooth=(pred[:,:,1:]-pred[:,:,:-1]).abs().mean() if D>1 else pred.new_tensor(0.0)
    total=set_loss+0.15*feas_loss+0.05*burden_loss+0.05*diversity+0.03*contrast+0.01*smooth
    stats={"loss":float(total.detach()),"set":float(set_loss.detach()),"feas":float(feas_loss.detach()),"burden":float(burden_loss.detach()),"diversity":float(diversity.detach()),"contrast":float(contrast.detach()),"smooth":float(smooth.detach())}
    return total,stats


def _eval(model, loader, device, query_count:int) -> dict[str,float]:
    model.eval(); sums={}; n=0
    with torch.no_grad():
        for b in loader:
            b=_move(b,device); loss,st=_loss(model,b,query_count)
            bs=int(b["root_tokens"].shape[0]); n+=bs
            for k,v in st.items(): sums[k]=sums.get(k,0.0)+v*bs
    return {k:v/max(n,1) for k,v in sums.items()}|{"examples":n}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--sidecar-root",required=True); ap.add_argument("--output-dir",required=True)
    ap.add_argument("--epochs",type=int,default=30); ap.add_argument("--batch-size",type=int,default=64); ap.add_argument("--lr",type=float,default=3e-4); ap.add_argument("--weight-decay",type=float,default=1e-4)
    ap.add_argument("--d-model",type=int,default=128); ap.add_argument("--nhead",type=int,default=4); ap.add_argument("--layers",type=int,default=2); ap.add_argument("--max-queries",type=int,default=16); ap.add_argument("--control-knots",type=int,default=8)
    ap.add_argument("--num-workers",type=int,default=4); ap.add_argument("--device",default="auto"); ap.add_argument("--seed",type=int,default=20260845)
    args=ap.parse_args(); torch.manual_seed(args.seed); np.random.seed(args.seed)
    device=torch.device("cuda" if args.device=="auto" and torch.cuda.is_available() else ("cpu" if args.device=="auto" else args.device))
    train=RecourseSidecarDataset(args.sidecar_root,"train"); val=RecourseSidecarDataset(args.sidecar_root,"val")
    loader=lambda ds,shuffle: DataLoader(ds,batch_size=args.batch_size,shuffle=shuffle,num_workers=args.num_workers,pin_memory=device.type=="cuda",collate_fn=collate_recourse_sidecar,persistent_workers=args.num_workers>0)
    tr=loader(train,True); va=loader(val,False)
    cfg=RCRSOConfig(d_model=args.d_model,nhead=args.nhead,encoder_layers=args.layers,max_queries=args.max_queries,control_knots=args.control_knots)
    model=RootConditionedRecourseSetTransformer(cfg).to(device); opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    outdir=Path(args.output_dir); outdir.mkdir(parents=True,exist_ok=True); best=float("inf"); hist=[]
    for epoch in range(1,args.epochs+1):
        model.train(); t0=time.time(); sums={}; n=0
        for b in tr:
            b=_move(b,device); opt.zero_grad(set_to_none=True); loss,st=_loss(model,b,cfg.max_queries); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step(); bs=int(b["root_tokens"].shape[0]); n+=bs
            for k,v in st.items(): sums[k]=sums.get(k,0.0)+v*bs
        train_stats={k:v/max(n,1) for k,v in sums.items()}; val_stats=_eval(model,va,device,cfg.max_queries)
        rec={"epoch":epoch,"train":train_stats,"val":val_stats,"seconds":time.time()-t0}; hist.append(rec); print(json.dumps(rec,sort_keys=True))
        payload={"version":"V16.8.45R1","model":model.state_dict(),"rcrso_config":cfg.to_dict(),"selected_k":cfg.max_queries,"epoch":epoch,"train_stats":train_stats,"val_stats":val_stats,"contract":{"base_cowp_frozen":True,"hard_verifier_not_learned":True,"selected_k_status":"provisional_until_stage0_validation"}}
        torch.save(payload,outdir/"rcrso_last.pt")
        if val_stats["set"]<best:
            best=val_stats["set"]; torch.save(payload,outdir/"rcrso_best_unselected.pt")
    (outdir/"training_history.json").write_text(json.dumps(hist,indent=2),encoding="utf-8")

if __name__=="__main__": main()
