import torch; torch.set_num_threads(4)
pass

import os, sys, json, argparse, numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.environ.get("CIVIC_PROJECT_ROOT", ".") + "/phaseC")
from model import CrossAttnVerifier, build, n_params
CACHE=os.environ.get("CIVIC_PROJECT_ROOT", ".") + "/cache"; DS=os.environ.get("CIVIC_PROJECT_ROOT", ".") + "/data/intervention_dataset"; HERE=os.path.dirname(os.path.abspath(__file__))

class InterventionRanker(nn.Module):
    pass
    def __init__(self, dz, ddelta, hidden=256):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(dz+ddelta,hidden),nn.ReLU(),nn.Linear(hidden,1))
    def forward(self, z, h_delta):
        return self.net(torch.cat([z,h_delta],-1)).squeeze(-1)

def load_data():

    a=np.load(f"{CACHE}/embdiff_arrays.npz"); idx=[json.loads(l) for l in open(f"{CACHE}/embdiff_index.jsonl")]

    bench=np.load(f"{CACHE}/bench_aacr_emb.npz"); bidx=[json.loads(l) for l in open(f"{CACHE}/bench_aacr_index.jsonl")]
    return a, idx, bench, bidx

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--model",default="crossattn"); ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--epochs",type=int,default=200); ap.add_argument("--lr",type=float,default=1e-3)
    ap.add_argument("--out",default="train_civic_result.json")

    ap.add_argument("--w_flip",type=float,default=1.0); ap.add_argument("--w_spec",type=float,default=1.0)
    ap.add_argument("--w_eqv",type=float,default=0.5); ap.add_argument("--w_rank",type=float,default=0.5)
    ap.add_argument("--margin",type=float,default=0.5)
    args=ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    a,idx,bench,bidx=load_data()

    crit=torch.tensor(a["crit"],dtype=torch.float32); cx=torch.tensor(a["cx"],dtype=torch.float32)
    cic=torch.tensor(a["cic"],dtype=torch.float32); cicp=torch.tensor(a["cicp"],dtype=torch.float32)

    import numpy as _np
    _cx=a["cx"]; _cicp=a["cicp"]
    _cosp=(_cx*_cicp).sum(1)/(_np.linalg.norm(_cx,axis=1)*_np.linalg.norm(_cicp,axis=1)+1e-9)
    clean=set(i for i in range(len(idx)) if _cosp[i]>0.95)
    q_tr=[i for i,r in enumerate(idx) if r["split"]=="train" and i in clean]
    print(f"clean specificity quads (cos(x,I_c')>0.95): {len(clean)}/{len(idx)} (剔除脏 {len(idx)-len(clean)})",flush=True)

    bcrit=torch.tensor(bench["crit"],dtype=torch.float32); bcode=torch.tensor(bench["code"],dtype=torch.float32)
    blab=torch.tensor(bench["labels"],dtype=torch.float32)
    bsplit=[r["split"] for r in bidx]
    b_tr=[i for i,s in enumerate(bsplit) if s=="train"]; b_va=[i for i,s in enumerate(bsplit) if s=="val"]; b_te=[i for i,s in enumerate(bsplit) if s=="test"]
    dc=cx.shape[1]; dt=crit.shape[1]
    print(f"quad 四态 train {len(q_tr)} | AACR cls train {len(b_tr)}/val {len(b_va)}/test {len(b_te)} (dc={dc} dt={dt})",flush=True)
    dev="cuda" if torch.cuda.is_available() else "cpu"
    model=build(args.model, dc, dt).to(dev)
    ranker=InterventionRanker(dz=512, ddelta=dc).to(dev)
    print(f"model params: {n_params(model)} + ranker {n_params(ranker)} = {n_params(model)+n_params(ranker)}",flush=True)
    opt=torch.optim.AdamW(list(model.parameters())+list(ranker.parameters()),lr=args.lr,weight_decay=1e-4)
    crit_d=crit.to(dev); cx_d=cx.to(dev); cic_d=cic.to(dev); cicp_d=cicp.to(dev)
    bcrit_d=bcrit.to(dev); bcode_d=bcode.to(dev); blab_d=blab.to(dev)
    qi=torch.tensor(q_tr,device=dev); bi=torch.tensor(b_tr,device=dev)
    def s_struct(code_emb, I):
        sc,z=model(code_emb[I], crit_d[I]); return sc,z
    for ep in range(args.epochs):
        model.train(); ranker.train(); opt.zero_grad()

        s_find,_=model(bcode_d[bi], bcrit_d[bi])
        L_cls=nn.functional.binary_cross_entropy_with_logits(s_find, blab_d[bi])

        sx,zx=s_struct(cx_d,qi); sic,_=s_struct(cic_d,qi); sicp,_=s_struct(cicp_d,qi)
        L_flip=torch.clamp(args.margin + sic - sx, min=0).mean()
        L_spec=torch.abs(sicp - sx).mean()
        h_dp=(cic_d[qi]-cx_d[qi]); h_dn=(cicp_d[qi]-cx_d[qi])
        r_pos=ranker(zx,h_dp); r_neg=ranker(zx,h_dn)
        L_rank=-torch.log(torch.exp(r_pos)/(torch.exp(r_pos)+torch.exp(r_neg)+1e-9)).mean()
        L_min=0.01*h_dp.norm(dim=-1).mean()
        L_eqv=torch.tensor(0.0,device=dev)
        loss=L_cls + args.w_flip*L_flip + args.w_spec*L_spec + args.w_rank*L_rank + L_min + args.w_eqv*L_eqv
        loss.backward(); opt.step()
        if ep%50==0 or ep==args.epochs-1:
            print(f"ep{ep} loss={loss.item():.4f} cls={L_cls.item():.3f} flip={L_flip.item():.3f} spec={L_spec.item():.3f} rank={L_rank.item():.3f}",flush=True)

    model.eval()
    from frozen_reviewer_baseline import metrics as cls_metrics, fp_at_recall
    with torch.no_grad():
        def eval_aacr(bidxs):
            I=torch.tensor(bidxs,device=dev)
            s=torch.sigmoid(model(bcode_d[I],bcrit_d[I])[0]).cpu().numpy()
            y=blab_d[I].cpu().numpy().astype(int).tolist()
            try:
                from sklearn.metrics import roc_auc_score
                au=roc_auc_score(y,s) if len(set(y))>1 else None
            except Exception: au=None
            return {"n":len(bidxs),"n_real":int(sum(y)),"auroc":au,
                    "thr0.5":cls_metrics(s.tolist(),y,0.5),"thr0.6":cls_metrics(s.tolist(),y,0.6),
                    "fp_at_recall0.8":fp_at_recall(s.tolist(),y,0.8)}
        def diag_quad(I_list):
            I=torch.tensor(I_list,device=dev)
            sx,_=s_struct(cx_d,I); sic,_=s_struct(cic_d,I); sicp,_=s_struct(cicp_d,I)
            dtm=(sx-sic); dum=(sx-sicp).abs()
            return {"Delta_target(sx-sic,>0)":round(float(dtm.mean()),4),"Delta_unrel(|sx-sicp|,~0)":round(float(dum.mean()),4),
                    "spec_ratio":round(float(dtm.mean()/(1e-6+dum.mean())),3),"flip_rate":round(float((sic<sx).float().mean()),3)}
        q_te=[i for i,r in enumerate(idx) if r["split"]=="test" and i in clean]
        res={"seed":args.seed,"model":args.model,"params":n_params(model)+n_params(ranker),
             "AACR_test_E1":eval_aacr(b_te),"AACR_val_E1":eval_aacr(b_va) if b_va else None,
             "quad_test_struct":diag_quad(q_te)}
    print(json.dumps(res,ensure_ascii=False),flush=True)
    json.dump(res,open(f"{HERE}/{args.out}","w"),indent=2,ensure_ascii=False)
    print("TRAIN_CIVIC DONE",flush=True)

if __name__=="__main__": main()
