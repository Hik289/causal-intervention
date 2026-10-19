from civic.paths import OUTPUT_DIR
pass

import os
import json, os, argparse, numpy as np
import torch, torch.nn as nn
from sklearn.metrics import roc_auc_score
from civic.models import build, n_params
CACHE=os.environ.get("CIVIC_PROJECT_ROOT", ".") + "/cache"; ARTIFACTS=str(OUTPUT_DIR)
PRE=os.environ.get("CIVIC_PROJECT_ROOT", ".") + "/data/anchors"

def load():
    A=np.load(f"{CACHE}/emb_tight_arrays.npz")
    idx=[json.loads(l) for l in open(f"{CACHE}/emb_tight_index.jsonl")]
    return A, idx

def raw_specificity(idx):
    pass
    from collections import defaultdict
    d=defaultdict(lambda: defaultdict(list)); repo={}
    for l in open(f"{PRE}/scores_reviewergen_claude.jsonl"):
        r=json.loads(l)
        if r.get("score") is None: continue
        d[r["triple_id"]][r["version"]].append(r["score"]); repo[r["triple_id"]]=r.get("repo","")
    agg={t:{v:float(np.mean(s)) for v,s in vv.items()} for t,vv in d.items()}
    TEST=["sphinx","matplotlib","xarray"]
    ts=[t for t in agg if "x" in agg[t] and "target" in agg[t] and any(k.startswith("unrel") for k in agg[t])
        and any(x in repo.get(t,"") for x in TEST)]
    if not ts: ts=[t for t in agg if "x" in agg[t] and "target" in agg[t] and any(k.startswith("unrel") for k in agg[t])]
    sx=np.array([agg[t]["x"] for t in ts]); stg=np.array([agg[t]["target"] for t in ts])
    su=np.array([float(np.mean([agg[t][k] for k in agg[t] if k.startswith("unrel")])) for t in ts])
    dt=sx-stg; du=np.abs(sx-su)
    y=np.r_[np.ones(len(ts)),np.zeros(len(ts))]
    return {"n":len(ts),"spec_ratio":round(float(dt.mean()/(1e-6+du.mean())),4),
            "target_flip_rate":round(float((stg<sx-0.1).mean()),4),
            "auroc_dt_vs_du":round(float(roc_auc_score(y,np.r_[dt,du])),4),
            "bug_vs_fixed_auroc":round(float(roc_auc_score(np.r_[np.ones(len(ts)),np.zeros(len(ts))],np.r_[sx,stg])),4)}

def train_rung(rung, A, idx, seed, epochs=25, lr=3e-4):
    torch.manual_seed(seed); np.random.seed(seed); torch.set_num_threads(8)
    crit=torch.tensor(A["crit"],dtype=torch.float32); cx=torch.tensor(A["cx"],dtype=torch.float32)
    cic=torch.tensor(A["cic"],dtype=torch.float32); cicp=torch.tensor(A["cicp"],dtype=torch.float32)
    sp=np.array([r["split"] for r in idx])
    tr=torch.tensor(sp=="train"); dc=cx.shape[1]; dt=crit.shape[1]
    method="mlp" if rung=="R2" else "crossattn"
    m=build(method,dc,dt,hidden=256 if rung=="R2" else 512,layers=4,heads=8,dropout=0.1)
    opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=0.01); bce=nn.BCEWithLogitsLoss()
    w_flip=1.0 if rung=="R4" else 0.0; w_spec=1.0 if rung=="R4" else 0.0
    def sc(code,c): return m(code,c)[0]
    tri=torch.where(tr)[0]
    for ep in range(epochs):
        m.train(); perm=tri[torch.randperm(len(tri))]
        for i in range(0,len(perm),64):
            b=perm[i:i+64]
            s_x=sc(cx[b],crit[b]); s_ic=sc(cic[b],crit[b]); s_icp=sc(cicp[b],crit[b])
            loss=bce(s_x,torch.ones_like(s_x))+bce(s_ic,torch.zeros_like(s_ic))
            if w_flip>0: loss=loss+w_flip*torch.clamp(0.5+s_ic-s_x,min=0).mean()
            if w_spec>0: loss=loss+w_spec*torch.abs(s_icp-s_x).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    m.eval()

    te=np.where(sp=="test")[0]
    with torch.no_grad():
        sx=torch.sigmoid(sc(cx[te],crit[te])).numpy(); sic=torch.sigmoid(sc(cic[te],crit[te])).numpy()
        sicp=torch.sigmoid(sc(cicp[te],crit[te])).numpy()
    dt_=sx-sic; du_=np.abs(sx-sicp)
    y=np.r_[np.ones(len(te)),np.zeros(len(te))]
    return {"spec_ratio":float(dt_.mean()/(1e-6+du_.mean())),
            "target_flip_rate":float((sic<sx-0.1).mean()),
            "auroc_dt_vs_du":float(roc_auc_score(y,np.r_[dt_,du_])) if len(te)>1 else float('nan'),
            "bug_vs_fixed_auroc":float(roc_auc_score(y,np.r_[sx,sic])) if len(te)>1 else float('nan')}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--seeds",type=int,nargs="+",default=[42,123,456])
    ap.add_argument("--out",default=f"{ARTIFACTS}/specificity_ablation_results.json"); args=ap.parse_args()
    A,idx=load()
    from collections import Counter
    print("quads:",len(idx),"splits:",dict(Counter(r["split"] for r in idx)),flush=True)
    res={"R1_raw_scalar":raw_specificity(idx)}
    print("R1 raw:",res["R1_raw_scalar"],flush=True)
    for rung in ["R2_MLP","R3_crossattn_noCausal","R4_CIVIC_Full"]:
        rk=rung.split("_")[0]
        accs=defaultdict(list) if False else {"spec_ratio":[],"target_flip_rate":[],"auroc_dt_vs_du":[],"bug_vs_fixed_auroc":[]}
        for seed in args.seeds:
            r=train_rung(rk,A,idx,seed)
            for k in accs: accs[k].append(r[k])
            print(f"  {rung} seed{seed}: SpecRatio={r['spec_ratio']:.3f} AUROC(Δt,Δu)={r['auroc_dt_vs_du']:.3f}",flush=True)
        res[rung]={k:[round(float(np.mean(v)),4),round(float(np.std(v)),4)] for k,v in accs.items()}
        print(f"{rung}:",res[rung],flush=True)
    res["verdict_note"]=("(a) if R4 SpecRatio & AUROC(Δt,Δu) significantly > R1(raw) & R2(MLP) and drops when L_spec removed (R3). "
                         "(b) if R4 ~ R1/R2. frozen emb=all-MiniLM (weak, see ISOLATE); this tests if trained nonlinear upper+causal loss recovers signal linear probe missed.")
    json.dump(res,open(args.out,"w"),indent=2)
    print("WROTE specificity_ablation_results.json",flush=True)

if __name__=="__main__":
    from collections import defaultdict
    main()
