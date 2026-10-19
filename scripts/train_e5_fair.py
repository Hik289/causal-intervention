from civic.paths import OUTPUT_DIR
pass

import os
import json, os, argparse, numpy as np
import torch, torch.nn as nn
from collections import defaultdict
from sklearn.metrics import roc_auc_score
PRE=os.environ.get("CIVIC_PROJECT_ROOT", ".") + "/data/anchors"; ARTIFACTS=str(OUTPUT_DIR)
SPLIT={"train":["django","sympy","astropy","pytest","seaborn","flask","pylint"],
 "val":["scikit-learn","requests"],"test":["sphinx","matplotlib","xarray"]}
def split_of(repo):
    r=repo.split("/")[-1] if "/" in repo else repo
    for s,rs in SPLIT.items():
        if any(x in r for x in rs): return s
    return None

def load():
    d=defaultdict(lambda: defaultdict(list)); repo={}
    for l in open(f"{PRE}/scores_reviewergen_claude.jsonl"):
        r=json.loads(l)
        if r.get("score") is None: continue
        d[r["triple_id"]][r["version"]].append(r["score"]); repo[r["triple_id"]]=r.get("repo","")
    agg={t:{v:float(np.mean(s)) for v,s in vv.items()} for t,vv in d.items()}
    rows=[]
    for t in agg:
        if "x" in agg[t] and "target" in agg[t] and any(k.startswith("unrel") for k in agg[t]):
            sx=agg[t]["x"]; stg=agg[t]["target"]
            su=float(np.mean([agg[t][k] for k in agg[t] if k.startswith("unrel")]))
            rows.append({"tid":t,"split":split_of(repo.get(t,"")),"sx":sx,"stg":stg,"su":su})
    return rows

class BatteryHead(nn.Module):
    pass

    def __init__(self, din=4, hid=32):
        super().__init__(); self.net=nn.Sequential(nn.Linear(din,hid),nn.ReLU(),nn.Linear(hid,hid),nn.ReLU(),nn.Linear(hid,1))
    def forward(self,x): return self.net(x).squeeze(-1)

def feats(sx,stg,su):
    pass

    fx=np.c_[sx, sx-stg, sx-su, su-stg]
    fic=np.c_[stg, np.zeros_like(sx), stg-su, su-stg]
    ficp=np.c_[su, sx-su, np.zeros_like(sx), su-stg]
    return fx,fic,ficp

def train(rung, rows, seed, epochs=200, lr=1e-2):
    torch.manual_seed(seed); np.random.seed(seed)
    tr=[r for r in rows if r["split"]=="train"]; te=[r for r in rows if r["split"]=="test"]
    if len(tr)<5: tr=rows
    sx=np.array([r["sx"] for r in tr]); stg=np.array([r["stg"] for r in tr]); su=np.array([r["su"] for r in tr])
    fx,fic,ficp=feats(sx,stg,su)
    fx,fic,ficp=[torch.tensor(a,dtype=torch.float32) for a in (fx,fic,ficp)]
    m=BatteryHead(din=4 if rung!="R1" else 1)
    if rung=="R1": return None
    opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=1e-3); bce=nn.BCEWithLogitsLoss()
    w_flip=1.0 if rung=="B-Full" else 0.0; w_spec=1.0 if rung=="B-Full" else 0.0
    for ep in range(epochs):
        m.train()
        s_x=m(fx); s_ic=m(fic); s_icp=m(ficp)
        loss=bce(s_x,torch.ones_like(s_x))+bce(s_ic,torch.zeros_like(s_ic))
        if w_flip>0: loss=loss+w_flip*torch.clamp(0.5+s_ic-s_x,min=0).mean()
        if w_spec>0: loss=loss+w_spec*torch.abs(s_icp-s_x).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval()

    txs=np.array([r["sx"] for r in te]); tstg=np.array([r["stg"] for r in te]); tsu=np.array([r["su"] for r in te])
    tfx,tfic,tficp=feats(txs,tstg,tsu)
    with torch.no_grad():
        px=torch.sigmoid(m(torch.tensor(tfx,dtype=torch.float32))).numpy()
        pic=torch.sigmoid(m(torch.tensor(tfic,dtype=torch.float32))).numpy()
        picp=torch.sigmoid(m(torch.tensor(tficp,dtype=torch.float32))).numpy()
    dt=px-pic; du=np.abs(px-picp); y=np.r_[np.ones(len(te)),np.zeros(len(te))]
    return {"n_test":len(te),"spec_ratio":float(dt.mean()/(1e-6+du.mean())),
            "target_flip_rate":float((pic<px-0.1).mean()),
            "auroc_dt_vs_du":float(roc_auc_score(y,np.r_[dt,du])) if len(te)>1 else float('nan'),
            "bug_vs_fixed_auroc":float(roc_auc_score(y,np.r_[px,pic])) if len(te)>1 else float('nan')}

def raw_eval(rows):
    te=[r for r in rows if r["split"]=="test"] or rows
    sx=np.array([r["sx"] for r in te]); stg=np.array([r["stg"] for r in te]); su=np.array([r["su"] for r in te])
    dt=sx-stg; du=np.abs(sx-su); y=np.r_[np.ones(len(te)),np.zeros(len(te))]
    return {"n_test":len(te),"spec_ratio":round(float(dt.mean()/(1e-6+du.mean())),4),
            "target_flip_rate":round(float((stg<sx-0.1).mean()),4),
            "auroc_dt_vs_du":round(float(roc_auc_score(y,np.r_[dt,du])),4),
            "bug_vs_fixed_auroc":round(float(roc_auc_score(y,np.r_[sx,stg])),4)}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--seeds",type=int,nargs="+",default=[42,123,456])
    ap.add_argument("--out",default=f"{ARTIFACTS}/E5_fair_results.json"); args=ap.parse_args()
    rows=load()
    from collections import Counter
    print("triples:",len(rows),"splits:",dict(Counter(r["split"] for r in rows)),flush=True)
    res={"R1_raw_scalar":raw_eval(rows)}
    print("R1 raw:",res["R1_raw_scalar"],flush=True)
    for rung in ["B-MLP","B-Full"]:
        acc={"spec_ratio":[],"target_flip_rate":[],"auroc_dt_vs_du":[],"bug_vs_fixed_auroc":[]}
        for seed in args.seeds:
            r=train(rung,rows,seed)
            for k in acc: acc[k].append(r[k])
            print(f"  {rung} seed{seed}: SpecRatio={r['spec_ratio']:.3f} AUROC(Δt,Δu)={r['auroc_dt_vs_du']:.3f}",flush=True)
        res[rung]={k:[round(float(np.mean(v)),4),round(float(np.std(v)),4)] for k,v in acc.items()}
        print(f"{rung}:",res[rung],flush=True)
    res["note"]=("FAIR E5: upper layer trained ON LLM intervention battery features (s(x),s(Ic),s(Icp),Δ), NOT MiniLM. "
                 "vs raw-scalar. verdict: B-Full > raw on SpecRatio/AUROC & drops without L_spec → (a); else (b).")
    json.dump(res,open(args.out,"w"),indent=2)
    print("WROTE E5_fair_results.json",flush=True)

if __name__=="__main__":
    main()
