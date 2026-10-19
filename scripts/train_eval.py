from civic.paths import OUTPUT_DIR
pass

import os
import json, os, argparse, numpy as np
import torch, torch.nn as nn
from civic.models import build, n_params
from sklearn.metrics import roc_auc_score, average_precision_score

CACHE=os.environ.get("CIVIC_PROJECT_ROOT", ".") + "/cache"
OUT=str(OUTPUT_DIR)

def load(prefix):
    a=np.load(prefix+"_arrays.npz")
    idx=[json.loads(l) for l in open(prefix+"_index.jsonl")]
    return a, idx

def split_mask(idx, sp): return np.array([r["split"]==sp for r in idx])

def train_one(method, A, idx, seed, epochs=20, lr=3e-4, device="cpu",
              w_cls=1.0, w_flip=1.0, w_spec=1.0, w_eqv=0.0, margin=0.5):
    torch.manual_seed(seed); np.random.seed(seed)
    torch.set_num_threads(8)
    crit=torch.tensor(A["crit"],dtype=torch.float32)
    cx=torch.tensor(A["cx"],dtype=torch.float32)
    cic=torch.tensor(A["cic"],dtype=torch.float32)
    cicp=torch.tensor(A["cicp"],dtype=torch.float32)
    tr=torch.tensor(split_mask(idx,"train")); va=torch.tensor(split_mask(idx,"val"))
    dc=cx.shape[1]; dt=crit.shape[1]
    hid=512 if method!="mlp" else 256
    m=build("crossattn" if method in ("civic","invariance","sensitivity") else method, dc, dt,
            hidden=hid, layers=4, heads=8, dropout=0.1).to(device)
    opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=0.01)
    bce=nn.BCEWithLogitsLoss()
    def score(code,c): return m(code.to(device),c.to(device))[0]
    tri=torch.where(tr)[0]
    for ep in range(epochs):
        m.train(); perm=tri[torch.randperm(len(tri))]
        for i in range(0,len(perm),64):
            b=perm[i:i+64]
            s_x=score(cx[b],crit[b])
            s_ic=score(cic[b],crit[b])
            s_icp=score(cicp[b],crit[b])
            loss=w_cls*bce(s_x,torch.ones_like(s_x))

            loss=loss+w_cls*bce(s_ic,torch.zeros_like(s_ic))
            if w_flip>0: loss=loss+w_flip*torch.clamp(margin+s_ic-s_x,min=0).mean()
            if w_spec>0: loss=loss+w_spec*torch.abs(s_icp-s_x).mean()
            if w_eqv>0: loss=loss+w_eqv*torch.abs(s_icp-s_x).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    return m, score

def evaluate_e2(m, score, A, idx, split, device="cpu"):
    pass

    import torch
    crit=torch.tensor(A["crit"],dtype=torch.float32); cx=torch.tensor(A["cx"],dtype=torch.float32)
    cic=torch.tensor(A["cic"],dtype=torch.float32); cicp=torch.tensor(A["cicp"],dtype=torch.float32)
    mask=split_mask(idx,split); mi=np.where(mask)[0]
    with torch.no_grad():
        sx=torch.sigmoid(score(cx[mi],crit[mi])).cpu().numpy()
        sic=torch.sigmoid(score(cic[mi],crit[mi])).cpu().numpy()
        sicp=torch.sigmoid(score(cicp[mi],crit[mi])).cpu().numpy()
    return sx, sic, sicp

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--emb",default=os.path.join(CACHE,"emb_full"))
    ap.add_argument("--seeds",type=int,nargs="+",default=[42,123,456])
    ap.add_argument("--out",default=os.path.join(OUT,"E2_results.json"))
    args=ap.parse_args()
    A,idx=load(args.emb)
    print(f"loaded {len(idx)} quads, dim={A['crit'].shape[1]}",flush=True)

    configs={
        "MLP":            dict(method="mlp",   w_flip=0, w_spec=0),
        "Invariance-Only":dict(method="invariance", w_flip=0, w_spec=0, w_eqv=1.0),
        "Sensitivity-Only":dict(method="sensitivity", w_flip=1.0, w_spec=0),
        "CIVIC-CR-Full":  dict(method="civic", w_flip=1.0, w_spec=1.0),
    }

    results={}
    for name,cfg in configs.items():
        pair_accs=[]; aurocs=[]; trap=[]; margins=[]; posrec=[]
        tau_used=[]
        for seed in args.seeds:
            import time as _t; _t0=_t.time()
            m,score=train_one(cfg["method"],A,idx,seed,
                              w_flip=cfg.get("w_flip",0),w_spec=cfg.get("w_spec",0),w_eqv=cfg.get("w_eqv",0))
            print(f"  {name} seed{seed} trained in {_t.time()-_t0:.0f}s",flush=True)

            vx,vic,_=evaluate_e2(m,score,A,idx,"val")
            best_tau,best=0.5,-1
            for tau in np.linspace(0.1,0.9,33):
                pa=np.mean((vx>tau)&(vic<=tau)) if len(vx) else 0
                if pa>best: best,best_tau=pa,tau

            sx,sic,sicp=evaluate_e2(m,score,A,idx,"test")
            n=len(sx)
            if n==0: continue
            pa=float(np.mean((sx>best_tau)&(sic<=best_tau)))
            pr=float(np.mean(sx>best_tau))
            tr_=float(np.mean(sic<=best_tau))
            y=np.r_[np.ones(n),np.zeros(n)]; s=np.r_[sx,sic]
            au=float(roc_auc_score(y,s)) if len(set(y))>1 else float('nan')
            mg=float(np.mean(sx-sic))
            pair_accs.append(pa); aurocs.append(au); trap.append(tr_); margins.append(mg); posrec.append(pr); tau_used.append(float(best_tau))
        def ms(v): return [round(float(np.mean(v)),4),round(float(np.std(v)),4)] if v else [None,None]
        results[name]={"PairAcc":ms(pair_accs),"AUROC":ms(aurocs),"TrapRejection":ms(trap),
                       "PositiveRecall":ms(posrec),"MeanPairMargin":ms(margins),
                       "tau_val":ms(tau_used),"n_test_pairs":len(sx),"seeds":args.seeds}
        print(f"{name}: PairAcc={results[name]['PairAcc']} AUROC={results[name]['AUROC']}",flush=True)
    out={"experiment":"E2_matched_bug_vs_benign","benchmark":"SWE-Review-Bench (via quadruples)",
         "note":"x^bug=base code, x^safe=I_c(x) target-fixed; same reviewer critique; threshold val-selected test-frozen; 3 seeds",
         "model_params":{"crossattn":n_params(build('crossattn',384,384,hidden=512,layers=4,heads=8))},
         "results":results}
    json.dump(out,open(args.out,"w"),indent=2)
    print("WROTE",args.out,flush=True)

if __name__=="__main__":
    main()
