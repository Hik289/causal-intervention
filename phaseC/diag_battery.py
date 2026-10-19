pass

import json, numpy as np
from collections import defaultdict
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

PRE=os.environ.get("CIVIC_PROJECT_ROOT", ".") + "/experiments/preexp_anchor0"

def load(path):
    d=defaultdict(lambda: defaultdict(list))
    for l in open(path):
        r=json.loads(l)
        if r.get("score") is None: continue
        d[r["triple_id"]][r["version"]].append(r["score"])
    return {t:{v:float(np.mean(s)) for v,s in vv.items()} for t,vv in d.items()}

def run(name, agg):
    tids=[t for t in agg if "x" in agg[t] and "target" in agg[t] and any(k.startswith("unrel") for k in agg[t])]
    rows=[]
    for t in tids:
        sx=agg[t]["x"]; stg=agg[t]["target"]
        unrel=[agg[t][k] for k in agg[t] if k.startswith("unrel")]
        su=float(np.mean(unrel))
        rows.append((t,sx,stg,su))
    sx=np.array([r[1] for r in rows]); stg=np.array([r[2] for r in rows]); su=np.array([r[3] for r in rows])
    n=len(rows)

    y=np.r_[np.ones(n),np.zeros(n)]

    Xs=np.r_[sx,stg].reshape(-1,1)
    aus=roc_auc_score(y,cross_val_predict(LogisticRegression(max_iter=1000),Xs,y,cv=5,method="predict_proba")[:,1])

    bug_feat=np.c_[sx, sx-stg, sx-su, su-stg]
    fix_feat=np.c_[stg, np.zeros(n), stg-su, su-stg]
    Xb=np.r_[bug_feat,fix_feat]
    aub=roc_auc_score(y,cross_val_predict(LogisticRegression(max_iter=1000),Xb,y,cv=5,method="predict_proba")[:,1])

    dt=sx-stg; du=np.abs(sx-su)
    au_spec=roc_auc_score(np.r_[np.ones(n),np.zeros(n)], np.r_[dt,du])
    print(f"[{name}] n={n} scalar-AUROC={aus:.4f} battery-AUROC={aub:.4f} spec(Δt vs Δu)AUROC={au_spec:.4f} "
          f"meanΔt={dt.mean():.3f} meanΔu={du.mean():.3f}",flush=True)
    return {"name":name,"n":n,"scalar_auroc":round(float(aus),4),"battery_auroc":round(float(aub),4),
            "spec_auroc":round(float(au_spec),4),"mean_dt":round(float(dt.mean()),4),"mean_du":round(float(du.mean()),4)}

def main():
    out={}
    out["reviewer_gen"]=run("reviewer-gen", load(f"{PRE}/scores_reviewergen_claude.jsonl"))
    out["detailed_A"]=run("detailed-A", load(f"{PRE}/scores_A_claude.jsonl"))
    import os
    json.dump(out,open(os.environ.get("CIVIC_PROJECT_ROOT", ".") + "/phaseC/diag_battery.json","w"),indent=2)
    print("WROTE diag_battery.json",flush=True)

if __name__=="__main__":
    main()
