pass

import os
import json, numpy as np
from collections import defaultdict
from sklearn.metrics import roc_auc_score
PRE=os.environ.get("CIVIC_PROJECT_ROOT", ".") + "/experiments/preexp_anchor0"; PHC=os.environ.get("CIVIC_PROJECT_ROOT", ".") + "/phaseC"
SPLIT={"train":["django","sympy","astropy","pytest","seaborn","flask","pylint"],
 "val":["scikit-learn","requests"],"test":["sphinx","matplotlib","xarray"]}
def split_of(repo):
    r=repo.split("/")[-1] if "/" in repo else repo
    for s,rs in SPLIT.items():
        if any(x in r for x in rs): return s
    return None

def main():
    d=defaultdict(lambda: defaultdict(list)); repo={}
    for l in open(f"{PRE}/scores_reviewergen_claude.jsonl"):
        r=json.loads(l)
        if r.get("score") is None: continue
        d[r["triple_id"]][r["version"]].append(r["score"]); repo[r["triple_id"]]=r.get("repo","")
    agg={t:{v:float(np.mean(s)) for v,s in vv.items()} for t,vv in d.items()}
    tids=[t for t in agg if "x" in agg[t] and "target" in agg[t] and any(k.startswith("unrel") for k in agg[t])]
    delta=0.1; eps=1e-6
    def compute(ts):
        sx=np.array([agg[t]["x"] for t in ts]); stg=np.array([agg[t]["target"] for t in ts])
        su=np.array([float(np.mean([agg[t][k] for k in agg[t] if k.startswith("unrel")])) for t in ts])
        d_target=sx-stg; d_unrel=np.abs(sx-su)
        tfr=float(np.mean(stg < sx-delta))
        leak=float(np.mean(d_unrel))
        specr=float(np.mean(d_target)/(eps+np.mean(d_unrel)))
        y=np.r_[np.ones(len(ts)),np.zeros(len(ts))]; s=np.r_[d_target,d_unrel]
        au=float(roc_auc_score(y,s)) if len(ts)>1 else float('nan')
        return {"n":len(ts),"target_flip_rate":round(tfr,4),"specificity_leakage":round(leak,4),
                "specificity_ratio":round(specr,4),"AUROC_target_vs_unrel_drop":round(au,4),
                "mean_d_target":round(float(d_target.mean()),4),"mean_d_unrel":round(float(d_unrel.mean()),4)}
    out={"experiment":"E3 Causal Specificity (reviewer-gen, SWE-Review executable interventions)",
         "overall":compute(tids),"by_split":{}}
    for sp in ["train","val","test"]:
        ts=[t for t in tids if split_of(repo.get(t,""))==sp]
        if ts: out["by_split"][sp]=compute(ts)
    out["note"]=("E3 = method's cleanest contribution: raw scalar confidence has NO target/unrel intervention dimension, "
                 "so this discrimination is structurally unavailable to the raw-threshold baseline. "
                 "BUT: on ecologically-valid reviewer-gen, pre-exp showed Δtarget≈Δunrel (reviewer lacks causal specificity). "
                 "So raw reviewer specificity is weak; the question is whether TRAINED upper layer improves it (E5/full train).")
    json.dump(out,open(f"{PHC}/E3_results.json","w"),indent=2)
    print(json.dumps(out,indent=2),flush=True)

if __name__=="__main__":
    main()
