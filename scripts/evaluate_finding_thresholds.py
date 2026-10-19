from civic.paths import OUTPUT_DIR
pass

import os
import json, os, argparse, numpy as np
from collections import defaultdict
from sklearn.metrics import roc_auc_score

ARTIFACTS=str(OUTPUT_DIR)

def load():
    rows=[json.loads(l) for l in open(f"{ARTIFACTS}/finding_dataset.jsonl")]
    rows=[r for r in rows if r.get("s_true") is not None and r.get("s_false") is not None]
    return rows

def findings(rows, split):
    pass
    out=[]
    for r in rows:
        if r["split"]!=split: continue
        out.append({"issue":r["issue_id"],"score":r["s_true"],"y":1})
        out.append({"issue":r["issue_id"],"score":r["s_false"],"y":0})
    return out

def metrics_at_tau(F, tau):
    tp=sum(1 for f in F if f["score"]>tau and f["y"]==1)
    fp=sum(1 for f in F if f["score"]>tau and f["y"]==0)
    fn=sum(1 for f in F if f["score"]<=tau and f["y"]==1)
    n_pr=len(set(f["issue"] for f in F))
    prec=tp/(tp+fp) if tp+fp else 0.0
    rec=tp/(tp+fn) if tp+fn else 0.0
    f1=2*prec*rec/(prec+rec) if prec+rec else 0.0
    fp_pr=fp/n_pr if n_pr else 0.0
    return {"precision":prec,"recall":rec,"f1":f1,"fp_per_pr":fp_pr,"tp":tp,"fp":fp,"n_pr":n_pr}

def tau_at_recall(F, target_recall):
    pass
    scores=sorted(set(f["score"] for f in F))
    best_tau=0.0
    for tau in [0.0]+scores:
        m=metrics_at_tau(F,tau)
        if m["recall"]>=target_recall: best_tau=tau
    return best_tau

def bootstrap_fp_pr(F, tau, n=1000, seed=0):
    rng=np.random.default_rng(seed)
    issues=list(set(f["issue"] for f in F))
    by=defaultdict(list)
    for f in F: by[f["issue"]].append(f)
    vals=[]
    for _ in range(n):
        samp=rng.choice(issues,len(issues),replace=True)
        FF=[x for iss in samp for x in by[iss]]
        vals.append(metrics_at_tau(FF,tau)["fp_per_pr"])
    return [float(np.percentile(vals,2.5)),float(np.percentile(vals,97.5))]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out",default=f"{ARTIFACTS}/finding_threshold_results.json"); args=ap.parse_args()
    rows=load()
    from collections import Counter
    print("usable rows:",len(rows),"by split:",dict(Counter(r["split"] for r in rows)),flush=True)
    Fval=findings(rows,"val"); Ftest=findings(rows,"test"); Ftr=findings(rows,"train")

    def auroc(F):
        y=[f["y"] for f in F]; s=[f["score"] for f in F]
        return roc_auc_score(y,s) if len(set(y))>1 else float('nan')
    res={"n_test_findings":len(Ftest),"test_PRs":len(set(f["issue"] for f in Ftest)),
         "raw_AUROC":{"train":round(auroc(Ftr),4),"val":round(auroc(Fval),4),"test":round(auroc(Ftest),4)},
         "raw_scalar_threshold_baseline":{}}

    for R in [0.5,0.7,0.8]:
        tau=tau_at_recall(Fval,R)
        mt=metrics_at_tau(Ftest,tau)
        ci=bootstrap_fp_pr(Ftest,tau)
        res["raw_scalar_threshold_baseline"][f"recall_{int(R*100)}"]={
            "tau_from_val":round(tau,4),"test":{k:round(v,4) if isinstance(v,float) else v for k,v in mt.items()},
            "fp_per_pr_95ci":[round(x,4) for x in ci]}

    tau70=tau_at_recall(Fval,0.7)
    by=defaultdict(lambda:{"fp":0})
    for f in Ftest:
        if f["score"]>tau70 and f["y"]==0: by[f["issue"]]["fp"]+=1
    fpc=[v["fp"] for v in by.values()]+[0]*(len(set(f["issue"] for f in Ftest))-len(by))
    res["per_pr_fp_dist_at_recall70"]={"mean":round(float(np.mean(fpc)),4),"median":float(np.median(fpc)),
        "max":int(max(fpc)) if fpc else 0,"std":round(float(np.std(fpc)),4)}
    res["note"]=("E1 FP reduction. finding=(issue,critique). pos=c_true, neg=c_false_hard (plausible nonexistent). "
                 "raw-scalar-threshold = headline baseline. Upper-layer verifier gap vs this = method increment (TBD in E5). "
                 "SWE-Review executable-grounded. 单条 s(x,c) 已是强信号(0.81 AUROC), 方法增量待 E3/E5 证明.")
    json.dump(res,open(args.out,"w"),indent=2)
    print(json.dumps(res,indent=2),flush=True)

if __name__=="__main__":
    main()
