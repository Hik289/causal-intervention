pass

import os
import json, os, re, random
import pandas as pd, glob
from scipy import stats
import numpy as np
random.seed(42); np.random.seed(42)
HERE=os.path.dirname(os.path.abspath(__file__))
DATA=os.environ.get("CIVIC_PROJECT_ROOT", ".") + "/data/swe-review-bench_raw/data"

def boot_ci(flips, n, iters=10000):
    pass
    if n==0: return (0,0)
    arr=np.array(flips)
    bs=[np.mean(np.random.choice(arr,size=n,replace=True)) for _ in range(iters)]
    return (round(float(np.percentile(bs,2.5)),4), round(float(np.percentile(bs,97.5)),4))

def main():
    d=json.load(open(f"{HERE}/spike_repair.json"))
    v=[r for r in d if r.get("base")=="fail" and not r.get("skip")]
    N=len(v)
    print(f"=== VALID n={N} ===",flush=True)

    res={}
    for lab in ["nocrit","false","true"]:
        prod=[r for r in v if r.get(f"{lab}_resolution") in ('pass','fail')]
        flips_all=[1 if r.get(f"{lab}_flip") else 0 for r in v]
        flips_ran=[1 if r.get(f"{lab}_flip") else 0 for r in prod]
        fl=sum(flips_all)
        ci_all=boot_ci(flips_all,N); ci_ran=boot_ci(flips_ran,len(prod)) if prod else (0,0)
        res[lab]={"flip":fl,"N":N,"ran":len(prod)}
        print(f"{lab}: all-gated {fl}/{N}={fl/N:.3f} CI{ci_all} | ran-only {fl}/{len(prod)}={fl/len(prod) if prod else 0:.3f} CI{ci_ran}",flush=True)

    print("\n=== PAIRED TESTS ===",flush=True)
    def mcnemar(la,lb):
        both=[r for r in v if r.get(f"{la}_resolution") in ('pass','fail') and r.get(f"{lb}_resolution") in ('pass','fail')]
        a=sum(1 for r in both if r.get(f"{la}_flip") and not r.get(f"{lb}_flip"))
        b=sum(1 for r in both if not r.get(f"{la}_flip") and r.get(f"{lb}_flip"))
        p=stats.binomtest(min(a,b),a+b,0.5).pvalue if (a+b)>0 else 1.0
        return len(both),a,b,p
    for la,lb in [("true","nocrit"),("true","false")]:
        n,a,b,p=mcnemar(la,lb); print(f"McNemar {la}-vs-{lb}: n={n} {la}_only={a} {lb}_only={b} p={p:.3e}",flush=True)

    for lb in ["false","nocrit"]:
        tf=res["true"]["flip"]; of=res[lb]["flip"]
        _,p=stats.fisher_exact([[tf,N-tf],[of,N-of]])
        print(f"Fisher true-vs-{lb} (all-gated): p={p:.3e}",flush=True)

    for la,lb in [("true","nocrit"),("true","false")]:
        pairs=[(r.get(f"{la}_pass_delta",0) or 0,r.get(f"{lb}_pass_delta",0) or 0) for r in v]
        diffs=[a-b for a,b in pairs if a-b!=0]
        if diffs:
            try: w=stats.wilcoxon([a for a,b in pairs],[b for a,b in pairs]).pvalue
            except Exception: w=1.0
            print(f"Wilcoxon pass_delta {la}-vs-{lb}: n_nonzero={len(diffs)} mean_diff={np.mean([a-b for a,b in pairs]):.3f} p={w:.3e}",flush=True)

    tpd=[r.get("true_pass_delta",0) or 0 for r in v]
    bs=[np.mean(np.random.choice(tpd,len(tpd),replace=True)) for _ in range(10000)]
    print(f"true mean pass_delta={np.mean(tpd):.3f} CI({np.percentile(bs,2.5):.3f},{np.percentile(bs,97.5):.3f})",flush=True)

    print("\n=== REPO 分层 (true flips 分布) ===",flush=True)
    from collections import Counter,defaultdict
    repo_tot=Counter(r["repo"] for r in v)
    repo_flip=Counter(r["repo"] for r in v if r.get("true_flip"))
    for repo in sorted(repo_tot,key=lambda x:-repo_flip[x]):
        print(f"  {repo}: true flip {repo_flip[repo]}/{repo_tot[repo]}",flush=True)

    print("\n=== 异常 flip 调查 (false/nocrit) ===",flush=True)
    for lab in ["false","nocrit"]:
        for r in v:
            if r.get(f"{lab}_flip"):
                print(f"  {lab} FLIP: {r['iid']} (repo {r['repo']}, window_src={r.get(f'{lab}_window_src')})",flush=True)
    print("CLOSEOUT_STATS DONE",flush=True)

if __name__=="__main__": main()
