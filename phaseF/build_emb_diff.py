pass

import os, sys, json, re, numpy as np
sys.path.insert(0,os.environ.get("CIVIC_PROJECT_ROOT", ".") + "/phaseC")
import pandas as pd, glob
DATA=os.environ.get("CIVIC_PROJECT_ROOT", ".") + "/data"; DS=f"{DATA}/intervention_dataset"; CACHE=os.environ.get("CIVIC_PROJECT_ROOT", ".") + "/cache"
REPO_DIR={"astropy/astropy":"astropy","django/django":"django","matplotlib/matplotlib":"matplotlib","mwaskom/seaborn":"seaborn",
 "pallets/flask":"flask","psf/requests":"requests","pydata/xarray":"xarray","pylint-dev/pylint":"pylint",
 "pytest-dev/pytest":"pytest","scikit-learn/scikit-learn":"scikit-learn","sphinx-doc/sphinx":"sphinx","sympy/sympy":"sympy"}

def srcf(p): return [f for f in re.findall(r'diff --git a/(\S+) b/',p or '') if 'test' not in f.lower()]

def changed_sides(patch, fp, ctx_window=8):
    pass
    before=[]; after=[]; infile=False
    for ln in (patch or "").split("\n"):
        m=re.match(r'diff --git a/(\S+) b/(\S+)',ln)
        if m: infile=(m.group(2)==fp or m.group(1)==fp); continue
        if not infile: continue
        if ln.startswith("@@"): before.append("# hunk"); after.append("# hunk"); continue
        if ln.startswith("+") and not ln.startswith("+++"): after.append(ln[1:])
        elif ln.startswith("-") and not ln.startswith("---"): before.append(ln[1:])
        elif ln.startswith(" "): before.append(ln[1:]); after.append(ln[1:])
    return "\n".join(before), "\n".join(after)

def main():
    quads=[json.loads(l) for l in open(f"{DS}/quadruples_final.jsonl")]
    qmap={q['quad_id']:q for q in quads}
    df=pd.concat([pd.read_parquet(f) for f in glob.glob(f"{DATA}/swe-review-bench_raw/data/*.parquet")],ignore_index=True)
    res=df[df.resolved==True].groupby('instance_id').first()
    pby={i:r.model_patch for i,r in res.iterrows()}
    fixidx=[json.loads(l) for l in open(f"{CACHE}/embfix_index.jsonl")]
    from sentence_transformers import SentenceTransformer
    m=SentenceTransformer("all-MiniLM-L6-v2")
    def emb(t): return m.encode([str(t) if t else " "],convert_to_numpy=True)[0]
    out={"crit":[],"cx":[],"cic":[],"cicp":[]}; keep=[]
    for rec in fixidx:
        q=qmap.get(rec['quad_id'])
        if not q: continue
        iid=q['issue_id']
        if iid not in pby: continue
        tgt=pby[iid]; dist=pby.get(q['I_cprime'].get('from_issue'))
        if not dist: continue
        src=srcf(tgt)
        if not src: continue
        fp=src[0]
        before,after=changed_sides(tgt,fp)
        if not (before.strip() or after.strip()): continue

        db,da=changed_sides(dist,fp)
        unrel_after = da if da.strip() else before
        cval=q.get('critique',''); ctext=cval.get('text','') if isinstance(cval,dict) else str(cval)
        out["cx"].append(emb(before)); out["cic"].append(emb(after)); out["cicp"].append(emb(unrel_after)); out["crit"].append(emb(ctext))
        keep.append(rec)
        if len(keep)%100==0: print(f"  diff-emb {len(keep)}",flush=True)
    np.savez(f"{CACHE}/embdiff_arrays.npz",**{k:np.array(v) for k,v in out.items()})
    with open(f"{CACHE}/embdiff_index.jsonl","w") as f:
        for r in keep: f.write(json.dumps(r)+"\n")

    cx=np.array(out["cx"]);cic=np.array(out["cic"]);cicp=np.array(out["cicp"])
    def cv(u,v):return (u*v).sum(1)/(np.linalg.norm(u,axis=1)*np.linalg.norm(v,axis=1)+1e-9)
    print(f"DONE diff-emb {len(keep)} quads | target 偏离 {1-cv(cx,cic).mean():.4f} | unrel 偏离 {1-cv(cx,cicp).mean():.4f}",flush=True)

if __name__=="__main__": main()
