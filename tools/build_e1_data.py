from civic.paths import OUTPUT_DIR
pass

import os
import json, os, re, random, numpy as np
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import pandas as pd, glob
random.seed(42)
DATA=os.environ.get("CIVIC_PROJECT_ROOT", ".") + "/data/swe-review-bench_raw/data"; PRE=os.environ.get("CIVIC_PROJECT_ROOT", ".") + "/data/anchors"
ARTIFACTS=str(OUTPUT_DIR); REPOS=os.environ.get("CIVIC_PROJECT_ROOT", ".") + "/repos_cache"
DS=os.environ.get("CIVIC_PROJECT_ROOT", ".") + "/data/intervention_dataset"
REPO_DIR={"astropy/astropy":"astropy","django/django":"django","matplotlib/matplotlib":"matplotlib",
 "mwaskom/seaborn":"seaborn","pallets/flask":"flask","psf/requests":"requests","pydata/xarray":"xarray",
 "pylint-dev/pylint":"pylint","pytest-dev/pytest":"pytest","scikit-learn/scikit-learn":"scikit-learn",
 "sphinx-doc/sphinx":"sphinx","sympy/sympy":"sympy"}
SPLIT={"train":["django/django","sympy/sympy","astropy/astropy","pytest-dev/pytest","mwaskom/seaborn","pallets/flask","pylint-dev/pylint"],
 "val":["scikit-learn/scikit-learn","psf/requests"],"test":["sphinx-doc/sphinx","matplotlib/matplotlib","pydata/xarray"]}
def split_of(r):
    for s,rs in SPLIT.items():
        if r in rs: return s
    return None
repo_locks={}
def sh(c,cwd=None,t=120):
    import subprocess; r=subprocess.run(c,cwd=cwd,shell=True,capture_output=True,text=True,timeout=t); return r.returncode,r.stdout,r.stderr
def region(repo,commit,patch,ctx=40):
    from threading import Lock as L
    d=os.path.join(REPOS,REPO_DIR[repo]); lk=repo_locks.setdefault(REPO_DIR[repo],L())
    with lk:
        sh("git reset --hard -q && git clean -fdq",cwd=d); rc,_,_=sh(f"git checkout -q {commit}",cwd=d)
        if rc!=0:
            sh("git fetch -q --all",cwd=d,t=600); rc,_,_=sh(f"git checkout -q {commit}",cwd=d)
            if rc!=0: return None
        src=[f for f in re.findall(r'diff --git a/(\S+) b/',patch) if 'test' not in f.lower()]
        out=[]
        for fp in src[:1]:
            full=os.path.join(d,fp)
            if not os.path.isfile(full): continue
            L2=open(full,encoding='utf-8',errors='replace').read().split("\n")
            hs=[int(m.group(1)) for m in re.finditer(r'@@ -\d+(?:,\d+)? \+(\d+)',patch)]
            if not hs: hs=[1]
            out.append("\n".join(L2[max(0,min(hs)-ctx):min(len(L2),max(hs)+ctx)]))
        return "\n".join(out)[:6000] if out else None

def anth(t=90):
    import anthropic; return anthropic.Anthropic(base_url=os.environ["ANTHROPIC_BASE_URL"],api_key=os.environ["ANTHROPIC_API_KEY"],max_retries=6,timeout=t)
FALSE_SYS=("You are constructing a HARD test case. Given code, write ONE plausible-sounding review comment claiming a "
 "SPECIFIC defect that does NOT exist in this code (null-deref where guard exists, off-by-one where bound is correct, "
 "missing check that is present). Must sound credible but be FALSE for THIS code. Output ONLY the comment (1-3 sent).")
def gen_false(code):
    return anth(90).messages.create(model="claude-sonnet-4-6",max_tokens=200,temperature=0.6,system=FALSE_SYS,
        messages=[{"role":"user","content":f"CODE:\n{code[:6000]}\n\nPlausible-but-FALSE critique:"}]).content[0].text.strip()
RUBRIC=("You are an expert code reviewer. Judge ONLY whether the specific defect in the critique is actually present in "
 "THIS code. Output 'SCORE: <int 0-100>' then 'REASON: <text>'.")
def score(code,crit):
    return anth().messages.create(model="claude-sonnet-4-6",max_tokens=180,temperature=0.4,system=RUBRIC,
        messages=[{"role":"user","content":f"CRITIQUE:\n{crit}\n\nCODE:\n{code[:6000]}\n\nIs the defect present?"}]).content[0].text
def parse(txt):
    m=re.search(r'SCORE:\s*(\d+)',txt or "",re.I)
    if m: return max(0,min(100,int(m.group(1))))/100.0
    c=[int(x) for x in re.findall(r'\b(\d{1,3})\b',txt or "") if 0<=int(x)<=100]
    return c[-1]/100.0 if c else None

def main():
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument("--n_per_split",type=int,default=120); ap.add_argument("--workers",type=int,default=8)
    args=ap.parse_args()
    df=pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{DATA}/*.parquet"))],ignore_index=True)
    res=df[df.resolved==True].groupby('instance_id').first().reset_index()
    meta={r.instance_id:r for _,r in res.iterrows()}

    crit={}
    for l in open(f"{DS}/quadruples_final.jsonl"):
        q=json.loads(l)
        if q["issue_id"] not in crit and q["critique"].get("text"): crit[q["issue_id"]]=q["critique"]["text"]

    d=defaultdict(lambda: defaultdict(list))
    for l in open(f"{PRE}/scores_reviewergen_claude.jsonl"):
        r=json.loads(l)
        if r.get("score") is not None: d[r["triple_id"]][r["version"]].append(r["score"])
    s_true_pre={t:float(np.mean(v["x"])) for t,v in d.items() if "x" in v}
    cf_pre={}
    if os.path.exists(f"{ARTIFACTS}/cleanfp_scores.jsonl"):
        for l in open(f"{ARTIFACTS}/cleanfp_scores.jsonl"):
            r=json.loads(l)
            if r.get("s_false") is not None: cf_pre[r["tid"]]=(r["s_false"],r.get("c_false",""))

    bysplit=defaultdict(list)
    for iid,r in meta.items():
        sp=split_of(r.repo)
        if sp and iid in crit: bysplit[sp].append(iid)
    sel=[]
    for sp,ids in bysplit.items():
        random.shuffle(ids); sel+=ids[:args.n_per_split]
    print("selected per split:",{sp:min(len(ids),args.n_per_split) for sp,ids in bysplit.items()},flush=True)

    outp=f"{ARTIFACTS}/e1_data.jsonl"; done=set()
    if os.path.exists(outp):
        for l in open(outp):
            try: done.add(json.loads(l)["issue_id"])
            except: pass
    todo=[i for i in sel if i not in done]
    print(f"{len(sel)} selected, {len(done)} done, {len(todo)} todo",flush=True)
    fo=open(outp,"a");lk=Lock();cnt=[0]
    def work(iid):
        r=meta[iid]; sp=split_of(r.repo)
        code=region(r.repo,r.base_commit,r.model_patch)
        if not code:
            with lk: fo.write(json.dumps({"issue_id":iid,"repo":r.repo,"split":sp,"s_true":None,"reason":"no_code"})+"\n");fo.flush()
            return

        st=s_true_pre.get(iid)
        if st is None:
            try: st=parse(score(code,crit[iid]))
            except: st=None

        if iid in cf_pre: sf,cf=cf_pre[iid]
        else:
            try: cf=gen_false(code); sf=parse(score(code,cf))
            except: cf,sf=None,None
        with lk:
            fo.write(json.dumps({"issue_id":iid,"repo":r.repo,"split":sp,"s_true":st,"s_false":sf,
                                 "c_true":crit[iid][:300],"c_false":(cf or "")[:300]})+"\n");fo.flush();cnt[0]+=1
            if cnt[0]%20==0: print(f"  {cnt[0]}/{len(todo)}",flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as ex: list(ex.map(work,todo))
    fo.close()

    rows=[json.loads(l) for l in open(outp) if json.loads(l).get("s_true") is not None and json.loads(l).get("s_false") is not None]
    from collections import Counter
    print("usable:",len(rows),"by split:",dict(Counter(r["split"] for r in rows)),flush=True)

if __name__=="__main__":
    main()
