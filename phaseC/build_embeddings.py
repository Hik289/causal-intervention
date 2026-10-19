pass

import os
import json, re, os, subprocess, argparse, hashlib
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import pandas as pd, glob

DS=os.environ.get("CIVIC_PROJECT_ROOT", ".") + "/data/intervention_dataset"
DATA=os.environ.get("CIVIC_PROJECT_ROOT", ".") + "/data/swe-review-bench_raw/data"
REPOS=os.environ.get("CIVIC_PROJECT_ROOT", ".") + "/repos_cache"
REPO_DIR={"astropy/astropy":"astropy","django/django":"django","matplotlib/matplotlib":"matplotlib",
 "mwaskom/seaborn":"seaborn","pallets/flask":"flask","psf/requests":"requests","pydata/xarray":"xarray",
 "pylint-dev/pylint":"pylint","pytest-dev/pytest":"pytest","scikit-learn/scikit-learn":"scikit-learn",
 "sphinx-doc/sphinx":"sphinx","sympy/sympy":"sympy"}
repo_locks={}

def sh(cmd,cwd=None,timeout=120):
    r=subprocess.run(cmd,cwd=cwd,shell=True,capture_output=True,text=True,timeout=timeout); return r.returncode,r.stdout,r.stderr

import ast as _ast
def _func_span(src_text, line):
    try: tree=_ast.parse(src_text)
    except: return None
    best=None
    for n in _ast.walk(tree):
        if isinstance(n,(_ast.FunctionDef,_ast.AsyncFunctionDef)):
            e=getattr(n,"end_lineno",None)
            if e and n.lineno<=line<=e and (best is None or e-n.lineno<best[1]-best[0]):
                best=(n.lineno,e)
    return best

def region_after_patch(repo, base_commit, patch, src_files, hunk_ref, ctx=3, func_level=False):
    pass

    from threading import Lock as L
    d=os.path.join(REPOS,REPO_DIR[repo])
    lk=repo_locks.setdefault(REPO_DIR[repo], L())
    with lk:
        sh("git reset --hard -q && git clean -fdq",cwd=d,timeout=120)
        rc,_,_=sh(f"git checkout -q {base_commit}",cwd=d,timeout=120)
        if rc!=0:
            sh("git fetch -q --all",cwd=d,timeout=600); rc,_,_=sh(f"git checkout -q {base_commit}",cwd=d,timeout=120)
            if rc!=0: return None
        if patch:
            pf=os.path.join(d,".e.diff"); open(pf,"w").write(patch)
            rc,_,_=sh("git apply --check .e.diff",cwd=d,timeout=60)
            if rc==0: sh("git apply .e.diff",cwd=d,timeout=60)
            else:
                rc2,_,_=sh("git apply --3way --check .e.diff",cwd=d,timeout=60)
                if rc2==0: sh("git apply --3way .e.diff",cwd=d,timeout=60)
                else: os.remove(pf); return None
            os.remove(pf)

        out=[]
        hunk_patch = hunk_ref or ""
        for fp in src_files:
            full=os.path.join(d,fp)
            if not os.path.isfile(full): continue
            try: lines=open(full,encoding='utf-8',errors='replace').read().split("\n")
            except: continue

            hs=[]
            infile=False
            for line in hunk_patch.splitlines():
                mm=re.match(r'diff --git a/(\S+) b/(\S+)',line)
                if mm: infile=(mm.group(2)==fp or mm.group(1)==fp); continue
                if infile:
                    h=re.match(r'@@ -\d+(?:,\d+)? \+(\d+)',line)
                    if h: hs.append(int(h.group(1)))
            if not hs: hs=[1]
            if func_level:
                sp=_func_span("\n".join(lines), min(hs))
                if sp: lo,hi=sp[0]-1,sp[1]
                else: lo=max(0,min(hs)-10); hi=min(len(lines),max(hs)+10)
            else:
                lo=max(0,min(hs)-ctx); hi=min(len(lines),max(hs)+ctx)
            out.append(f"# {fp}\n"+"\n".join(lines[lo:hi]))
        return "\n".join(out)[:8000] if out else None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--quads",default=os.path.join(DS,"quadruples_final.jsonl"))
    ap.add_argument("--out_prefix",default=os.path.join(os.environ.get("CIVIC_PROJECT_ROOT", ".") + "/cache","emb"))
    ap.add_argument("--model",default="all-MiniLM-L6-v2")
    ap.add_argument("--ctx",type=int,default=3)
    ap.add_argument("--func_level",action="store_true")
    ap.add_argument("--workers",type=int,default=8)
    args=ap.parse_args()
    os.environ["HF_HOME"]=os.environ.get("CIVIC_PROJECT_ROOT", ".") + "/cache/hf"

    df=pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{DATA}/*.parquet"))],ignore_index=True)
    res=df[df.resolved==True].groupby('instance_id').first().reset_index()
    meta={r.instance_id:r for _,r in res.iterrows()}
    patch_by_issue={r.instance_id:r.model_patch for _,r in res.iterrows()}

    quads=[json.loads(l) for l in open(args.quads)]
    def srcf(p): return [f for f in re.findall(r'diff --git a/(\S+) b/',p) if 'test' not in f.lower()]

    idx_path=args.out_prefix+"_index.jsonl"
    done=set()
    if os.path.exists(idx_path):
        for l in open(idx_path):
            try: done.add(json.loads(l)["quad_id"])
            except: pass
    todo=[q for q in quads if q["quad_id"] not in done]
    print(f"{len(quads)} quads, {len(done)} done, {len(todo)} todo",flush=True)

    from collections import defaultdict
    by_repo=defaultdict(list)
    for q in todo: by_repo[q["repo"]].append(q)

    base_cache={}
    tgt_cache={}
    dist_cache={}
    lk=Lock(); built=[]; prog=[0]

    def proc_repo(repo, qs):
        local=[]
        for q in qs:
            iid=q["issue_id"]; r=meta.get(iid)
            if r is None: prog[0]+=1; continue
            tgt_patch=patch_by_issue.get(iid)
            if not tgt_patch: prog[0]+=1; continue
            src=srcf(tgt_patch)
            dist_iid=q["I_cprime"]["from_issue"]; dist_patch=patch_by_issue.get(dist_iid)

            union_src=list(dict.fromkeys(src))
            hunk_ref=(tgt_patch or "")
            if iid not in base_cache:
                base_cache[iid]=region_after_patch(repo,r.base_commit,None,union_src,hunk_ref,ctx=args.ctx,func_level=args.func_level)
                tgt_cache[iid]=region_after_patch(repo,r.base_commit,tgt_patch,union_src,hunk_ref,ctx=args.ctx,func_level=args.func_level)
            rx_base=base_cache[iid]; ric=tgt_cache[iid]
            dk=(iid,dist_iid)
            if dk not in dist_cache:
                dist_cache[dk]=region_after_patch(repo,r.base_commit,dist_patch,union_src,hunk_ref,ctx=args.ctx,func_level=args.func_level) if dist_patch else None
            ricp=dist_cache[dk]
            prog[0]+=1
            if prog[0]%100==0: print(f"  text {prog[0]}/{len(todo)} (ok {len(built)+len(local)})",flush=True)
            if rx_base is None or ric is None: continue
            local.append({"quad_id":q["quad_id"],"issue_id":iid,"repo":repo,"split":q["split"],
                "is_base":q.get("is_base"),"critique":q["critique"]["text"],
                "code_x":rx_base,"code_ic":ric,"code_icp":ricp,
                "has_tx":bool(q.get("T_x") and q["T_x"].get("verified"))})
        with lk: built.extend(local)

    with ThreadPoolExecutor(max_workers=min(args.workers,len(by_repo))) as ex:
        list(ex.map(lambda kv: proc_repo(*kv), by_repo.items()))
    print(f"built {len(built)} code-region records",flush=True)

    from sentence_transformers import SentenceTransformer
    m=SentenceTransformer(args.model)
    def emb(texts): return m.encode(texts,batch_size=64,show_progress_bar=False,convert_to_numpy=True)
    crit=emb([b["critique"] for b in built])
    cx=emb([b["code_x"] for b in built])
    cic=emb([b["code_ic"] for b in built])
    cicp=emb([b["code_icp"] if b["code_icp"] else "" for b in built])

    arr={"crit":crit,"cx":cx,"cic":cic,"cicp":cicp}
    np.savez(args.out_prefix+"_arrays.npz",**arr)
    with open(idx_path,"w") as f:
        for b in built: f.write(json.dumps({k:b[k] for k in ("quad_id","issue_id","repo","split","is_base","has_tx")})+"\n")
    print(f"DONE embeddings: {len(built)} quads, dim={crit.shape[1]}",flush=True)

if __name__=="__main__":
    main()
