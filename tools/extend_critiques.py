from civic.paths import OUTPUT_DIR
pass

import os
import json, os, re, subprocess, random
import pandas as pd, glob
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
DATA=os.environ.get("CIVIC_PROJECT_ROOT", ".") + "/data/swe-review-bench_raw/data"; PRE=os.environ.get("CIVIC_PROJECT_ROOT", ".") + "/data/anchors"
HERE=str(OUTPUT_DIR)
random.seed(42)
def gpt(t=180):
    import openai; return openai.OpenAI(base_url=os.environ["OPENAI_BASE_URL"],api_key=os.environ["OPENAI_API_KEY"],max_retries=6,timeout=t)
def gpt_chat(system,user,t=180):
    r=gpt(t).chat.completions.create(model="gpt-5.5",messages=[{"role":"system","content":system},{"role":"user","content":user}])
    return r.choices[0].message.content or ""
def dbl(x):
    x=json.loads(x) if isinstance(x,str) else x
    return json.loads(x) if isinstance(x,str) else x
def sh(c,t=120):
    r=subprocess.run(c,shell=True,capture_output=True,text=True,timeout=t); return r.returncode,r.stdout,r.stderr
import shutil as _sh
def free_gb(): return _sh.disk_usage("/data2")[2]//(1024**3)

REVIEWER_SYS=("You are an expert code reviewer. You are given a bug report and the current source file(s). Based on the report, "
 "identify the SPECIFIC code-level defect: name the function, the lines, and the exact mechanism that causes the reported "
 "behavior. Write a precise technical critique (3-6 sentences) describing WHAT is wrong in the code and WHY it produces the "
 "bug. Reference concrete code (function names, line numbers, variables). Do NOT just restate the report; diagnose the "
 "mechanism in the code. Output ONLY the critique.")
def gen_true(report, code, repo):
    return gpt_chat(REVIEWER_SYS, f"REPOSITORY: {repo}\n\nBUG REPORT:\n{report[:2000]}\n\nSOURCE FILE(S):\n{code[:9000]}\n\nTechnical mechanism critique:", t=360)

FALSE_SYS=("You are an expert code reviewer. Invent a PLAUSIBLE-sounding but FALSE defect critique about this code: it must "
 "describe a specific defect (function/lines/mechanism) that sounds reasonable but does NOT actually exist in this code. "
 "Same style/specificity as a real critique. 3-6 sentences. Output ONLY the critique.")
def gen_false(code, repo):
    return gpt_chat(FALSE_SYS, f"REPOSITORY: {repo}\n\nCODE:\n{code[:9000]}\n\nPlausible-but-FALSE defect critique:", t=360)

def get_src(img, patch):
    src=[f for f in re.findall(r'diff --git a/(\S+) b/',patch) if 'test' not in f.lower()][:2]
    if not src: return None
    out=[]
    for fp in src:
        try:
            rc=subprocess.run(["docker","run","--rm","--network=none",img,"bash","-c",
                f"cd /testbed && git reset --hard -q 2>/dev/null; cat {fp} 2>/dev/null"],capture_output=True,text=True,timeout=300)
            if rc.stdout.strip(): out.append(f"=== {fp} ===\n"+rc.stdout[:8000])
        except Exception:
            return None
    return "\n\n".join(out) if out else None

def main():
    import argparse; ap=argparse.ArgumentParser()
    ap.add_argument("--n",type=int,default=130); ap.add_argument("--workers",type=int,default=6); args=ap.parse_args()
    df=pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{DATA}/*.parquet"))],ignore_index=True)
    res=df[df.resolved==True].groupby('instance_id').first().reset_index()
    meta={r.instance_id:r for _,r in res.iterrows()}
    used=set(json.loads(l)["instance_id"] for l in open(f"{PRE}/triples_reviewergen.jsonl") if json.loads(l).get("critique"))
    cand=[i for i in meta if i not in used]

    pref=["psf__requests","pallets__flask","pydata__xarray","scikit-learn","sphinx","sympy__sympy","matplotlib","django","astropy","pytest","pylint"]
    def rank(i):
        for k,p in enumerate(pref):
            if i.startswith(p): return k
        return 99
    random.shuffle(cand)
    from collections import defaultdict; import itertools
    byr=defaultdict(list)
    for i in sorted(cand,key=rank): byr[rank(i)].append(i)
    ids=[i for i in itertools.chain(*itertools.zip_longest(*byr.values())) if i is not None][:args.n]
    print(f"{len(ids)} new issues for critique gen (of {len(cand)} available)",flush=True)
    outp=f"{HERE}/extend_triples.jsonl"; done=set()
    if os.path.exists(outp):
        for l in open(outp):
            try: done.add(json.loads(l)["iid"])
            except: pass
    todo=[i for i in ids if i not in done]
    print(f"{len(todo)} todo ({len(done)} done)",flush=True)
    fo=open(outp,"a"); lk=Lock(); cnt=[0]; pull_lk=Lock()
    def work(iid):
      try:
        r=meta[iid]; img=r.image_name
        with pull_lk:
            _,have,_=sh("docker images --format '{{.Repository}}:{{.Tag}}'",t=60)
            if img not in have:
                if free_gb()<60:
                    with lk: fo.write(json.dumps({"iid":iid,"skip":"disk_low"})+"\n"); fo.flush()
                    return
                rc,_,_=sh(f"docker pull {img}",t=1800)
                if rc!=0:
                    with lk: fo.write(json.dumps({"iid":iid,"skip":"pull_fail"})+"\n"); fo.flush()
                    return
        code=get_src(img, r.model_patch)
        if not code:
            with lk: fo.write(json.dumps({"iid":iid,"skip":"no_code"})+"\n"); fo.flush()
            return
        report=str(r.get('problem_statement','') or r.get('pr_title','') or '')
        try:
            ct=gen_true(report, code, r.repo); cfx=gen_false(code, r.repo)
        except Exception as e:
            with lk: fo.write(json.dumps({"iid":iid,"skip":f"gen_err:{str(e)[:50]}"})+"\n"); fo.flush()
            return
        with lk:
            fo.write(json.dumps({"iid":iid,"repo":r.repo,"critique":ct,"c_false":cfx})+"\n"); fo.flush(); cnt[0]+=1
            if cnt[0]%10==0: print(f"  gen {cnt[0]}/{len(todo)} free={free_gb()}GB",flush=True)
      except Exception as e:
        try:
            with lk: fo.write(json.dumps({"iid":iid,"skip":f"work_err:{str(e)[:60]}"})+"\n"); fo.flush()
        except Exception: pass
    with ThreadPoolExecutor(max_workers=args.workers) as ex: list(ex.map(work,todo))
    fo.close()
    n=sum(1 for l in open(outp) if json.loads(l).get("critique"))
    print(f"DONE extend_critiques: {n} new (true+false) critiques generated",flush=True)

if __name__=="__main__": main()
