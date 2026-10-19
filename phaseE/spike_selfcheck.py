pass

import os
import json, os, re, base64, subprocess, sys
import pandas as pd, glob
DATA=os.environ.get("CIVIC_PROJECT_ROOT", ".") + "/data/swe-review-bench_raw/data"; PRE=os.environ.get("CIVIC_PROJECT_ROOT", ".") + "/experiments/preexp_anchor0"
PHC=os.environ.get("CIVIC_PROJECT_ROOT", ".") + "/phaseC"
def dbl(x):
    x=json.loads(x) if isinstance(x,str) else x
    return json.loads(x) if isinstance(x,str) else x
def anth(t=90):
    import anthropic; return anthropic.Anthropic(base_url=os.environ["ANTHROPIC_BASE_URL"],api_key=os.environ["ANTHROPIC_API_KEY"],max_retries=6,timeout=t)
def sh(c,t=120):
    r=subprocess.run(c,shell=True,capture_output=True,text=True,timeout=t); return r.returncode,r.stdout,r.stderr

GEN_SYS=("You are an expert code reviewer. You are given a code snippet and a candidate defect critique. "
 "If the claimed defect is REAL, write a SELF-CONTAINED Python script that EXECUTABLY demonstrates it, to run from the "
 "repository root with the package importable. The script must: (1) import the relevant module(s) from the repository, "
 "(2) construct the minimal input/call that exercises the claimed defect, (3) check for the specific WRONG behavior the "
 "defect would cause. CONTRACT: call sys.exit(0) if the predicted BUGGY behavior IS observed (defect confirmed present), "
 "or sys.exit(1) if the behavior is correct (defect NOT present). Wrap the core check in try/except; on unexpected "
 "exception that itself is the predicted failure, exit 0, else exit 1. Output STRICT JSON: "
 "{\"mechanism\":\"<one sentence>\",\"check_script\":\"<complete python, uses sys.exit(0/1)>\"}. Output ONLY JSON. "
 "Do NOT assume the defect is real; if after writing you think it is not present, the script should exit 1 on this code.")
def gen_check(code,crit,repo,retries=3):
    pass
    last=""
    for att in range(retries):
        try:
            r=anth(120).messages.create(model="claude-sonnet-4-6",max_tokens=1200,temperature=0.3 if att==0 else 0.5,system=GEN_SYS,
                messages=[{"role":"user","content":f"REPOSITORY: {repo}\n\nCRITIQUE (claimed defect):\n{crit}\n\nCODE:\n{code[:6000]}\n\nSelf-contained executable check JSON:"}])
            last=r.content[0].text
            o=parse_json(last)
            if o and o.get("check_script"): return last
        except Exception: pass
    return last
def parse_json(t):
    t=re.sub(r'^```(?:json)?','',t.strip()); t=re.sub(r'```$','',t.strip()); i=t.find("{")
    if i<0: return None
    try: o,_=json.JSONDecoder().raw_decode(t[i:]); return o
    except: return None

def code_region_docker(img, patch):
    pass
    src=[f for f in re.findall(r'diff --git a/(\S+) b/',patch) if 'test' not in f.lower()][:1]
    if not src: return None,None
    fp=src[0]
    script=(f"cd /testbed && git reset --hard -q 2>/dev/null; sed -n '1,400p' {fp} 2>/dev/null")
    rc=subprocess.run(["docker","run","--rm","--network=none",img,"bash","-c",script],capture_output=True,text=True,timeout=120)
    return fp, rc.stdout[:6000]

def run_check(img, check_script, timeout=120):
    pass

    b64=base64.b64encode(check_script.encode()).decode()
    script=("source /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed && cd /testbed && "
            "git reset --hard -q && git clean -fdq && "
            f"echo {b64} | base64 -d > /tmp/chk.py && timeout 90 python /tmp/chk.py; echo EXIT=$?")
    try:
        rc=subprocess.run(["docker","run","--rm","--network=none",img,"bash","-c",script],capture_output=True,text=True,timeout=timeout)
        out=rc.stdout+rc.stderr
    except Exception as e:
        return False, False, f"docker_timeout:{e}"
    m=re.search(r'EXIT=(\d+)',out); ec=int(m.group(1)) if m else -1
    low=out.lower()

    broke = bool(re.search(r'(syntaxerror|importerror|modulenotfounderror|no module named|cannot import|nameerror|attributeerror.*has no attribute)', low))

    runnable = (not broke) and ec in (0,1)
    triggered = (ec==0) and runnable
    tail="\n".join(out.splitlines()[-6:])
    return runnable, triggered, tail

def main():
    import argparse,random; ap=argparse.ArgumentParser()
    ap.add_argument("--n",type=int,default=5); ap.add_argument("--clean_ids",default="")
    ap.add_argument("--out",default="spike_selfcheck.json")
    args=ap.parse_args(); random.seed(42)
    global OUTNAME; OUTNAME=args.out
    df=pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{DATA}/*.parquet"))],ignore_index=True)
    res=df[df.resolved==True].groupby('instance_id').first().reset_index()
    meta={r.instance_id:r for _,r in res.iterrows()}
    tri={json.loads(l)["instance_id"]:json.loads(l)["critique"] for l in open(f"{PRE}/triples_reviewergen.jsonl") if json.loads(l).get("critique")}
    cfalse={}
    for l in open(f"{PHC}/cleanfp_scores.jsonl"):
        r=json.loads(l)
        if r.get("c_false"): cfalse[r["tid"]]=r["c_false"]

    if args.clean_ids and os.path.exists(args.clean_ids):
        clean=json.load(open(args.clean_ids))
        pool=[i for i in clean if i in tri and i in meta and i in cfalse]
    else:
        pool=[t for t in tri if t in meta and t in cfalse]

    pref=["psf__requests","pallets__flask","pydata__xarray","scikit-learn","sphinx","sympy__sympy","matplotlib","django","pylint","astropy","pytest"]
    def rank(i):
        for k,p in enumerate(pref):
            if i.startswith(p): return k
        return 99
    random.shuffle(pool)

    from collections import defaultdict
    byrepo=defaultdict(list)
    for i in sorted(pool,key=rank): byrepo[rank(i)].append(i)
    ids=[]; import itertools
    for i in itertools.chain(*itertools.zip_longest(*byrepo.values())):
        if i is not None: ids.append(i)
        if len(ids)>=args.n: break
    ids=ids[:args.n]
    print(f"spike on {len(ids)} CLEAN issues: {ids}",flush=True)
    out=[]
    for n,iid in enumerate(ids):
        r=meta[iid]; img=r.image_name
        _,have,_=sh("docker images --format '{{.Repository}}:{{.Tag}}'",t=60)
        if img not in have:
            print(f"[{n}] pulling {img}",flush=True); rc,_,_=sh(f"docker pull {img}",t=1800)
            if rc!=0: print(f"[{n}] pull FAIL",flush=True); continue
        fp,code=code_region_docker(img, r.model_patch)
        if not code: print(f"[{n}] {iid} no code",flush=True); continue
        rec={"iid":iid,"repo":r.repo}
        for lab,crit in [("true",tri[iid]),("false",cfalse.get(iid))]:
            if not crit: continue
            obj=parse_json(gen_check(code,crit,r.repo)) or {}
            chk=obj.get("check_script","")
            if not chk:
                rec[f"{lab}_runnable"]=False; rec[f"{lab}_triggered"]=False; rec[f"{lab}_note"]="no_script"; continue
            runnable,triggered,tail=run_check(img, chk)
            rec[f"{lab}_runnable"]=runnable; rec[f"{lab}_triggered"]=triggered
            rec[f"{lab}_mech"]=str(obj.get("mechanism",""))[:120]; rec[f"{lab}_tail"]=tail[:200]
            print(f"[{n}] {iid} {lab}: runnable={runnable} triggered={triggered}",flush=True)
        out.append(rec)
        json.dump(out,open(f"{os.path.dirname(os.path.abspath(__file__))}/{OUTNAME}","w"),indent=2,ensure_ascii=False)

    tr=[r for r in out]
    rr=[r.get("true_runnable") for r in tr]+[r.get("false_runnable") for r in tr]
    rr=[x for x in rr if x is not None]
    runnable_rate=sum(rr)/len(rr) if rr else 0
    tt=[r.get("true_triggered") for r in tr if r.get("true_runnable")]
    ft=[r.get("false_triggered") for r in tr if r.get("false_runnable")]
    print("="*50,flush=True)
    print(f"RUNNABLE RATE: {runnable_rate:.2f} ({sum(rr)}/{len(rr)})",flush=True)
    print(f"TRUE triggered: {sum(x for x in tt)}/{len(tt)}" if tt else "TRUE: none runnable",flush=True)
    print(f"FALSE triggered: {sum(x for x in ft)}/{len(ft)}" if ft else "FALSE: none runnable",flush=True)
    print("SPIKE DONE",flush=True)

if __name__=="__main__": main()
