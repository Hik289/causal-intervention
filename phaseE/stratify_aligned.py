pass

import os
import json, os, re
import pandas as pd, glob
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
DATA=os.environ.get("CIVIC_PROJECT_ROOT", ".") + "/data/swe-review-bench_raw/data"; PRE=os.environ.get("CIVIC_PROJECT_ROOT", ".") + "/experiments/preexp_anchor0"
HERE=os.path.dirname(os.path.abspath(__file__))
def gpt(t=120):
    import openai; return openai.OpenAI(base_url=os.environ["OPENAI_BASE_URL"],api_key=os.environ["OPENAI_API_KEY"],max_retries=6,timeout=t)
def gpt_chat(system,user,t=120):
    r=gpt(t).chat.completions.create(model="gpt-5.5",messages=[{"role":"system","content":system},{"role":"user","content":user}])
    return r.choices[0].message.content or ""

def gold_locus(patch):
    pass
    files=[]; cur=None; loci=[]
    for ln in patch.split("\n"):
        m=re.match(r'diff --git a/(\S+) b/',ln)
        if m:
            cur=m.group(1)
            if 'test' not in cur.lower(): files.append(cur)
            continue
        if cur and 'test' not in cur.lower():
            hm=re.match(r'@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@(.*)$',ln)
            if hm: loci.append(f"{cur}: {hm.group(1).strip()[:80]}")
    return files, loci

JUDGE_SYS=("You are an impartial evaluator. You are given (1) a reviewer's defect critique and (2) the ground-truth location "
 "of the real bug (the files and function/code contexts that the actual fix modified). Judge whether the reviewer's critique "
 "points to the SAME location and mechanism as the real bug. Answer 'ALIGNED: <0-100>' where 100 = the critique identifies "
 "the same file(s) and the same code mechanism that the real fix addresses, 0 = the critique points to a different file or a "
 "different mechanism entirely. Judge only location/mechanism correspondence. Output ONLY 'ALIGNED: <int>'.")
def judge_aligned(critique, gold_files, gold_loci):
    loc=f"FILES: {', '.join(gold_files)}\nCODE CONTEXTS (function signatures near the fix):\n"+"\n".join(gold_loci[:6])
    try:
        t=gpt_chat(JUDGE_SYS,f"REVIEWER CRITIQUE:\n{critique[:1500]}\n\nGROUND-TRUTH BUG LOCATION:\n{loc}\n\nALIGNED:",t=120)
        m=re.search(r'ALIGNED:\s*(\d+)',t,re.I)
        return max(0,min(100,int(m.group(1))))/100.0 if m else None
    except Exception: return None

def main():
    import argparse; ap=argparse.ArgumentParser(); ap.add_argument("--spike",default="spike_repair.json"); args=ap.parse_args()
    d=json.load(open(f"{HERE}/{args.spike}"))
    valid=[r for r in d if r.get("base")=="fail" and not r.get("skip")]

    tri={json.loads(l)["instance_id"]:json.loads(l)["critique"] for l in open(f"{PRE}/triples_reviewergen.jsonl") if json.loads(l).get("critique")}
    extp=f"{HERE}/extend_triples.jsonl"
    if os.path.exists(extp):
        for l in open(extp):
            r=json.loads(l)
            if r.get("critique"): tri[r["iid"]]=r["critique"]
    df=pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{DATA}/*.parquet"))],ignore_index=True)
    res=df[df.resolved==True].groupby('instance_id').first().reset_index()
    meta={r.instance_id:r for _,r in res.iterrows()}
    print(f"{len(valid)} valid issues to judge alignment",flush=True)
    out=[]; lk=Lock()
    def work(r):
        iid=r["iid"]
        if iid not in tri or iid not in meta: return
        gf,gl=gold_locus(meta[iid].model_patch)
        al=None
        for _ in range(3):
            al=judge_aligned(tri[iid], gf, gl)
            if al is not None: break
        with lk: out.append({"iid":iid,"aligned_score":al,"true_flip":r.get("true_flip"),
                             "true_resolution":r.get("true_resolution"),"true_pass_delta":r.get("true_pass_delta"),
                             "false_flip":r.get("false_flip"),"nocrit_flip":r.get("nocrit_flip"),"gold_files":gf})
    with ThreadPoolExecutor(max_workers=4) as ex: list(ex.map(work,valid))
    json.dump(out,open(f"{HERE}/stratify_aligned.json","w"),indent=2,ensure_ascii=False)

    judged=[r for r in out if r["aligned_score"] is not None]
    aligned=[r for r in judged if r["aligned_score"]>=0.5]
    misaligned=[r for r in judged if r["aligned_score"]<0.5]
    def rate(sub,key="true_flip"):
        prod=[r for r in sub if r.get("true_resolution") in ('pass','fail')]
        fl=sum(1 for r in prod if r.get(key))
        return fl,len(prod)
    print("="*50,flush=True)
    print(f"judged {len(judged)}/{len(out)} (judge_err {len(out)-len(judged)})",flush=True)
    af,an=rate(aligned); mf,mn=rate(misaligned)
    print(f"ALIGNED-true (critique 指对 gold): {len(aligned)} issues, resolution {af}/{an} = {af/an:.3f}" if an else f"ALIGNED: {len(aligned)}, none produced",flush=True)
    print(f"MISALIGNED-true (指错): {len(misaligned)} issues, resolution {mf}/{mn} = {mf/mn:.3f}" if mn else f"MISALIGNED: {len(misaligned)}, none produced",flush=True)

    tp=[r for r in judged if r.get("true_resolution") in ('pass','fail')]; tf=sum(1 for r in tp if r.get("true_flip"))
    ff=sum(1 for r in judged if r.get("false_flip")); nf=sum(1 for r in judged if r.get("nocrit_flip"))
    print(f"总体 true: {tf}/{len(tp)} | false flips: {ff} | nocrit flips: {nf}",flush=True)

    try:
        from scipy import stats
        if an:
            _,p=stats.fisher_exact([[af,an-af],[ff,len(judged)-ff]])
            print(f"Fisher aligned-true vs false(pooled): p={p:.4f}",flush=True)
    except Exception as e: print("fisher err",e,flush=True)
    print(f"aligned_score 分布: "+str(sorted(round(r['aligned_score'],2) for r in judged)),flush=True)
    print("STRATIFY DONE",flush=True)

if __name__=="__main__": main()
