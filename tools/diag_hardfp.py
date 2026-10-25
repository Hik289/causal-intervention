from civic.services import message_client, message_options, review_model
from civic.paths import OUTPUT_DIR
pass

import os
import json, os, re, random, time, numpy as np
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from sklearn.metrics import roc_auc_score
import urllib.request
random.seed(42)
AACR=os.environ.get("CIVIC_PROJECT_ROOT", ".") + "/data/aacr-bench_raw/dataset.json"
ARTIFACTS=str(OUTPUT_DIR)

RUBRIC=("You are an expert code reviewer. Below is a code snippet and a critique claiming a specific defect exists. "
 "Judge ONLY whether the specific defect described in the critique is actually present in THIS code as shown. "
 "Output a single integer 0-100 = confidence the claimed defect is genuinely present, then one short reason. "
 "Format exactly: SCORE: <int>\\nREASON: <text>")
def score_llm(code, crit, timeout=90):

    c=message_client()
    r=c.messages.create(model=review_model(), system=RUBRIC, messages=[{'role': 'user', 'content': f'CRITIQUE:\n{crit}\n\nCODE:\n{code[:6000]}\n\nIs the specific defect present in THIS code?'}], **message_options())
    return r.content[0].text
def parse(txt):
    m=re.search(r'SCORE:\s*(\d+)',txt or "",re.I)
    if m: return max(0,min(100,int(m.group(1))))/100.0
    c=[int(x) for x in re.findall(r'\b(\d{1,3})\b',txt or "") if 0<=int(x)<=100]
    return c[-1]/100.0 if c else None

def fetch_region(repo_slug, commit, path, frm, to, ctx=40, timeout=25):
    url=f"https://raw.githubusercontent.com/{repo_slug}/{commit}/{path}"
    try:
        req=urllib.request.Request(url,headers={"User-Agent":"civic"})
        txt=urllib.request.urlopen(req,timeout=timeout).read().decode("utf-8","replace")
    except Exception: return None
    lines=txt.split("\n")
    try: f=int(frm); t=int(to)
    except: f,t=1,min(len(lines),80)
    lo=max(0,f-1-ctx); hi=min(len(lines),t+ctx)
    return f"# {path} lines {lo+1}-{hi}\n"+"\n".join(lines[lo:hi])

def repo_slug(pr_url):
    m=re.match(r'https://github.com/([^/]+/[^/]+)/pull',pr_url or "")
    return m.group(1) if m else None

def main():
    a=json.load(open(AACR))

    pos=[r for r in a if r["label"]==1 and r.get("is_ai_comment")]
    neg=[r for r in a if r["label"]==0 and r.get("is_ai_comment")]
    random.shuffle(pos); random.shuffle(neg)
    N=120
    pos=pos[:N]; neg=neg[:N]
    print(f"pos(true AI critique)={len(pos)} neg(hard plausible FP)={len(neg)}",flush=True)

    outp=f"{ARTIFACTS}/hardfp_scores.jsonl"
    done={}
    if os.path.exists(outp):
        for l in open(outp):
            r=json.loads(l)
            if r.get("score") is not None: done[r["key"]]=r["score"]

    jobs=[]
    for lab,group in [("pos",pos),("neg",neg)]:
        for i,r in enumerate(group):
            key=f"{lab}_{i}"
            if key in done: continue
            jobs.append((key,lab,r))
    print(f"{len(jobs)} to fetch+score ({len(done)} done)",flush=True)

    fo=open(outp,"a"); lk=Lock(); cnt=[0]
    def work(j):
        key,lab,r=j
        slug=repo_slug(r.get("pr_url"))
        if not slug:
            with lk: fo.write(json.dumps({"key":key,"label":lab,"score":None,"reason":"no_slug"})+"\n"); fo.flush()
            return
        code=fetch_region(slug, r.get("pr_source_commit"), r.get("path"), r.get("from_line"), r.get("to_line"))
        if not code:
            with lk: fo.write(json.dumps({"key":key,"label":lab,"score":None,"reason":"fetch_fail"})+"\n"); fo.flush()
            return
        crit=str(r.get("note") or "")
        try: s=parse(score_llm(code,crit))
        except Exception as e: s=None
        with lk:
            fo.write(json.dumps({"key":key,"label":lab,"score":s})+"\n"); fo.flush(); cnt[0]+=1
            if cnt[0]%20==0: print(f"  {cnt[0]}/{len(jobs)}",flush=True)
    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(work,jobs))
    fo.close()

    sc={}
    for l in open(outp):
        r=json.loads(l)
        if r.get("score") is not None: sc[r["key"]]=(r["label"],r["score"])
    ps=[v[1] for v in sc.values() if v[0]=="pos"]; ns=[v[1] for v in sc.values() if v[0]=="neg"]
    y=np.r_[np.ones(len(ps)),np.zeros(len(ns))]; s=np.r_[ps,ns]
    au=roc_auc_score(y,s) if ps and ns else float('nan')
    out={"task":"HARD plausible FP detection (AACR label=1 AI-critique vs label=0 AI-FP)",
         "n_pos":len(ps),"n_neg":len(ns),"scorer":review_model(),
         "raw_LLM_scalar_AUROC":round(float(au),4),
         "mean_s_true":round(float(np.mean(ps)),4) if ps else None,
         "mean_s_hardFP":round(float(np.mean(ns)),4) if ns else None,
         "compare_easy_FP_AUROC":0.956}
    json.dump(out,open(f"{ARTIFACTS}/diag_hardfp.json","w"),indent=2)
    print("WROTE diag_hardfp.json\n"+json.dumps(out,indent=2),flush=True)

if __name__=="__main__":
    main()
