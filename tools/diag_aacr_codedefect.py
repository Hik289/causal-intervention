from civic.services import message_client, message_options, review_model
from civic.paths import OUTPUT_DIR
pass

import os
import json, os, re, random, numpy as np
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from sklearn.metrics import roc_auc_score
import urllib.request
random.seed(42)
AACR=os.environ.get("CIVIC_PROJECT_ROOT", ".") + "/data/aacr-bench_raw/dataset.json"; ARTIFACTS=str(OUTPUT_DIR)

def provider_client(t=90):
    return message_client()
RUBRIC=("You are an expert code reviewer. Judge ONLY whether the specific defect described in the critique is "
 "actually present in THIS code as shown. Output 'SCORE: <int 0-100>' then 'REASON: <text>'.")
def score(code,crit):
    r=provider_client().messages.create(model=review_model(), system=RUBRIC, messages=[{'role': 'user', 'content': f'CRITIQUE:\n{crit}\n\nCODE:\n{code[:6000]}\n\nIs the defect present?'}], **message_options())
    return r.content[0].text
def parse(txt):
    m=re.search(r'SCORE:\s*(\d+)',txt or "",re.I)
    if m: return max(0,min(100,int(m.group(1))))/100.0
    c=[int(x) for x in re.findall(r'\b(\d{1,3})\b',txt or "") if 0<=int(x)<=100]
    return c[-1]/100.0 if c else None
def fetch(slug,commit,path,frm,to,ctx=40):
    try:
        req=urllib.request.Request(f"https://raw.githubusercontent.com/{slug}/{commit}/{path}",headers={"User-Agent":"civic"})
        txt=urllib.request.urlopen(req,timeout=25).read().decode("utf-8","replace")
    except Exception: return None
    L=txt.split("\n")
    try: f=int(frm);t=int(to)
    except: f,t=1,min(len(L),80)
    return "\n".join(L[max(0,f-1-ctx):min(len(L),t+ctx)])
def slug(u):
    m=re.match(r'https://github.com/([^/]+/[^/]+)/pull',u or ""); return m.group(1) if m else None

def main():
    a=json.load(open(AACR))
    from collections import Counter
    print("category dist:",dict(Counter(r.get("category") for r in a)),flush=True)

    cd=[r for r in a if r.get("category")=="Code Defect"]
    pos=[r for r in cd if r["label"]==1 and r.get("is_ai_comment")]
    neg=[r for r in cd if r["label"]==0 and r.get("is_ai_comment")]
    random.shuffle(pos);random.shuffle(neg); N=90; pos=pos[:N];neg=neg[:N]
    print(f"Code-Defect subset: total cd={len(cd)} ({len(cd)/len(a)*100:.0f}% of AACR); pos={len(pos)} neg={len(neg)}",flush=True)

    outp=f"{ARTIFACTS}/aacr_codedefect_scores.jsonl"; done={}
    if os.path.exists(outp):
        for l in open(outp):
            r=json.loads(l)
            if r.get("score") is not None: done[r["key"]]=r
    jobs=[]
    for lab,g in [("pos",pos),("neg",neg)]:
        for i,r in enumerate(g):
            if f"{lab}_{i}" not in done: jobs.append((f"{lab}_{i}",lab,r))
    print(f"{len(jobs)} to score ({len(done)} done)",flush=True)
    fo=open(outp,"a");lk=Lock();cnt=[0]
    def work(j):
        key,lab,r=j; s=slug(r.get("pr_url"))
        rec={"key":key,"label":lab,"score":None}
        if s:
            code=fetch(s,r.get("pr_source_commit"),r.get("path"),r.get("from_line"),r.get("to_line"))
            if code:
                try: rec["score"]=parse(score(code,str(r.get("note") or "")))
                except Exception as e: rec["err"]=str(e)[:60]
        with lk: fo.write(json.dumps(rec)+"\n");fo.flush();cnt[0]+=1
        if cnt[0]%20==0: print(f"  {cnt[0]}/{len(jobs)}",flush=True)
    with ThreadPoolExecutor(max_workers=8) as ex: list(ex.map(work,jobs))
    fo.close()
    sc={}
    for l in open(outp):
        r=json.loads(l)
        if r.get("score") is not None: sc[r["key"]]=(r["label"],r["score"])
    ps=[v[1] for v in sc.values() if v[0]=="pos"]; ns=[v[1] for v in sc.values() if v[0]=="neg"]
    y=np.r_[np.ones(len(ps)),np.zeros(len(ns))]; s=np.r_[ps,ns]
    au=roc_auc_score(y,s) if ps and ns else float('nan')
    out={"task":"AACR Code-Defect subset (principled: category==Code Defect) true vs plausible-FP",
         "aacr_total":len(a),"code_defect_total":len(cd),"code_defect_frac":round(len(cd)/len(a),3),
         "n_pos":len(ps),"n_neg":len(ns),"raw_scalar_AUROC":round(float(au),4),
         "mean_s_true":round(float(np.mean(ps)),4) if ps else None,
         "mean_s_fp":round(float(np.mean(ns)),4) if ns else None,
         "compare_all_category_AUROC":0.483}
    json.dump(out,open(f"{ARTIFACTS}/diag_aacr_codedefect.json","w"),indent=2)
    print("WROTE\n"+json.dumps(out,indent=2),flush=True)

if __name__=="__main__":
    main()
