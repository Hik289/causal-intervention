from civic.services import review_score_path
from civic.services import message_client, message_options, review_model
from civic.paths import OUTPUT_DIR
pass

import os
import json, os, re, random, numpy as np
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
import urllib.request
random.seed(42)
AACR=os.environ.get("CIVIC_PROJECT_ROOT", ".") + "/data/aacr-bench_raw/dataset.json"
ARTIFACTS=str(OUTPUT_DIR); PRE=os.environ.get("CIVIC_PROJECT_ROOT", ".") + "/data/anchors"

def provider_client(timeout=90):

    return message_client()
RUBRIC=("You are an expert code reviewer. Judge ONLY whether the specific defect described in the critique is "
 "actually present in THIS code as shown. Output 'SCORE: <int 0-100>' (confidence defect present) then 'REASON: <text>'.")
def score(code,crit):
    r=provider_client().messages.create(model=review_model(), system=RUBRIC, messages=[{'role': 'user', 'content': f'CRITIQUE:\n{crit}\n\nCODE:\n{code[:6000]}\n\nIs the defect present?'}], **message_options())
    return r.content[0].text
FIX_SYS=("You are a developer. The review comment below claims a specific defect in the code. Apply a MINIMAL edit "
 "that fixes exactly the claimed defect. If the claimed defect does NOT actually exist, make the smallest change you "
 "would make to address the comment anyway (may be a redundant guard or no-op). Output ONLY the full edited code region.")
def gen_fix(code,crit):
    r=provider_client(120).messages.create(model=review_model(), system=FIX_SYS, messages=[{'role': 'user', 'content': f'COMMENT:\n{crit}\n\nCODE:\n{code[:6000]}\n\nEdited code region:'}], **message_options())
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
    lo=max(0,f-1-ctx);hi=min(len(L),t+ctx); return "\n".join(L[lo:hi])
def slug(u):
    m=re.match(r'https://github.com/([^/]+/[^/]+)/pull',u or ""); return m.group(1) if m else None

def cvau(X,y):
    try: return float(roc_auc_score(y,cross_val_predict(LogisticRegression(max_iter=2000),X,y,cv=5,method="predict_proba")[:,1]))
    except Exception: return float('nan')

def aacr_battery():
    a=json.load(open(AACR))
    pos=[r for r in a if r["label"]==1 and r.get("is_ai_comment")]
    neg=[r for r in a if r["label"]==0 and r.get("is_ai_comment")]
    random.shuffle(pos);random.shuffle(neg); N=70; pos=pos[:N];neg=neg[:N]
    outp=f"{ARTIFACTS}/hardfp_battery_aacr.jsonl"; done={}
    if os.path.exists(outp):
        for l in open(outp):
            r=json.loads(l)
            if r.get("sx") is not None: done[r["key"]]=r
    jobs=[]
    for lab,g in [("pos",pos),("neg",neg)]:
        for i,r in enumerate(g):
            if f"{lab}_{i}" not in done: jobs.append((f"{lab}_{i}",lab,r))
    print(f"AACR battery: {len(jobs)} to process ({len(done)} done)",flush=True)
    fo=open(outp,"a");lk=Lock();cnt=[0]
    def work(j):
        key,lab,r=j; s=slug(r.get("pr_url"))
        rec={"key":key,"label":lab}
        if not s: rec["sx"]=None;
        else:
            code=fetch(s,r.get("pr_source_commit"),r.get("path"),r.get("from_line"),r.get("to_line"))
            crit=str(r.get("note") or "")
            if not code: rec["sx"]=None
            else:
                try:
                    sx=parse(score(code,crit))
                    fix=gen_fix(code,crit); s_ic=parse(score(fix,crit))

                    rec.update({"sx":sx,"s_ic":s_ic})
                except Exception as e: rec["sx"]=None; rec["err"]=str(e)[:80]
        with lk: fo.write(json.dumps(rec)+"\n");fo.flush();cnt[0]+=1
        if cnt[0]%20==0: print(f"  aacr {cnt[0]}/{len(jobs)}",flush=True)
    with ThreadPoolExecutor(max_workers=6) as ex: list(ex.map(work,jobs))
    fo.close()

    rows=[json.loads(l) for l in open(outp)]
    rows=[r for r in rows if r.get("sx") is not None and r.get("s_ic") is not None]
    lab=np.array([1 if r["label"]=="pos" else 0 for r in rows])
    sx=np.array([r["sx"] for r in rows]); sic=np.array([r["s_ic"] for r in rows])
    dt=sx-sic

    F1=cvau(sx.reshape(-1,1),lab); F2=cvau(np.c_[sx,dt],lab); F4=cvau(dt.reshape(-1,1),lab)
    return {"n":len(rows),"n_pos":int(lab.sum()),"F1_sx_only":round(F1,4),"F2_sx+dtarget":round(F2,4),
            "F4_dtarget_only":round(F4,4),"mean_dt_true":round(float(dt[lab==1].mean()),4),
            "mean_dt_false":round(float(dt[lab==0].mean()),4),
            "note":"AACR: true(label1) vs plausible-FP(label0). I_c=LLM fix (UNVERIFIED, confound). Δtarget=s(x)-s(Ic)."}

def swe_executable():
    pass

    d=defaultdict(dict)
    for l in open(review_score_path(PRE)):
        r=json.loads(l)
        if r.get("score") is None: continue
        d[r["triple_id"]].setdefault(r["version"],[]).append(r["score"])
    agg={t:{v:float(np.mean(s)) for v,s in vv.items()} for t,vv in d.items()}
    tids=[t for t in agg if "x" in agg[t] and "target" in agg[t] and any(k.startswith("unrel") for k in agg[t])]
    sx=np.array([agg[t]["x"] for t in tids]); stg=np.array([agg[t]["target"] for t in tids])
    su=np.array([float(np.mean([agg[t][k] for k in agg[t] if k.startswith("unrel")])) for t in tids])
    d_real=sx-stg; d_unrel=sx-su

    y=np.r_[np.ones(len(tids)),np.zeros(len(tids))]
    au=roc_auc_score(y,np.r_[d_real,d_unrel])
    return {"n":len(tids),"AUROC_realfix_vs_unrelfix":round(float(au),4),
            "mean_d_realfix":round(float(d_real.mean()),4),"mean_d_unrelfix":round(float(d_unrel.mean()),4),
            "note":"SWE-Review reviewer-gen: real-fix(resolved,executable) Δ vs unrel-fix Δ. tests if validity tracks real removal."}

def main():
    out={}
    out["AACR_hardfp_battery"]=aacr_battery()
    out["SWE_executable_intervention"]=swe_executable()
    json.dump(out,open(f"{ARTIFACTS}/diag_hardfp_battery.json","w"),indent=2)
    print("WROTE\n"+json.dumps(out,indent=2),flush=True)

if __name__=="__main__":
    main()
