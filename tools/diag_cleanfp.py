from civic.paths import OUTPUT_DIR
pass

import os
import json, os, re, random, numpy as np
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from sklearn.metrics import roc_auc_score
random.seed(42)
PRE=os.environ.get("CIVIC_PROJECT_ROOT", ".") + "/data/anchors"; ARTIFACTS=str(OUTPUT_DIR)

def anth(timeout=90):
    import anthropic
    return anthropic.Anthropic(base_url=os.environ["ANTHROPIC_BASE_URL"],api_key=os.environ["ANTHROPIC_API_KEY"],max_retries=6,timeout=timeout)

FALSE_SYS=("You are constructing a HARD test case for a code-review verifier. Given a code snippet, write ONE "
 "plausible-sounding review comment that claims a SPECIFIC defect which does NOT actually exist in this code. "
 "The claimed defect must be the KIND of bug this code could plausibly have (e.g. a null-dereference where a guard "
 "already exists, an off-by-one where the bound is actually correct, a missing check that is in fact present). "
 "It must sound technically credible to a reviewer skimming the code, but be FALSE for THIS specific code. "
 "Do NOT describe a real bug. Output ONLY the review comment (1-3 sentences).")
def gen_false(code):
    r=anth(90).messages.create(model="claude-sonnet-4-6",max_tokens=200,temperature=0.6,system=FALSE_SYS,
        messages=[{"role":"user","content":f"CODE:\n{code[:6000]}\n\nPlausible-but-FALSE defect critique:"}])
    return r.content[0].text.strip()

RUBRIC=("You are an expert code reviewer. Judge ONLY whether the specific defect described in the critique is "
 "actually present in THIS code as shown. Output 'SCORE: <int 0-100>' (confidence defect present) then 'REASON: <text>'.")
def score(code,crit):
    r=anth().messages.create(model="claude-sonnet-4-6",max_tokens=180,temperature=0.4,system=RUBRIC,
        messages=[{"role":"user","content":f"CRITIQUE:\n{crit}\n\nCODE:\n{code[:6000]}\n\nIs the defect present in THIS code?"}])
    return r.content[0].text
def parse(txt):
    m=re.search(r'SCORE:\s*(\d+)',txt or "",re.I)
    if m: return max(0,min(100,int(m.group(1))))/100.0
    c=[int(x) for x in re.findall(r'\b(\d{1,3})\b',txt or "") if 0<=int(x)<=100]
    return c[-1]/100.0 if c else None

def main():
    blob=json.load(open(f"{PRE}/render_A.json")); cache={tuple(k.split("\t")):v for k,v in blob["cache"].items()}

    d=defaultdict(lambda: defaultdict(list))
    for l in open(f"{PRE}/scores_reviewergen_claude.jsonl"):
        r=json.loads(l)
        if r.get("score") is not None: d[r["triple_id"]].setdefault(r["version"],[]).append(r["score"])
    s_true={t:float(np.mean(v["x"])) for t,v in d.items() if "x" in v}
    s_ic_true={t:float(np.mean(v["target"])) for t,v in d.items() if "target" in v}
    tri=[json.loads(l) for l in open(f"{PRE}/triples_reviewergen.jsonl") if json.loads(l).get("critique")]
    items=[t for t in tri if (t["instance_id"],"x") in cache and t["instance_id"] in s_true]
    print(f"{len(items)} items (SWE-Review true-bug, has x code + s_true)",flush=True)

    outp=f"{ARTIFACTS}/cleanfp_scores.jsonl"; done={}
    if os.path.exists(outp):
        for l in open(outp):
            r=json.loads(l)
            if r.get("s_false") is not None: done[r["tid"]]=r
    todo=[t for t in items if t["instance_id"] not in done]
    print(f"{len(todo)} to gen c_false_hard + score ({len(done)} done)",flush=True)
    fo=open(outp,"a");lk=Lock();cnt=[0]
    def work(t):
        tid=t["instance_id"]; code=cache[(tid,"x")]
        try:
            cf=gen_false(code); s_false=parse(score(code,cf))
            rec={"tid":tid,"c_false":cf[:400],"s_false":s_false}
        except Exception as e: rec={"tid":tid,"s_false":None,"err":str(e)[:80]}
        with lk: fo.write(json.dumps(rec)+"\n");fo.flush();cnt[0]+=1
        if cnt[0]%20==0: print(f"  {cnt[0]}/{len(todo)}",flush=True)
    with ThreadPoolExecutor(max_workers=8) as ex: list(ex.map(work,todo))
    fo.close()

    sf={}
    cf_text={}
    for l in open(outp):
        r=json.loads(l)
        if r.get("s_false") is not None: sf[r["tid"]]=r["s_false"]; cf_text[r["tid"]]=r.get("c_false","")
    common=[t for t in s_true if t in sf]
    pos=np.array([s_true[t] for t in common])
    neg=np.array([sf[t] for t in common])
    y=np.r_[np.ones(len(common)),np.zeros(len(common))]; s=np.r_[pos,neg]
    au_raw=roc_auc_score(y,s)
    pairwise=float(np.mean(pos>neg))

    dt_true=np.array([s_true[t]-s_ic_true.get(t,s_true[t]) for t in common])

    F1=np.r_[pos,neg].reshape(-1,1)
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    def cvau(X):
        try: return float(roc_auc_score(y,cross_val_predict(LogisticRegression(max_iter=2000),X,y,cv=5,method="predict_proba")[:,1]))
        except: return float('nan')
    Xb=np.r_[np.c_[pos,dt_true], np.c_[neg,np.zeros(len(common))]]
    au_batt=cvau(Xb)
    out={"task":"CLEAN hard FP on SWE-Review (true-bug code: c_true vs c_false_hard plausible-nonexistent)",
         "n":len(common),"scorer":"claude-sonnet-4-6",
         "raw_scalar_AUROC":round(float(au_raw),4),"pairwise_true>false":round(pairwise,4),
         "battery_AUROC":round(float(au_batt),4),
         "mean_s_true":round(float(pos.mean()),4),"mean_s_false_hard":round(float(neg.mean()),4),
         "sample_c_false":[cf_text[t] for t in common[:5]]}
    json.dump(out,open(f"{ARTIFACTS}/diag_cleanfp.json","w"),indent=2)
    print("WROTE diag_cleanfp.json\n"+json.dumps({k:v for k,v in out.items() if k!='sample_c_false'},indent=2),flush=True)
    print("\n=== 5 c_false_hard 抽查 ===")
    for i,t in enumerate(common[:5]): print(f"[{i}] {cf_text[t][:200]}")

if __name__=="__main__":
    main()
