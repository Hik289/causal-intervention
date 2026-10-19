from civic.paths import OUTPUT_DIR
pass

import os
import json, os, re, argparse
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import numpy as np
DATA=os.environ.get("CIVIC_PROJECT_ROOT", ".") + "/data"
def gpt(t=120):
    import openai; return openai.OpenAI(base_url=os.environ["OPENAI_BASE_URL"],api_key=os.environ["OPENAI_API_KEY"],max_retries=6,timeout=t)
def chat(system,user,t=120):
    r=gpt(t).chat.completions.create(model=os.environ.get("LLM_MODEL_NAME", "gpt-4"),messages=[{"role":"system","content":system},{"role":"user","content":user}])
    return r.choices[0].message.content or ""
def parse_valid(txt):
    m=re.search(r'VALID:\s*(\d+)',txt,re.I); return max(0,min(100,int(m.group(1))))/100.0 if m else None

REVIEW_SYS=("You are an expert code reviewer judging whether a review finding describes a REAL defect vs a false-positive. "
 "Output ONLY 'VALID: <int 0-100>' (100=real, 0=false-positive).")

def b_frozen(c,ctx,cat):
    return parse_valid(chat(REVIEW_SYS,f"FINDING:\n{c[:1500]}\nCODE CONTEXT:\n{str(ctx)[:3000]}\nVALID:"))
def b_self_reflection(c,ctx,cat):
    s1=chat(REVIEW_SYS,f"FINDING:\n{c[:1500]}\nCODE CONTEXT:\n{str(ctx)[:3000]}\nVALID:")
    v1=parse_valid(s1)
    s2=chat(REVIEW_SYS+" Reflect critically on your first judgment, check for over-flagging, then give final.",
            f"FINDING:\n{c[:1500]}\nCODE CONTEXT:\n{str(ctx)[:2000]}\nYour first judgment: {s1[:200]}\nReflect: is this finding truly a real defect or are you over-flagging? Final VALID:")
    return parse_valid(s2) if parse_valid(s2) is not None else v1
def b_llm_verifier(c,ctx,cat):
    v1=parse_valid(chat(REVIEW_SYS,f"FINDING:\n{c[:1500]}\nCODE CONTEXT:\n{str(ctx)[:3000]}\nVALID:"))
    ver=chat("You are an independent verifier. A reviewer flagged a finding. Verify whether the claimed defect truly exists. Output ONLY 'VALID: <int 0-100>'.",
             f"FINDING:\n{c[:1500]}\nCODE CONTEXT:\n{str(ctx)[:2500]}\nVerify the claim. VALID:")
    v2=parse_valid(ver)
    return (v1+v2)/2 if (v1 is not None and v2 is not None) else (v2 if v2 is not None else v1)
def b_fix_guided(c,ctx,cat):
    fix=chat("You are a code-fix assistant. Given a review finding, describe the minimal fix IF the defect is real, or say 'NO FIX NEEDED' if the code is already correct / finding is wrong.",
             f"FINDING:\n{c[:1500]}\nCODE CONTEXT:\n{str(ctx)[:2500]}\nMinimal fix or NO FIX NEEDED:")
    if 'NO FIX' in fix.upper(): return 0.2
    j=chat(REVIEW_SYS+" You have a proposed fix; judge if the original defect was real given a concrete fix exists.",
           f"FINDING:\n{c[:1200]}\nPROPOSED FIX:\n{fix[:1500]}\nVALID:")
    return parse_valid(j)

BASELINES={"frozen":b_frozen,"self_reflection":b_self_reflection,"llm_verifier":b_llm_verifier,"fix_guided":b_fix_guided}

def load_aacr():
    d=json.load(open(f"{DATA}/aacr-bench_raw/dataset.json"))
    import random; random.seed(42)
    items=[{"critique":r.get("note",""),"context":r.get("context",""),"category":r.get("category",""),
            "label":int(r.get("label")),"pr":r.get("pr_url","")} for r in d if r.get("note") and r.get("label") is not None]
    random.shuffle(items); return items[:200]

def metrics(scores,labels,thr):
    pred=[1 if s>=thr else 0 for s in scores]
    tp=sum(1 for p,l in zip(pred,labels) if p==1 and l==1);fp=sum(1 for p,l in zip(pred,labels) if p==1 and l==0)
    fn=sum(1 for p,l in zip(pred,labels) if p==0 and l==1);tn=sum(1 for p,l in zip(pred,labels) if p==0 and l==0)
    P=tp/(tp+fp) if tp+fp else 0;R=tp/(tp+fn) if tp+fn else 0;F1=2*P*R/(P+R) if P+R else 0
    return {"P":round(P,3),"R":round(R,3),"F1":round(F1,3),"fp":fp,"tp":tp}
def fp_per_pr(scores,labels,prs,thr):
    pass
    from collections import defaultdict
    byp=defaultdict(lambda:{"fp":0,"surfaced":0})
    for s,l,p in zip(scores,labels,prs):
        if s>=thr:
            byp[p]["surfaced"]+=1
            if l==0: byp[p]["fp"]+=1
    npr=len(set(prs))
    tot_fp=sum(v["fp"] for v in byp.values()); tot_surf=sum(v["surfaced"] for v in byp.values())
    return round(tot_fp/max(1,npr),3), round(tot_surf/max(1,npr),3)
def fp_at_recall(scores,labels,tr=0.7):
    pos=sum(labels);
    if not pos: return None
    order=sorted(range(len(scores)),key=lambda i:-scores[i]);tp=0;fp=0
    for i in order:
        if labels[i]==1:tp+=1
        else:fp+=1
        if tp/pos>=tr: break
    return round(fp/pos if pos else 0,3)

def main():
    items=load_aacr()
    print(f"AACR E1: {len(items)} findings, {sum(x['label'] for x in items)} real, {len(set(x['pr'] for x in items))} PRs",flush=True)
    out={}
    for bname,bfn in BASELINES.items():
        scores=[None]*len(items); lock=Lock()
        def work(idx):
            x=items[idx]; s=bfn(x["critique"],x["context"],x["category"])
            with lock: scores[idx]=s
        with ThreadPoolExecutor(max_workers=8) as ex: list(ex.map(work,range(len(items))))
        valid=[(s,items[i]["label"],items[i]["pr"]) for i,s in enumerate(scores) if s is not None]
        sc=[v[0] for v in valid];lb=[v[1] for v in valid];pr=[v[2] for v in valid]
        try:
            from sklearn.metrics import roc_auc_score; au=round(roc_auc_score(lb,sc),4) if len(set(lb))>1 else None
        except: au=None
        thr=float(np.median(sc))
        m=metrics(sc,lb,thr); fppr,cpr=fp_per_pr(sc,lb,pr,thr); fpr70=fp_at_recall(sc,lb,0.7)
        out[bname]={"n":len(valid),"auroc":au,"P":m["P"],"R":m["R"],"F1":m["F1"],
                    "FP_per_PR":fppr,"Comments_per_PR":cpr,"FP_per_PR_at_70recall":fpr70}
        print(f"  [{bname}] AUROC {au} P {m['P']} R {m['R']} F1 {m['F1']} FP/PR {fppr} Comments/PR {cpr} FP/PR@70R {fpr70}",flush=True)
        json.dump(out,open(os.environ.get('CIVIC_PROJECT_ROOT', '.') + '/outputs/e1_baselines_result.json','w'),indent=2)
    print("E1_BASELINES DONE",flush=True)
if __name__=="__main__": main()
