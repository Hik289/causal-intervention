pass

import os
import json, os, re, argparse
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); DATA=os.environ.get("CIVIC_PROJECT_ROOT", ".") + "/data"
def gpt(t=120):
    import openai; return openai.OpenAI(base_url=os.environ["OPENAI_BASE_URL"],api_key=os.environ["OPENAI_API_KEY"],max_retries=6,timeout=t)
def gpt_chat(system,user,t=120):
    r=gpt(t).chat.completions.create(model=os.environ.get("LLM_MODEL_NAME", "gpt-4"),messages=[{"role":"system","content":system},{"role":"user","content":user}])
    return r.choices[0].message.content or ""

REVIEW_SYS=("You are an expert code reviewer judging whether a specific review finding describes a REAL defect/issue that "
 "actually exists in the code, versus a false-positive (incorrect, non-existent, or unjustified) finding. Consider the "
 "finding's claim and the code context. Output 'VALID: <int 0-100>' = your confidence the finding is a genuine real issue "
 "(100=definitely real, 0=definitely false-positive/wrong). Output ONLY 'VALID: <int>'.")
def score_finding(critique, context, category=""):
    try:
        t=gpt_chat(REVIEW_SYS, f"FINDING CATEGORY: {category}\nFINDING (review comment):\n{critique[:1500]}\n\nCODE CONTEXT:\n{str(context)[:3000]}\n\nIs this a genuine real defect?\nVALID:", t=120)
        m=re.search(r'VALID:\s*(\d+)',t,re.I)
        return max(0,min(100,int(m.group(1))))/100.0 if m else None
    except Exception: return None

def load_aacr(n):
    import random; random.seed(42)
    d=json.load(open(f"{DATA}/aacr-bench_raw/dataset.json"))
    items=[{"id":i,"critique":r.get("note",""),"context":r.get("context",""),"category":r.get("category",""),
            "label":int(r.get("label"))} for i,r in enumerate(d) if r.get("note") and r.get("label") is not None]
    if not n: return items

    pos=[x for x in items if x["label"]==1]; neg=[x for x in items if x["label"]==0]
    random.shuffle(pos); random.shuffle(neg)
    k=n//2
    return pos[:k]+neg[:n-k]

def metrics(scores, labels, thr):
    pass
    pred=[1 if s>=thr else 0 for s in scores]
    tp=sum(1 for p,l in zip(pred,labels) if p==1 and l==1)
    fp=sum(1 for p,l in zip(pred,labels) if p==1 and l==0)
    fn=sum(1 for p,l in zip(pred,labels) if p==0 and l==1)
    tn=sum(1 for p,l in zip(pred,labels) if p==0 and l==0)
    P=tp/(tp+fp) if tp+fp else 0; R=tp/(tp+fn) if tp+fn else 0
    F1=2*P*R/(P+R) if P+R else 0
    FPR=fp/(fp+tn) if fp+tn else 0
    return {"P":round(P,3),"R":round(R,3),"F1":round(F1,3),"FPR":round(FPR,3),"tp":tp,"fp":fp,"fn":fn,"tn":tn}

def fp_at_recall(scores, labels, target_recall=0.8):
    pass
    pos=sum(labels)
    if pos==0: return None
    order=sorted(range(len(scores)),key=lambda i:-scores[i])
    tp=0; fp=0
    for rank,i in enumerate(order,1):
        if labels[i]==1: tp+=1
        else: fp+=1
        if tp/pos>=target_recall:
            return {"recall":round(tp/pos,3),"fp_at_recall":fp,"fp_rate":round(fp/(len(scores)),3),"thr_rank":rank}
    return {"recall":round(tp/pos,3),"fp_at_recall":fp,"fp_rate":round(fp/len(scores),3)}

def load_crb(n):
    pass

    import pandas as pd, random; random.seed(42)
    df=pd.read_parquet(f"{DATA}/codereviewbench_raw/human_annotations_task2.parquet")
    items=[]
    for i,r in df.iterrows():
        fa=str(r.get("human_factual_accuracy",""))
        if fa not in ("Yes","No"): continue
        bc=r.get("bot_comment","")
        crit=bc.get("description",str(bc)) if isinstance(bc,dict) else str(bc)
        items.append({"id":f"crb_{i}","critique":crit,"context":str(r.get("diff_lines",""))[:2000],
                      "category":"","label":1 if fa=="Yes" else 0})
    random.shuffle(items)
    return items[:n] if n else items

def load_swepr(n):
    pass

    import pandas as pd, random, os, glob; random.seed(42)
    items=[]
    for f in glob.glob(f"{DATA}/swe-prbench_raw/dataset/annotations/*_human.json"):
        try: ann=json.load(open(f))
        except: continue
        pr=os.path.basename(f).replace("_human.json","")
        for j,c in enumerate(ann.get("comments",[]) if isinstance(ann,dict) else []):
            if not isinstance(c,dict): continue
            body=c.get("body",""); rc=str(c.get("requires_change",""))
            if not body or rc not in ("True","False"): continue
            items.append({"id":f"swepr_{pr}_{j}","critique":str(body),"context":str(c.get("diff_hunk",""))[:2000],
                          "category":str(c.get("severity","")),"label":1 if rc=="True" else 0})
    random.shuffle(items)
    return items[:n] if n else items

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--bench",default="aacr"); ap.add_argument("--n",type=int,default=20)
    ap.add_argument("--workers",type=int,default=6); args=ap.parse_args()
    if args.bench=="aacr": items=load_aacr(args.n)
    elif args.bench=="crb": items=load_crb(args.n)
    elif args.bench=="swepr": items=load_swepr(args.n)
    else: print(f"bench {args.bench} loader TBD"); return
    print(f"{args.bench}: {len(items)} findings (label 1=real: {sum(x['label'] for x in items)})",flush=True)
    lk=Lock(); done=[0]
    def work(it):
        s=score_finding(it["critique"],it["context"],it["category"])
        it["score"]=s
        with lk:
            done[0]+=1
            if done[0]%10==0: print(f"  scored {done[0]}/{len(items)}",flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as ex: list(ex.map(work,items))
    scored=[it for it in items if it["score"] is not None]
    scores=[it["score"] for it in scored]; labels=[it["label"] for it in scored]
    print(f"scored {len(scored)}/{len(items)} (judge_err {len(items)-len(scored)})",flush=True)

    try:
        from sklearn.metrics import roc_auc_score
        auroc=roc_auc_score(labels,scores) if len(set(labels))>1 else None
    except Exception: auroc=None
    print(f"Frozen Reviewer AUROC (finding validity vs expert label): {auroc}",flush=True)
    for thr in [0.5,0.6,0.7]:
        print(f"  @thr{thr}: {metrics(scores,labels,thr)}",flush=True)
    print(f"  FP@recall0.8: {fp_at_recall(scores,labels,0.8)}",flush=True)
    out={"bench":args.bench,"n":len(scored),"n_real":sum(labels),"auroc":auroc,
         "metrics_thr":{str(t):metrics(scores,labels,t) for t in [0.5,0.6,0.7]},
         "fp_at_recall_0.8":fp_at_recall(scores,labels,0.8),
         "scored_items":[{"id":it["id"],"label":it["label"],"score":it["score"]} for it in scored]}
    json.dump(out,open(f"{HERE}/frozen_{args.bench}_n{args.n}.json","w"),indent=2)
    print("FROZEN_BASELINE DONE",flush=True)

if __name__=="__main__": main()
