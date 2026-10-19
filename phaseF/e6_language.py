pass

import os
import json, os, re, numpy as np
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
DATA=os.environ.get("CIVIC_PROJECT_ROOT", ".") + "/data"
def gpt(t=120):
    import openai; return openai.OpenAI(base_url=os.environ["OPENAI_BASE_URL"],api_key=os.environ["OPENAI_API_KEY"],max_retries=6,timeout=t)
def chat(system,user,t=120):
    r=gpt(t).chat.completions.create(model=os.environ.get("LLM_MODEL_NAME", "gpt-4"),messages=[{"role":"system","content":system},{"role":"user","content":user}]);return r.choices[0].message.content or ""
REVIEW_SYS=("You are an expert code reviewer judging whether a review finding describes a REAL defect vs a false-positive. "
 "Output ONLY 'VALID: <int 0-100>' (100=real, 0=false-positive).")
def score(c,ctx):
    try:
        t=chat(REVIEW_SYS,f"FINDING:\n{c[:1500]}\nCODE CONTEXT:\n{str(ctx)[:3000]}\nVALID:")
        m=re.search(r'VALID:\s*(\d+)',t,re.I);return int(m.group(1))/100.0 if m else None
    except: return None
def main():
    d=json.load(open(f"{DATA}/aacr-bench_raw/dataset.json"))
    import random; random.seed(42)
    items=[{"c":r.get("note",""),"ctx":r.get("context",""),"lang":r.get("project_main_language"),"label":int(r.get("label"))}
           for r in d if r.get("note") and r.get("label") is not None and r.get("project_main_language")]

    from collections import defaultdict
    byl=defaultdict(list)
    for x in items: byl[x["lang"]].append(x)
    sample=[]
    for lang,xs in byl.items():
        random.shuffle(xs); sample+=xs[:40]
    print(f"E6-lang: {len(sample)} findings across {len(byl)} langs",flush=True)
    scores=[None]*len(sample); lock=Lock()
    def work(i):
        s=score(sample[i]["c"],sample[i]["ctx"])
        with lock: scores[i]=s
    with ThreadPoolExecutor(max_workers=8) as ex: list(ex.map(work,range(len(sample))))
    from sklearn.metrics import roc_auc_score
    out={}
    byl2=defaultdict(lambda:{"s":[],"l":[]})
    for i,x in enumerate(sample):
        if scores[i] is not None: byl2[x["lang"]]["s"].append(scores[i]); byl2[x["lang"]]["l"].append(x["label"])
    for lang,v in sorted(byl2.items()):
        au=round(roc_auc_score(v["l"],v["s"]),4) if len(set(v["l"]))>1 else None
        out[lang]={"n":len(v["l"]),"n_real":int(sum(v["l"])),"frozen_reviewer_auroc":au,"held_out":lang!="Python"}
        print(f"  [{lang}] n={len(v['l'])} FrozenReviewer AUROC={au} {'(held-out lang)' if lang!='Python' else '(in-domain)'}",flush=True)

    nps=[(s,l) for i,x in enumerate(sample) if x["lang"]!="Python" and scores[i] is not None for s,l in [(scores[i],x["label"])]]
    ps=[(scores[i],x["label"]) for i,x in enumerate(sample) if x["lang"]=="Python" and scores[i] is not None]
    if nps:
        au_np=round(roc_auc_score([l for _,l in nps],[s for s,_ in nps]),4)
        out["_heldout_nonPython_pooled"]={"n":len(nps),"frozen_reviewer_auroc":au_np}
        print(f"  [POOLED non-Python held-out] n={len(nps)} AUROC={au_np}",flush=True)
    if ps:
        au_p=round(roc_auc_score([l for _,l in ps],[s for s,_ in ps]),4)
        out["_inDomain_Python"]={"n":len(ps),"frozen_reviewer_auroc":au_p}
        print(f"  [Python in-domain] n={len(ps)} AUROC={au_p}",flush=True)
    json.dump(out,open(os.environ.get('CIVIC_PROJECT_ROOT', '.') + '/phaseF/e6_language_result.json','w'),indent=2)
    print("E6_LANG DONE",flush=True)
if __name__=="__main__": main()
