from civic.paths import OUTPUT_DIR
pass

import os
import json, os, re, random, time, numpy as np
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from sklearn.metrics import roc_auc_score
os.environ["HF_HOME"]=os.environ.get("CIVIC_PROJECT_ROOT", ".") + "/cache/hf"
PRE=os.environ.get("CIVIC_PROJECT_ROOT", ".") + "/data/anchors"
random.seed(42)

RUBRIC=("You are an expert code reviewer. Below is a code snippet and a critique claiming a specific defect exists. "
 "Judge ONLY whether the specific defect described in the critique is actually present in THIS code as shown. "
 "Output a single integer 0-100 = confidence the claimed defect is genuinely present, then one short reason. "
 "Format exactly: SCORE: <int>\\nREASON: <text>")
def score_llm(code, crit, timeout=90):
    import anthropic
    c=anthropic.Anthropic(base_url=os.environ["ANTHROPIC_BASE_URL"],api_key=os.environ["ANTHROPIC_API_KEY"],max_retries=6,timeout=timeout)
    r=c.messages.create(model="claude-sonnet-4-6",max_tokens=200,temperature=0.0 if False else 0.4,
        system=RUBRIC,messages=[{"role":"user","content":f"CRITIQUE:\n{crit}\n\nCODE:\n{code[:6000]}\n\nIs the specific defect present in THIS code?"}])
    return r.content[0].text
def parse(txt):
    m=re.search(r'SCORE:\s*(\d+)',txt or "",re.I)
    if m: return max(0,min(100,int(m.group(1))))/100.0
    c=[int(x) for x in re.findall(r'\b(\d{1,3})\b',txt or "") if 0<=int(x)<=100]
    return c[-1]/100.0 if c else None

def main():

    blob=json.load(open(f"{PRE}/render_A.json")); cache={tuple(k.split("\t")):v for k,v in blob["cache"].items()}
    tri=[json.loads(l) for l in open(f"{PRE}/triples_reviewergen.jsonl") if json.loads(l).get("critique")]

    sc=defaultdict(lambda: defaultdict(list))
    for l in open(f"{PRE}/scores_reviewergen_claude.jsonl"):
        r=json.loads(l)
        if r.get("score") is not None: sc[r["triple_id"]][r["version"]].append(r["score"])
    s_true={t:float(np.mean(v["x"])) for t,v in sc.items() if "x" in v}

    items=[]
    for t in tri:
        tid=t["instance_id"]
        if (tid,"x") in cache and tid in s_true:
            items.append({"tid":tid,"repo":t["repo"],"crit":t["critique"],"code":cache[(tid,"x")],"s_true":s_true[tid]})
    print(f"{len(items)} true-critique items",flush=True)

    byrepo=defaultdict(list)
    for it in items: byrepo[it["repo"]].append(it)
    fp_jobs=[]
    for it in items:
        pool=[o for o in byrepo[it["repo"]] if o["tid"]!=it["tid"]]
        if not pool: continue
        other=random.choice(pool)
        fp_jobs.append((it["tid"], it["code"], other["crit"]))
    print(f"{len(fp_jobs)} FP (mismatched critique) to score",flush=True)

    outp=os.environ.get("CIVIC_PROJECT_ROOT", ".") + "/outputs/fp_scores.jsonl"
    done={}
    if os.path.exists(outp):
        for l in open(outp):
            r=json.loads(l)
            if r.get("score") is not None: done[r["tid"]]=r["score"]
    todo=[j for j in fp_jobs if j[0] not in done]
    print(f"{len(todo)} FP to score ({len(done)} done)",flush=True)
    fo=open(outp,"a"); lk=Lock(); cnt=[0]
    def work(j):
        tid,code,crit=j
        try: s=parse(score_llm(code,crit))
        except Exception as e: s=None
        with lk:
            fo.write(json.dumps({"tid":tid,"score":s})+"\n"); fo.flush(); cnt[0]+=1
            if cnt[0]%20==0: print(f"  {cnt[0]}/{len(todo)}",flush=True)
        return (tid,s)
    with ThreadPoolExecutor(max_workers=8) as ex:
        for tid,s in ex.map(work,todo):
            if s is not None: done[tid]=s
    fo.close()

    pos=[it["s_true"] for it in items if it["tid"] in done]
    neg=[done[it["tid"]] for it in items if it["tid"] in done]
    y=np.r_[np.ones(len(pos)),np.zeros(len(neg))]; s=np.r_[pos,neg]
    au=roc_auc_score(y,s)
    pairwise=float(np.mean(np.array(pos)>np.array(neg)))
    out={"task":"FP-detection (true critique vs mismatched false critique, reviewer-gen dist)",
         "n_pairs":len(pos),"scorer":"claude-sonnet-4-6",
         "LLM_scalar_AUROC":round(float(au),4),"pairwise_true>false":round(pairwise,4),
         "mean_s_true":round(float(np.mean(pos)),4),"mean_s_false":round(float(np.mean(neg)),4)}
    json.dump(out,open(os.environ.get("CIVIC_PROJECT_ROOT", ".") + "/outputs/diag_fp.json","w"),indent=2)
    print("WROTE diag_fp.json\n"+json.dumps(out,indent=2),flush=True)

if __name__=="__main__":
    main()
