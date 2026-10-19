pass

import os
import json, os, re, numpy as np
from collections import defaultdict
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

PRE=os.environ.get("CIVIC_PROJECT_ROOT", ".") + "/experiments/preexp_anchor0"
os.environ["HF_HOME"]=os.environ.get("CIVIC_PROJECT_ROOT", ".") + "/cache/hf"

def load_scores():
    rows=[json.loads(l) for l in open(f"{PRE}/scores_reviewergen_claude.jsonl") if json.loads(l).get("score") is not None]
    d=defaultdict(lambda: defaultdict(list))
    raw=defaultdict(dict)
    for r in rows:
        d[r["triple_id"]][r["version"]].append(r["score"])
        if "raw" in r: raw[r["triple_id"]][r["version"]]=r["raw"]
    agg={t:{v:float(np.mean(s)) for v,s in vv.items()} for t,vv in d.items()}
    return agg, raw

def parse_reason(txt):
    m=re.search(r'REASON:\s*(.+)',txt or "",re.S)
    return m.group(1).strip()[:500] if m else (txt or "")[:500]

def main():
    agg, raw = load_scores()
    tids=[t for t in agg if "x" in agg[t] and "target" in agg[t]]
    print(f"{len(tids)} reviewer-gen E2 pairs (x^bug vs x^safe=I_c)",flush=True)

    s_bug=np.array([agg[t]["x"] for t in tids])
    s_safe=np.array([agg[t]["target"] for t in tids])
    y=np.r_[np.ones(len(tids)),np.zeros(len(tids))]; s=np.r_[s_bug,s_safe]
    au2=roc_auc_score(y,s)

    pa2=float(np.mean((s_bug>s_safe)))
    print(f"[#2 LLM scalar] AUROC={au2:.4f} PairwiseAcc(s_bug>s_safe)={pa2:.4f} mean_margin={float((s_bug-s_safe).mean()):.4f}",flush=True)

    from sentence_transformers import SentenceTransformer
    mt=SentenceTransformer("all-MiniLM-L6-v2")
    rb=[parse_reason(raw[t].get("x","")) for t in tids]
    rs=[parse_reason(raw[t].get("target","")) for t in tids]
    eb=mt.encode(rb,convert_to_numpy=True); es=mt.encode(rs,convert_to_numpy=True)
    X3=np.vstack([eb,es]); y3=np.r_[np.ones(len(tids)),np.zeros(len(tids))]

    try:
        p3=cross_val_predict(LogisticRegression(max_iter=1000),X3,y3,cv=5,method="predict_proba")[:,1]
        au3=roc_auc_score(y3,p3)
    except Exception as e:
        au3=float('nan'); print("reason-emb cv err",e)

    X3b=np.hstack([X3, np.r_[s_bug,s_safe].reshape(-1,1)])
    try:
        p3b=cross_val_predict(LogisticRegression(max_iter=1000),X3b,y3,cv=5,method="predict_proba")[:,1]
        au3b=roc_auc_score(y3,p3b)
    except: au3b=float('nan')
    print(f"[#3 LLM reason-emb] 5fold-LR AUROC={au3:.4f} | reason+scalar AUROC={au3b:.4f}",flush=True)

    au1=float('nan'); pa1=None
    try:
        blob=json.load(open(f"{PRE}/render_A.json"))
        cache={tuple(k.split("\t")):v for k,v in blob["cache"].items()}
        have=[t for t in tids if (t,"x") in cache and (t,"target") in cache]
        if have:
            import torch
            from transformers import AutoTokenizer, AutoModel
            name="microsoft/codebert-base"
            tok=AutoTokenizer.from_pretrained(name); mdl=AutoModel.from_pretrained(name)
            dev="cuda" if torch.cuda.is_available() else "cpu"; mdl=mdl.to(dev).eval()
            def embed(texts):
                out=[]
                for i in range(0,len(texts),16):
                    b=texts[i:i+16]
                    enc=tok(b,truncation=True,max_length=512,padding=True,return_tensors="pt").to(dev)
                    with torch.no_grad(): h=mdl(**enc).last_hidden_state[:,0,:]
                    out.append(h.cpu().numpy())
                return np.vstack(out)
            cb=embed([cache[(t,"x")] for t in have]); cs=embed([cache[(t,"target")] for t in have])

            X1=np.vstack([cb,cs]); y1=np.r_[np.ones(len(have)),np.zeros(len(have))]
            p1=cross_val_predict(LogisticRegression(max_iter=1000),X1,y1,cv=5,method="predict_proba")[:,1]
            au1=roc_auc_score(y1,p1)

            def cos(u,v): return (u*v).sum(1)/(np.linalg.norm(u,axis=1)*np.linalg.norm(v,axis=1)+1e-9)
            print(f"[#1 CodeBERT CLS] 5fold-LR AUROC={au1:.4f} (n={len(have)}) cos(x,fix)={float(cos(cb,cs).mean()):.4f}",flush=True)
        else:
            print("[#1 CodeBERT] no cached code for reviewer-gen triples",flush=True)
    except Exception as e:
        print("[#1 CodeBERT] err",str(e)[:200],flush=True)

    out={"n_pairs":len(tids),
         "rep2_llm_scalar":{"AUROC":round(au2,4),"pairwise_acc":round(pa2,4)},
         "rep3_llm_reason_emb":{"AUROC_5foldLR":round(au3,4),"AUROC_reason+scalar":round(au3b,4)},
         "rep1_codebert":{"AUROC_5foldLR":round(au1,4) if au1==au1 else None},
         "note":"raw separability (no upper-layer training). E2 reviewer-gen x^bug vs x^safe. ISOLATE diagnostic."}
    json.dump(out,open(os.environ.get("CIVIC_PROJECT_ROOT", ".") + "/phaseC/diag_isolate.json","w"),indent=2)
    print("WROTE diag_isolate.json",flush=True)
    print(json.dumps(out,indent=2),flush=True)

if __name__=="__main__":
    main()
