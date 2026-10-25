from civic.services import review_score_path
from civic.paths import OUTPUT_DIR
pass

import os
import json, numpy as np, os
from collections import defaultdict
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
os.environ["HF_HOME"]=os.environ.get("CIVIC_PROJECT_ROOT", ".") + "/cache/hf"
PRE=os.environ.get("CIVIC_PROJECT_ROOT", ".") + "/data/anchors"

def reviewer_gen_tids():
    d=defaultdict(dict)
    for l in open(review_score_path(PRE)):
        r=json.loads(l)
        if r.get("score") is None: continue
        d[r["triple_id"]][r["version"]]=1
    return set(d.keys())

def main():
    blob=json.load(open(f"{PRE}/render_A.json"))
    cache={tuple(k.split("\t")):v for k,v in blob["cache"].items()}
    rg=reviewer_gen_tids()
    tids=[t for t in rg if (t,"x") in cache and (t,"target") in cache and (t,"unrel0") in cache]
    print(f"{len(tids)} triples with x/target/unrel0 code",flush=True)

    import torch
    from transformers import AutoTokenizer, AutoModel
    name="microsoft/codebert-base"
    tok=AutoTokenizer.from_pretrained(name); mdl=AutoModel.from_pretrained(name)
    dev="cuda" if torch.cuda.is_available() else "cpu"; mdl=mdl.to(dev).eval()
    print(f"device={dev}",flush=True)
    def embed(texts):
        out=[]
        for i in range(0,len(texts),16):
            enc=tok(texts[i:i+16],truncation=True,max_length=512,padding=True,return_tensors="pt").to(dev)
            with torch.no_grad(): h=mdl(**enc).last_hidden_state[:,0,:]
            out.append(h.cpu().numpy())
        return np.vstack(out)
    hx=embed([cache[(t,"x")] for t in tids])
    hic=embed([cache[(t,"target")] for t in tids])
    hicp=embed([cache[(t,"unrel0")] for t in tids])
    n=len(tids)

    y=np.r_[np.ones(n),np.zeros(n)]
    def cvauroc(X): return roc_auc_score(y,cross_val_predict(LogisticRegression(max_iter=2000),X,y,cv=5,method="predict_proba")[:,1])
    A_static=cvauroc(np.r_[hx,hic])
    bug=np.c_[hx, hx-hic, hx-hicp]; fix=np.c_[hic, np.zeros_like(hic), hic-hicp]
    A_diff=cvauroc(np.r_[bug,fix])
    print(f"[A bug-vs-fixed] static-CLS AUROC={A_static:.4f} | diff-struct AUROC={A_diff:.4f}",flush=True)

    dt=hx-hic; du=hx-hicp
    yb=np.r_[np.ones(n),np.zeros(n)]
    B_vec=cvauroc_x=roc_auc_score(yb,cross_val_predict(LogisticRegression(max_iter=2000),np.r_[dt,du],yb,cv=5,method="predict_proba")[:,1])

    nt=np.linalg.norm(dt,axis=1); nu=np.linalg.norm(du,axis=1)
    B_norm=roc_auc_score(yb,np.r_[nt,nu])
    print(f"[B specificity] drop-vector AUROC={B_vec:.4f} | drop-norm AUROC={B_norm:.4f} meanNorm_t={nt.mean():.3f} meanNorm_u={nu.mean():.3f}",flush=True)

    out={"n":n,
         "taskA_bug_vs_fixed":{"static_cls":round(float(A_static),4),"diff_struct":round(float(A_diff),4)},
         "taskB_specificity":{"drop_vector":round(float(B_vec),4),"drop_norm":round(float(B_norm),4),
                              "mean_norm_target":round(float(nt.mean()),4),"mean_norm_unrel":round(float(nu.mean()),4)},
         "encoder":"microsoft/codebert-base CLS","distribution":"reviewer-gen (ecologically valid)"}
    json.dump(out,open(os.environ.get("CIVIC_PROJECT_ROOT", ".") + "/outputs/diag_codediff.json","w"),indent=2)
    print("WROTE diag_codediff.json\n"+json.dumps(out,indent=2),flush=True)

if __name__=="__main__":
    main()
