from civic.paths import OUTPUT_DIR
pass

import os
import torch; torch.set_num_threads(8)
import numpy as np, json, sys
from civic.models import build
from civic.ranking import InterventionRanker
from sentence_transformers import SentenceTransformer
from sklearn.metrics import roc_auc_score
from collections import defaultdict
CACHE=os.environ.get("CIVIC_PROJECT_ROOT", ".") + "/cache"; DATA=os.environ.get("CIVIC_PROJECT_ROOT", ".") + "/data"
a=np.load(f'{CACHE}/embdiff_arrays.npz'); idx=[json.loads(l) for l in open(f'{CACHE}/embdiff_index.jsonl')]
cx,cic,cicp=a['cx'],a['cic'],a['cicp']
def cv(u,v):return (u*v).sum(1)/(np.linalg.norm(u,axis=1)*np.linalg.norm(v,axis=1)+1e-9)
clean=[i for i in range(len(idx)) if cv(cx,cicp)[i]>0.95 and cv(cx,cic)[i]<0.999]
tr=[i for i in clean if idx[i]['split']=='train']
T=lambda arr,ii: torch.tensor(arr[ii],dtype=torch.float32)

def train(seed):
    torch.manual_seed(seed);np.random.seed(seed)
    m=build('crossattn',384,384);rk=InterventionRanker(dz=512,ddelta=384)
    opt=torch.optim.AdamW(list(m.parameters())+list(rk.parameters()),lr=2e-3);I=np.array(tr)
    for ep in range(150):
        m.train();opt.zero_grad()
        sx,zx=m(T(cx,I),T(a['crit'],I));sic,_=m(T(cic,I),T(a['crit'],I));sicp,_=m(T(cicp,I),T(a['crit'],I))
        L=torch.clamp(0.5+sic-sx,min=0).mean()+0.1*torch.abs(sicp-sx).mean()
        hdp=T(cic,I)-T(cx,I);hdn=T(cicp,I)-T(cx,I)
        L=L+0.5*(-torch.log(torch.exp(rk(zx,hdp))/(torch.exp(rk(zx,hdp))+torch.exp(rk(zx,hdn))+1e-9)).mean())+0.01*hdp.norm(dim=-1).mean()
        L.backward();opt.step()
    m.eval();return m

print("embedding AACR findings...",flush=True)
d=json.load(open(f"{DATA}/aacr-bench_raw/dataset.json"))
import random; random.seed(42)
byl=defaultdict(list)
for r in d:
    if r.get("note") and r.get("label") is not None and r.get("project_main_language"):
        byl[r["project_main_language"]].append(r)
emb_m=SentenceTransformer("all-MiniLM-L6-v2"); E=lambda t:emb_m.encode([str(t) if t else " "],convert_to_numpy=True)[0]
samp={}
for lang,xs in byl.items():
    random.shuffle(xs); xs=xs[:40]
    code=np.array([E(x.get("context","")) for x in xs]); crit=np.array([E(x.get("note","")) for x in xs]); lab=[int(x["label"]) for x in xs]
    samp[lang]=(code,crit,lab)
print(f"embedded {sum(len(v[2]) for v in samp.values())} findings, {len(samp)} langs",flush=True)

models=[train(s) for s in [42,123,456]]
print("sθ trained 3 seed",flush=True)
out={}
allnp_s=[];allnp_l=[];allp_s=[];allp_l=[]
for lang,(code,crit,lab) in sorted(samp.items()):
    aus=[]
    for m in models:
        with torch.no_grad(): s=torch.sigmoid(m(torch.tensor(code,dtype=torch.float32),torch.tensor(crit,dtype=torch.float32))[0]).numpy()
        if len(set(lab))>1: aus.append(roc_auc_score(lab,s))
    civic_au=round(float(np.mean(aus)),4) if aus else None
    out[lang]={"n":len(lab),"civic_cr_auroc":civic_au,"held_out":lang!="Python"}
    print(f"  [{lang}] n={len(lab)} CIVIC-CR sθ AUROC={civic_au} {'(held-out)' if lang!='Python' else '(in-domain)'}",flush=True)

    with torch.no_grad(): savg=np.mean([torch.sigmoid(mm(torch.tensor(code,dtype=torch.float32),torch.tensor(crit,dtype=torch.float32))[0]).numpy() for mm in models],0)
    if lang!="Python": allnp_s+=list(savg);allnp_l+=lab
    else: allp_s+=list(savg);allp_l+=lab
if allnp_l: out["_heldout_nonPython_pooled"]={"n":len(allnp_l),"civic_cr_auroc":round(roc_auc_score(allnp_l,allnp_s),4)}
if allp_l: out["_inDomain_Python"]={"n":len(allp_l),"civic_cr_auroc":round(roc_auc_score(allp_l,allp_s),4)}
print(f"  POOLED non-Python held-out CIVIC-CR AUROC={out.get('_heldout_nonPython_pooled',{}).get('civic_cr_auroc')}",flush=True)
print(f"  Python in-domain CIVIC-CR AUROC={out.get('_inDomain_Python',{}).get('civic_cr_auroc')}",flush=True)
json.dump(out,open(os.environ.get('CIVIC_PROJECT_ROOT', '.') + '/outputs/verifier_cross_language_results.json','w'),indent=2)
print("E6_CIVIC_LANG DONE",flush=True)
