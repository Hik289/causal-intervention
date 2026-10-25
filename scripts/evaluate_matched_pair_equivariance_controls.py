from civic.services import text_encoder
from civic.paths import OUTPUT_DIR
pass

import os
import torch; torch.set_num_threads(8)
import numpy as np, json, re, sys
import pandas as pd, glob
from civic.models import build
from civic.ranking import InterventionRanker
from sklearn.metrics import roc_auc_score, average_precision_score
from sentence_transformers import SentenceTransformer
import torch.nn as nn
DATA=os.environ.get("CIVIC_PROJECT_ROOT", ".") + "/data"; DS=f"{DATA}/intervention_dataset"; CACHE=os.environ.get("CIVIC_PROJECT_ROOT", ".") + "/cache"
a=np.load(f'{CACHE}/embdiff_arrays.npz'); idx=[json.loads(l) for l in open(f'{CACHE}/embdiff_index.jsonl')]
cx,cic,cicp=a['cx'],a['cic'],a['cicp']
def cv(u,v):return (u*v).sum(1)/(np.linalg.norm(u,axis=1)*np.linalg.norm(v,axis=1)+1e-9)
clean=[i for i in range(len(idx)) if cv(cx,cicp)[i]>0.95 and cv(cx,cic)[i]<0.999]
def sp(s):return [i for i in clean if idx[i]['split']==s]
tr,te=sp('train'),sp('test'); T=lambda arr,ii: torch.tensor(arr[ii],dtype=torch.float32)

RENAME={r'\bi\b':'idx',r'\bx\b':'val',r'\btmp\b':'temp',r'\bn\b':'count',r'\bs\b':'text'}
def txr(c):
    o=c
    for p,r in RENAME.items(): o=re.sub(p,r,o)
    return "# refactored\n"+o
def srcf(p): return [f for f in re.findall(r'diff --git a/(\S+) b/',p or '') if 'test' not in f.lower()]
def chsides(patch,fp):
    b=[];af=[];inf=False
    for ln in (patch or "").split("\n"):
        mm=re.match(r'diff --git a/(\S+) b/(\S+)',ln)
        if mm: inf=(mm.group(2)==fp or mm.group(1)==fp);continue
        if not inf: continue
        if ln.startswith("@@"):b.append("# h");af.append("# h");continue
        if ln.startswith("+") and not ln.startswith("+++"):af.append(ln[1:])
        elif ln.startswith("-") and not ln.startswith("---"):b.append(ln[1:])
        elif ln.startswith(" "):b.append(ln[1:]);af.append(ln[1:])
    return "\n".join(b),"\n".join(af)
print("build tx emb (E4)...",flush=True)
quads=[json.loads(l) for l in open(f"{DS}/quadruples_final.jsonl")];qmap={q['quad_id']:q for q in quads}
df=pd.concat([pd.read_parquet(f) for f in glob.glob(f"{DATA}/swe-review-bench_raw/data/*.parquet")],ignore_index=True)
res=df[df.resolved==True].groupby('instance_id').first();pby={i:r.model_patch for i,r in res.iterrows()}
m=SentenceTransformer(text_encoder());emb=lambda t:m.encode([str(t) if t else " "],convert_to_numpy=True)[0]
txcx=np.zeros_like(cx);txcic=np.zeros_like(cic);ht=np.zeros(len(idx),dtype=bool)
for i in te:
    q=qmap.get(idx[i]['quad_id'])
    if not q: continue
    iid=q['issue_id']
    if iid not in pby: continue
    src=srcf(pby[iid])
    if not src: continue
    b,af=chsides(pby[iid],src[0])
    if not(b.strip() or af.strip()):continue
    txcx[i]=emb(txr(b));txcic[i]=emb(txr(af));ht[i]=True
te_tx=[i for i in te if ht[i]]
class MLP(nn.Module):
    def __init__(s):
        super().__init__();s.net=nn.Sequential(nn.Linear(768,256),nn.ReLU(),nn.Linear(256,1))
    def forward(s,code,crit):
        return s.net(torch.cat([code,crit],-1)).squeeze(-1), None
def train(seed,variant):
    torch.manual_seed(seed);np.random.seed(seed)
    if variant=="mlp": mod=MLP()
    else: mod=build('crossattn',384,384)
    rk=InterventionRanker(dz=512,ddelta=384)
    params=list(mod.parameters())+(list(rk.parameters()) if variant=="full" else [])
    opt=torch.optim.AdamW(params,lr=2e-3);I=np.array(tr)
    for ep in range(150):
        mod.train();opt.zero_grad()
        sx,zx=mod(T(cx,I),T(a['crit'],I));sic,_=mod(T(cic,I),T(a['crit'],I));sicp,_=mod(T(cicp,I),T(a['crit'],I))
        if variant=="mlp":

            lab=torch.cat([torch.ones(len(I)),torch.zeros(len(I))]);pred=torch.cat([sx,sic])
            L=nn.functional.binary_cross_entropy_with_logits(pred,lab)
        elif variant=="invariance": L=torch.abs(sicp-sx).mean()+0.01*(T(cic,I)-T(cx,I)).norm(dim=-1).mean()
        elif variant=="sensitivity": L=torch.clamp(0.5+sic-sx,min=0).mean()
        else:
            L=torch.clamp(0.5+sic-sx,min=0).mean()+0.1*torch.abs(sicp-sx).mean()
            hdp=T(cic,I)-T(cx,I);hdn=T(cicp,I)-T(cx,I)
            L=L+0.5*(-torch.log(torch.exp(rk(zx,hdp))/(torch.exp(rk(zx,hdp))+torch.exp(rk(zx,hdn))+1e-9)).mean())+0.01*hdp.norm(dim=-1).mean()
        L.backward();opt.step()
    mod.eval();return mod
def e2eval(mod):
    I=np.array(te)
    with torch.no_grad():sx,_=mod(T(cx,I),T(a['crit'],I));sic,_=mod(T(cic,I),T(a['crit'],I))
    sx,sic=sx.numpy(),sic.numpy();thr=np.median(np.concatenate([sx,sic]))
    pr=float((sx>thr).mean());tj=float((sic<=thr).mean());pa=float((sx>sic).mean())
    sc=np.concatenate([sx,sic]);lb=np.concatenate([np.ones(len(sx)),np.zeros(len(sx))])
    return pr,tj,pa,roc_auc_score(lb,sc),average_precision_score(lb,sc),float((sx-sic).mean())
def e4eval(mod):
    I=np.array(te_tx)
    with torch.no_grad():
        sx,_=mod(T(cx,I),T(a['crit'],I));sic,_=mod(T(cic,I),T(a['crit'],I))
        sxt,_=mod(T(txcx,I),T(a['crit'],I));sict,_=mod(T(txcic,I),T(a['crit'],I))
    dt=(sx-sic).numpy();dtt=(sxt-sict).numpy()
    vc=float(((dt>0)==(dtt>0)).mean());corr=float(np.corrcoef(dt,dtt)[0,1]) if len(dt)>1 else 0
    fo=float((sic.numpy()<sx.numpy()).mean());ft=float((sict.numpy()<sxt.numpy()).mean())
    return vc,corr,abs(fo-ft)
out={"E2":{},"E4":{}}
for v,nm in [("mlp","MLP Classifier"),("invariance","Invariance-Only"),("sensitivity","Sensitivity-Only"),("full","CIVIC-CR full")]:
    e2=[];e4=[]
    for s in [42,123,456]:
        mod=train(s,v);e2.append(e2eval(mod));e4.append(e4eval(mod))
    e2=np.array(e2);e4=np.array(e4)
    out["E2"][nm]={"posrec":round(e2[:,0].mean(),4),"trap_rej":round(e2[:,1].mean(),4),"pairacc":round(e2[:,2].mean(),4),"auroc":round(e2[:,3].mean(),4),"auprc":round(e2[:,4].mean(),4),"margin":round(e2[:,5].mean(),4)}
    out["E4"][nm]={"verdict_consistency":round(e4[:,0].mean(),4),"evidence_eqv":round(e4[:,1].mean(),4),"fp_change":round(e4[:,2].mean(),4)}
    print(f"  [{nm}] E2 PairAcc {out['E2'][nm]['pairacc']} AUROC {out['E2'][nm]['auroc']} | E4 VC {out['E4'][nm]['verdict_consistency']} corr {out['E4'][nm]['evidence_eqv']}",flush=True)
json.dump(out,open(os.environ.get('CIVIC_PROJECT_ROOT', '.') + '/outputs/matched_pair_equivariance_control_results.json','w'),indent=2)
print("E2E4_CONTROLS DONE",flush=True)
