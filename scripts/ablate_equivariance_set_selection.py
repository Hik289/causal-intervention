from civic.paths import OUTPUT_DIR
pass

import os
import torch; torch.set_num_threads(8)
import numpy as np, json, re, sys
import pandas as pd, glob
from civic.models import build
from civic.ranking import InterventionRanker
from sklearn.metrics import roc_auc_score
from sentence_transformers import SentenceTransformer
DATA=os.environ.get("CIVIC_PROJECT_ROOT", ".") + "/data"; DS=f"{DATA}/intervention_dataset"; CACHE=os.environ.get("CIVIC_PROJECT_ROOT", ".") + "/cache"
a=np.load(f'{CACHE}/embdiff_arrays.npz'); idx=[json.loads(l) for l in open(f'{CACHE}/embdiff_index.jsonl')]
cx,cic,cicp=a['cx'],a['cic'],a['cicp']
def cv(u,v):return (u*v).sum(1)/(np.linalg.norm(u,axis=1)*np.linalg.norm(v,axis=1)+1e-9)
clean=[i for i in range(len(idx)) if cv(cx,cicp)[i]>0.95 and cv(cx,cic)[i]<0.999]

def srcf(p): return [f for f in re.findall(r'diff --git a/(\S+) b/',p or '') if 'test' not in f.lower()]
def changed_sides(patch, fp):
    before=[];after=[];inf=False
    for ln in (patch or "").split("\n"):
        mm=re.match(r'diff --git a/(\S+) b/(\S+)',ln)
        if mm: inf=(mm.group(2)==fp or mm.group(1)==fp); continue
        if not inf: continue
        if ln.startswith("@@"): before.append("# hunk");after.append("# hunk");continue
        if ln.startswith("+") and not ln.startswith("+++"): after.append(ln[1:])
        elif ln.startswith("-") and not ln.startswith("---"): before.append(ln[1:])
        elif ln.startswith(" "): before.append(ln[1:]);after.append(ln[1:])
    return "\n".join(before),"\n".join(after)
RENAME={r'\bi\b':'idx',r'\bx\b':'val',r'\btmp\b':'temp',r'\bn\b':'count',r'\bs\b':'text',r'\bres\b':'result'}
def txr(code):
    o=code
    for p,r in RENAME.items(): o=re.sub(p,r,o)
    return o

print("building transform embeddings...",flush=True)
quads=[json.loads(l) for l in open(f"{DS}/quadruples_final.jsonl")]; qmap={q['quad_id']:q for q in quads}
df=pd.concat([pd.read_parquet(f) for f in glob.glob(f"{DATA}/swe-review-bench_raw/data/*.parquet")],ignore_index=True)
res=df[df.resolved==True].groupby('instance_id').first();pby={i:r.model_patch for i,r in res.iterrows()}
m=SentenceTransformer("all-MiniLM-L6-v2"); emb=lambda t:m.encode([str(t) if t else " "],convert_to_numpy=True)[0]
txcx=np.zeros_like(cx); txcic=np.zeros_like(cic); havetx=np.zeros(len(idx),dtype=bool)
for i in clean:
    q=qmap.get(idx[i]['quad_id']);
    if not q: continue
    iid=q['issue_id']
    if iid not in pby: continue
    tgt=pby[iid];src=srcf(tgt)
    if not src: continue
    b,af=changed_sides(tgt,src[0])
    if not (b.strip() or af.strip()): continue
    txcx[i]=emb(txr(b)); txcic[i]=emb(txr(af)); havetx[i]=True
cl2=[i for i in clean if havetx[i]]
print(f"transform emb built for {len(cl2)}/{len(clean)} clean quads",flush=True)
def sp(s):return [i for i in cl2 if idx[i]['split']==s]
tr,te=sp('train'),sp('test'); T=lambda arr,ii: torch.tensor(arr[ii],dtype=torch.float32)
def run(seed, use_eqv, use_set):
    torch.manual_seed(seed);np.random.seed(seed)
    mod=build('crossattn',384,384);rk=InterventionRanker(dz=512,ddelta=384)
    opt=torch.optim.AdamW(list(mod.parameters())+list(rk.parameters()),lr=2e-3);I=np.array(tr)
    for ep in range(150):
        mod.train();opt.zero_grad()
        sx,zx=mod(T(cx,I),T(a['crit'],I));sic,_=mod(T(cic,I),T(a['crit'],I));sicp,_=mod(T(cicp,I),T(a['crit'],I))
        L=torch.clamp(0.5+sic-sx,min=0).mean()+0.1*torch.abs(sicp-sx).mean()
        hdp=T(cic,I)-T(cx,I);hdn=T(cicp,I)-T(cx,I)
        L=L+0.5*(-torch.log(torch.exp(rk(zx,hdp))/(torch.exp(rk(zx,hdp))+torch.exp(rk(zx,hdn))+1e-9)).mean())+0.01*hdp.norm(dim=-1).mean()
        if use_eqv:
            sxt,_=mod(T(txcx,I),T(a['crit'],I));sict,_=mod(T(txcic,I),T(a['crit'],I))
            L=L+0.1*(torch.abs(sx-sxt).mean()+torch.abs(sic-sict).mean())
        if use_set:

            target_std=0.5; L=L+0.02*torch.abs(sic.std()-target_std)
        L.backward();opt.step()
    mod.eval();I=np.array(te)
    with torch.no_grad():
        sx,_=mod(T(cx,I),T(a['crit'],I));sic,_=mod(T(cic,I),T(a['crit'],I));sicp,_=mod(T(cicp,I),T(a['crit'],I))
    sx,sic,sicp=sx.numpy(),sic.numpy(),sicp.numpy()
    sc=np.concatenate([sx-sic,sx-sicp]);lb=np.concatenate([np.ones(len(sx)),np.zeros(len(sx))])
    return roc_auc_score(lb,sc), float((sic<sx).mean()), float((sx-sic).mean())
out={}
for name,ue,us in [("Full (flip+spec+rank+eqv+set)",True,True),("Full - Equivariance",False,True),("Full - Set Selection",True,False)]:
    rows=np.array([run(s,ue,us) for s in [42,123,456]])
    out[name]={"pair_auroc":[round(rows[:,0].mean(),4),round(rows[:,0].std(),4)],"flip":round(rows[:,1].mean(),4),"dtarget":round(rows[:,2].mean(),4)}
    print(f"  [{name}] pair_auroc {rows[:,0].mean():.4f}±{rows[:,0].std():.4f} flip {rows[:,1].mean():.3f} dtarget {rows[:,2].mean():.3f}",flush=True)
json.dump(out,open(os.environ.get('CIVIC_PROJECT_ROOT', '.') + '/outputs/equivariance_set_selection_results.json','w'),indent=2)
print("E5_EQVSET_V2 DONE",flush=True)
