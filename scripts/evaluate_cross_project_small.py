from civic.paths import OUTPUT_DIR
pass

import os
import torch; torch.set_num_threads(4)
import numpy as np, json, sys
from civic.models import build
from civic.ranking import InterventionRanker
from sklearn.metrics import roc_auc_score
a=np.load(os.environ.get('CIVIC_PROJECT_ROOT', '.') + '/cache/embdiff_arrays.npz'); idx=[json.loads(l) for l in open(os.environ.get('CIVIC_PROJECT_ROOT', '.') + '/cache/embdiff_index.jsonl')]
cx,cic,cicp=a['cx'],a['cic'],a['cicp']
def cv(u,v):return (u*v).sum(1)/(np.linalg.norm(u,axis=1)*np.linalg.norm(v,axis=1)+1e-9)
clean=[i for i in range(len(idx)) if cv(cx,cicp)[i]>0.95 and cv(cx,cic)[i]<0.999]
T=lambda k,ii: torch.tensor(a[k][ii],dtype=torch.float32)
def train(tr_idx, seed, epochs=60):
    torch.manual_seed(seed); np.random.seed(seed)
    m=build('crossattn',384,384); ranker=InterventionRanker(dz=512,ddelta=384)
    opt=torch.optim.AdamW(list(m.parameters())+list(ranker.parameters()),lr=2e-3)
    I=np.array(tr_idx)
    for ep in range(epochs):
        m.train(); ranker.train(); opt.zero_grad()
        sx,zx=m(T('cx',I),T('crit',I)); sic,_=m(T('cic',I),T('crit',I)); sicp,_=m(T('cicp',I),T('crit',I))
        L=torch.clamp(0.5+sic-sx,min=0).mean()+0.1*torch.abs(sicp-sx).mean()
        hdp=T('cic',I)-T('cx',I); hdn=T('cicp',I)-T('cx',I)
        rp=ranker(zx,hdp); rn=ranker(zx,hdn)
        L=L+0.5*(-torch.log(torch.exp(rp)/(torch.exp(rp)+torch.exp(rn)+1e-9)).mean())+0.01*hdp.norm(dim=-1).mean()
        L.backward(); opt.step()
    m.eval(); return m
def pa(m, te_idx):
    I=np.array(te_idx)
    with torch.no_grad():
        sx,_=m(T('cx',I),T('crit',I)); sic,_=m(T('cic',I),T('crit',I)); sicp,_=m(T('cicp',I),T('crit',I))
    sx,sic,sicp=sx.numpy(),sic.numpy(),sicp.numpy()
    sc=np.concatenate([sx-sic,sx-sicp]); lb=np.concatenate([np.ones(len(sx)),np.zeros(len(sx))])
    return roc_auc_score(lb,sc)
print("===== E6 Cross-project (leave-project-out, min: 3 repo 60ep 2seed) =====",flush=True)
targets=['django/django','sympy/sympy','scikit-learn/scikit-learn']
e6={"_base_ref":"clsonly global pair_auroc 0.495 (无机制对照)","_config":"60ep 2seed (饱和机器精简)"}
for held in targets:
    teh=[i for i in clean if idx[i]['repo']==held]; trh=[i for i in clean if idx[i]['repo']!=held]
    if not teh: continue
    aus=[]
    for s in [42,123]:
        m=train(trh,s); aus.append(pa(m,teh))
    e6[held]={"n_test":len(teh),"n_train":len(trh),"civic_pair_auroc":[round(np.mean(aus),4),round(np.std(aus),4)]}
    print(f"  [{held}] n_test={len(teh)} n_train={len(trh)} CIVIC pair_auroc {round(np.mean(aus),4)}±{round(np.std(aus),4)} (vs base 0.495)",flush=True)
    json.dump(e6,open(os.environ.get('CIVIC_PROJECT_ROOT', '.') + '/outputs/cross_project_results.json','w'),indent=2)
print("E6 DONE",flush=True)
