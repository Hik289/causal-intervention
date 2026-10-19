pass

import os
import torch; torch.set_num_threads(4)
import numpy as np, json, sys
sys.path.insert(0,os.environ.get('CIVIC_PROJECT_ROOT', '.') + '/phaseC')
sys.path.insert(0,os.environ.get('CIVIC_PROJECT_ROOT', '.') + '/phaseF')
from model import build
from train_civic import InterventionRanker
from sklearn.metrics import roc_auc_score
a=np.load(os.environ.get('CIVIC_PROJECT_ROOT', '.') + '/cache/embdiff_arrays.npz'); idx=[json.loads(l) for l in open(os.environ.get('CIVIC_PROJECT_ROOT', '.') + '/cache/embdiff_index.jsonl')]
cx,cic,cicp=a['cx'],a['cic'],a['cicp']
def cv(u,v):return (u*v).sum(1)/(np.linalg.norm(u,axis=1)*np.linalg.norm(v,axis=1)+1e-9)
clean=[i for i in range(len(idx)) if cv(cx,cicp)[i]>0.95 and cv(cx,cic)[i]<0.999]
T=lambda k,ii: torch.tensor(a[k][ii],dtype=torch.float32)
repos=sorted(set(idx[i]['repo'] for i in clean))

def train(tr_idx, seed, epochs=100):
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
    return m, ranker
def pair_auroc(m, te_idx):
    I=np.array(te_idx)
    with torch.no_grad():
        sx,_=m(T('cx',I),T('crit',I)); sic,_=m(T('cic',I),T('crit',I)); sicp,_=m(T('cicp',I),T('crit',I))
    sx,sic,sicp=sx.numpy(),sic.numpy(),sicp.numpy()
    sc=np.concatenate([sx-sic,sx-sicp]); lb=np.concatenate([np.ones(len(sx)),np.zeros(len(sx))])
    return roc_auc_score(lb,sc), float((sic<sx).mean())

a_crit='crit'

assert 'crit' in a.files

print("===== E7 Intervention Ranking =====",flush=True)
tr=[i for i in clean if idx[i]['split']=='train']; te=[i for i in clean if idx[i]['split']=='test']
e7={}
for K in [4,8,16]:
    top1=[];recall5=[];mrr=[]
    for s in [42,123,456]:
        m,ranker=train(tr,s)
        rng=np.random.RandomState(s); t1=0;r5=0;rr=0.0;nq=0
        with torch.no_grad():
            for i in te:
                others=[j for j in te if j!=i]
                if len(others)<K: continue
                dist=rng.choice(others,K,replace=False); cand=[i]+list(dist)
                zx=m(T('cx',[i]),T('crit',[i]))[1]
                deltas=torch.tensor(a['cic'][cand]-a['cx'][cand],dtype=torch.float32)
                scores=ranker(zx.repeat(len(cand),1),deltas).numpy().flatten()
                order=np.argsort(-scores); rk=int(np.where(order==0)[0][0])+1
                t1+=(rk==1);r5+=(rk<=5);rr+=1.0/rk;nq+=1
        top1.append(t1/max(1,nq));recall5.append(r5/max(1,nq));mrr.append(rr/max(1,nq))
    e7[f"K={K}"]={"top1":[round(np.mean(top1),4),round(np.std(top1),4)],
                  "recall5":[round(np.mean(recall5),4),round(np.std(recall5),4)],
                  "mrr":[round(np.mean(mrr),4),round(np.std(mrr),4)],"n_query":nq}
    print(f"  [K={K}] Top-1 {round(np.mean(top1),4)}±{round(np.std(top1),4)} Recall@5 {round(np.mean(recall5),4)} MRR {round(np.mean(mrr),4)} (nq={nq})",flush=True)
json.dump(e7,open(os.environ.get('CIVIC_PROJECT_ROOT', '.') + '/phaseF/e7_result.json','w'),indent=2)
print("E7 DONE",flush=True)

print("\n===== E6 Cross-project (leave-project-out, lean) =====",flush=True)
big=[r for r in repos if sum(1 for i in clean if idx[i]['repo']==r)>=20]
e6={"_base_ref":"clsonly global pair_auroc 0.495 (主结果对照, 无机制基线)"}
for held in big:
    te=[i for i in clean if idx[i]['repo']==held]; tr2=[i for i in clean if idx[i]['repo']!=held]
    aus=[]
    for s in [42,123,456]:
        m,_=train(tr2,s); au,fl=pair_auroc(m,te); aus.append(au)
    e6[held]={"n_test":len(te),"civic_pair_auroc":[round(np.mean(aus),4),round(np.std(aus),4)]}
    print(f"  [{held}] n_test={len(te)} CIVIC pair_auroc {round(np.mean(aus),4)}±{round(np.std(aus),4)} (vs base ref 0.495)",flush=True)
json.dump(e6,open(os.environ.get('CIVIC_PROJECT_ROOT', '.') + '/phaseF/e6_result.json','w'),indent=2)
print("E6 DONE",flush=True)
print("E6_E7 ALL DONE",flush=True)
