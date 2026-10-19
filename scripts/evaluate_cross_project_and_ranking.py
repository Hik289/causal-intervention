from civic.paths import OUTPUT_DIR
pass

import os
import torch; torch.set_num_threads(4)
import numpy as np, json, sys
from civic.models import build
from civic.ranking import InterventionRanker
from sklearn.metrics import roc_auc_score
a=np.load(os.environ.get('CIVIC_PROJECT_ROOT', '.') + '/cache/embdiff_arrays.npz'); idx=[json.loads(l) for l in open(os.environ.get('CIVIC_PROJECT_ROOT', '.') + '/cache/embdiff_index.jsonl')]
cx,cic,cicp,cr=a['cx'],a['cic'],a['cicp'],a['crit']
def cv(u,v):return (u*v).sum(1)/(np.linalg.norm(u,axis=1)*np.linalg.norm(v,axis=1)+1e-9)
cp=cv(cx,cicp); ci=cv(cx,cic)
clean=[i for i in range(len(idx)) if cp[i]>0.95 and ci[i]<0.999]
T=lambda k,ii: torch.tensor(a[k][ii],dtype=torch.float32)
repos=sorted(set(idx[i]['repo'] for i in clean))

def train(tr_idx, seed, w=dict(flip=1.0,spec=0.1,rank=0.5), epochs=150):
    torch.manual_seed(seed); np.random.seed(seed)
    m=build('crossattn',384,384); ranker=InterventionRanker(dz=512,ddelta=384)
    opt=torch.optim.AdamW(list(m.parameters())+list(ranker.parameters()),lr=2e-3)
    I=np.array(tr_idx)
    for ep in range(epochs):
        m.train(); ranker.train(); opt.zero_grad()
        sx,zx=m(T('cx',I),T('crit',I)); sic,_=m(T('cic',I),T('crit',I)); sicp,_=m(T('cicp',I),T('crit',I))
        L=w['flip']*torch.clamp(0.5+sic-sx,min=0).mean()+w['spec']*torch.abs(sicp-sx).mean()
        hdp=T('cic',I)-T('cx',I); hdn=T('cicp',I)-T('cx',I)
        rp=ranker(zx,hdp); rn=ranker(zx,hdn)
        L=L+w['rank']*(-torch.log(torch.exp(rp)/(torch.exp(rp)+torch.exp(rn)+1e-9)).mean())
        L=L+0.01*hdp.norm(dim=-1).mean(); L.backward(); opt.step()
    return m, ranker

def pair_auroc(m, te_idx):
    I=np.array(te_idx)
    with torch.no_grad():
        sx,_=m(T('cx',I),T('crit',I)); sic,_=m(T('cic',I),T('crit',I)); sicp,_=m(T('cicp',I),T('crit',I))
    sx,sic,sicp=sx.numpy(),sic.numpy(),sicp.numpy()
    sc=np.concatenate([sx-sic,sx-sicp]); lb=np.concatenate([np.ones(len(sx)),np.zeros(len(sx))])
    return roc_auc_score(lb,sc), float((sic<sx).mean())

print("===== E6 Cross-project (leave-project-out) =====",flush=True)
big=[r for r in repos if sum(1 for i in clean if idx[i]['repo']==r)>=20]
e6={}
for held in big:
    te=[i for i in clean if idx[i]['repo']==held]
    tr=[i for i in clean if idx[i]['repo']!=held]
    aus=[]; base_aus=[]
    for s in [42,123,456]:
        m,_=train(tr,s); au,fl=pair_auroc(m,te); aus.append(au)

        torch.manual_seed(s); np.random.seed(s)
        mb=build('crossattn',384,384); ob=torch.optim.AdamW(mb.parameters(),lr=2e-3); I=np.array(tr)
        for ep in range(150):
            mb.train(); ob.zero_grad()
            sx,_=mb(T('cx',I),T('crit',I))
            Lb=0.01*(T('cic',I)-T('cx',I)).norm(dim=-1).mean()
            if Lb.requires_grad: Lb.backward(); ob.step()
        bau,_=pair_auroc(mb,te); base_aus.append(bau)
    e6[held]={"n_test":len(te),"civic_pair_auroc":[round(np.mean(aus),4),round(np.std(aus),4)],
              "base_pair_auroc":[round(np.mean(base_aus),4),round(np.std(base_aus),4)]}
    print(f"  [{held}] n_test={len(te)} CIVIC {round(np.mean(aus),4)}±{round(np.std(aus),4)} | base {round(np.mean(base_aus),4)}",flush=True)
json.dump(e6,open(os.environ.get('CIVIC_PROJECT_ROOT', '.') + '/outputs/cross_project_results.json','w'),indent=2)
print("E6 DONE",flush=True)

print("\n===== E7 Intervention Ranking =====",flush=True)

tr=[i for i in clean if idx[i]['split']=='train']; te=[i for i in clean if idx[i]['split']=='test']
e7={}
for K in [4,8,16]:
    top1=[]; recall5=[]; mrr=[]
    for s in [42,123,456]:
        m,ranker=train(tr,s)
        rng=np.random.RandomState(s)
        t1=0; r5=0; rr=0.0; nq=0
        with torch.no_grad():
            for i in te:

                others=[j for j in te if j!=i]
                if len(others)<K: continue
                dist=rng.choice(others,K,replace=False)
                zx=m(T('cx',[i]),T('crit',[i]))[1]
                cand_idx=[i]+list(dist)
                deltas=torch.tensor(a['cic'][cand_idx]-a['cx'][cand_idx],dtype=torch.float32)
                zrep=zx.repeat(len(cand_idx),1)
                scores=ranker(zrep,deltas).numpy().flatten()
                order=np.argsort(-scores)
                rank_of_target=int(np.where(order==0)[0][0])+1
                t1+=(rank_of_target==1); r5+=(rank_of_target<=5); rr+=1.0/rank_of_target; nq+=1
        top1.append(t1/max(1,nq)); recall5.append(r5/max(1,nq)); mrr.append(rr/max(1,nq))
    e7[f"K={K}"]={"top1":[round(np.mean(top1),4),round(np.std(top1),4)],
                  "recall5":[round(np.mean(recall5),4),round(np.std(recall5),4)],
                  "mrr":[round(np.mean(mrr),4),round(np.std(mrr),4)],"n_query":nq}
    print(f"  [K={K}] Top-1 {round(np.mean(top1),4)}±{round(np.std(top1),4)} | Recall@5 {round(np.mean(recall5),4)} | MRR {round(np.mean(mrr),4)} (nq={nq})",flush=True)
json.dump(e7,open(os.environ.get('CIVIC_PROJECT_ROOT', '.') + '/outputs/intervention_ranking_results.json','w'),indent=2)
print("E7 DONE",flush=True)
print("E6_E7 ALL DONE",flush=True)
