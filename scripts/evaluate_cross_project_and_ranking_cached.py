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
    m.eval(); ranker.eval(); return m, ranker
def pair_auroc(m, te_idx):
    I=np.array(te_idx)
    with torch.no_grad():
        sx,_=m(T('cx',I),T('crit',I)); sic,_=m(T('cic',I),T('crit',I)); sicp,_=m(T('cicp',I),T('crit',I))
    sx,sic,sicp=sx.numpy(),sic.numpy(),sicp.numpy()
    sc=np.concatenate([sx-sic,sx-sicp]); lb=np.concatenate([np.ones(len(sx)),np.zeros(len(sx))])
    return roc_auc_score(lb,sc), float((sic<sx).mean())

tr=[i for i in clean if idx[i]['split']=='train']; te=[i for i in clean if idx[i]['split']=='test']

print("===== E7 Intervention Ranking (fast) =====",flush=True)

te_arr=np.array(te)
all_delta=torch.tensor(a['cic'][te_arr]-a['cx'][te_arr],dtype=torch.float32)
per_seed={42:{},123:{},456:{}}
models={}
for s in [42,123,456]:
    m,ranker=train(tr,s); models[s]=(m,ranker)
    with torch.no_grad():
        zx_all=m(T('cx',te),T('crit',te))[1]

    rng=np.random.RandomState(s)
    for K in [4,8,16]:
        t1=0;r5=0;rr=0.0;nq=0
        with torch.no_grad():
            for qi in range(len(te)):
                others=np.delete(np.arange(len(te)),qi)
                if len(others)<K: continue
                dist=rng.choice(others,K,replace=False); cand=np.concatenate([[qi],dist])
                z=zx_all[qi:qi+1].repeat(len(cand),1)
                sc=ranker(z,all_delta[cand]).numpy().flatten()
                rk=int(np.where(np.argsort(-sc)==0)[0][0])+1
                t1+=(rk==1);r5+=(rk<=5);rr+=1.0/rk;nq+=1
        per_seed[s][K]=(t1/max(1,nq),r5/max(1,nq),rr/max(1,nq),nq)
    print(f"  seed{s} done",flush=True)
e7={}
for K in [4,8,16]:
    t1=[per_seed[s][K][0] for s in [42,123,456]]; r5=[per_seed[s][K][1] for s in [42,123,456]]; mr=[per_seed[s][K][2] for s in [42,123,456]]
    nq=per_seed[42][K][3]
    e7[f"K={K}"]={"top1":[round(np.mean(t1),4),round(np.std(t1),4)],"recall5":[round(np.mean(r5),4),round(np.std(r5),4)],"mrr":[round(np.mean(mr),4),round(np.std(mr),4)],"n_query":nq}
    print(f"  [K={K}] Top-1 {round(np.mean(t1),4)}±{round(np.std(t1),4)} Recall@5 {round(np.mean(r5),4)} MRR {round(np.mean(mr),4)} (nq={nq})",flush=True)
json.dump(e7,open(os.environ.get('CIVIC_PROJECT_ROOT', '.') + '/outputs/intervention_ranking_results.json','w'),indent=2)
print("E7 DONE",flush=True)

print("\n===== E6 Cross-project (leave-project-out) =====",flush=True)
big=[r for r in repos if sum(1 for i in clean if idx[i]['repo']==r)>=20]
e6={"_base_ref":"clsonly global pair_auroc 0.495 (无机制对照)"}
for held in big:
    teh=[i for i in clean if idx[i]['repo']==held]; trh=[i for i in clean if idx[i]['repo']!=held]
    aus=[]
    for s in [42,123,456]:
        m,_=train(trh,s); au,_=pair_auroc(m,teh); aus.append(au)
    e6[held]={"n_test":len(teh),"civic_pair_auroc":[round(np.mean(aus),4),round(np.std(aus),4)]}
    print(f"  [{held}] n_test={len(teh)} CIVIC pair_auroc {round(np.mean(aus),4)}±{round(np.std(aus),4)}",flush=True)
json.dump(e6,open(os.environ.get('CIVIC_PROJECT_ROOT', '.') + '/outputs/cross_project_results.json','w'),indent=2)
print("E6 DONE\nE6_E7 ALL DONE",flush=True)
