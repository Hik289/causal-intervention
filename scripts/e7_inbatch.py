from civic.paths import OUTPUT_DIR
pass

import os
import torch; torch.set_num_threads(4)
import numpy as np, json, sys
from civic.models import build
from civic.ranking import InterventionRanker
a=np.load(os.environ.get('CIVIC_PROJECT_ROOT', '.') + '/cache/embdiff_arrays.npz'); idx=[json.loads(l) for l in open(os.environ.get('CIVIC_PROJECT_ROOT', '.') + '/cache/embdiff_index.jsonl')]
cx,cic,cicp=a['cx'],a['cic'],a['cicp']
def cv(u,v):return (u*v).sum(1)/(np.linalg.norm(u,axis=1)*np.linalg.norm(v,axis=1)+1e-9)
clean=[i for i in range(len(idx)) if cv(cx,cicp)[i]>0.95 and cv(cx,cic)[i]<0.999]
def sp(s):return [i for i in clean if idx[i]['split']==s]
tr,te=sp('train'),sp('test')
T=lambda k,ii: torch.tensor(a[k][ii],dtype=torch.float32)

def train_inbatch(seed, epochs=100, tau=0.1):
    torch.manual_seed(seed); np.random.seed(seed)
    m=build('crossattn',384,384); ranker=InterventionRanker(dz=512,ddelta=384)
    opt=torch.optim.AdamW(list(m.parameters())+list(ranker.parameters()),lr=2e-3)
    I=np.array(tr); N=len(I)
    for ep in range(epochs):
        m.train(); ranker.train(); opt.zero_grad()
        sx,zx=m(T('cx',I),T('crit',I)); sic,_=m(T('cic',I),T('crit',I)); sicp,_=m(T('cicp',I),T('crit',I))

        L=torch.clamp(0.5+sic-sx,min=0).mean()+0.1*torch.abs(sicp-sx).mean()

        dtar=T('cic',I)-T('cx',I)

        M=min(8,N-1)
        rng_idx=torch.randint(0,N,(N,M))
        zi=zx.unsqueeze(1).expand(N,M+1,zx.shape[1]).reshape(-1,zx.shape[1])
        cand=torch.cat([dtar.unsqueeze(1), dtar[rng_idx]],dim=1).reshape(-1,dtar.shape[1])
        sc=ranker(zi,cand).reshape(N,M+1)/tau
        L_rank=torch.nn.functional.cross_entropy(sc, torch.zeros(N,dtype=torch.long))
        L=L+0.5*L_rank+0.01*dtar.norm(dim=-1).mean()
        L.backward(); opt.step()
    m.eval(); ranker.eval(); return m, ranker

def eval_ranking(m, ranker):
    te_arr=np.array(te); all_delta=torch.tensor(a['cic'][te_arr]-a['cx'][te_arr],dtype=torch.float32)
    with torch.no_grad(): zx_all=m(T('cx',te),T('crit',te))[1]
    res={}
    for K in [4,8,16]:
        rng=np.random.RandomState(0); t1=0;r5=0;rr=0.0;nq=0
        with torch.no_grad():
            for qi in range(len(te)):
                others=np.delete(np.arange(len(te)),qi)
                if len(others)<K: continue
                dist=rng.choice(others,K,replace=False); cand=np.concatenate([[qi],dist])
                z=zx_all[qi:qi+1].repeat(len(cand),1)
                sc=ranker(z,all_delta[cand]).numpy().flatten()
                rk=int(np.where(np.argsort(-sc)==0)[0][0])+1
                t1+=(rk==1);r5+=(rk<=5);rr+=1.0/rk;nq+=1
        res[K]=(t1/max(1,nq),r5/max(1,nq),rr/max(1,nq),nq)
    return res
out={}; per={}
for s in [42,123,456]:
    m,ranker=train_inbatch(s); per[s]=eval_ranking(m,ranker)
    print(f"  seed{s} done",flush=True)
for K in [4,8,16]:
    t1=[per[s][K][0] for s in [42,123,456]];r5=[per[s][K][1] for s in [42,123,456]];mr=[per[s][K][2] for s in [42,123,456]]
    out[f"K={K}"]={"top1":[round(np.mean(t1),4),round(np.std(t1),4)],"recall5":[round(np.mean(r5),4),round(np.std(r5),4)],"mrr":[round(np.mean(mr),4),round(np.std(mr),4)],"n_query":per[42][K][3]}
    print(f"  [K={K}] Top-1 {round(np.mean(t1),4)}±{round(np.std(t1),4)} Recall@5 {round(np.mean(r5),4)} MRR {round(np.mean(mr),4)} (vs 原E7 Top1 {'0.230' if K==4 else '0.104' if K==8 else '0.058'})",flush=True)
json.dump(out,open(os.environ.get('CIVIC_PROJECT_ROOT', '.') + '/outputs/e7_inbatch_result.json','w'),indent=2)
print("E7_INBATCH DONE",flush=True)
