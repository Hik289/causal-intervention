pass

import os
import torch; torch.set_num_threads(8)
import numpy as np, json, sys
sys.path.insert(0,os.environ.get('CIVIC_PROJECT_ROOT', '.') + '/phaseC'); sys.path.insert(0,os.environ.get('CIVIC_PROJECT_ROOT', '.') + '/phaseF')
from model import build
from train_civic import InterventionRanker
a=np.load(os.environ.get('CIVIC_PROJECT_ROOT', '.') + '/cache/embdiff_arrays.npz'); idx=[json.loads(l) for l in open(os.environ.get('CIVIC_PROJECT_ROOT', '.') + '/cache/embdiff_index.jsonl')]
cx,cic,cicp=a['cx'],a['cic'],a['cicp']
def cv(u,v):return (u*v).sum(1)/(np.linalg.norm(u,axis=1)*np.linalg.norm(v,axis=1)+1e-9)
clean=[i for i in range(len(idx)) if cv(cx,cicp)[i]>0.95 and cv(cx,cic)[i]<0.999]
def sp(s):return [i for i in clean if idx[i]['split']==s]
tr,te=sp('train'),sp('test'); T=lambda k,ii: torch.tensor(a[k][ii],dtype=torch.float32)
def train(seed, use_spec):
    torch.manual_seed(seed);np.random.seed(seed)
    m=build('crossattn',384,384);rk=InterventionRanker(dz=512,ddelta=384)
    opt=torch.optim.AdamW(list(m.parameters())+list(rk.parameters()),lr=2e-3);I=np.array(tr)
    for ep in range(100):
        m.train();opt.zero_grad()
        sx,zx=m(T('cx',I),T('crit',I));sic,_=m(T('cic',I),T('crit',I));sicp,_=m(T('cicp',I),T('crit',I))
        L=torch.clamp(0.5+sic-sx,min=0).mean()
        if use_spec: L=L+0.1*torch.abs(sicp-sx).mean()
        hdp=T('cic',I)-T('cx',I);hdn=T('cicp',I)-T('cx',I)
        L=L+0.5*(-torch.log(torch.exp(rk(zx,hdp))/(torch.exp(rk(zx,hdp))+torch.exp(rk(zx,hdn))+1e-9)).mean())+0.01*hdp.norm(dim=-1).mean()
        L.backward();opt.step()
    m.eval();rk.eval();return m,rk
te_arr=np.array(te); all_delta=a['cic'][te_arr]-a['cx'][te_arr]
def ev_ranker(m,rk):
    with torch.no_grad(): zx=m(T('cx',te),T('crit',te))[1]
    res={}
    for K in [4,8,16]:
        rng=np.random.RandomState(0);t1=0;r5=0;rr=0.0;nq=0
        with torch.no_grad():
            for qi in range(len(te)):
                o=np.delete(np.arange(len(te)),qi)
                if len(o)<K: continue
                cand=np.concatenate([[qi],rng.choice(o,K,replace=False)])
                sc=rk(zx[qi:qi+1].repeat(len(cand),1),torch.tensor(all_delta[cand],dtype=torch.float32)).numpy().flatten()
                rk_=int(np.where(np.argsort(-sc)==0)[0][0])+1;t1+=(rk_==1);r5+=(rk_<=5);rr+=1.0/rk_;nq+=1
        res[K]=(t1/nq,r5/nq,rr/nq)
    return res
def ev_embsim(m):

    with torch.no_grad(): zx=m(T('cx',te),T('crit',te))[1].numpy()
    res={}
    for K in [4,8,16]:
        rng=np.random.RandomState(0);t1=0;r5=0;rr=0.0;nq=0
        for qi in range(len(te)):
            o=np.delete(np.arange(len(te)),qi)
            if len(o)<K: continue
            cand=np.concatenate([[qi],rng.choice(o,K,replace=False)])

            dq=all_delta[qi]; sc=np.array([cv(all_delta[c:c+1],dq[None])[0] for c in cand])
            rk_=int(np.where(np.argsort(-sc)==0)[0][0])+1;t1+=(rk_==1);r5+=(rk_<=5);rr+=1.0/rk_;nq+=1
        res[K]=(t1/nq,r5/nq,rr/nq)
    return res
def ev_random():
    res={}
    for K in [4,8,16]: res[K]=(1.0/(K+1),min(1.0,5.0/(K+1)),sum(1.0/r for r in range(1,K+2))/(K+1))
    return res
out={}

rnd=ev_random(); out["Random Candidate"]={f"K={K}":{"top1":round(rnd[K][0],4),"recall5":round(rnd[K][1],4),"mrr":round(rnd[K][2],4)} for K in [4,8,16]}
print(f"  [Random] K4 Top1 {rnd[4][0]:.3f}",flush=True)

for name,usespec,useranker in [("Embedding Similarity",None,False),("Ranker w/o Specificity",False,True),("CIVIC Ranker (full)",True,True)]:
    accs={4:[],8:[],16:[]}
    for s in [42,123,456]:
        if useranker: m,rk=train(s,usespec); r=ev_ranker(m,rk)
        else: m,_=train(s,True); r=ev_embsim(m)
        for K in [4,8,16]: accs[K].append(r[K])
    out[name]={}
    for K in [4,8,16]:
        arr=np.array(accs[K]); out[name][f"K={K}"]={"top1":round(arr[:,0].mean(),4),"recall5":round(arr[:,1].mean(),4),"mrr":round(arr[:,2].mean(),4)}
    print(f"  [{name}] K4 Top1 {out[name]['K=4']['top1']} K8 {out[name]['K=8']['top1']} K16 {out[name]['K=16']['top1']}",flush=True)
json.dump(out,open(os.environ.get('CIVIC_PROJECT_ROOT', '.') + '/phaseF/e7_baselines_result.json','w'),indent=2)
print("E7_BASELINES DONE",flush=True)
