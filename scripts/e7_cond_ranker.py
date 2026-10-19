from civic.paths import OUTPUT_DIR
pass

import os
import torch; torch.set_num_threads(8)
import torch.nn as nn
import numpy as np, json, sys
a=np.load(os.environ.get('CIVIC_PROJECT_ROOT', '.') + '/cache/embdiff_arrays.npz'); idx=[json.loads(l) for l in open(os.environ.get('CIVIC_PROJECT_ROOT', '.') + '/cache/embdiff_index.jsonl')]
cx,cic,cicp,cr=a['cx'],a['cic'],a['cicp'],a['crit']
def cv(u,v):return (u*v).sum(1)/(np.linalg.norm(u,axis=1)*np.linalg.norm(v,axis=1)+1e-9)
clean=[i for i in range(len(idx)) if cv(cx,cicp)[i]>0.95 and cv(cx,cic)[i]<0.999]
def sp(s):return [i for i in clean if idx[i]['split']==s]
tr,te=sp('train'),sp('test')
T=lambda arr,ii: torch.tensor(arr[ii],dtype=torch.float32)

class CondRanker(nn.Module):
    pass
    def __init__(s,d=384,h=256):
        super().__init__()
        s.net=nn.Sequential(nn.Linear(d*3,h),nn.ReLU(),nn.Linear(h,h),nn.ReLU(),nn.Linear(h,1))
    def forward(s,c,x,delta):
        return s.net(torch.cat([c,x,delta],-1)).squeeze(-1)

def train(seed):
    torch.manual_seed(seed);np.random.seed(seed)
    m=CondRanker();opt=torch.optim.AdamW(m.parameters(),lr=1e-3)
    I=np.array(tr);N=len(I)
    cxt=T(cx,I);crt=T(cr,I)
    dpos=T(cic,I)-T(cx,I)
    dunrel=T(cicp,I)-T(cx,I)
    for ep in range(200):
        m.train();opt.zero_grad()

        perm=torch.randperm(N)
        dneg_cross=dpos[perm]
        sp_=m(crt,cxt,dpos)
        sn1=m(crt,cxt,dunrel)
        sn2=m(crt,cxt,dneg_cross)

        L=torch.clamp(0.3+sn1-sp_,min=0).mean()+torch.clamp(0.3+sn2-sp_,min=0).mean()
        L.backward();opt.step()
    m.eval();return m
def eval_rank(m):
    te_arr=np.array(te);allc=T(cr,te_arr);allx=T(cx,te_arr);alld=torch.tensor(a['cic'][te_arr]-a['cx'][te_arr],dtype=torch.float32)
    res={}
    for K in [4,8,16]:
        rng=np.random.RandomState(0);t1=0;r5=0;rr=0.0;nq=0
        with torch.no_grad():
            for qi in range(len(te)):
                o=np.delete(np.arange(len(te)),qi)
                if len(o)<K: continue
                cand=np.concatenate([[qi],rng.choice(o,K,replace=False)])
                c_=allc[qi:qi+1].repeat(len(cand),1);x_=allx[qi:qi+1].repeat(len(cand),1)
                sc=m(c_,x_,alld[cand]).numpy()
                rkk=int(np.where(np.argsort(-sc)==0)[0][0])+1;t1+=(rkk==1);r5+=(rkk<=5);rr+=1.0/rkk;nq+=1
        res[K]=(t1/nq,r5/nq,rr/nq,nq)
    return res
out={};per={}
for s in [42,123,456]:
    m=train(s);per[s]=eval_rank(m);print(f"  seed{s} done",flush=True)
for K in [4,8,16]:
    t1=[per[s][K][0] for s in [42,123,456]];r5=[per[s][K][1] for s in [42,123,456]];mr=[per[s][K][2] for s in [42,123,456]]
    rand=1.0/(K+1)
    out[f"K={K}"]={"top1":[round(np.mean(t1),4),round(np.std(t1),4)],"recall5":[round(np.mean(r5),4),round(np.std(r5),4)],"mrr":[round(np.mean(mr),4),round(np.std(mr),4)],"random_top1":round(rand,4),"n_query":per[42][K][3]}
    print(f"  [K={K}] Top-1 {round(np.mean(t1),4)}±{round(np.std(t1),4)} (random {rand:.3f}) Recall@5 {round(np.mean(r5),4)} MRR {round(np.mean(mr),4)}",flush=True)
json.dump(out,open(os.environ.get('CIVIC_PROJECT_ROOT', '.') + '/outputs/e7_cond_result.json','w'),indent=2)
print("E7_COND DONE",flush=True)
