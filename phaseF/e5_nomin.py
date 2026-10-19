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
def sp(s):return [i for i in clean if idx[i]['split']==s]
tr,te=sp('train'),sp('test')
T=lambda k,ii: torch.tensor(a[k][ii],dtype=torch.float32)
def run(seed, use_min):
    torch.manual_seed(seed); np.random.seed(seed)
    m=build('crossattn',384,384); ranker=InterventionRanker(dz=512,ddelta=384)
    opt=torch.optim.AdamW(list(m.parameters())+list(ranker.parameters()),lr=2e-3)
    I=np.array(tr)
    for ep in range(150):
        m.train(); ranker.train(); opt.zero_grad()
        sx,zx=m(T('cx',I),T('crit',I)); sic,_=m(T('cic',I),T('crit',I)); sicp,_=m(T('cicp',I),T('crit',I))
        L=torch.clamp(0.5+sic-sx,min=0).mean()+0.1*torch.abs(sicp-sx).mean()
        hdp=T('cic',I)-T('cx',I); hdn=T('cicp',I)-T('cx',I)
        rp=ranker(zx,hdp); rn=ranker(zx,hdn)
        L=L+0.5*(-torch.log(torch.exp(rp)/(torch.exp(rp)+torch.exp(rn)+1e-9)).mean())
        if use_min: L=L+0.01*hdp.norm(dim=-1).mean()
        L.backward(); opt.step()
    m.eval(); I=np.array(te)
    with torch.no_grad():
        sx,_=m(T('cx',I),T('crit',I)); sic,_=m(T('cic',I),T('crit',I)); sicp,_=m(T('cicp',I),T('crit',I))
    sx,sic,sicp=sx.numpy(),sic.numpy(),sicp.numpy()
    sc=np.concatenate([sx-sic,sx-sicp]); lb=np.concatenate([np.ones(len(sx)),np.zeros(len(sx))])
    return roc_auc_score(lb,sc), float((sic<sx).mean()), float((sx-sic).mean())
out={}
for name,um in [("full_with_min",True),("minus_minimality",False)]:
    aus=[];fls=[];dts=[]
    for s in [42,123,456]:
        au,fl,dt=run(s,um); aus.append(au);fls.append(fl);dts.append(dt)
    out[name]={"pair_auroc":[round(np.mean(aus),4),round(np.std(aus),4)],"flip":[round(np.mean(fls),4)],"dtarget":[round(np.mean(dts),4)]}
    print(f"  [{name}] pair_auroc {round(np.mean(aus),4)}±{round(np.std(aus),4)} flip {round(np.mean(fls),4)} dtarget {round(np.mean(dts),4)}",flush=True)
json.dump(out,open(os.environ.get('CIVIC_PROJECT_ROOT', '.') + '/phaseF/e5_nomin_result.json','w'),indent=2)
print("E5_NOMIN DONE",flush=True)
