import os
import torch; torch.set_num_threads(4)
import numpy as np, json, sys
sys.path.insert(0,os.environ.get('CIVIC_PROJECT_ROOT', '.') + '/phaseC')
from model import build
from sklearn.metrics import roc_auc_score
a=np.load(os.environ.get('CIVIC_PROJECT_ROOT', '.') + '/cache/embdiff_arrays.npz'); idx=[json.loads(l) for l in open(os.environ.get('CIVIC_PROJECT_ROOT', '.') + '/cache/embdiff_index.jsonl')]
cx,cic,cicp,cr=a['cx'],a['cic'],a['cicp'],a['crit']
def cv(u,v):return (u*v).sum(1)/(np.linalg.norm(u,axis=1)*np.linalg.norm(v,axis=1)+1e-9)
clean=[i for i in range(len(idx)) if cv(cx,cicp)[i]>0.95 and cv(cx,cic)[i]<0.999]
def sp(s):return [i for i in clean if idx[i]['split']==s]
tr,te=sp('train'),sp('test')
T=lambda k,ii: torch.tensor(a[k][ii],dtype=torch.float32)
def run(spec,seed):
    torch.manual_seed(seed); np.random.seed(seed)
    m=build('crossattn',384,384); opt=torch.optim.AdamW(m.parameters(),lr=2e-3); I=np.array(tr)
    for ep in range(150):
        m.train(); opt.zero_grad()
        sx,_=m(T('cx',I),T('crit',I)); sic,_=m(T('cic',I),T('crit',I)); sicp,_=m(T('cicp',I),T('crit',I))
        L=torch.clamp(0.5+sic-sx,min=0).mean()
        if spec>0: L=L+spec*torch.abs(sicp-sx).mean()
        L=L+0.01*(T('cic',I)-T('cx',I)).norm(dim=-1).mean(); L.backward(); opt.step()
    m.eval(); I=np.array(te)
    with torch.no_grad():
        sx,_=m(T('cx',I),T('crit',I)); sic,_=m(T('cic',I),T('crit',I)); sicp,_=m(T('cicp',I),T('crit',I))
    sx,sic,sicp=sx.numpy(),sic.numpy(),sicp.numpy()
    sc=np.concatenate([sx-sic,sx-sicp]); lb=np.concatenate([np.ones(len(sx)),np.zeros(len(sx))])
    return roc_auc_score(lb,sc), float((sic<sx).mean())
print('=== flip_spec (spec=0.1) 3 seed per-seed ===',flush=True)
for s in [42,123,456]:
    au,fl=run(0.1,s); print(f'  seed{s}: pair_auroc={au:.4f} flip={fl:.3f}',flush=True)
print('DIAG2 DONE',flush=True)
