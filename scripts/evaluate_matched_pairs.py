from civic.paths import OUTPUT_DIR
pass

import os
import torch; torch.set_num_threads(4)
import numpy as np, json, sys
from civic.models import build
from civic.ranking import InterventionRanker
from sklearn.metrics import roc_auc_score, average_precision_score
a=np.load(os.environ.get('CIVIC_PROJECT_ROOT', '.') + '/cache/embdiff_arrays.npz'); idx=[json.loads(l) for l in open(os.environ.get('CIVIC_PROJECT_ROOT', '.') + '/cache/embdiff_index.jsonl')]
cx,cic,cicp=a['cx'],a['cic'],a['cicp']
def cv(u,v):return (u*v).sum(1)/(np.linalg.norm(u,axis=1)*np.linalg.norm(v,axis=1)+1e-9)
clean=[i for i in range(len(idx)) if cv(cx,cicp)[i]>0.95 and cv(cx,cic)[i]<0.999]
def sp(s):return [i for i in clean if idx[i]['split']==s]
tr,te=sp('train'),sp('test'); T=lambda k,ii: torch.tensor(a[k][ii],dtype=torch.float32)
CONF={"Classifier-Only":dict(flip=0,spec=0,rank=0),"Sensitivity-Only":dict(flip=0,spec=1,rank=0),
      "CIVIC-CR-full":dict(flip=1.0,spec=0.1,rank=0.5)}
def run(w,seed):
    torch.manual_seed(seed);np.random.seed(seed)
    m=build('crossattn',384,384);rk=InterventionRanker(dz=512,ddelta=384)
    opt=torch.optim.AdamW(list(m.parameters())+list(rk.parameters()),lr=2e-3);I=np.array(tr)
    for ep in range(150):
        m.train();opt.zero_grad()
        sx,zx=m(T('cx',I),T('crit',I));sic,_=m(T('cic',I),T('crit',I));sicp,_=m(T('cicp',I),T('crit',I))
        L=torch.tensor(0.0)
        if w['flip']>0:L=L+w['flip']*torch.clamp(0.5+sic-sx,min=0).mean()
        if w['spec']>0:L=L+w['spec']*torch.abs(sicp-sx).mean()
        if w['rank']>0:
            hdp=T('cic',I)-T('cx',I);hdn=T('cicp',I)-T('cx',I)
            L=L+w['rank']*(-torch.log(torch.exp(rk(zx,hdp))/(torch.exp(rk(zx,hdp))+torch.exp(rk(zx,hdn))+1e-9)).mean())
        hdp=T('cic',I)-T('cx',I);L=L+0.01*hdp.norm(dim=-1).mean()
        if L.requires_grad:L.backward();opt.step()
    m.eval();I=np.array(te)
    with torch.no_grad():
        sx,_=m(T('cx',I),T('crit',I));sic,_=m(T('cic',I),T('crit',I))
    sx,sic=sx.numpy(),sic.numpy()

    thr=np.median(np.concatenate([sx,sic]))
    pos_recall=float((sx>thr).mean())
    trap_rej=float((sic<=thr).mean())
    pairacc=float((sx>sic).mean())
    scores=np.concatenate([sx,sic]);labels=np.concatenate([np.ones(len(sx)),np.zeros(len(sic))])
    au=roc_auc_score(labels,scores);ap=average_precision_score(labels,scores)
    margin=float((sx-sic).mean())
    return pos_recall,trap_rej,pairacc,au,ap,margin
out={}
for name,w in CONF.items():
    rows=np.array([run(w,s) for s in [42,123,456]])
    mean=rows.mean(0);std=rows.std(0)
    out[name]={"positive_recall":[round(mean[0],4),round(std[0],4)],"trap_rejection":[round(mean[1],4),round(std[1],4)],
               "pairacc":[round(mean[2],4),round(std[2],4)],"auroc":[round(mean[3],4),round(std[3],4)],
               "auprc":[round(mean[4],4),round(std[4],4)],"mean_pair_margin":[round(mean[5],4),round(std[5],4)]}
    print(f"  [{name}] PosRecall {mean[0]:.3f} TrapRej {mean[1]:.3f} PairAcc {mean[2]:.3f} AUROC {mean[3]:.3f} AUPRC {mean[4]:.3f} Margin {mean[5]:.3f}",flush=True)
json.dump(out,open(os.environ.get('CIVIC_PROJECT_ROOT', '.') + '/outputs/matched_pair_results.json','w'),indent=2)
print("E2_COMPLETE DONE",flush=True)
