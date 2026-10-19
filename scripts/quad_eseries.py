from civic.paths import OUTPUT_DIR
pass

import os
import torch; torch.set_num_threads(4)
import numpy as np, json, sys
from civic.models import build
from sklearn.metrics import roc_auc_score

a=np.load(os.environ.get('CIVIC_PROJECT_ROOT', '.') + '/cache/embdiff_arrays.npz'); idx=[json.loads(l) for l in open(os.environ.get('CIVIC_PROJECT_ROOT', '.') + '/cache/embdiff_index.jsonl')]
cx,cic,cicp,cr=a['cx'],a['cic'],a['cicp'],a['crit']
def cv(u,v):return (u*v).sum(1)/(np.linalg.norm(u,axis=1)*np.linalg.norm(v,axis=1)+1e-9)
cp=cv(cx,cicp); ci=cv(cx,cic)
clean=[i for i in range(len(idx)) if cp[i]>0.95 and ci[i]<0.999]
def sp(s):return [i for i in clean if idx[i]['split']==s]
tr,va,te=sp('train'),sp('val'),sp('test')
print(f"quad clean: train {len(tr)} val {len(va)} test {len(te)} | raw target 偏离 {1-ci[clean].mean():.4f}",flush=True)
T=lambda k,ii: torch.tensor(a[k][ii],dtype=torch.float32)

CONFIGS={
 "full":      dict(flip=1.0,spec=0.1,rank=0.5),
 "flip_spec": dict(flip=1.0,spec=0.1,rank=0.0),
 "flip_only": dict(flip=1.0,spec=0.0,rank=0.0),
 "spec_only": dict(flip=0.0,spec=1.0,rank=0.0),
 "clsonly":   dict(flip=0.0,spec=0.0,rank=0.0),
}

def run(cfg_name, w, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    m=build('crossattn',384,384)
    try:
        from civic.ranking import InterventionRanker; ranker=InterventionRanker(dz=512,ddelta=384); has_rank=True
        opt=torch.optim.AdamW(list(m.parameters())+list(ranker.parameters()),lr=2e-3)
    except Exception:
        has_rank=False; opt=torch.optim.AdamW(m.parameters(),lr=2e-3)
    I=np.array(tr)
    for ep in range(150):
        m.train(); opt.zero_grad()
        sx,zx=m(T('cx',I),T('crit',I)); sic,_=m(T('cic',I),T('crit',I)); sicp,_=m(T('cicp',I),T('crit',I))
        L=torch.tensor(0.0)
        if w['flip']>0: L=L+w['flip']*torch.clamp(0.5+sic-sx,min=0).mean()
        if w['spec']>0: L=L+w['spec']*torch.abs(sicp-sx).mean()
        if w['rank']>0 and has_rank:
            hdp=T('cic',I)-T('cx',I); hdn=T('cicp',I)-T('cx',I)
            rp=ranker(zx,hdp); rn=ranker(zx,hdn)
            L=L+w['rank']*(-torch.log(torch.exp(rp)/(torch.exp(rp)+torch.exp(rn)+1e-9)).mean())
        hdp=T('cic',I)-T('cx',I); L=L+0.01*hdp.norm(dim=-1).mean()
        if L.requires_grad: L.backward(); opt.step()
    m.eval()
    def ev(ii):
        I=np.array(ii)
        with torch.no_grad():
            sx,_=m(T('cx',I),T('crit',I)); sic,_=m(T('cic',I),T('crit',I)); sicp,_=m(T('cicp',I),T('crit',I))
        sx,sic,sicp=sx.numpy(),sic.numpy(),sicp.numpy()
        dt=(sx-sic); du=np.abs(sx-sicp)

        d_t=sx-sic; d_u=sx-sicp
        scores=np.concatenate([d_t,d_u]); labels=np.concatenate([np.ones(len(d_t)),np.zeros(len(d_u))])
        try: pau=roc_auc_score(labels,scores)
        except Exception: pau=None

        dum=float(du.mean()); sr=round(float(dt.mean()/dum),3) if dum>1e-3 else None
        return dict(dtarget=round(float(dt.mean()),4),dunrel=round(dum,4),
                    spec_ratio=sr,
                    flip_rate=round(float((sic<sx).mean()),3),
                    pair_auroc=round(float(pau),4) if pau is not None else None)
    return ev(te)

def main():
    out={}
    for cfg,w in CONFIGS.items():
        rows=[run(cfg,w,s) for s in [42,123,456]]
        agg={}
        for k in rows[0]:
            vals=[r[k] for r in rows if r[k] is not None]
            agg[k]={"mean":round(float(np.mean(vals)),4),"std":round(float(np.std(vals)),4)} if vals else None
        out[cfg]=agg
        print(f"[{cfg}] Δtarget={agg['dtarget']['mean']:.3f}±{agg['dtarget']['std']:.3f} "
              f"Δunrel={agg['dunrel']['mean']:.3f} spec_ratio={agg['spec_ratio']['mean'] if agg['spec_ratio'] else None} "
              f"flip={agg['flip_rate']['mean']:.2f} pair_auroc={agg['pair_auroc']['mean'] if agg['pair_auroc'] else None}",flush=True)
    json.dump(out,open(os.environ.get('CIVIC_PROJECT_ROOT', '.') + '/outputs/quad_eseries_result.json','w'),indent=2)
    print("QUAD_ESERIES DONE",flush=True)

if __name__=="__main__": main()
