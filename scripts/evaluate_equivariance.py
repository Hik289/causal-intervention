from civic.paths import OUTPUT_DIR
pass

import os
import torch; torch.set_num_threads(4)
import numpy as np, json, re, sys
import pandas as pd, glob
from civic.models import build
from civic.ranking import InterventionRanker
from sklearn.metrics import roc_auc_score
from sentence_transformers import SentenceTransformer
DATA=os.environ.get("CIVIC_PROJECT_ROOT", ".") + "/data"; DS=f"{DATA}/intervention_dataset"; CACHE=os.environ.get("CIVIC_PROJECT_ROOT", ".") + "/cache"

def srcf(p): return [f for f in re.findall(r'diff --git a/(\S+) b/',p or '') if 'test' not in f.lower()]
def changed_sides(patch, fp):
    before=[]; after=[]; infile=False
    for ln in (patch or "").split("\n"):
        m=re.match(r'diff --git a/(\S+) b/(\S+)',ln)
        if m: infile=(m.group(2)==fp or m.group(1)==fp); continue
        if not infile: continue
        if ln.startswith("@@"): before.append("# hunk"); after.append("# hunk"); continue
        if ln.startswith("+") and not ln.startswith("+++"): after.append(ln[1:])
        elif ln.startswith("-") and not ln.startswith("---"): before.append(ln[1:])
        elif ln.startswith(" "): before.append(ln[1:]); after.append(ln[1:])
    return "\n".join(before), "\n".join(after)

RENAME={r'\bi\b':'idx',r'\bj\b':'jdx',r'\bx\b':'val',r'\btmp\b':'temp',r'\bres\b':'result',
        r'\bret\b':'retval',r'\bbuf\b':'buffer',r'\bn\b':'count',r'\bs\b':'text',r'\bobj\b':'instance'}
def transform(code):
    out=code
    for pat,rep in RENAME.items(): out=re.sub(pat,rep,out)

    return "# refactored\n"+out

def main():
    quads=[json.loads(l) for l in open(f"{DS}/quadruples_final.jsonl")]
    qmap={q['quad_id']:q for q in quads}
    df=pd.concat([pd.read_parquet(f) for f in glob.glob(f"{DATA}/swe-review-bench_raw/data/*.parquet")],ignore_index=True)
    res=df[df.resolved==True].groupby('instance_id').first(); pby={i:r.model_patch for i,r in res.iterrows()}
    fixidx=[json.loads(l) for l in open(f"{CACHE}/embdiff_index.jsonl")]
    m=SentenceTransformer("all-MiniLM-L6-v2")
    def emb(t): return m.encode([str(t) if t else " "],convert_to_numpy=True)[0]

    recs=[]
    for rec in fixidx:
        q=qmap.get(rec['quad_id']);
        if not q: continue
        iid=q['issue_id']
        if iid not in pby: continue
        tgt=pby[iid]; src=srcf(tgt)
        if not src: continue
        fp=src[0]; before,after=changed_sides(tgt,fp)
        if not (before.strip() or after.strip()): continue
        cval=q.get('critique',''); ctext=cval.get('text','') if isinstance(cval,dict) else str(cval)
        recs.append({"split":rec['split'],"before":before,"after":after,"crit":ctext})
    print(f"rebuilt {len(recs)} quads",flush=True)

    def E(txts): return np.array([emb(t) for t in txts])
    tr=[r for r in recs if r['split']=='train']; te=[r for r in recs if r['split']=='test']
    cx_tr=E([r['before'] for r in tr]); cic_tr=E([r['after'] for r in tr]); crit_tr=E([r['crit'] for r in tr])
    cx_te=E([r['before'] for r in te]); cic_te=E([r['after'] for r in te]); crit_te=E([r['crit'] for r in te])

    cx_te_t=E([transform(r['before']) for r in te]); cic_te_t=E([transform(r['after']) for r in te]); crit_te_t=crit_te
    T=lambda arr: torch.tensor(arr,dtype=torch.float32)
    def run_seed(seed):
        torch.manual_seed(seed); np.random.seed(seed)
        mod=build('crossattn',384,384); opt=torch.optim.AdamW(mod.parameters(),lr=2e-3)
        for ep in range(150):
            mod.train(); opt.zero_grad()
            sx,_=mod(T(cx_tr),T(crit_tr)); sic,_=mod(T(cic_tr),T(crit_tr))
            L=torch.clamp(0.5+sic-sx,min=0).mean()+0.01*T(cic_tr-cx_tr).norm(dim=-1).mean()
            L.backward(); opt.step()
        mod.eval()
        with torch.no_grad():
            sx,_=mod(T(cx_te),T(crit_te)); sic,_=mod(T(cic_te),T(crit_te))
            sxt,_=mod(T(cx_te_t),T(crit_te_t)); sict,_=mod(T(cic_te_t),T(crit_te_t))
        dt=(sx-sic).numpy(); dtt=(sxt-sict).numpy()

        vc=float(((dt>0)==(dtt>0)).mean())

        corr=float(np.corrcoef(dt,dtt)[0,1]) if len(dt)>1 else None

        def pa(sx,sic,sicp=None):
            return float((sic<sx).mean())
        flip_orig=float((sic.numpy()<sx.numpy()).mean()); flip_tx=float((sict.numpy()<sxt.numpy()).mean())
        return vc, corr, flip_orig, flip_tx
    vcs=[];corrs=[];fo=[];ft=[]
    for s in [42,123,456]:
        vc,corr,flip_o,flip_t=run_seed(s); vcs.append(vc);corrs.append(corr);fo.append(flip_o);ft.append(flip_t)
        print(f"  seed{s}: verdict_consistency={vc:.4f} Δtarget_corr={corr:.4f} flip_orig={flip_o:.3f} flip_tx={flip_t:.3f}",flush=True)
    out={"n_test":len(te),"verdict_consistency":[round(np.mean(vcs),4),round(np.std(vcs),4)],
         "dtarget_corr":[round(np.nanmean(corrs),4),round(np.nanstd(corrs),4)],
         "flip_orig":[round(np.mean(fo),4)],"flip_tx":[round(np.mean(ft),4)],
         "fp_change_rate":round(abs(np.mean(fo)-np.mean(ft)),4)}
    json.dump(out,open(os.environ.get('CIVIC_PROJECT_ROOT', '.') + '/outputs/equivariance_results.json','w'),indent=2)
    print(f"E4: verdict_consistency {out['verdict_consistency']} | Δtarget_corr {out['dtarget_corr']} | FP_change {out['fp_change_rate']}",flush=True)
    print("E4 DONE",flush=True)

if __name__=="__main__": main()
