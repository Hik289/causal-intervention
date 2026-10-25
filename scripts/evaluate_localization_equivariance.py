from civic.services import text_encoder
from civic.paths import OUTPUT_DIR
pass

import os
import torch; torch.set_num_threads(4)
import numpy as np, json, re, sys
import pandas as pd, glob
from civic.models import build
from sentence_transformers import SentenceTransformer
DATA=os.environ.get("CIVIC_PROJECT_ROOT", ".") + "/data"; DS=f"{DATA}/intervention_dataset"; CACHE=os.environ.get("CIVIC_PROJECT_ROOT", ".") + "/cache"
def srcf(p): return [f for f in re.findall(r'diff --git a/(\S+) b/',p or '') if 'test' not in f.lower()]
def changed_sides(patch, fp):
    before=[];after=[];inf=False
    for ln in (patch or "").split("\n"):
        m=re.match(r'diff --git a/(\S+) b/(\S+)',ln)
        if m: inf=(m.group(2)==fp or m.group(1)==fp); continue
        if not inf: continue
        if ln.startswith("@@"): before.append("# hunk");after.append("# hunk");continue
        if ln.startswith("+") and not ln.startswith("+++"): after.append(ln[1:])
        elif ln.startswith("-") and not ln.startswith("---"): before.append(ln[1:])
        elif ln.startswith(" "): before.append(ln[1:]);after.append(ln[1:])
    return "\n".join(before),"\n".join(after)
RENAME={r'\bi\b':'idx',r'\bj\b':'jdx',r'\bx\b':'val',r'\btmp\b':'temp',r'\bres\b':'result',r'\bn\b':'count',r'\bs\b':'text'}
def tx_rename(code):
    out=code
    for p,r in RENAME.items(): out=re.sub(p,r,out)
    return out, 0
def tx_shift(code, N=2):
    return "\n".join(["# header"]*N)+"\n"+code, N
def main():
    quads=[json.loads(l) for l in open(f"{DS}/quadruples_final.jsonl")]; qmap={q['quad_id']:q for q in quads}
    df=pd.concat([pd.read_parquet(f) for f in glob.glob(f"{DATA}/swe-review-bench_raw/data/*.parquet")],ignore_index=True)
    res=df[df.resolved==True].groupby('instance_id').first();pby={i:r.model_patch for i,r in res.iterrows()}
    fixidx=[json.loads(l) for l in open(f"{CACHE}/embdiff_index.jsonl")]
    m=SentenceTransformer(text_encoder()); emb=lambda t:m.encode([str(t) if t else " "],convert_to_numpy=True)[0]
    recs=[]
    for rec in fixidx:
        q=qmap.get(rec['quad_id']);
        if not q: continue
        iid=q['issue_id']
        if iid not in pby: continue
        tgt=pby[iid];src=srcf(tgt)
        if not src: continue
        fp=src[0];before,after=changed_sides(tgt,fp)
        if not (before.strip() or after.strip()): continue
        cval=q.get('critique','');ctext=cval.get('text','') if isinstance(cval,dict) else str(cval)
        recs.append({"split":rec['split'],"before":before,"after":after,"crit":ctext})
    tr=[r for r in recs if r['split']=='train'];te=[r for r in recs if r['split']=='test']
    E=lambda ts:np.array([emb(t) for t in ts])
    cx_tr=E([r['before'] for r in tr]);cic_tr=E([r['after'] for r in tr]);cr_tr=E([r['crit'] for r in tr])
    cx_te=E([r['before'] for r in te]);cic_te=E([r['after'] for r in te]);cr_te=E([r['crit'] for r in te])

    cx_rn=E([tx_rename(r['before'])[0] for r in te]);cic_rn=E([tx_rename(r['after'])[0] for r in te])
    cx_sh=E([tx_shift(r['before'])[0] for r in te]);cic_sh=E([tx_shift(r['after'])[0] for r in te])
    T=lambda arr:torch.tensor(arr,dtype=torch.float32)
    def run(seed):
        torch.manual_seed(seed);np.random.seed(seed)
        mod=build('crossattn',384,384);opt=torch.optim.AdamW(mod.parameters(),lr=2e-3)
        for ep in range(150):
            mod.train();opt.zero_grad()
            sx,_=mod(T(cx_tr),T(cr_tr));sic,_=mod(T(cic_tr),T(cr_tr))
            L=torch.clamp(0.5+sic-sx,min=0).mean()+0.01*T(cic_tr-cx_tr).norm(dim=-1).mean();L.backward();opt.step()
        mod.eval()
        with torch.no_grad():
            dt=(mod(T(cx_te),T(cr_te))[0]-mod(T(cic_te),T(cr_te))[0]).numpy()
            dt_rn=(mod(T(cx_rn),T(cr_te))[0]-mod(T(cic_rn),T(cr_te))[0]).numpy()
            dt_sh=(mod(T(cx_sh),T(cr_te))[0]-mod(T(cic_sh),T(cr_te))[0]).numpy()

        loc_rn=float(((dt>0)==(dt_rn>0)).mean()); loc_sh=float(((dt>0)==(dt_sh>0)).mean())
        LocEqv=(loc_rn+loc_sh)/2

        ev_rn=float(np.corrcoef(dt,dt_rn)[0,1]); ev_sh=float(np.corrcoef(dt,dt_sh)[0,1])
        EvidenceEqv=(ev_rn+ev_sh)/2
        return LocEqv, EvidenceEqv
    rows=np.array([run(s) for s in [42,123,456]])
    out={"CIVIC-CR":{"LocEqv":[round(rows[:,0].mean(),4),round(rows[:,0].std(),4)],
                     "EvidenceEqv":[round(rows[:,1].mean(),4),round(rows[:,1].std(),4)],"n_test":len(te)}}
    print(f"  CIVIC-CR LocEqv {rows[:,0].mean():.4f}±{rows[:,0].std():.4f} EvidenceEqv {rows[:,1].mean():.4f}±{rows[:,1].std():.4f}",flush=True)
    json.dump(out,open(os.environ.get('CIVIC_PROJECT_ROOT', '.') + '/outputs/localization_equivariance_results.json','w'),indent=2)
    print("E4_LOC DONE",flush=True)
if __name__=="__main__": main()
