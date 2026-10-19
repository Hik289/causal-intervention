pass

import os
import json, os, re, sys
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from spike_repair import resolution_check, dbl, sh, free_gb
import pandas as pd, glob
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
DATA=os.environ.get("CIVIC_PROJECT_ROOT", ".") + "/data/swe-review-bench_raw/data"; HERE=os.path.dirname(os.path.abspath(__file__))

def strip_test_hunks(patch):
    blocks=re.split(r'(?=diff --git )',patch)
    return "".join(b for b in blocks if b.startswith('diff --git') and 'test' not in b.split("\n")[0].lower())

def main():
    d=json.load(open(f"{HERE}/spike_repair.json"))
    valid=[r for r in d if r.get('base')=='fail' and not r.get('skip')]
    ids=[r['iid'] for r in valid]
    df=pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{DATA}/*.parquet"))],ignore_index=True)
    res=df[df.resolved==True].groupby('instance_id').first().reset_index()
    meta={r.instance_id:r for _,r in res.iterrows()}
    outp=f"{HERE}/gold_flip.json"
    gf={}
    if os.path.exists(outp):
        try: gf=json.load(open(outp))
        except: gf={}
    lk=Lock()
    def work(iid):
        if iid in gf: return
        r=meta[iid]; img=r.image_name; f2p=dbl(r.FAIL_TO_PASS)
        gold_src=strip_test_hunks(r.model_patch)
        oc,rep_ok,cnt=resolution_check(img, r.repo, getattr(r,'test_patch',''), f2p, repair_patch=gold_src, timeout=900)
        with lk:
            gf[iid]={"gold_oc":oc,"gold_flip":(oc=='pass'),"gold_rep_ok":rep_ok}
            json.dump(gf,open(outp,"w"),indent=2)
            print(f"  {iid}: gold_oc={oc} flip={oc=='pass'} rep_ok={rep_ok}",flush=True)
    todo=[i for i in ids if i not in gf]
    print(f"{len(todo)} to compute gold-flip (of {len(ids)})",flush=True)
    with ThreadPoolExecutor(max_workers=10) as ex: list(ex.map(work,todo))

    print("="*50,flush=True)
    repairable=[iid for iid in ids if gf.get(iid,{}).get("gold_flip")]
    unrepairable=[iid for iid in ids if not gf.get(iid,{}).get("gold_flip")]
    print(f"可修子集 (gold-flip=True): {len(repairable)}",flush=True)
    print(f"不可修子集 (gold 修不动/无法验证): {len(unrepairable)}",flush=True)
    recs={r['iid']:r for r in valid}

    tflip=[r['iid'] for r in valid if r.get('true_flip')]
    print(f"TRUE flips: {tflip}; 都在可修子集? {all(i in repairable for i in tflip)}",flush=True)

    from scipy import stats
    def arm_rate(subset, lab):
        v=[recs[i] for i in subset if recs[i].get(f'{lab}_resolution') in ('pass','fail')]
        flip=sum(1 for r in v if r.get(f'{lab}_flip'))
        return flip,len(v)
    print("--- 可修子集内三臂 resolution ---",flush=True)
    rates={}
    for lab in ['nocrit','false','true']:
        fl,n=arm_rate(repairable,lab); rates[lab]=(fl,n)
        print(f"  {lab}: {fl}/{n} = {fl/n:.3f}" if n else f"  {lab}: 0/0",flush=True)

    tf,tn=rates['true']; ff,fn=rates['false']; nf,nn=rates['nocrit']
    try:
        _,p_tf=stats.fisher_exact([[tf,tn-tf],[ff,fn-ff]])
        print(f"  Fisher true-vs-false (可修子集): p={p_tf:.4f}",flush=True)
        if nn>0:
            _,p_tn=stats.fisher_exact([[tf,tn-tf],[nf,nn-nf]])
            print(f"  Fisher true-vs-nocrit (可修子集): p={p_tn:.4f}",flush=True)
    except Exception as e: print("fisher err",e,flush=True)
    print("GOLD_FLIP DONE",flush=True)

if __name__=="__main__": main()
