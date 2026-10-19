pass

import os, sys, json, time, numpy as np, urllib.request, re
import tools.build_embeddings as BE
import pandas as pd, glob
DATA=os.environ.get("CIVIC_PROJECT_ROOT", ".") + "/data"; DS=f"{DATA}/intervention_dataset"; CACHE=os.environ.get("CIVIC_PROJECT_ROOT", ".") + "/cache"
AZ_URL=os.environ.get("AZURE_EMBEDDING_ENDPOINT", "")
AZ_KEY=os.environ.get("AZURE_EMBEDDING_API_KEY", "")
def az_embed(text, retries=6):
    text=str(text) if text else " "; body=json.dumps({"model":"text-embedding-3-small","input":text[:8000]}).encode()
    for att in range(retries):
        try:
            req=urllib.request.Request(AZ_URL,data=body,headers={"api-key":AZ_KEY,"Content-Type":"application/json"})
            with urllib.request.urlopen(req,timeout=40) as r:
                return np.array(json.loads(r.read())["data"][0]["embedding"],dtype=np.float32)
        except Exception:
            if att<retries-1: time.sleep(2*(att+1)); continue
            return None
    return None

def srcf(p): return [f for f in re.findall(r'diff --git a/(\S+) b/',p or '') if 'test' not in f.lower()]

def main():
    quads=[json.loads(l) for l in open(f"{DS}/quadruples_final.jsonl")]
    df=pd.concat([pd.read_parquet(f) for f in glob.glob(f"{DATA}/swe-review-bench_raw/data/*.parquet")],ignore_index=True)
    res=df[df.resolved==True].groupby('instance_id').first()
    patch_by={i:r.model_patch for i,r in res.iterrows()}; base_by={i:r.base_commit for i,r in res.iterrows()}

    fixidx=[json.loads(l) for l in open(f"{CACHE}/embfix_index.jsonl")]
    qmap={q['quad_id']:q for q in quads}
    out={"crit":[],"cx":[],"cic":[],"cicp":[]}; keep_idx=[]

    import os as _os
    done_ids=set()
    if _os.path.exists(f"{CACHE}/embaz_arrays.npz") and _os.path.exists(f"{CACHE}/embaz_index.jsonl"):
        try:
            prev=np.load(f"{CACHE}/embaz_arrays.npz"); pidx=[json.loads(l) for l in open(f"{CACHE}/embaz_index.jsonl")]
            for k in out: out[k]=list(prev[k])
            keep_idx=list(pidx); done_ids=set(r['quad_id'] for r in pidx)
            print(f"resume: {len(done_ids)} already done",flush=True)
        except Exception as e: print("resume load fail",e,flush=True)
    done=len(done_ids)
    for rec in fixidx:
        qid=rec['quad_id']; q=qmap.get(qid)
        if not q or qid in done_ids: continue
        iid=q['issue_id']; repo=q['repo']
        if iid not in patch_by or repo not in BE.REPO_DIR: continue
        tgt=patch_by[iid]; dist=patch_by.get(q['I_cprime'].get('from_issue'))
        if not dist: continue
        src=srcf(tgt)
        if not src: continue
        union=list(dict.fromkeys(src)); hunk=tgt or ""
        from concurrent.futures import ThreadPoolExecutor as _TPE
        def _regions():
            return (BE.region_after_patch(repo,base_by[iid],None,union,hunk,ctx=3),
                    BE.region_after_patch(repo,base_by[iid],tgt,union,hunk,ctx=3),
                    BE.region_after_patch(repo,base_by[iid],dist,union,hunk,ctx=3))
        try:
            with _TPE(max_workers=1) as _ex:
                rx,ric,ricp=_ex.submit(_regions).result(timeout=120)
        except Exception: continue
        if not(rx and ric and ricp): continue
        hx=az_embed(rx); hic=az_embed(ric); hicp=az_embed(ricp); cval=q.get('critique',''); ctext=cval.get('text','') if isinstance(cval,dict) else str(cval); hc=az_embed(ctext)
        time.sleep(0.25)
        if any(x is None for x in [hx,hic,hicp,hc]): continue
        out["cx"].append(hx);out["cic"].append(hic);out["cicp"].append(hicp);out["crit"].append(hc)
        keep_idx.append(rec); done+=1
        if done%25==0:
            print(f"  azure {done} quads",flush=True)
            np.savez(f"{CACHE}/embaz_arrays.npz",**{k:np.array(v) for k,v in out.items()})
            with open(f"{CACHE}/embaz_index.jsonl","w") as f:
                for r in keep_idx: f.write(json.dumps(r)+"\n")
    np.savez(f"{CACHE}/embaz_arrays.npz",**{k:np.array(v) for k,v in out.items()})
    with open(f"{CACHE}/embaz_index.jsonl","w") as f:
        for r in keep_idx: f.write(json.dumps(r)+"\n")
    print(f"DONE azure materialize: {done} quads, dim={len(out['cx'][0]) if out['cx'] else 0}",flush=True)

if __name__=="__main__": main()
