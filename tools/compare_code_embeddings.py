from civic.services import code_encoder, text_encoder
pass

import os, sys, json, numpy as np, torch
import tools.build_embeddings as BE
import pandas as pd, glob
DATA=os.environ.get("CIVIC_PROJECT_ROOT", ".") + "/data"; DS=f"{DATA}/intervention_dataset"

def get_regions(n=40):
    pass
    quads=[json.loads(l) for l in open(f"{DS}/quadruples_final.jsonl")]
    df=pd.concat([pd.read_parquet(f) for f in glob.glob(f"{DATA}/swe-review-bench_raw/data/*.parquet")],ignore_index=True)
    res=df[df.resolved==True].groupby('instance_id').first()
    patch_by={i:r.model_patch for i,r in res.iterrows()}
    base_by={i:r.base_commit for i,r in res.iterrows()}
    def srcf(p):
        import re; return [f for f in re.findall(r'diff --git a/(\S+) b/',p or '') if 'test' not in f.lower()]
    out=[]

    seen=set()
    for q in quads:
        iid=q['issue_id']
        if iid in seen or iid not in patch_by: continue
        repo=q['repo']
        if repo not in BE.REPO_DIR: continue
        tgt=patch_by[iid]; dist_iid=q['I_cprime'].get('from_issue'); dist=patch_by.get(dist_iid)
        if not dist: continue
        src=srcf(tgt);
        if not src: continue
        union=list(dict.fromkeys(src)); hunk=tgt or ""
        try:
            rx=BE.region_after_patch(repo,base_by[iid],None,union,hunk,ctx=3)
            ric=BE.region_after_patch(repo,base_by[iid],tgt,union,hunk,ctx=3)
            ricp=BE.region_after_patch(repo,base_by[iid],dist,union,hunk,ctx=3)
        except Exception: continue
        if rx and ric and ricp:
            out.append((rx,ric,ricp)); seen.add(iid)
        if len(out)>=n: break
    return out

def encode_text(texts):
    from sentence_transformers import SentenceTransformer
    m=SentenceTransformer(text_encoder())
    return m.encode(texts,convert_to_numpy=True,show_progress_bar=False)

def encode_code(texts):
    from transformers import AutoTokenizer, AutoModel
    tok=AutoTokenizer.from_pretrained(code_encoder())
    mdl=AutoModel.from_pretrained(code_encoder()); mdl.eval()
    embs=[]
    with torch.no_grad():
        for t in texts:
            ids=tok(t[:4000],return_tensors="pt",truncation=True,max_length=512)
            o=mdl(**ids)

            h=o.last_hidden_state.mean(1).squeeze(0).numpy()
            embs.append(h)
    return np.array(embs)

def sep(enc, trips, name):
    xs=[t[0] for t in trips]; ics=[t[1] for t in trips]; icps=[t[2] for t in trips]
    hx=enc(xs); hic=enc(ics); hicp=enc(icps)
    td=np.linalg.norm(hx-hic,axis=1); ud=np.linalg.norm(hx-hicp,axis=1)
    def cos(u,v): return (u*v).sum(1)/(np.linalg.norm(u,axis=1)*np.linalg.norm(v,axis=1)+1e-9)
    ct=cos(hx,hic); cu=cos(hx,hicp)
    print(f"=== {name} (dim={hx.shape[1]}) ===",flush=True)
    print(f"  target_diff ||h_x-h_Ic|| mean={td.mean():.4f}  unrel_diff ||h_x-h_Icp|| mean={ud.mean():.4f}",flush=True)
    print(f"  cos(x,I_c) mean={ct.mean():.4f}  cos(x,I_c') mean={cu.mean():.4f}",flush=True)
    print(f"  分离度 target_diff/unrel_diff = {td.mean()/(ud.mean()+1e-9):.3f} (want target 改动>>unrel; 但 unrel=disjoint 应≈0)",flush=True)
    print(f"  target 可见比例 (cos(x,I_c)<0.999): {int((ct<0.999).sum())}/{len(ct)}",flush=True)
    return td.mean(), ud.mean(), ct.mean()

def main():
    trips=get_regions(40)
    print(f"got {len(trips)} quad region triples",flush=True)
    if len(trips)<10: print("too few, abort"); return
    sep(encode_text, trips, "Text encoder (baseline 0.28)")
    sep(encode_code, trips, "Code encoder (code-aware)")
    print("SMOKE DONE",flush=True)

if __name__=="__main__": main()
