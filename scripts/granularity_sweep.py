from civic.services import remote_embedding
from civic.services import text_encoder
pass

import os, sys, json, time, numpy as np, urllib.request, re
import tools.build_embeddings as BE
import pandas as pd, glob
DATA=os.environ.get("CIVIC_PROJECT_ROOT", ".") + "/data"; DS=f"{DATA}/intervention_dataset"

def az(t):
    return np.asarray(remote_embedding(t), dtype=np.float32)

def make_text_encoder():
    from sentence_transformers import SentenceTransformer
    m=SentenceTransformer(text_encoder())
    return lambda texts: m.encode(texts,convert_to_numpy=True,show_progress_bar=False)

def srcf(p): return [f for f in re.findall(r'diff --git a/(\S+) b/',p or '') if 'test' not in f.lower()]

def patch_changed_lines(patch, fp):
    pass
    old=[]; new=[]; infile=False
    for ln in (patch or "").split("\n"):
        m=re.match(r'diff --git a/(\S+) b/(\S+)',ln)
        if m: infile=(m.group(2)==fp or m.group(1)==fp); continue
        if infile:
            if ln.startswith("+") and not ln.startswith("+++"): new.append(ln[1:])
            elif ln.startswith("-") and not ln.startswith("---"): old.append(ln[1:])
    return "\n".join(old), "\n".join(new)

def get_samples(n=40):
    quads=[json.loads(l) for l in open(f"{DS}/quadruples_final.jsonl")]
    df=pd.concat([pd.read_parquet(f) for f in glob.glob(f"{DATA}/swe-review-bench_raw/data/*.parquet")],ignore_index=True)
    res=df[df.resolved==True].groupby('instance_id').first()
    pby={i:r.model_patch for i,r in res.iterrows()}; bby={i:r.base_commit for i,r in res.iterrows()}
    out=[]; seen=set()
    for q in quads:
        iid=q['issue_id']
        if iid in seen or iid not in pby or q['repo'] not in BE.REPO_DIR: continue
        tgt=pby[iid]; dist=pby.get(q['I_cprime'].get('from_issue'))
        if not dist: continue
        src=srcf(tgt)
        if not src: continue
        fp=src[0]; union=[fp]; hunk=tgt
        try:
            rx=BE.region_after_patch(q['repo'],bby[iid],None,union,hunk,ctx=3)
            ric=BE.region_after_patch(q['repo'],bby[iid],tgt,union,hunk,ctx=3)
            rx1=BE.region_after_patch(q['repo'],bby[iid],None,union,hunk,ctx=1)
            ric1=BE.region_after_patch(q['repo'],bby[iid],tgt,union,hunk,ctx=1)
        except Exception: continue
        xo,xn=patch_changed_lines(tgt,fp)
        if rx and ric and rx1 and ric1 and (xo or xn):
            out.append({"region_x":rx,"region_ic":ric,"hunk_x":rx1,"hunk_ic":ric1,
                        "diff_x":xo or " ","diff_ic":xn or " "}); seen.add(iid)
        if len(out)>=n: break
    return out

def analyze(enc_az, enc_mini, samples, xkey, ickey, name):
    xs=[s[xkey] for s in samples]; ics=[s[ickey] for s in samples]
    for label,enc in [("Text encoder",enc_mini),("Remote",enc_az)]:
        if label=="Remote":
            hx=np.array([az(t) for t in xs]); hic=np.array([az(t) for t in ics])
        else:
            hx=enc(xs); hic=enc(ics)
        cos=(hx*hic).sum(1)/(np.linalg.norm(hx,axis=1)*np.linalg.norm(hic,axis=1)+1e-9)
        dev=1-cos
        print(f"  [{name}/{label}] target cos 偏离 mean={dev.mean():.4f} sd={dev.std():.4f} Cohen_d(vs0)={dev.mean()/(dev.std()+1e-9):.2f}",flush=True)

def main():
    s=get_samples(40)
    print(f"got {len(s)} samples",flush=True)
    mini=make_text_encoder()
    print("=== 粒度 1: region ctx=3 (现在) ===",flush=True); analyze(az,mini,s,"region_x","region_ic","ctx3")
    print("=== 粒度 2: hunk ctx=1 (聚焦) ===",flush=True); analyze(az,mini,s,"hunk_x","hunk_ic","hunk")
    print("=== 粒度 3: diff 纯 changed 行 (最聚焦) ===",flush=True); analyze(az,mini,s,"diff_x","diff_ic","diff")
    print("参照: Text encoder region 的 target 偏离 0.041 (the study 要 Remote 聚焦粒度下 > 它)",flush=True)
    print("SWEEP DONE",flush=True)

if __name__=="__main__": main()
