pass

import os, sys, json, time, numpy as np, urllib.request, re
sys.path.insert(0,os.environ.get("CIVIC_PROJECT_ROOT", ".") + "/phaseC")
import build_embeddings as BE
import pandas as pd, glob
DATA=os.environ.get("CIVIC_PROJECT_ROOT", ".") + "/data"; DS=f"{DATA}/intervention_dataset"
AZ_URL=os.environ.get("AZURE_EMBEDDING_ENDPOINT", "")
AZ_KEY=os.environ.get("AZURE_EMBEDDING_API_KEY", "")

def az(t):
    t=str(t) if t else " "
    body=json.dumps({"model":"text-embedding-3-small","input":t[:8000]}).encode()
    for _ in range(5):
        try:
            r=urllib.request.Request(AZ_URL,data=body,headers={"api-key":AZ_KEY,"Content-Type":"application/json"})
            with urllib.request.urlopen(r,timeout=40) as x: return np.array(json.loads(x.read())["data"][0]["embedding"],dtype=np.float32)
        except Exception: time.sleep(2)
    return None

def minilm_enc():
    from sentence_transformers import SentenceTransformer
    m=SentenceTransformer("all-MiniLM-L6-v2")
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
    for label,enc in [("all-MiniLM",enc_mini),("Azure",enc_az)]:
        if label=="Azure":
            hx=np.array([az(t) for t in xs]); hic=np.array([az(t) for t in ics])
        else:
            hx=enc(xs); hic=enc(ics)
        cos=(hx*hic).sum(1)/(np.linalg.norm(hx,axis=1)*np.linalg.norm(hic,axis=1)+1e-9)
        dev=1-cos
        print(f"  [{name}/{label}] target cos 偏离 mean={dev.mean():.4f} sd={dev.std():.4f} Cohen_d(vs0)={dev.mean()/(dev.std()+1e-9):.2f}",flush=True)

def main():
    s=get_samples(40)
    print(f"got {len(s)} samples",flush=True)
    mini=minilm_enc()
    print("=== 粒度 1: region ctx=3 (现在) ===",flush=True); analyze(az,mini,s,"region_x","region_ic","ctx3")
    print("=== 粒度 2: hunk ctx=1 (聚焦) ===",flush=True); analyze(az,mini,s,"hunk_x","hunk_ic","hunk")
    print("=== 粒度 3: diff 纯 changed 行 (最聚焦) ===",flush=True); analyze(az,mini,s,"diff_x","diff_ic","diff")
    print("参照: all-MiniLM region 的 target 偏离 0.041 (the study 要 Azure 聚焦粒度下 > 它)",flush=True)
    print("SWEEP DONE",flush=True)

if __name__=="__main__": main()
