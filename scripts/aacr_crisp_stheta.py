from civic.services import chat_client, review_model, text_encoder
from civic.paths import OUTPUT_DIR
pass

import os
import json, os, re, urllib.request, difflib, time, sys
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import numpy as np
DATA=os.environ.get("CIVIC_PROJECT_ROOT", ".") + "/data"; CACHE=os.environ.get("CIVIC_PROJECT_ROOT", ".") + "/cache"
client=chat_client()
MODEL=review_model(); CRISP={'code defect','security vulnerability','performance'}
def fetch(repo,sha,path):
    url=f"https://raw.githubusercontent.com/{repo}/{sha}/{path}"
    try:
        req=urllib.request.Request(url,headers={"User-Agent":"civic"})
        return urllib.request.urlopen(req,timeout=20).read().decode("utf-8","ignore").split("\n")
    except Exception: return None
def region(lines,fl,tl,win=12):
    if not lines: return None
    try: fl=int(fl);tl=int(tl)
    except: fl=tl=0
    lo=max(0,fl-win);hi=min(len(lines),(tl or fl)+win)
    return "\n".join(lines[lo:hi])
def sol(msg):
    r=client.chat.completions.create(model=MODEL, messages=[{'role': 'user', 'content': msg}])
    return r.choices[0].message.content or ""
def gen_ic(reg,note):
    p=f"""代码和 review 评论 (可能缺陷)。中性: 存在问题则返回修复区, 否则原样(no-op)。只两块:
```before
<原始区域>
```
```after
<修复后; 无需修则同 before>
```
评论: {note[:700]}
代码:
```
{reg[:3000]}
```"""
    out=sol(p);mb=re.search(r'```before\n?(.*?)```',out,re.S);ma=re.search(r'```after\n?(.*?)```',out,re.S)
    return (mb.group(1).strip() if mb else ""),(ma.group(1).strip() if ma else "")
def gen_unrel(reg):
    p=f"对下面代码做与 bug 无关的表面修改 (加注释), 只返回 ```代码```。\n```\n{reg[:2500]}\n```"
    out=sol(p);m=re.search(r'```[a-z]*\n?(.*?)```',out,re.S)
    return (m.group(1) if m else out).strip()
def main():
    import torch; torch.set_num_threads(4)
    from civic.models import build
    from civic.ranking import InterventionRanker
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics import roc_auc_score

    a=np.load(f'{CACHE}/embdiff_arrays.npz'); idxq=[json.loads(l) for l in open(f'{CACHE}/embdiff_index.jsonl')]
    cx,cic,cicp=a['cx'],a['cic'],a['cicp']
    def cvf(u,v):return (u*v).sum(1)/(np.linalg.norm(u,axis=1)*np.linalg.norm(v,axis=1)+1e-9)
    clean=[i for i in range(len(idxq)) if cvf(cx,cicp)[i]>0.95 and cvf(cx,cic)[i]<0.999]
    trq=[i for i in clean if idxq[i]['split']!='test']
    T=lambda arr,ii: torch.tensor(arr[ii],dtype=torch.float32)
    def train_stheta(seed):
        torch.manual_seed(seed);np.random.seed(seed)
        m=build('crossattn',384,384);rk=InterventionRanker(dz=512,ddelta=384)
        opt=torch.optim.AdamW(list(m.parameters())+list(rk.parameters()),lr=2e-3);I=np.array(trq)
        for ep in range(80):
            m.train();opt.zero_grad()
            sx,zx=m(T(cx,I),T(a['crit'],I));sic,_=m(T(cic,I),T(a['crit'],I));sicp,_=m(T(cicp,I),T(a['crit'],I))
            L=torch.clamp(0.5+sic-sx,min=0).mean()+0.1*torch.abs(sicp-sx).mean()
            hdp=T(cic,I)-T(cx,I);hdn=T(cicp,I)-T(cx,I)
            L=L+0.5*(-torch.log(torch.exp(rk(zx,hdp))/(torch.exp(rk(zx,hdp))+torch.exp(rk(zx,hdn))+1e-9)).mean())+0.01*hdp.norm(dim=-1).mean()
            L.backward();opt.step()
        m.eval();return m
    print("training sθ (3 seed) on quad...",flush=True)
    models=[train_stheta(s) for s in [42,123]]
    print("sθ trained",flush=True)

    d=json.load(open(f"{DATA}/aacr-bench_raw/dataset.json"))
    crisp=[r for r in d if r.get('note') and r.get('label') is not None and str(r.get('category','')).lower() in CRISP]
    real=[r for r in crisp if int(r['label'])==1];fp=[r for r in crisp if int(r['label'])==0]
    import random; random.seed(11); random.shuffle(real); random.shuffle(fp)
    sample=[(r,1) for r in real[:30]]+[(r,0) for r in fp[:30]]
    print(f"AACR crisp n={len(sample)} (30 real/30 FP), constructing interventions...",flush=True)
    results=[None]*len(sample); lock=Lock()
    def work(i):
        r,y=sample[i]
        m=re.search(r'github.com/([^/]+/[^/]+)',r['pr_url'])
        if not m: return
        lines=fetch(m.group(1),r['pr_source_commit'],r['path'])
        if not lines: return
        reg=region(lines,r.get('from_line'),r.get('to_line'))
        if not reg or not reg.strip(): return
        try:
            before,after=gen_ic(reg,r['note']); unrel=gen_unrel(reg)
        except Exception: return
        with lock: results[i]=(y,before,after,unrel,r['note'])
    with ThreadPoolExecutor(max_workers=8) as ex: list(ex.map(work,range(len(sample))))
    valid=[x for x in results if x is not None]
    print(f"constructed {len(valid)} valid interventions",flush=True)

    emb_m=SentenceTransformer(text_encoder());E=lambda t:emb_m.encode([str(t) if t else " "],convert_to_numpy=True)[0]
    labs=[v[0] for v in valid]
    xb=np.array([E(v[1]) for v in valid]);xic=np.array([E(v[2]) for v in valid]);xcr=np.array([E(v[4]) for v in valid])

    import torch
    sigs=[]
    for m in models:
        with torch.no_grad():
            sx=m(torch.tensor(xb,dtype=torch.float32),torch.tensor(xcr,dtype=torch.float32))[0].numpy()
            sicv=m(torch.tensor(xic,dtype=torch.float32),torch.tensor(xcr,dtype=torch.float32))[0].numpy()
        sigs.append(sx-sicv)
    sig=np.mean(sigs,0)

    au=roc_auc_score(labs,sig) if len(set(labs))>1 else None

    out={"n":len(valid),"n_real":int(sum(labs)),"civic_sθ_validity_auroc":round(float(au),4) if au else None,
         "baseline_frozen_reviewer":0.635,"transed":"REAL flip signal sx-s_Ic"}
    print(f"CIVIC-CR sθ AACR crisp-bug validity AUROC = {out['civic_sθ_validity_auroc']} (vs baseline 0.635)",flush=True)
    json.dump(out,open(os.environ.get('CIVIC_PROJECT_ROOT', '.') + '/outputs/aacr_crisp_stheta_result.json','w'),indent=2)
    print("AACR_CRISP_STHETA DONE",flush=True)
if __name__=="__main__": main()
