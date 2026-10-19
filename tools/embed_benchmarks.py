from civic.paths import OUTPUT_DIR
pass

import os, json, argparse, numpy as np
HERE=str(OUTPUT_DIR); DATA=os.environ.get("CIVIC_PROJECT_ROOT", ".") + "/data"; CACHE=os.environ.get("CIVIC_PROJECT_ROOT", ".") + "/cache"

def _aacr_repo(r):
    import re
    m=re.search(r'github.com/([^/]+/[^/]+)',r.get("pr_url","") or "")
    return m.group(1) if m else "unknown"

def load_aacr():
    import hashlib
    d=json.load(open(f"{DATA}/aacr-bench_raw/dataset.json"))
    items=[]
    for i,r in enumerate(d):
        if r.get("note") and r.get("label") is not None:
            repo=_aacr_repo(r)

            h=int(hashlib.sha256(repo.encode()).hexdigest(),16)%100
            split="train" if h<71 else ("val" if h<81 else "test")
            items.append({"id":f"aacr_{i}","crit":str(r["note"]),"code":str(r.get("context","")),
                          "label":int(r["label"]),"category":str(r.get("category","")),"is_ai":bool(r.get("is_ai_comment")),
                          "repo":repo,"split":split})
    return items

def load_swepr():

    import glob
    items=[]
    for f in glob.glob(f"{DATA}/swe-prbench_raw/dataset/annotations/*_human.json"):
        try: ann=json.load(open(f))
        except: continue
        pr=os.path.basename(f).replace("_human.json","")

        findings=ann if isinstance(ann,list) else ann.get("findings",ann.get("comments",[]))
        if not isinstance(findings,list): continue
        for j,fd in enumerate(findings):
            if not isinstance(fd,dict): continue
            crit=fd.get("comment",fd.get("note",fd.get("text","")))
            lab=fd.get("label",fd.get("verdict",fd.get("category","")))

            y=0 if str(lab).upper() in ("FABRICATED","UNSUPPORTED","FALSE","INVALID") else (1 if lab else None)
            if crit and y is not None:
                items.append({"id":f"swepr_{pr}_{j}","crit":str(crit),"code":str(fd.get("context",fd.get("code",""))),"label":y,"category":str(lab)})
    return items

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--bench",default="aacr"); ap.add_argument("--model",default="all-MiniLM-L6-v2")
    args=ap.parse_args()
    if args.bench=="aacr": items=load_aacr()
    elif args.bench=="swepr": items=load_swepr()
    else: print("crb loader TBD (crscore relevance-score format, 需先探测)"); return
    print(f"{args.bench}: {len(items)} findings, y1={sum(1 for x in items if x['label']==1)} y0={sum(1 for x in items if x['label']==0)}",flush=True)
    if not items: print("no items, abort"); return
    from sentence_transformers import SentenceTransformer
    m=SentenceTransformer(args.model)
    crit=m.encode([x["crit"][:2000] for x in items],batch_size=64,convert_to_numpy=True,show_progress_bar=False)
    code=m.encode([x["code"][:2000] for x in items],batch_size=64,convert_to_numpy=True,show_progress_bar=False)
    labels=np.array([x["label"] for x in items])
    np.savez(f"{CACHE}/bench_{args.bench}_emb.npz",crit=crit,code=code,labels=labels)
    with open(f"{CACHE}/bench_{args.bench}_index.jsonl","w") as f:
        for x in items: f.write(json.dumps({k:x.get(k) for k in ('id','label','category','repo','split')})+"\n")
    from collections import Counter
    print("split dist:",Counter(x.get('split') for x in items),flush=True)
    print("split x label:",{s:Counter(x['label'] for x in items if x.get('split')==s) for s in ('train','val','test')},flush=True)
    print(f"DONE embed {args.bench}: crit{crit.shape} code{code.shape} labels{labels.shape}",flush=True)

if __name__=="__main__": main()
