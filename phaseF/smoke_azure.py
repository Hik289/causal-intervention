pass

import os, sys, json, time, numpy as np, urllib.request
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from smoke_unixcoder import get_regions, encode_minilm
AZ_URL=os.environ.get("AZURE_EMBEDDING_ENDPOINT", "")
AZ_KEY=os.environ.get("AZURE_EMBEDDING_API_KEY", "")

def az_embed(text, retries=5):
    body=json.dumps({"model":"text-embedding-3-small","input":text[:8000]}).encode()
    for att in range(retries):
        try:
            req=urllib.request.Request(AZ_URL,data=body,headers={"api-key":AZ_KEY,"Content-Type":"application/json"})
            with urllib.request.urlopen(req,timeout=40) as r:
                d=json.loads(r.read())
                return np.array(d["data"][0]["embedding"],dtype=np.float32)
        except Exception as e:
            if att<retries-1: time.sleep(2*(att+1)); continue
            raise
    return None

def encode_azure(texts):
    embs=[]
    for i,t in enumerate(texts):
        embs.append(az_embed(t)); time.sleep(0.3)
        if (i+1)%20==0: print(f"  azure {i+1}/{len(texts)}",flush=True)
    return np.array(embs)

def sep(enc, trips, name, ref=""):
    xs=[t[0] for t in trips]; ics=[t[1] for t in trips]; icps=[t[2] for t in trips]
    hx=enc(xs); hic=enc(ics); hicp=enc(icps)
    def cos(u,v): return (u*v).sum(1)/(np.linalg.norm(u,axis=1)*np.linalg.norm(v,axis=1)+1e-9)
    ct=cos(hx,hic); cu=cos(hx,hicp)
    tdev=1-ct.mean(); udev=1-cu.mean()
    print(f"=== {name} (dim={hx.shape[1]}) ===",flush=True)
    print(f"  cos(x,I_c) target mean={ct.mean():.4f} → 偏离 {tdev:.4f} {ref}",flush=True)
    print(f"  cos(x,I_c') unrel mean={cu.mean():.4f} → 偏离 {udev:.4f} (disjoint 应≈0)",flush=True)
    print(f"  target 可见 (cos<0.999): {int((ct<0.999).sum())}/{len(ct)}",flush=True)
    return tdev, udev

def main():
    trips=get_regions(40)
    print(f"got {len(trips)} quad region triples",flush=True)
    if len(trips)<10: print("too few"); return
    t_mini,_=sep(encode_minilm, trips, "all-MiniLM", "(参照偏离 0.041)")
    t_az,u_az=sep(encode_azure, trips, "Azure text-embedding-3-small", "(参照 all-MiniLM 0.041 / UniXcoder 0.061)")
    print(f"\n判据: Azure target 偏离 {t_az:.4f} vs all-MiniLM {t_mini:.4f} —— ratio {t_az/(t_mini+1e-9):.2f}x",flush=True)
    print("SMOKE_AZURE DONE",flush=True)

if __name__=="__main__": main()
