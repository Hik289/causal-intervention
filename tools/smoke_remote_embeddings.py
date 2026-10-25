from civic.services import remote_embedding
pass

import os, sys, json, time, numpy as np, urllib.request
from tools.compare_code_embeddings import get_regions, encode_text

def az_embed(text, retries=5):
    return np.asarray(remote_embedding(text), dtype=np.float32)

def encode_remote(texts):
    embs=[]
    for i,t in enumerate(texts):
        embs.append(az_embed(t)); time.sleep(0.3)
        if (i+1)%20==0: print(f"  remote {i+1}/{len(texts)}",flush=True)
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
    t_mini,_=sep(encode_text, trips, "Text encoder", "(参照偏离 0.041)")
    t_az,u_az=sep(encode_remote, trips, "Remote encoder", "(参照 Text encoder 0.041 / Code encoder 0.061)")
    print(f"\n判据: Remote target 偏离 {t_az:.4f} vs Text encoder {t_mini:.4f} —— ratio {t_az/(t_mini+1e-9):.2f}x",flush=True)
    print("SMOKE_REMOTE_EMBEDDINGS DONE",flush=True)

if __name__=="__main__": main()
