pass

import torch; torch.set_num_threads(4)
import os, json, argparse, numpy as np, sys
sys.path.insert(0,os.environ.get("CIVIC_PROJECT_ROOT", ".") + "/phaseC")
from model import build, n_params
CACHE=os.environ.get("CIVIC_PROJECT_ROOT", ".") + "/cache"; HERE=os.path.dirname(os.path.abspath(__file__))

class Ranker(torch.nn.Module):
    def __init__(self,dz,dd,h=256):
        super().__init__(); self.net=torch.nn.Sequential(torch.nn.Linear(dz+dd,h),torch.nn.ReLU(),torch.nn.Linear(h,1))
    def forward(self,z,d): return self.net(torch.cat([z,d],-1)).squeeze(-1)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--seed",type=int,default=42); ap.add_argument("--epochs",type=int,default=150); ap.add_argument("--lr",type=float,default=2e-3)
    ap.add_argument("--w_flip",type=float,default=1.0); ap.add_argument("--w_spec",type=float,default=0.1)
    ap.add_argument("--w_rank",type=float,default=0.3); ap.add_argument("--w_cls",type=float,default=1.0)
    ap.add_argument("--margin",type=float,default=0.5); ap.add_argument("--out",default="unified.json")
    args=ap.parse_args(); torch.manual_seed(args.seed); np.random.seed(args.seed)
    a=np.load(f"{CACHE}/embfix_arrays.npz"); idx=[json.loads(l) for l in open(f"{CACHE}/embfix_index.jsonl")]
    cx,cic,cicp,cr=a["cx"],a["cic"],a["cicp"],a["crit"]
    def cosv(u,v): return (u*v).sum(1)/(np.linalg.norm(u,axis=1)*np.linalg.norm(v,axis=1)+1e-9)
    cp=cosv(cx,cicp); ci=cosv(cx,cic)

    clean=[i for i in range(len(idx)) if cp[i]>0.95 and ci[i]<0.999]
    def sp(s): return [i for i in clean if idx[i]["split"]==s]
    tr,va,te=sp("train"),sp("val"),sp("test")
    print(f"clean usable: train {len(tr)} val {len(va)} test {len(te)}",flush=True)
    dev="cpu"
    T=lambda k,ii: torch.tensor(a[k][ii],dtype=torch.float32)
    model=build("crossattn",cx.shape[1],cr.shape[1]); ranker=Ranker(512,cx.shape[1])
    opt=torch.optim.AdamW(list(model.parameters())+list(ranker.parameters()),lr=args.lr,weight_decay=1e-4)
    def batch(ii):
        return T("cx",ii),T("cic",ii),T("cicp",ii),T("crit",ii)
    tri=np.array(tr)
    for ep in range(args.epochs):
        model.train();ranker.train();opt.zero_grad()
        bx,bic,bicp,bc=batch(tri)
        sx,zx=model(bx,bc); sic,_=model(bic,bc); sicp,_=model(bicp,bc)

        s_pos=sx; s_neg=sic
        L_cls=torch.nn.functional.binary_cross_entropy_with_logits(
            torch.cat([s_pos,s_neg]), torch.cat([torch.ones_like(s_pos),torch.zeros_like(s_neg)]))
        L_flip=torch.clamp(args.margin+sic-sx,min=0).mean()
        L_spec=torch.abs(sicp-sx).mean()
        hdp=bic-bx; hdn=bicp-bx
        rp=ranker(zx,hdp); rn=ranker(zx,hdn)
        L_rank=-torch.log(torch.exp(rp)/(torch.exp(rp)+torch.exp(rn)+1e-9)).mean()
        loss=args.w_cls*L_cls+args.w_flip*L_flip+args.w_spec*L_spec+args.w_rank*L_rank
        loss.backward();opt.step()
        if ep%50==0: print(f"ep{ep} loss={loss.item():.4f} cls={L_cls.item():.3f} flip={L_flip.item():.3f} spec={L_spec.item():.3f}",flush=True)
    model.eval()
    def ev(ii):
        if not ii: return None
        I=np.array(ii)
        with torch.no_grad():
            bx,bic,bicp,bc=batch(I)
            sx,_=model(bx,bc); sic,_=model(bic,bc); sicp,_=model(bicp,bc)
        dt=(sx-sic); du=(sx-sicp).abs()

        from sklearn.metrics import roc_auc_score
        sc=torch.cat([torch.sigmoid(sx),torch.sigmoid(sic)]).numpy()
        y=np.concatenate([np.ones(len(ii)),np.zeros(len(ii))])
        au=roc_auc_score(y,sc)
        return {"n":len(ii),"Delta_target":round(float(dt.mean()),4),"Delta_unrel":round(float(du.mean()),4),
                "spec_ratio":round(float(dt.mean()/(1e-6+du.mean())),3),"flip_rate":round(float((sic<sx).float().mean()),3),
                "cls_auroc_x_vs_Ic":round(float(au),4)}
    res={"seed":args.seed,"w_spec":args.w_spec,"params":n_params(model)+n_params(ranker),
         "train":ev(tr),"val":ev(va),"test":ev(te)}
    print(json.dumps(res,ensure_ascii=False),flush=True)
    json.dump(res,open(f"{HERE}/{args.out}","w"),indent=2,ensure_ascii=False)
    print("UNIFIED DONE",flush=True)

if __name__=="__main__": main()
