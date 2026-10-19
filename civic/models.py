pass

import torch, torch.nn as nn

class MLPVerifier(nn.Module):
    def __init__(self, dc, dt, hidden=512, dropout=0.1):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(dc+dt, hidden), nn.ReLU(), nn.Dropout(dropout),
                               nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.head=nn.Linear(hidden,1)
    def forward(self, hx, hc):
        z=self.net(torch.cat([hx,hc],-1)); return self.head(z).squeeze(-1), z

class BilinearVerifier(nn.Module):
    def __init__(self, dc, dt, hidden=512):
        super().__init__()
        self.px=nn.Linear(dc,hidden); self.pc=nn.Linear(dt,hidden)
        self.W=nn.Bilinear(hidden,hidden,1)
    def forward(self, hx, hc):
        zx=self.px(hx); zc=self.pc(hc); return self.W(zx,zc).squeeze(-1), zx*zc

class CrossAttnVerifier(nn.Module):
    pass

    def __init__(self, dc, dt, hidden=512, layers=4, heads=8, dropout=0.1):
        super().__init__()
        self.px=nn.Linear(dc,hidden); self.pc=nn.Linear(dt,hidden)
        self.type_emb=nn.Parameter(torch.randn(2,hidden)*0.02)
        enc=nn.TransformerEncoderLayer(hidden,heads,hidden*2,dropout,batch_first=True)
        self.tr=nn.TransformerEncoder(enc,layers)
        self.head=nn.Linear(hidden,1)
    def forward(self, hx, hc):
        zx=self.px(hx)+self.type_emb[0]; zc=self.pc(hc)+self.type_emb[1]
        seq=torch.stack([zx,zc],1)
        z=self.tr(seq)
        pooled=z.mean(1)
        return self.head(pooled).squeeze(-1), pooled

def build(name, dc, dt, **kw):
    if name=="mlp": return MLPVerifier(dc,dt,**{k:v for k,v in kw.items() if k in ('hidden','dropout')})
    if name=="bilinear": return BilinearVerifier(dc,dt,hidden=kw.get('hidden',512))
    if name=="crossattn": return CrossAttnVerifier(dc,dt,**{k:v for k,v in kw.items() if k in ('hidden','layers','heads','dropout')})
    raise ValueError(name)

def n_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)
