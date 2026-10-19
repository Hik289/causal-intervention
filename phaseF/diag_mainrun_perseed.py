pass

import os
import sys
sys.path.insert(0,os.environ.get('CIVIC_PROJECT_ROOT', '.') + '/phaseF')
sys.path.insert(0,os.environ.get('CIVIC_PROJECT_ROOT', '.') + '/phaseC')
import quad_eseries as QE
import numpy as np
CFG={"flip_only":dict(flip=1.0,spec=0.0,rank=0.0),
     "flip_spec":dict(flip=1.0,spec=0.1,rank=0.0)}
for cfg,w in CFG.items():
    print(f"=== {cfg} (EXACT main-run path, with ranker in opt) ===",flush=True)
    aus=[]
    for s in [42,123,456]:
        r=QE.run(cfg,w,s)
        au=r['pair_auroc']; aus.append(au)
        print(f"  seed{s}: pair_auroc={au:.4f} flip={r['flip_rate']:.3f} Δtarget={r['dtarget']:.4f}",flush=True)
    print(f"  → mean={np.mean(aus):.4f} std={np.std(aus):.4f}",flush=True)
print("MAINRUN_PERSEED DONE",flush=True)
