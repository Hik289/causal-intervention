import os
import sys; from scripts.run_3arm import llm_judge_match, executable_verify, dbl
import pandas as pd, glob, subprocess
df=pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.environ.get('CIVIC_PROJECT_ROOT', '.') + '/data/swe-review-bench_raw/data/*.parquet'))],ignore_index=True)
res=df[df.resolved==True].groupby('instance_id').first().reset_index()
cases={
 'sphinx-doc__sphinx-9320':("type annotations merged wrong by record_typehints","autodoc a function def foo(x:int)->bool, assert rtype"),
 'pydata__xarray-3095':("shallow copy deep=False shares _data so mutating copy._attrs also mutates original","call variable.copy(deep=False), modify copy _attrs, assert original unchanged"),
}
for iid,(mech,hint) in cases.items():
    r=res[res.instance_id==iid].iloc[0]; img=r.image_name; f2p=dbl(r.FAIL_TO_PASS)
    if img not in subprocess.run("docker images --format '{{.Repository}}:{{.Tag}}'",shell=True,capture_output=True,text=True).stdout:
        subprocess.run(f"docker pull {img}",shell=True,capture_output=True,timeout=1800)
    anchor,tsrc=executable_verify(img,r.repo,r.model_patch,getattr(r,'test_patch',''),f2p,timeout=900)
    f2p_str="; ".join(str(x) for x in f2p[:4])
    j=llm_judge_match(mech,hint,f2p_str,tsrc)
    print(f"{iid}: anchor={anchor} tsrc_len={len(tsrc)} judge={j}")
    print(f"   tsrc_head={tsrc[:200]!r}")
