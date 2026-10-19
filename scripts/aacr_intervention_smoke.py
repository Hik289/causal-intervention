pass

import os, sys, json, re, urllib.request, difflib, time
from openai import OpenAI
AACR=os.environ.get("CIVIC_PROJECT_ROOT", ".") + "/data/aacr-bench_raw/dataset.json"
client=OpenAI(base_url=os.environ["OPENAI_BASE_URL"], api_key=os.environ["OPENAI_API_KEY"], max_retries=6)
MODEL=os.environ.get("LLM_MODEL_NAME", "gpt-4")

def fetch(repo,sha,path):
    url=f"https://raw.githubusercontent.com/{repo}/{sha}/{path}"
    try:
        req=urllib.request.Request(url,headers={"User-Agent":"civic"})
        return urllib.request.urlopen(req,timeout=20).read().decode("utf-8","ignore")
    except Exception: return None

def region_around(code, note, win=40):
    pass
    syms=re.findall(r'`([A-Za-z_][\w:]+)`',note)
    lines=code.split("\n")
    for s in syms:
        for i,l in enumerate(lines):
            if s in l:
                lo=max(0,i-win//2); hi=min(len(lines),i+win//2)
                return "\n".join(lines[lo:hi]), lo

    return "\n".join(lines[:win]), 0

def sol(msg):
    r=client.chat.completions.create(model=MODEL,messages=[{"role":"user","content":msg}],timeout=120)
    return r.choices[0].message.content

def gen_ic_blind(region, note):
    pass
    p=f"""下面是一段代码和一条 code review 评论。评论指出该位置*可能*存在问题。

你的任务: 判断该评论指出的问题**是否真实存在于代码中**。
- 如果存在真实问题, 返回修复后的代码区域。
- 如果代码**已经正确** (评论是误报/主观偏好/不适用), **原样返回代码, 不做任何修改** (no-op)。

只返回代码区域本身 (修复后或原样), 不要解释。用 ```代码``` 包裹。

评论: {note[:600]}

代码区域:
```
{region[:3000]}
```"""
    out=sol(p)
    m=re.search(r'```[a-z]*\n?(.*?)```',out,re.S)
    return (m.group(1) if m else out).strip()

def gen_icprime(region):
    pass
    p=f"""下面是一段代码。请做一个**与任何 bug 无关的表面修改** (比如给某行加一句注释, 或微调空白), 返回修改后的代码区域。不要改任何逻辑。只返回代码, ```包裹```。

代码:
```
{region[:3000]}
```"""
    out=sol(p)
    m=re.search(r'```[a-z]*\n?(.*?)```',out,re.S)
    return (m.group(1) if m else out).strip()

def diff_size(a,b):
    pass
    sm=difflib.SequenceMatcher(None,a.split("\n"),b.split("\n"))
    ch=sum((i2-i1)+(j2-j1) for tag,i1,i2,j1,j2 in sm.get_opcodes() if tag!='equal')
    return ch

def main():
    d=json.load(open(AACR))
    real=[r for r in d if r.get('label')==1 and r.get('note') and r.get('pr_source_commit')]
    fp=[r for r in d if r.get('label')==0 and r.get('note') and r.get('pr_source_commit')]
    import random; random.seed(42); random.shuffle(real); random.shuffle(fp)
    sample=[(r,1) for r in real[:10]]+[(r,0) for r in fp[:10]]
    out=[]
    for r,y in sample:
        repo=re.search(r'github.com/([^/]+/[^/]+)/pull',r['pr_url'])
        if not repo: continue
        repo=repo.group(1)
        code=fetch(repo,r['pr_source_commit'],r['path'])
        if not code: print(f"  SKIP fetch fail {r['path']}",flush=True); continue
        region,lo=region_around(code,r['note'])
        try:
            ic=gen_ic_blind(region,r['note']); time.sleep(0.3)
            icp=gen_icprime(region); time.sleep(0.3)
        except Exception as e: print(f"  SKIP gen {e}",flush=True); continue
        ds_ic=diff_size(region,ic); ds_icp=diff_size(region,icp)
        out.append({"label":y,"path":r['path'],"ic_diff":ds_ic,"icp_diff":ds_icp,
                    "region":region,"ic":ic,"icp":icp,"note":r['note'][:300]})
        print(f"  [{'REAL' if y else 'FP  '}] ic_diff={ds_ic:3d} icp_diff={ds_icp:3d} {r['path'][:40]}",flush=True)
    json.dump(out,open(os.environ.get("CIVIC_PROJECT_ROOT", ".") + "/cache/aacr_smoke_interv.json","w"))

    rd=[o['ic_diff'] for o in out if o['label']==1]; fd=[o['ic_diff'] for o in out if o['label']==0]
    import statistics as st
    print(f"\n== 泄漏诊断 ==",flush=True)
    print(f"REAL finding I_c diff: mean={st.mean(rd):.1f} median={st.median(rd)} (应有实质 diff)",flush=True)
    print(f"FP   finding I_c diff: mean={st.mean(fd):.1f} median={st.median(fd)} (应 ≈no-op=0)",flush=True)
    fp_noop=sum(1 for x in fd if x<=2)
    print(f"FP 中 ≈no-op (diff<=2): {fp_noop}/{len(fd)} (越多越干净, 无泄漏)",flush=True)
    print(f"判据: FP 多数 no-op + REAL 有 diff → sol 没虚构 FP 修复 → 无泄漏, 可全量",flush=True)
    print("SMOKE DONE",flush=True)

if __name__=="__main__": main()
