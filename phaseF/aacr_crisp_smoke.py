pass

import os
import json, os, re, urllib.request, difflib, time, statistics as st
from openai import OpenAI
DATA=os.environ.get("CIVIC_PROJECT_ROOT", ".") + "/data"
client=OpenAI(base_url=os.environ["OPENAI_BASE_URL"],api_key=os.environ["OPENAI_API_KEY"],max_retries=6)
MODEL=os.environ.get("LLM_MODEL_NAME", "gpt-4")
CRISP={'code defect','security vulnerability','performance'}
def fetch(repo,sha,path):
    url=f"https://raw.githubusercontent.com/{repo}/{sha}/{path}"
    try:
        req=urllib.request.Request(url,headers={"User-Agent":"civic"})
        return urllib.request.urlopen(req,timeout=20).read().decode("utf-8","ignore").split("\n")
    except Exception: return None
def region(lines, fl, tl, win=12):
    if not lines: return None
    try: fl=int(fl); tl=int(tl)
    except: fl=tl=0
    lo=max(0,fl-win); hi=min(len(lines),(tl or fl)+win)
    return "\n".join(lines[lo:hi]) if lines else None
def sol(msg):
    r=client.chat.completions.create(model=MODEL,messages=[{"role":"user","content":msg}],timeout=150)
    return r.choices[0].message.content or ""
def locate_gen(code_region, note):
    p=f"""下面是一段代码和一条 code review 评论 (指出可能的缺陷)。保持中立:
- 若存在真实问题, 返回修复后的代码区域。
- 若代码已正确 (评论误报/不适用), 原样返回 (no-op)。
严格格式 (只两块, 不解释):
```before
<原始代码区域, 逐字复制>
```
```after
<修复后; 无需修则与 before 完全一致>
```
评论: {note[:700]}
代码:
```
{code_region[:3000]}
```"""
    out=sol(p)
    mb=re.search(r'```before\n?(.*?)```',out,re.S);ma=re.search(r'```after\n?(.*?)```',out,re.S)
    return (mb.group(1).strip() if mb else ""),(ma.group(1).strip() if ma else "")
def diff_size(x,y):
    sm=difflib.SequenceMatcher(None,x.split("\n"),y.split("\n"))
    return sum((i2-i1)+(j2-j1) for t,i1,i2,j1,j2 in sm.get_opcodes() if t!='equal')
def main():
    d=json.load(open(f"{DATA}/aacr-bench_raw/dataset.json"))
    crisp=[r for r in d if r.get('note') and r.get('label') is not None and str(r.get('category','')).lower() in CRISP and r.get('pr_source_commit') and r.get('path')]
    real=[r for r in crisp if int(r['label'])==1]; fp=[r for r in crisp if int(r['label'])==0]
    import random; random.seed(7); random.shuffle(real); random.shuffle(fp)
    print(f"AACR crisp subset: {len(crisp)} ({len(real)} real/{len(fp)} FP). smoke 10+10.",flush=True)
    sample=[(r,1) for r in real[:10]]+[(r,0) for r in fp[:10]]
    out=[]
    for r,y in sample:
        m=re.search(r'github.com/([^/]+/[^/]+)',r['pr_url'])
        if not m: print("  SKIP no repo",flush=True); continue
        repo=m.group(1); lines=fetch(repo,r['pr_source_commit'],r['path'])
        if not lines: print(f"  SKIP fetch {r['path'][:40]}",flush=True); continue
        reg=region(lines,r.get('from_line'),r.get('to_line'))
        if not reg or not reg.strip(): print("  SKIP empty region",flush=True); continue
        try:
            before,after=locate_gen(reg,r['note']); time.sleep(0.3)
        except Exception as e: print(f"  SKIP gen {e}",flush=True); continue
        ds=diff_size(before,after)

        rl=set(l.strip() for l in reg.split("\n") if l.strip()); bl=set(l.strip() for l in before.split("\n") if l.strip())
        san=round(len(bl&rl)/max(1,len(bl)),2) if bl else 0.0
        out.append({"label":y,"cat":r['category'],"ic_diff":ds,"sanity":san,"note":r['note'][:150]})
        print(f"  [{'REAL' if y else 'FP  '}] ic_diff={ds:3d} sanity={san} cat={r['category'][:20]} | {r['note'][:45]}",flush=True)
    json.dump(out,open(os.environ.get('CIVIC_PROJECT_ROOT', '.') + '/cache/aacr_crisp_smoke.json','w'))
    rd=[o['ic_diff'] for o in out if o['label']==1];fd=[o['ic_diff'] for o in out if o['label']==0]
    san=[o['sanity'] for o in out if o['label']==1]
    print(f"\n== AACR crisp smoke 三检查 ==",flush=True)
    if rd: print(f"① REAL I_c diff: mean={st.mean(rd):.2f} median={st.median(rd)} (crisp 应有 diff)",flush=True)
    if fd: print(f"② FP I_c diff: mean={st.mean(fd):.2f} no-op(<=2)={sum(1 for x in fd if x<=2)}/{len(fd)} (泄漏护栏)",flush=True)
    if san: print(f"③ REAL 定位 sanity: mean={st.mean(san):.2f} (from_line 区命中)",flush=True)
    print("AACR_CRISP_SMOKE DONE",flush=True)
if __name__=="__main__": main()
