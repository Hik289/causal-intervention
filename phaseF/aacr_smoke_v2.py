pass

import os, sys, json, re, urllib.request, difflib, time, statistics as st
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

def sol(msg):
    r=client.chat.completions.create(model=MODEL,messages=[{"role":"user","content":msg}],timeout=150)
    return r.choices[0].message.content

def locate_and_gen(file_content, note):
    pass

    fc=file_content[:14000]
    p=f"""下面是一个源代码文件和一条 code review 评论。评论指出某处*可能*存在问题。

你的任务 (保持中立, 不要假设评论一定对):
1. 在文件中定位评论所指的代码区域 (若能定位)。
2. 判断该区域**是否真的存在评论所述问题**:
   - 若存在真实问题: 返回该区域**修复后**的版本。
   - 若代码已正确 (评论是误报/主观偏好/不适用): 返回该区域**原样** (不改)。

严格按下面格式返回 (只返回这两个块, 不要解释):
```before
<你定位到的原始代码区域, 逐字复制>
```
```after
<修复后的该区域; 若无需修复则与 before 完全一致>
```

评论: {note[:800]}

文件内容:
```
{fc}
```"""
    out=sol(p)
    mb=re.search(r'```before\n?(.*?)```',out,re.S); ma=re.search(r'```after\n?(.*?)```',out,re.S)
    before=(mb.group(1).strip() if mb else ""); after=(ma.group(1).strip() if ma else "")
    return before, after

def gen_unrelated(before):
    pass
    p=f"""对下面代码做一个与任何 bug 无关的表面修改 (加一行注释即可), 返回修改后代码。只返回 ```代码```。
```
{before[:3000]}
```"""
    out=sol(p); m=re.search(r'```[a-z]*\n?(.*?)```',out,re.S)
    return (m.group(1) if m else out).strip()

def diff_size(a,b):
    sm=difflib.SequenceMatcher(None,a.split("\n"),b.split("\n"))
    return sum((i2-i1)+(j2-j1) for tag,i1,i2,j1,j2 in sm.get_opcodes() if tag!='equal')

def pr_changed_lines(src_code, tgt_code):
    pass
    sm=difflib.SequenceMatcher(None,src_code.split("\n"),tgt_code.split("\n"))
    ch=[]
    for tag,i1,i2,j1,j2 in sm.get_opcodes():
        if tag!='equal': ch.extend(src_code.split("\n")[i1:i2])
    return set(l.strip() for l in ch if l.strip())

def main():
    d=json.load(open(AACR))
    real=[r for r in d if r.get('label')==1 and r.get('note') and r.get('pr_source_commit') and r.get('pr_target_commit')]
    fp=[r for r in d if r.get('label')==0 and r.get('note') and r.get('pr_source_commit')]
    import random; random.seed(7); random.shuffle(real); random.shuffle(fp)
    sample=[(r,1) for r in real[:10]]+[(r,0) for r in fp[:10]]
    out=[]
    for r,y in sample:
        m=re.search(r'github.com/([^/]+/[^/]+)/pull',r['pr_url'])
        if not m: continue
        repo=m.group(1); code=fetch(repo,r['pr_source_commit'],r['path'])
        if not code: print(f"  SKIP fetch {r['path'][:40]}",flush=True); continue
        try:
            before,after=locate_and_gen(code,r['note']); time.sleep(0.3)
            icp=gen_unrelated(before) if before else ""; time.sleep(0.3)
        except Exception as e: print(f"  SKIP gen {e}",flush=True); continue
        ds_ic=diff_size(before,after); ds_icp=diff_size(before,icp)

        sanity=None
        if y==1:
            tgt=fetch(repo,r['pr_target_commit'],r['path'])
            if tgt and before:
                prlines=pr_changed_lines(code,tgt); blines=set(l.strip() for l in before.split("\n") if l.strip())
                ov=len(blines & prlines); sanity=round(ov/max(1,len(blines)),2)
        out.append({"label":y,"path":r['path'],"ic_diff":ds_ic,"icp_diff":ds_icp,"locate_sanity":sanity,
                    "before":before,"after":after,"icp":icp,"note":r['note'][:300]})
        print(f"  [{'REAL' if y else 'FP  '}] ic_diff={ds_ic:3d} icp_diff={ds_icp:3d} sanity={sanity} {r['path'][:38]}",flush=True)
    json.dump(out,open(os.environ.get("CIVIC_PROJECT_ROOT", ".") + "/cache/aacr_smoke_v2.json","w"))
    rd=[o['ic_diff'] for o in out if o['label']==1]; fd=[o['ic_diff'] for o in out if o['label']==0]
    san=[o['locate_sanity'] for o in out if o['label']==1 and o['locate_sanity'] is not None]
    print(f"\n== 三检查 ==",flush=True)
    if rd: print(f"① REAL I_c diff: mean={st.mean(rd):.1f} median={st.median(rd)} (应有实质 diff, 定位修好)",flush=True)
    if fd:
        fp_noop=sum(1 for x in fd if x<=2)
        print(f"② FP I_c diff: mean={st.mean(fd):.1f} median={st.median(fd)}; FP no-op(≤2)={fp_noop}/{len(fd)} (泄漏护栏)",flush=True)
    if san: print(f"③ REAL 定位 sanity (before∩PR改动区): mean={st.mean(san):.2f} (>0 = sol 定位落到真缺陷区)",flush=True)
    print(f"定位方式: REAL/FP 同一 prompt (sol 盲读整文件自定位), PR diff 仅 REAL sanity, 不作定位来源 → 无定位 artifact",flush=True)
    print("SMOKE_V2 DONE",flush=True)

if __name__=="__main__": main()
