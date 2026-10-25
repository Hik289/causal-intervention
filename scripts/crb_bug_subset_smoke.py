from civic.services import chat_client, review_model
pass

import os, sys, json, re, urllib.request, difflib, time, statistics as st
import pandas as pd
client=chat_client()
MODEL=review_model()
DATA=os.environ.get("CIVIC_PROJECT_ROOT", ".") + "/data"
CRISP={"bug","performance","security"}

def fetch_lines(repo,sha,path,line,win=20):
    url=f"https://raw.githubusercontent.com/{repo}/{sha}/{path}"
    try:
        req=urllib.request.Request(url,headers={"User-Agent":"civic"})
        c=urllib.request.urlopen(req,timeout=20).read().decode("utf-8","ignore").split("\n")
    except Exception: return None
    if line and str(line).isdigit():
        li=int(line); lo=max(0,li-win); hi=min(len(c),li+win); return "\n".join(c[lo:hi])
    return "\n".join(c[:2*win])

def sol(msg):
    r=client.chat.completions.create(model=MODEL, messages=[{'role': 'user', 'content': msg}])
    return r.choices[0].message.content

def locate_gen(region, note):
    p=f"""下面是一段代码和一条 code review 评论。评论指出该位置*可能*存在问题。

保持中立 (不假设评论一定对):
- 若存在真实问题, 返回该区域**修复后**版本。
- 若代码已正确 (误报/主观/不适用), 返回**原样** (no-op)。

严格格式 (只两块, 不解释):
```before
<原始代码区域>
```
```after
<修复后; 无需修则与 before 一致>
```

评论: {note[:600]}

代码:
```
{region[:3000]}
```"""
    out=sol(p)
    mb=re.search(r'```before\n?(.*?)```',out,re.S); ma=re.search(r'```after\n?(.*?)```',out,re.S)
    return (mb.group(1).strip() if mb else ""),(ma.group(1).strip() if ma else "")

def gen_unrel(before):
    p=f"对下面代码做一个与 bug 无关的表面修改 (加注释), 返回修改后代码, ```包裹```。\n```\n{before[:3000]}\n```"
    out=sol(p); m=re.search(r'```[a-z]*\n?(.*?)```',out,re.S)
    return (m.group(1) if m else out).strip()

def diff_size(a,b):
    sm=difflib.SequenceMatcher(None,a.split("\n"),b.split("\n"))
    return sum((i2-i1)+(j2-j1) for tag,i1,i2,j1,j2 in sm.get_opcodes() if tag!='equal')

def main():
    df=pd.read_parquet(f"{DATA}/codereviewbench_raw/human_annotations_task2.parquet")
    real=[]; fp=[]
    for _,r in df.iterrows():
        bc=r['bot_comment']
        try: bc=json.loads(bc) if isinstance(bc,str) else bc
        except: continue
        cat=str(bc.get('category','')).lower()
        if cat not in CRISP: continue
        fa=str(r.get('human_factual_accuracy',''))
        y=1 if 'Yes' in fa else (0 if 'No' in fa else None)
        if y is None: continue
        m=re.search(r'github.com/([^/]+/[^/]+)',str(r.get('pr_url','')))
        if not m: continue
        item={"note":bc.get('description',''),"repo":m.group(1),"path":bc.get('file_path',''),
              "line":bc.get('line_number'),"cat":cat,"task_id":str(r.get('task_id',''))}
        (real if y else fp).append(item)
    import random; random.seed(7); random.shuffle(real); random.shuffle(fp)
    sample=[(x,1) for x in real[:10]]+[(x,0) for x in fp[:10]]
    print(f"CRB crisp-bug 子集: {len(real)} REAL / {len(fp)} FP (category∈{CRISP})",flush=True)
    out=[]
    for item,y in sample:

        region=None
        for sha in ['HEAD','main','master']:
            region=fetch_lines(item['repo'],sha,item['path'],item['line'])
            if region and region.strip(): break
        if not region or not region.strip(): print(f"  SKIP fetch {item['path'][:40]}",flush=True); continue
        try:
            before,after=locate_gen(region,item['note']); time.sleep(0.3)
            icp=gen_unrel(before) if before else ""; time.sleep(0.3)
        except Exception as e: print(f"  SKIP gen {e}",flush=True); continue
        ds_ic=diff_size(before,after); ds_icp=diff_size(before,icp)
        out.append({"y":y,"cat":item['cat'],"ic_diff":ds_ic,"icp_diff":ds_icp,"before":before,"after":after,"note":item['note'][:150]})
        print(f"  [{'REAL' if y else 'FP  '}] cat={item['cat']:11s} ic_diff={ds_ic:3d} icp_diff={ds_icp:3d} | {item['note'][:45]}",flush=True)
    json.dump(out,open(os.environ.get("CIVIC_PROJECT_ROOT", ".") + "/cache/crb_bug_smoke.json","w"))
    rd=[o['ic_diff'] for o in out if o['y']==1]; fd=[o['ic_diff'] for o in out if o['y']==0]
    print(f"\n== CRB crisp-bug 子集三检查 ==",flush=True)
    if rd: print(f"① REAL(bug类) I_c diff: mean={st.mean(rd):.1f} median={st.median(rd)} (假设: bug 类应有实质 diff)",flush=True)
    if fd:
        noop=sum(1 for x in fd if x<=2); print(f"② FP I_c diff: mean={st.mean(fd):.1f}; FP no-op(≤2)={noop}/{len(fd)} (泄漏护栏)",flush=True)
    print("CRB_BUG_SMOKE DONE",flush=True)

if __name__=="__main__": main()
