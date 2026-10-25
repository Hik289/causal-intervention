from civic.services import chat_client, review_model
pass

import os, sys, json, re, urllib.request, difflib, time, glob, statistics as st
client=chat_client()
MODEL=review_model()
DATA=os.environ.get("CIVIC_PROJECT_ROOT", ".") + "/data"

def sol(msg):
    r=client.chat.completions.create(model=MODEL, messages=[{'role': 'user', 'content': msg}])
    return r.choices[0].message.content

def hunk_region(diff_hunk):
    pass
    lines=[]
    for ln in diff_hunk.split("\n"):
        if ln.startswith("@@"): continue
        if ln.startswith("+"): continue
        lines.append(ln[1:] if ln and ln[0] in " -" else ln)
    return "\n".join(lines)

def hunk_changed(diff_hunk):
    pass
    return set(ln[1:].strip() for ln in diff_hunk.split("\n") if ln.startswith("-") and not ln.startswith("---") and ln[1:].strip())

def locate_gen(region, note):
    pass
    p=f"""下面是一段代码和一条 code review 评论。评论指出该位置*可能*存在问题。

保持中立 (不要假设评论一定对):
- 若存在真实问题, 返回该区域**修复后**版本。
- 若代码已正确 (评论是误报/主观偏好/不适用), 返回该区域**原样** (no-op)。

严格格式 (只返回两块, 不解释):
```before
<原始代码区域, 逐字复制>
```
```after
<修复后; 若无需修复则与 before 完全一致>
```

评论: {note[:600]}

代码区域:
```
{region[:3000]}
```"""
    out=sol(p)
    mb=re.search(r'```before\n?(.*?)```',out,re.S); ma=re.search(r'```after\n?(.*?)```',out,re.S)
    return (mb.group(1).strip() if mb else ""), (ma.group(1).strip() if ma else "")

def gen_unrel(before):
    p=f"""对下面代码做一个与任何 bug 无关的表面修改 (加一行注释即可), 返回修改后代码。只返回 ```代码```。
```
{before[:3000]}
```"""
    out=sol(p); m=re.search(r'```[a-z]*\n?(.*?)```',out,re.S)
    return (m.group(1) if m else out).strip()

def diff_size(a,b):
    sm=difflib.SequenceMatcher(None,a.split("\n"),b.split("\n"))
    return sum((i2-i1)+(j2-j1) for tag,i1,i2,j1,j2 in sm.get_opcodes() if tag!='equal')

def load_swepr(n):
    fs=glob.glob(f"{DATA}/swe-prbench_raw/**/*human*.json",recursive=True)
    real=[]; fp=[]
    for f in fs:
        try: d=json.load(open(f))
        except: continue
        for c in d.get('comments',[]):
            dh=c.get('diff_hunk')
            if not dh or dh=='None' or not c.get('body'): continue
            rc=str(c.get('requires_change'))=='True'
            item={"note":c['body'],"diff_hunk":dh,"y":1 if rc else 0}
            (real if rc else fp).append(item)
    import random; random.seed(7); random.shuffle(real); random.shuffle(fp)
    return [(x,1) for x in real[:n//2]]+[(x,0) for x in fp[:n//2]]

def load_crb(n):
    import pandas as pd
    df=pd.read_parquet(f"{DATA}/codereviewbench_raw/human_annotations_task2.parquet")
    items=[]
    for _,r in df.iterrows():
        bc=r['bot_comment']
        try: bc=json.loads(bc) if isinstance(bc,str) else bc
        except: continue
        desc=bc.get('description','') if isinstance(bc,dict) else str(bc)
        dl=str(r.get('diff_lines',''))
        fa=str(r.get('human_factual_accuracy',''))
        y=1 if 'Yes' in fa else (0 if 'No' in fa else None)
        if y is None or not desc: continue
        items.append({"note":desc,"region":dl,"y":y})
    import random; random.seed(7); random.shuffle(items)
    real=[x for x in items if x['y']==1][:n//2]; fp=[x for x in items if x['y']==0][:n//2]
    return [(x,1) for x in real]+[(x,0) for x in fp]

def main():
    import argparse; ap=argparse.ArgumentParser(); ap.add_argument("--bench",required=True); ap.add_argument("--n",type=int,default=20)
    args=ap.parse_args()
    sample=load_swepr(args.n) if args.bench=="swepr" else load_crb(args.n)
    out=[]
    for item,y in sample:
        if args.bench=="swepr":
            region=hunk_region(item['diff_hunk']); changed=hunk_changed(item['diff_hunk'])
        else:
            region=item['region']; changed=set()
        if not region.strip(): print("  SKIP empty region",flush=True); continue
        try:
            before,after=locate_gen(region,item['note']); time.sleep(0.3)
            icp=gen_unrel(before) if before else ""; time.sleep(0.3)
        except Exception as e: print(f"  SKIP gen {e}",flush=True); continue
        ds_ic=diff_size(before,after); ds_icp=diff_size(before,icp)

        sanity=None
        if y==1 and changed and before:
            bl=set(l.strip() for l in before.split("\n") if l.strip())
            sanity=round(len(bl & changed)/max(1,len(changed)),2)
        out.append({"y":y,"ic_diff":ds_ic,"icp_diff":ds_icp,"sanity":sanity,
                    "before":before,"after":after,"note":item['note'][:200]})
        print(f"  [{'REAL' if y else 'FP  '}] ic_diff={ds_ic:3d} icp_diff={ds_icp:3d} sanity={sanity} | {item['note'][:50]}",flush=True)
    json.dump(out,open(fos.environ.get("CIVIC_PROJECT_ROOT", ".") + "/cache/{args.bench}_smoke.json","w"))
    rd=[o['ic_diff'] for o in out if o['y']==1]; fd=[o['ic_diff'] for o in out if o['y']==0]
    san=[o['sanity'] for o in out if o['y']==1 and o['sanity'] is not None]
    print(f"\n== {args.bench.upper()} 三检查 ==",flush=True)
    if rd: print(f"① REAL I_c diff: mean={st.mean(rd):.1f} median={st.median(rd)} (应有实质 diff)",flush=True)
    if fd:
        noop=sum(1 for x in fd if x<=2); print(f"② FP I_c diff: mean={st.mean(fd):.1f}; FP no-op(≤2)={noop}/{len(fd)} (泄漏护栏)",flush=True)
    if san: print(f"③ REAL sanity (sol before ∩ diff_hunk 改动): mean={st.mean(san):.2f} (SWEPR 应 >>0.25)",flush=True)
    print(f"{args.bench.upper()}_SMOKE DONE",flush=True)

if __name__=="__main__": main()
