from civic.paths import OUTPUT_DIR
pass

import os
import json, os, re, base64, subprocess, random
import pandas as pd, glob
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
DATA=os.environ.get("CIVIC_PROJECT_ROOT", ".") + "/data/swe-review-bench_raw/data"; PRE=os.environ.get("CIVIC_PROJECT_ROOT", ".") + "/data/anchors"
ARTIFACTS=str(OUTPUT_DIR); HERE=str(OUTPUT_DIR)
random.seed(42)
def dbl(x):
    x=json.loads(x) if isinstance(x,str) else x
    return json.loads(x) if isinstance(x,str) else x

def gpt(t=180):
    import openai; return openai.OpenAI(base_url=os.environ["OPENAI_BASE_URL"],api_key=os.environ["OPENAI_API_KEY"],max_retries=6,timeout=t)
def gpt_chat(system, user, t=180):
    pass
    r=gpt(t).chat.completions.create(model="gpt-5.5",
        messages=[{"role":"system","content":system},{"role":"user","content":user}])
    return r.choices[0].message.content or ""
def sh(c,t=120):
    r=subprocess.run(c,shell=True,capture_output=True,text=True,timeout=t); return r.returncode,r.stdout,r.stderr
import shutil as _sh
def free_gb(): return _sh.disk_usage("/data2")[2]//(1024**3)

def django_ids(p2p):
    out=[]
    for t in p2p:
        m=re.search(r'\(([\w.]+)\)',t)
        if m and "." in m.group(1): out.append(m.group(1))
        elif "::" in t:
            a=t.split("::"); mod=a[0].replace("tests/","").replace("/",".")
            if mod.endswith(".py"): mod=mod[:-3]
            out.append(mod+"."+".".join(a[1:]))
    return out
def _parse_outcome(out, is_dj):
    o=out.lower()
    if is_dj:
        if re.search(r'ran 0 test', o): return 'norun'
        if re.search(r'\bfailed\b.*\b(failures|errors)=', o) or re.search(r'failures=[1-9]|errors=[1-9]', o): return 'fail'
        if re.search(r'\nok\b|\bok\b\s*$', o) or re.search(r'ran \d+ test.*\n\s*ok', o): return 'pass'
        if 'traceback' in o and 'ran 0' not in o: return 'fail'
        return 'norun'
    if 'no tests ran' in o: return 'norun'
    m=re.search(r'(\d+) failed', o)
    if m and int(m.group(1))>0: return 'fail'
    if re.search(r'\d+ error', o) and 'passed' not in o and 'failed' not in o: return 'norun'
    if re.search(r'\d+ passed', o): return 'pass'
    return 'norun'

def resolution_check(img, repo, test_patch, f2p, repair_patch, timeout=900):
    pass

    is_dj="django" in repo
    ids=django_ids(f2p)[:4] if is_dj else f2p[:4]
    if is_dj and not ids: return ('norun',None)
    run=("python tests/runtests.py --settings=test_sqlite -v2 "+" ".join(ids)) if is_dj else ("python -m pytest "+" ".join("'"+s+"'" for s in ids)+" -rA -q")
    tp_b64=base64.b64encode((test_patch or "").encode()).decode()
    rp_b64=base64.b64encode((repair_patch or "").encode()).decode()
    rep=(f"echo {rp_b64} | base64 -d > /tmp/rep.patch && (git apply /tmp/rep.patch 2>/tmp/rep.err || patch -p1 --fuzz=3 < /tmp/rep.patch 2>>/tmp/rep.err) && echo '===REP_OK===' || echo '===REP_FAIL==='; ") if repair_patch else "echo '===REP_OK==='; "
    script=("source /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed && cd /testbed && "
            "git reset --hard -q && git clean -fdq && "
            + rep +
            f"echo {tp_b64} | base64 -d > /tmp/tp.patch && (git apply /tmp/tp.patch 2>/tmp/tp.err || patch -p1 < /tmp/tp.patch 2>>/tmp/tp.err) && echo '===TP_OK===' || echo '===TP_FAIL==='; "
            "echo '===RUN==='; "+run+" 2>&1 | tail -20")
    try:
        rc=subprocess.run(["docker","run","--rm",img,"bash","-c",script],capture_output=True,text=True,timeout=timeout)
        out=rc.stdout+rc.stderr
    except Exception: return ('norun',None,(0,0))
    rep_ok='===REP_OK===' in out
    if '===TP_OK===' not in out: return ('tp_fail',rep_ok,(0,0))
    res=out.split('===RUN===',1)[1] if '===RUN===' in out else out
    npass,nfail=_count_tests(res,is_dj)
    return (_parse_outcome(res,is_dj), rep_ok, (npass,nfail))

def _count_tests(out, is_dj):
    pass
    o=out.lower()
    if is_dj:

        npass=len(re.findall(r'\.\.\. ok',o)); nfail=len(re.findall(r'\.\.\. (fail|error)',o))
        if npass==0 and nfail==0:
            m=re.search(r'failures=(\d+)',o); e=re.search(r'errors=(\d+)',o); rn=re.search(r'ran (\d+) test',o)
            nfail=(int(m.group(1)) if m else 0)+(int(e.group(1)) if e else 0)
            if rn: npass=max(0,int(rn.group(1))-nfail)
        return npass,nfail

    p=re.search(r'(\d+) passed',o); f=re.search(r'(\d+) failed',o); e=re.search(r'(\d+) error',o)
    return (int(p.group(1)) if p else 0), (int(f.group(1)) if f else 0)+(int(e.group(1)) if e else 0)

REPAIR_SYS=("You are an independent software repair agent. You are given one or more buggy source files and a SPECIFIC defect "
 "critique written by someone else. Fix ONLY the specific defect described in the critique, at the location it indicates. "
 "Do NOT look for or fix any other issues. Do NOT refactor or reformat unrelated code. If the critique describes a defect you "
 "think does not actually exist, make the minimal change the critique implies anyway (fix what it describes). "
 "Express your fix as one or more SEARCH/REPLACE edits. For each edit output EXACTLY this format:\n"
 "```EDIT path/to/file.py\n<<<<<<< SEARCH\n<exact existing code to find, verbatim incl. indentation>\n=======\n"
 "<replacement code>\n>>>>>>> REPLACE\n```\n"
 "The SEARCH block must match the file content EXACTLY (copy it verbatim, enough lines to be unique). Keep edits minimal. "
 "Output ONLY the ```EDIT ...``` block(s), nothing else.")
REPAIR_SYS_NOCRIT=("You are an independent software repair agent. You are given one or more source files that contain a bug. "
 "Find the bug and fix it. Produce minimal SEARCH/REPLACE edits. For each edit output EXACTLY:\n"
 "```EDIT path/to/file.py\n<<<<<<< SEARCH\n<exact existing code, verbatim incl. indentation>\n=======\n"
 "<replacement code>\n>>>>>>> REPLACE\n```\n"
 "The SEARCH block must match the file content EXACTLY. Keep edits minimal. Output ONLY the ```EDIT ...``` block(s).")
def _strip_lineno(s):
    pass
    lines=s.split("\n"); out=[]
    for ln in lines:
        m=re.match(r'^\s*\d+:\s?(.*)$',ln)
        out.append(m.group(1) if m else ln)
    return "\n".join(out)

def _parse_edits(t, code_files):
    edits={}
    for m in re.finditer(r'```EDIT\s+(\S+)\s*\n<<<<<<<\s*SEARCH\s*\n(.*?)\n=======\s*\n(.*?)\n>>>>>>>\s*REPLACE',t,re.S):
        p,s,rep=m.group(1),m.group(2),m.group(3)
        if p in code_files: edits.setdefault(p,[]).append((_strip_lineno(s),_strip_lineno(rep)))
    return edits

def gen_repair(code_files, crit, repo, retries=3, feedback=None):
    pass

    ctx="\n\n".join(f"=== FILE: {p} (lines prefixed with 'N: ' for reference only; do NOT include the 'N: ' prefix in SEARCH blocks) ===\n{c}" for p,c in code_files.items())
    if crit is None:
        sys_p=REPAIR_SYS_NOCRIT
        user=f"REPOSITORY: {repo}\n\nSOURCE FILE(S) (contain a bug, find and fix it):\n{ctx}\n\nSEARCH/REPLACE edit(s):"
    else:
        sys_p=REPAIR_SYS
        user=f"REPOSITORY: {repo}\n\nDEFECT CRITIQUE (fix only this):\n{crit}\n\nBUGGY SOURCE FILE(S):\n{ctx}\n\nSEARCH/REPLACE edit(s) fixing only the described defect:"
    if feedback:
        user+=f"\n\nYour previous fix attempt did NOT resolve the failing test. Test output:\n{feedback[:2500]}\n\nRevise your SEARCH/REPLACE edit(s) to actually fix the defect:"
    for att in range(retries):
        try:
            t=gpt_chat(sys_p, user, t=200)
            edits=_parse_edits(t, code_files)
            if edits: return edits
        except Exception: pass
    return {}

def _fuzzy_locate(file_lines, search_lines):
    pass

    sl=[x.strip() for x in search_lines if x.strip()!=""]
    if not sl: return None
    norm=[x.strip() for x in file_lines]
    hits=[]
    n=len(sl)

    for i in range(len(norm)):
        if norm[i]!=sl[0]: continue

        j=i; k=0
        while j<len(norm) and k<n:
            if norm[j]==sl[k]: k+=1; j+=1
            elif norm[j]=="": j+=1
            else: break
        if k==n: hits.append((i,j))
    if len(hits)==1: return hits[0]
    return None

def apply_edits(code_files, edits):
    pass

    new={}
    for p,elist in edits.items():
        c=code_files.get(p)
        if c is None: continue
        lines=c.split("\n"); applied=False
        for s,rep in elist:
            if not s.strip(): continue

            if s in c:
                c="\n".join(lines) if False else c
                c2=c.replace(s,rep,1)
                if c2!=c: c=c2; lines=c.split("\n"); applied=True; continue

            loc=_fuzzy_locate(lines, s.split("\n"))
            if loc:
                a,b=loc
                lines=lines[:a]+rep.split("\n")+lines[b:]
                c="\n".join(lines); applied=True
        if applied: new[p]=c
    return new

def make_diff(orig_files, new_files):
    import difflib
    parts=[]
    for p,newc in new_files.items():
        orig=orig_files.get(p,"")
        ud=difflib.unified_diff(orig.splitlines(keepends=True), newc.splitlines(keepends=True),
                                fromfile=f"a/{p}", tofile=f"b/{p}")
        d="".join(ud)
        if d:
            if not d.endswith("\n"): d+="\n"
            parts.append(f"diff --git a/{p} b/{p}\n"+d)
    return "".join(parts)

def patched_files(nf):
    return set(nf.keys()) if isinstance(nf,dict) else set()

def _hunk_lines(patch, fp):
    pass
    lines=[]; cur=None
    for ln in patch.split("\n"):
        m=re.match(r'diff --git a/(\S+) b/',ln)
        if m: cur=m.group(1); continue
        if cur==fp:
            hm=re.match(r'@@ -(\d+)',ln)
            if hm: lines.append(int(hm.group(1)))
    return lines

def get_files(img, patch, maxlen=400000):
    pass

    src=[f for f in re.findall(r'diff --git a/(\S+) b/',patch) if 'test' not in f.lower()][:2]
    if not src: return {}, set()
    full={}
    try:
        for fp in src:
            rc=subprocess.run(["docker","run","--rm","--network=none",img,"bash","-c",
                f"cd /testbed && git reset --hard -q 2>/dev/null; cat {fp} 2>/dev/null"],capture_output=True,text=True,timeout=300)
            c=rc.stdout
            if c.strip() and len(c)<=maxlen: full[fp]=c
    except Exception:
        return {}, set()
    return full, set(src)

SMALL=40000
def _crit_lines(critique):
    pass
    nums=[]
    for m in re.finditer(r'\blines?\s+(\d+)', critique or "", re.I): nums.append(int(m.group(1)))
    for m in re.finditer(r'\bline\s+(\d+)', critique or "", re.I): nums.append(int(m.group(1)))

    for m in re.finditer(r'\b(?:at|around|near)\s+(\d+)', critique or "", re.I): nums.append(int(m.group(1)))
    return sorted(set(nums))

def _window(text, centers, win=150):
    pass
    lines=text.split("\n"); N=len(lines)
    if not centers:
        a,b=0,min(N,2*win)
        seg=[(i+1,lines[i]) for i in range(a,b)]
    else:
        keep=set()
        for c in centers:
            for i in range(max(0,c-1-win), min(N,c-1+win)): keep.add(i)
        seg=[(i+1,lines[i]) for i in sorted(keep)]
    return "\n".join(f"{n}: {ln}" for n,ln in seg)

def window_for_arm(full, lab, critique):
    pass

    shown={}
    centers=_crit_lines(critique) if (lab in ("true","false") and critique) else []
    for p,c in full.items():
        if len(c)<=SMALL:
            shown[p]="\n".join(f"{i+1}: {ln}" for i,ln in enumerate(c.split("\n")))
        elif lab=="nocrit":
            shown[p]=_window(c, [], win=150)
        else:
            shown[p]=_window(c, centers, win=150)
    return shown

def main():
    import argparse; ap=argparse.ArgumentParser()
    ap.add_argument("--n",type=int,default=12); ap.add_argument("--clean_ids",default="clean_ids_confirmed.json")
    ap.add_argument("--out",default="spike_repair.json"); ap.add_argument("--workers",type=int,default=5); args=ap.parse_args()
    df=pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{DATA}/*.parquet"))],ignore_index=True)
    res=df[df.resolved==True].groupby('instance_id').first().reset_index()
    meta={r.instance_id:r for _,r in res.iterrows()}
    tri={json.loads(l)["instance_id"]:json.loads(l)["critique"] for l in open(f"{PRE}/triples_reviewergen.jsonl") if json.loads(l).get("critique")}
    cf={}
    for l in open(f"{ARTIFACTS}/cleanfp_scores.jsonl"):
        r=json.loads(l)
        if r.get("c_false"): cf[r["tid"]]=r["c_false"]

    ext_path=f"{HERE}/extend_triples.jsonl"
    if os.path.exists(ext_path):
        for l in open(ext_path):
            r=json.loads(l)
            if r.get("critique") and r.get("c_false"):
                tri[r["iid"]]=r["critique"]; cf[r["iid"]]=r["c_false"]

    clean=[]
    for cp in [args.clean_ids, "extend_clean_ids.json"]:
        p=f"{HERE}/{cp}"
        if os.path.exists(p):
            try: clean+=json.load(open(p))
            except: pass
    clean=list(dict.fromkeys(clean))
    if not clean: clean=list(tri)
    pool=[i for i in clean if i in tri and i in cf and i in meta]
    pref=["psf__requests","pallets__flask","pydata__xarray","scikit-learn","sphinx","sympy__sympy","matplotlib","django","astropy","pytest"]
    def rank(i):
        for k,p in enumerate(pref):
            if i.startswith(p): return k
        return 99
    random.shuffle(pool)
    from collections import defaultdict; import itertools
    byr=defaultdict(list)
    for i in sorted(pool,key=rank): byr[rank(i)].append(i)
    ids=[]
    for i in itertools.chain(*itertools.zip_longest(*byr.values())):
        if i is not None: ids.append(i)

    print(f"candidate pool {len(ids)}, target n={args.n} (after baseline-fail gate)",flush=True)
    outp=f"{HERE}/{args.out}"; done=set()
    if os.path.exists(outp):
        try:
            for r in json.load(open(outp)): done.add(r["iid"])
        except: pass
    out=[]
    if os.path.exists(outp):
        try: out=json.load(open(outp))
        except: out=[]
    lk=Lock(); pull_lk=Lock(); state={"n_ok":len([r for r in out if r.get("base")=="fail"])}
    def save(): json.dump(out,open(outp,"w"),indent=2,ensure_ascii=False)
    def process_issue(iid):
        if iid in done: return
        with lk:
            if state["n_ok"]>=args.n: return
        r=meta[iid]; img=r.image_name; f2p=dbl(r.FAIL_TO_PASS)

        rc_i,have,_=sh("docker image inspect "+img+" >/dev/null 2>&1 && echo YES || echo NO",t=60)
        if "YES" not in have:
            with pull_lk:
                rc_i2,have2,_=sh("docker image inspect "+img+" >/dev/null 2>&1 && echo YES || echo NO",t=60)
                if "YES" not in have2:
                    if free_gb()<60:
                        with lk: out.append({"iid":iid,"skip":"disk_low"}); save()
                        return
                    rc,_,_=sh(f"docker pull {img}",t=1800)
                    if rc!=0:
                        with lk: out.append({"iid":iid,"skip":"pull_fail"}); save()
                        return
        base_oc,_,base_cnt=resolution_check(img, r.repo, getattr(r,'test_patch',''), f2p, repair_patch="", timeout=900)
        rec={"iid":iid,"repo":r.repo,"base":base_oc,"base_pass":base_cnt[0],"base_fail":base_cnt[1]}
        if base_oc!='fail':
            rec["skip"]=f"base_{base_oc}"
            with lk: out.append(rec); save()
            print(f"  {iid} baseline={base_oc} SKIP",flush=True); return
        full,critregion=get_files(img, r.model_patch)
        if not full:
            rec["skip"]="no_files"
            with lk: out.append(rec); save()
            return
        tp=getattr(r,'test_patch','')
        for lab,crit in [("nocrit",None),("false",cf[iid]),("true",tri[iid])]:
            shown=window_for_arm(full, lab, crit)
            rec[f"{lab}_window_src"]=("neutral_head" if (lab=="nocrit" and any(len(c)>SMALL for c in full.values())) else ("critique_lines" if lab!="nocrit" else "full_small"))
            feedback=None; best_oc='fail'; best_pass=base_cnt[0]; rep_ok_any=False; touched_all=set()
            status="no_edits"
            for k in range(2):
                edits=gen_repair(shown, crit, r.repo, feedback=feedback)
                edits={p:e for p,e in edits.items() if 'test' not in p.lower()}
                if not edits:
                    if k==0: status="no_edits"
                    break
                newf=apply_edits(full, edits)
                if not newf:
                    if k==0: status="search_miss"
                    break
                diff=make_diff(full, newf)
                if not diff:
                    if k==0: status="empty_diff"
                    break
                touched_all|=patched_files(newf)
                oc,rep_ok,cnt=resolution_check(img, r.repo, tp, f2p, repair_patch=diff, timeout=900)
                rep_ok_any=rep_ok_any or rep_ok; status="ran"
                if cnt[0]>best_pass: best_pass=cnt[0]
                if oc=='pass': best_oc='pass'
                feedback=f"FAIL_TO_PASS still failing: {cnt[1]} failed, {cnt[0]} passed."
                if oc=='pass': break
            rec[f"{lab}_repair"]=status
            rec[f"{lab}_resolution"]=(best_oc if status=="ran" else None)
            rec[f"{lab}_flip"]=(best_oc=='pass')
            rec[f"{lab}_best_pass"]=best_pass
            rec[f"{lab}_pass_delta"]=best_pass-base_cnt[0]
            rec[f"{lab}_repair_applied"]=rep_ok_any
            rec[f"{lab}_outside_critregion"]=list(touched_all-critregion)[:5]
            print(f"  {iid} {lab}: status={status} flip={best_oc=='pass'} pass_delta={best_pass-base_cnt[0]}",flush=True)
        with lk:
            out.append(rec); save(); state["n_ok"]+=1
            print(f"== {iid} done ({state['n_ok']}/{args.n} gated), free={free_gb()}GB ==",flush=True)
    todo=[i for i in ids if i not in done]

    _,havelist,_=sh("docker images --format '{{.Repository}}:{{.Tag}}'",t=60)
    resident=set(havelist.split("\n"))
    BATCH=max(args.workers*2, 16)
    def prepull(img):
        if free_gb()<80: return
        sh(f"docker pull {img}",t=1800)
    for bstart in range(0,len(todo),BATCH):
        batch=todo[bstart:bstart+BATCH]

        _,hl,_=sh("docker images --format '{{.Repository}}:{{.Tag}}'",t=60)
        cached=set(hl.split("\n"))
        need=list(dict.fromkeys(meta[i].image_name for i in batch if meta[i].image_name not in cached))
        if need:
            print(f"[batch {bstart//BATCH}] pre-pull {len(need)} images, free={free_gb()}GB",flush=True)
            with ThreadPoolExecutor(max_workers=6) as pex: list(pex.map(prepull,need))

        with ThreadPoolExecutor(max_workers=args.workers) as ex: list(ex.map(process_issue, batch))

        _,hl2,_=sh("docker images --format '{{.Repository}}:{{.Tag}}'",t=60)
        for im in set(hl2.split("\n"))-resident:
            if im: sh(f"docker rmi {im} 2>/dev/null",t=120)
        print(f"[batch {bstart//BATCH}] done, cleaned, free={free_gb()}GB",flush=True)

    valid=[r for r in out if r.get("base")=="fail" and not r.get("skip")]
    def arm(lab):
        v=[r for r in valid if r.get(f"{lab}_resolution") in ('pass','fail')]
        flip=sum(1 for r in v if r.get(f"{lab}_flip"))
        return v,flip
    nv,nflip=arm("nocrit"); fv,fflip=arm("false"); tv,tflip=arm("true")
    print("="*50,flush=True)
    print(f"VALID (base=fail) issues: {len(valid)}",flush=True)
    print(f"NOCRIT resolution rate: {nflip}/{len(nv)} = {nflip/len(nv):.3f}" if nv else "NOCRIT: none valid",flush=True)
    print(f"FALSE  resolution rate: {fflip}/{len(fv)} = {fflip/len(fv):.3f}" if fv else "FALSE: none valid",flush=True)
    print(f"TRUE   resolution rate: {tflip}/{len(tv)} = {tflip/len(tv):.3f}" if tv else "TRUE: none valid",flush=True)

    try:
        from scipy import stats
        def mcnemar(la,lb):
            both=[r for r in valid if r.get(f"{la}_resolution") in ('pass','fail') and r.get(f"{lb}_resolution") in ('pass','fail')]
            a=sum(1 for r in both if r.get(f"{la}_flip") and not r.get(f"{lb}_flip"))
            b=sum(1 for r in both if not r.get(f"{la}_flip") and r.get(f"{lb}_flip"))
            p=stats.binomtest(min(a,b),a+b,0.5).pvalue if (a+b)>0 else 1.0
            return len(both),a,b,p
        n1,a1,b1,p1=mcnemar("true","nocrit"); n2,a2,b2,p2=mcnemar("true","false")
        print(f"McNemar true-vs-nocrit: n={n1} true_only={a1} nocrit_only={b1} p={p1:.3f}",flush=True)
        print(f"McNemar true-vs-false:  n={n2} true_only={a2} false_only={b2} p={p2:.3f}",flush=True)

        def wilcox(la,lb):
            pairs=[(r.get(f"{la}_pass_delta"),r.get(f"{lb}_pass_delta")) for r in valid if r.get(f"{la}_pass_delta") is not None and r.get(f"{lb}_pass_delta") is not None]
            diffs=[a-b for a,b in pairs if a-b!=0]
            if len(diffs)<1: return len(pairs),0,1.0,0.0
            mean_d=sum(a-b for a,b in pairs)/len(pairs)
            try: w=stats.wilcoxon([a for a,b in pairs],[b for a,b in pairs]).pvalue
            except Exception: w=1.0
            return len(pairs),len(diffs),w,mean_d
        for la,lb in [("true","nocrit"),("true","false")]:
            npr,nd,wp,md=wilcox(la,lb)
            print(f"pass_delta {la}-vs-{lb}: n_pairs={npr} nonzero_diff={nd} mean_diff={md:.2f} wilcoxon_p={wp:.3f}",flush=True)
    except Exception as e: print("stats err",e,flush=True)

    for lab in ["nocrit","false","true"]:
        pds=[r.get(f"{lab}_pass_delta") for r in valid if r.get(f"{lab}_pass_delta") is not None]
        if pds: print(f"{lab} mean pass_delta: {sum(pds)/len(pds):.2f} (n={len(pds)})",flush=True)

    for lab in ["nocrit","false","true"]:
        noe=sum(1 for r in valid if r.get(f"{lab}_repair") in ('no_edits','search_miss'))
        print(f"{lab} repair 未产出(no_edits/search_miss): {noe}/{len(valid)}",flush=True)
    print(f"越界 (true repair 改 critregion 外文件): {sum(1 for r in valid if r.get('true_outside_critregion'))}",flush=True)
    print("SPIKE_REPAIR DONE",flush=True)

if __name__=="__main__": main()
