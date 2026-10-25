from civic.services import review_score_path
from civic.services import message_client, message_options, review_model
from civic.paths import OUTPUT_DIR
pass

import os
import json, os, re, random, subprocess, numpy as np
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import pandas as pd, glob, shutil
random.seed(42)
def shutil_free(): return shutil.disk_usage(os.environ.get("CIVIC_PROJECT_ROOT", "."))[2]//(1024**3)
DATA=os.environ.get("CIVIC_PROJECT_ROOT", ".") + "/data/swe-review-bench_raw/data"; PRE=os.environ.get("CIVIC_PROJECT_ROOT", ".") + "/data/anchors"
ARTIFACTS=str(OUTPUT_DIR); RESULTS=str(OUTPUT_DIR); REPOS=os.environ.get("CIVIC_PROJECT_ROOT", ".") + "/repos_cache"
os.makedirs(RESULTS,exist_ok=True)
REPO_DIR={"astropy/astropy":"astropy","django/django":"django","matplotlib/matplotlib":"matplotlib","mwaskom/seaborn":"seaborn",
 "pallets/flask":"flask","psf/requests":"requests","pydata/xarray":"xarray","pylint-dev/pylint":"pylint",
 "pytest-dev/pytest":"pytest","scikit-learn/scikit-learn":"scikit-learn","sphinx-doc/sphinx":"sphinx","sympy/sympy":"sympy"}
repo_locks={}
def sh(c,cwd=None,t=120):
    r=subprocess.run(c,cwd=cwd,shell=True,capture_output=True,text=True,timeout=t); return r.returncode,r.stdout,r.stderr
def dbl(x):
    x=json.loads(x) if isinstance(x,str) else x
    return json.loads(x) if isinstance(x,str) else x
def provider_client(t=90):
    return message_client()

GEN_SYS=("You are an expert code reviewer. Examine the code and the candidate defect critique. "
 "If the claimed defect is REAL, predict a concrete, FALSIFIABLE consequence: name the kind of test or input that "
 "should FAIL on this (buggy) code because of this specific defect, and would PASS once the defect is fixed. "
 "Be specific about the failing behavior (function/method under test, the condition that triggers it). "
 "Output STRICT JSON: {\"mechanism\":\"<one-sentence falsifiable mechanism>\","
 "\"predicted_failing_test_hint\":\"<test name / module / behavior you predict currently fails>\","
 "\"self_confidence\":<int 0-100 that this prediction would actually hold>}. Output ONLY JSON. Do NOT assume the defect is real.")
def gen_mech(code,crit):
    r=provider_client(90).messages.create(model=review_model(), system=GEN_SYS, messages=[{'role': 'user', 'content': f'CRITIQUE:\n{crit}\n\nCODE:\n{code[:6000]}\n\nFalsifiable mechanism prediction JSON:'}], **message_options())
    return r.content[0].text
def parse_json(t):
    t=re.sub(r'^```(?:json)?','',t.strip()); t=re.sub(r'```$','',t.strip()); i=t.find("{")
    if i<0: return None
    try: o,_=json.JSONDecoder().raw_decode(t[i:]); return o
    except: return None

RUBRIC=("You are an expert code reviewer. Judge ONLY whether the specific defect in the critique is actually present in "
 "THIS code. Output 'SCORE: <int 0-100>' then 'REASON: <text>'.")
def raw_score(code,crit):
    r=provider_client().messages.create(model=review_model(), system=RUBRIC, messages=[{'role': 'user', 'content': f'CRITIQUE:\n{crit}\n\nCODE:\n{code[:6000]}\n\nIs the defect present?'}], **message_options())
    return r.content[0].text
def parse_score(t):
    m=re.search(r'SCORE:\s*(\d+)',t or "",re.I)
    if m: return max(0,min(100,int(m.group(1))))/100.0
    c=[int(x) for x in re.findall(r'\b(\d{1,3})\b',t or "") if 0<=int(x)<=100]
    return c[-1]/100.0 if c else None

def fuzzy_hit(pred_hint, f2p_list):
    pass
    if not pred_hint: return 0.0
    ph=set(t for t in re.findall(r'[A-Za-z_][A-Za-z0-9_]+', pred_hint.lower()) if len(t)>3)
    if not ph: return 0.0
    best=0.0
    for t in f2p_list:
        tn=set(x for x in re.findall(r'[A-Za-z_][A-Za-z0-9_]+', t.lower()) if len(x)>3)
        if tn: best=max(best, len(ph & tn)/len(tn))
    return best

JUDGE_SYS=("You are an impartial evaluator. A code reviewer predicted a falsifiable failure mechanism for a claimed defect. "
 "You are given (1) the reviewer's predicted mechanism + the test/behavior it predicts would fail, and (2) the names of the "
 "tests that ACTUALLY fail on the buggy code (ground-truth failing tests). Judge: does the reviewer's predicted failure "
 "mechanism CORRESPOND to the root cause exercised by these actual failing tests? Answer 'MATCH: <0-100>' where 100 = the "
 "prediction precisely identifies the behavior/mechanism these tests check, 0 = unrelated. Judge only correspondence, do not "
 "assume the reviewer is right. Output ONLY 'MATCH: <int>'.")
def llm_judge_match(mech, hint, f2p_str, test_bodies=""):
    if not (mech or hint): return 0.0
    tb=("\n\nACTUAL FAILING TEST CODE (what these tests check):\n"+test_bodies[:4000]) if test_bodies else ""
    try:
        r=provider_client(60).messages.create(model=review_model(), system=JUDGE_SYS, messages=[{'role': 'user', 'content': f'REVIEWER PREDICTED MECHANISM:\n{mech}\nPREDICTED FAILING TEST/BEHAVIOR:\n{hint}\n\nACTUAL FAILING TESTS (ground truth, names):\n{f2p_str}{tb}\n\nMATCH:'}], **message_options())
        m=re.search(r'MATCH:\s*(\d+)',r.content[0].text,re.I)
        return max(0,min(100,int(m.group(1))))/100.0 if m else 0.0
    except Exception: return 0.0

def extract_test_bodies(test_patch):
    pass
    if not test_patch: return ""
    added=[]
    for l in test_patch.split("\n"):
        if l.startswith("+") and not l.startswith("+++"):
            added.append(l[1:])
    return "\n".join(added)[:4000]

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
    pass

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

EXTRACT_PY=r'''
import re,sys,ast
f2p=sys.argv[1:]
seen=set()
for t in f2p:
    # pytest: path::Class::test / path::test ; django: mod.Class.test
    if "::" in t:
        path=t.split("::")[0]; fn=t.split("::")[-1]
    else:
        parts=t.replace("(","").replace(")","").split("."); fn=None
        # django 形式: app.tests.Class.method 或 module.method —— 取最后一段作 test 名
        fn=parts[-1]; path=None
    fn=re.sub(r"\[.*\]","",fn)  # 去参数化后缀
    # 搜文件
    import glob,os
    cands=[path] if path and os.path.isfile(path) else []
    if not cands:
        for g in glob.glob("**/*.py",recursive=True):
            if fn in open(g,encoding="utf-8",errors="replace").read(): cands.append(g)
    for c in cands[:3]:
        try: src=open(c,encoding="utf-8",errors="replace").read(); tree=ast.parse(src)
        except: continue
        for node in ast.walk(tree):
            if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name==fn and (c,fn) not in seen:
                seg=ast.get_source_segment(src,node)
                if seg: print("### %s::%s\n%s\n"%(c,fn,seg[:1800])); seen.add((c,fn))
'''
def executable_verify(img, repo, resolved_patch, test_patch, f2p, timeout=900):
    pass

    import subprocess, base64
    p2p=f2p
    if not p2p: return (None,"")
    is_dj="django" in repo
    ids=django_ids(p2p)[:4] if is_dj else p2p[:4]
    if is_dj and not ids: return (None,"")
    run=("python tests/runtests.py --settings=test_sqlite -v2 "+" ".join(ids)) if is_dj else ("python -m pytest "+" ".join("'"+s+"'" for s in ids)+" -rA -q")
    tp_b64=base64.b64encode((test_patch or "").encode()).decode()
    ex_b64=base64.b64encode(EXTRACT_PY.encode()).decode()
    args=" ".join("'"+s+"'" for s in p2p[:4])
    script=("source /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed && cd /testbed && "
            "git reset --hard -q && git clean -fdq && "
            f"echo {tp_b64} | base64 -d > /tmp/tp.patch && "
            "(git apply -v /tmp/tp.patch 2>/tmp/ap.err || patch -p1 < /tmp/tp.patch 2>>/tmp/ap.err) && "
            "echo '===APPLIED==='; "+run+" 2>&1 | tail -20; "
            f"echo '===TESTSRC==='; echo {ex_b64} | base64 -d > /tmp/ex.py && python /tmp/ex.py {args} 2>/dev/null")
    try:
        rc=subprocess.run(["docker","run","--rm",img,"bash","-c",script],capture_output=True,text=True,timeout=timeout)
        out=rc.stdout+rc.stderr
    except Exception: return (None,"")
    if '===APPLIED===' not in out: return (None,"")
    body=out.split('===APPLIED===',1)[1]
    res=body.split('===TESTSRC===',1)[0]
    tsrc=body.split('===TESTSRC===',1)[1].strip() if '===TESTSRC===' in body else ""
    oc=_parse_outcome(res, is_dj)
    anchor=1.0 if oc=='fail' else (0.0 if oc=='pass' else None)
    return (anchor, tsrc[:4000])

def code_region(repo,commit,patch,ctx=40):
    d=os.path.join(REPOS,REPO_DIR.get(repo,""))
    from threading import Lock as L
    lk=repo_locks.setdefault(repo,L())
    with lk:
        sh("git reset --hard -q && git clean -fdq",cwd=d); rc,_,_=sh(f"git checkout -q {commit}",cwd=d)
        if rc!=0:
            sh("git fetch -q --all",cwd=d,t=600); rc,_,_=sh(f"git checkout -q {commit}",cwd=d)
            if rc!=0: return None
        src=[f for f in re.findall(r'diff --git a/(\S+) b/',patch) if 'test' not in f.lower()]
        out=[]
        for fp in src[:1]:
            full=os.path.join(d,fp)
            if not os.path.isfile(full): continue
            L2=open(full,encoding='utf-8',errors='replace').read().split("\n")
            hs=[int(m.group(1)) for m in re.finditer(r'@@ -\d+(?:,\d+)? \+(\d+)',patch)]
            if not hs: hs=[1]
            out.append("\n".join(L2[max(0,min(hs)-ctx):min(len(L2),max(hs)+ctx)]))
        return "\n".join(out)[:6000] if out else None

def main():
    import argparse; ap=argparse.ArgumentParser(); ap.add_argument("--n",type=int,default=100); ap.add_argument("--workers",type=int,default=6); args=ap.parse_args()
    df=pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{DATA}/*.parquet"))],ignore_index=True)
    res=df[df.resolved==True].groupby('instance_id').first().reset_index()
    meta={r.instance_id:r for _,r in res.iterrows()}

    tri={json.loads(l)["instance_id"]:json.loads(l)["critique"] for l in open(f"{PRE}/triples_reviewergen.jsonl") if json.loads(l).get("critique")}
    cfalse={}
    for l in open(f"{ARTIFACTS}/cleanfp_scores.jsonl"):
        r=json.loads(l)
        if r.get("c_false"): cfalse[r["tid"]]=r["c_false"]

    draw=defaultdict(lambda: defaultdict(list))
    for l in open(review_score_path(PRE)):
        r=json.loads(l)
        if r.get("score") is not None: draw[r["triple_id"]][r["version"]].append(r["score"])
    s_true_A={t:float(np.mean(v["x"])) for t,v in draw.items() if "x" in v}
    s_false_A={}
    for l in open(f"{ARTIFACTS}/cleanfp_scores.jsonl"):
        r=json.loads(l)
        if r.get("s_false") is not None: s_false_A[r["tid"]]=r["s_false"]
    ids=[t for t in tri if t in cfalse and t in meta and t in s_true_A and t in s_false_A]
    random.shuffle(ids); ids=ids[:args.n]
    print(f"{len(ids)} issues with true+false critique + A scores",flush=True)
    outp=f"{RESULTS}/arm3_scores.jsonl"; done=set()
    if os.path.exists(outp):
        for l in open(outp):
            try: done.add(json.loads(l)["iid"])
            except: pass
    todo=[i for i in ids if i not in done]
    print(f"{len(todo)} todo ({len(done)} done)",flush=True)
    fo=open(outp,"a");lk=Lock();cnt=[0]
    img_resident=set()
    def work(iid):
        r=meta[iid]; code=code_region(r.repo,r.base_commit,r.model_patch)
        if not code:
            with lk: fo.write(json.dumps({"iid":iid,"skip":"no_code"})+"\n");fo.flush()
            return
        f2p=dbl(r.FAIL_TO_PASS)

        img=r.image_name
        _,o,_=sh("docker images --format '{{.Repository}}:{{.Tag}}'",t=60)
        if img not in o:
            if shutil_free()<60:
                with lk: fo.write(json.dumps({"iid":iid,"skip":"disk_low"})+"\n");fo.flush(); return
            rc,_,_=sh(f"docker pull {img}",t=1800)
            if rc!=0:
                with lk: fo.write(json.dumps({"iid":iid,"skip":"pull_fail"})+"\n");fo.flush(); return

        tpatch=getattr(r,'test_patch','')
        exec_anchor, test_src=executable_verify(img, r.repo, r.model_patch, tpatch, f2p)
        f2p_str="; ".join(str(x) for x in f2p[:6])

        test_bodies=test_src if test_src else extract_test_bodies(tpatch)
        rec={"iid":iid,"repo":r.repo,"A_true":s_true_A[iid],"A_false":s_false_A[iid],"exec_anchor_fails":exec_anchor}
        try:
            for lab,crit in [("true",tri[iid]),("false",cfalse[iid])]:
                obj=parse_json(gen_mech(code,crit)) or {}
                hint=obj.get("predicted_failing_test_hint",""); mech=str(obj.get("mechanism","")); conf=obj.get("self_confidence")
                rec[f"B_{lab}"]=(conf/100.0 if isinstance(conf,(int,float)) else None)
                hit=fuzzy_hit(hint, f2p)
                judge=llm_judge_match(mech, hint, f2p_str, test_bodies)

                rec[f"C_{lab}"]=float(judge * (exec_anchor if exec_anchor is not None else 0.0)) if exec_anchor is not None else None
                rec[f"judge_{lab}"]=round(judge,3); rec[f"hit_{lab}"]=round(hit,3)
                rec[f"mech_{lab}"]=mech[:150]; rec[f"hint_{lab}"]=str(hint)[:120]
        except Exception as e: rec["err"]=str(e)[:100]
        with lk: fo.write(json.dumps(rec)+"\n");fo.flush();cnt[0]+=1
        if cnt[0]%5==0: print(f"  {cnt[0]}/{len(todo)} free={shutil_free()}GB",flush=True)
    _,resident_o,_=sh("docker images --format '{{.Repository}}:{{.Tag}}'",t=60)
    resident=set(resident_o.splitlines())
    B=args.workers*2
    for bi in range(0,len(todo),B):
        with ThreadPoolExecutor(max_workers=args.workers) as ex: list(ex.map(work,todo[bi:bi+B]))
        _,cur,_=sh("docker images --format '{{.Repository}}:{{.Tag}}'",t=60)
        for im in set(cur.splitlines())-resident: sh(f"docker rmi {im} 2>/dev/null",t=120)
        print(f"== batch {bi//B} done, cleaned, free={shutil_free()}GB ==",flush=True)
    fo.close(); print("DONE gen+C-check",flush=True)

if __name__=="__main__": main()
