pass

import os
import json, os, re
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
HERE=os.path.dirname(os.path.abspath(__file__))
def gpt(t=120):
    import openai; return openai.OpenAI(base_url=os.environ["OPENAI_BASE_URL"],api_key=os.environ["OPENAI_API_KEY"],max_retries=6,timeout=t)
def gpt_chat(system,user,t=120):
    r=gpt(t).chat.completions.create(model="gpt-5.5",messages=[{"role":"system","content":system},{"role":"user","content":user}])
    return r.choices[0].message.content or ""
def toks(s): return set(w for w in re.findall(r'[a-zA-Z_][a-zA-Z0-9_]+',(s or "").lower()) if len(w)>3)
def jaccard(a,b):
    ta,tb=toks(a),toks(b)
    return len(ta&tb)/len(ta|tb) if (ta|tb) else 0.0
JUDGE_SYS=("You are an impartial evaluator checking dataset quality. Two defect critiques about the SAME code: T (describes a "
 "REAL defect) and F (SUPPOSED to describe a plausible-but-NONEXISTENT defect). Determine whether F actually describes the "
 "SAME underlying defect as T (which makes F a dirty/mislabeled sample). Answer 'SAME: <0-100>' where 100=F describes the "
 "same real defect as T (DIRTY), 0=F is a genuinely different nonexistent mechanism (clean). Output ONLY 'SAME: <int>'.")
def judge_same(ct,cf):
    try:
        t=gpt_chat(JUDGE_SYS,f"CRITIQUE T (real):\n{ct}\n\nCRITIQUE F (should be nonexistent):\n{cf}\n\nSAME:",t=120)
        m=re.search(r'SAME:\s*(\d+)',t,re.I)
        return max(0,min(100,int(m.group(1))))/100.0 if m else None
    except Exception: return None
def main():
    rows=[json.loads(l) for l in open(f"{HERE}/extend_triples.jsonl") if json.loads(l).get("critique")]
    print(f"{len(rows)} new triples to clean",flush=True)
    out=[]; lk=Lock()
    def work(r):
        ct,cf=r["critique"],r["c_false"]; iid=r["iid"]
        jac=jaccard(ct,cf); same=None
        for _ in range(3):
            same=judge_same(ct,cf)
            if same is not None: break
        dirty=(jac>=0.5) or (same is not None and same>=0.5)
        with lk: out.append({"iid":iid,"jaccard":round(jac,3),"judge_same":same,"dirty":dirty})
    with ThreadPoolExecutor(max_workers=4) as ex: list(ex.map(work,rows))
    json.dump(out,open(f"{HERE}/extend_clean_audit.json","w"),indent=2)
    dirty=[r for r in out if r["dirty"]]
    err=[r for r in out if r["judge_same"] is None]

    clean=[r["iid"] for r in out if not r["dirty"] and r["judge_same"] is not None]
    json.dump(clean,open(f"{HERE}/extend_clean_ids.json","w"))
    print(f"TOTAL {len(out)} | DIRTY {len(dirty)} | judge_err {len(err)} | CONFIRMED CLEAN {len(clean)}",flush=True)
    print("CLEAN_EXTEND DONE",flush=True)
if __name__=="__main__": main()
