from civic.paths import OUTPUT_DIR
pass

import os
import json, re, os
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
PRE=os.environ.get("CIVIC_PROJECT_ROOT", ".") + "/data/anchors"; ARTIFACTS=str(OUTPUT_DIR)
HERE=str(OUTPUT_DIR)
def anth(t=60):
    import anthropic; return anthropic.Anthropic(base_url=os.environ["ANTHROPIC_BASE_URL"],api_key=os.environ["ANTHROPIC_API_KEY"],max_retries=6,timeout=t)
def toks(s): return set(w for w in re.findall(r'[a-zA-Z_][a-zA-Z0-9_]+',(s or "").lower()) if len(w)>3)
def jaccard(a,b):
    ta,tb=toks(a),toks(b)
    return len(ta&tb)/len(ta|tb) if (ta|tb) else 0.0

JUDGE_SYS=("You are an impartial evaluator checking dataset quality. You are given two defect critiques about the SAME code: "
 "critique T (describes a REAL defect that exists) and critique F (SUPPOSED to describe a plausible-but-NONEXISTENT defect, "
 "i.e. a false positive). Your job: determine whether F actually describes the SAME underlying defect as T (which would make "
 "F a mislabeled/dirty sample, since F is supposed to be about a defect that does NOT exist). Answer 'SAME: <0-100>' where "
 "100 = F describes the same real defect as T (DIRTY sample), 0 = F describes a genuinely different mechanism that is not the "
 "real defect (clean false-positive). Output ONLY 'SAME: <int>'.")
def judge_same(c_true, c_false):
    try:
        r=anth(60).messages.create(model="claude-sonnet-4-6",max_tokens=20,temperature=0.0,system=JUDGE_SYS,
            messages=[{"role":"user","content":f"CRITIQUE T (real defect):\n{c_true}\n\nCRITIQUE F (should be nonexistent defect):\n{c_false}\n\nDoes F describe the same real defect as T?\nSAME:"}])
        m=re.search(r'SAME:\s*(\d+)',r.content[0].text,re.I)
        return max(0,min(100,int(m.group(1))))/100.0 if m else None
    except Exception: return None

def main():
    tri={json.loads(l)["instance_id"]:json.loads(l)["critique"] for l in open(f"{PRE}/triples_reviewergen.jsonl") if json.loads(l).get("critique")}
    cf={}
    for l in open(f"{ARTIFACTS}/cleanfp_scores.jsonl"):
        r=json.loads(l)
        if r.get("c_false"): cf[r["tid"]]=r["c_false"]
    ids=[t for t in tri if t in cf]
    print(f"{len(ids)} issues with c_true+c_false",flush=True)
    out=[];lk=Lock()
    def work(iid):
        ct,cfx=tri[iid],cf[iid]
        jac=jaccard(ct,cfx); same=judge_same(ct,cfx)

        dirty = (jac>=0.5) or (same is not None and same>=0.5)
        rec={"iid":iid,"jaccard":round(jac,3),"judge_same":same,"dirty":dirty}
        with lk: out.append(rec)
    with ThreadPoolExecutor(max_workers=6) as ex: list(ex.map(work,ids))
    out.sort(key=lambda r:-(r["judge_same"] or 0))
    json.dump(out,open(f"{HERE}/clean_cfalse.json","w"),indent=2,ensure_ascii=False)
    dirty=[r for r in out if r["dirty"]]; clean=[r for r in out if not r["dirty"]]
    err=[r for r in out if r["judge_same"] is None]
    print("="*50,flush=True)
    print(f"TOTAL: {len(out)}  DIRTY (剔除): {len(dirty)}  CLEAN (保留): {len(clean)}  judge_err: {len(err)}",flush=True)
    print(f"dirty by high-jaccard(>=0.5): {sum(1 for r in dirty if r['jaccard']>=0.5)}",flush=True)
    print(f"dirty by judge_same(>=0.5): {sum(1 for r in dirty if (r['judge_same'] or 0)>=0.5)}",flush=True)
    print("--- top 8 dirtiest ---",flush=True)
    for r in out[:8]: print(f"  {r['iid']}: jac={r['jaccard']} same={r['judge_same']} dirty={r['dirty']}",flush=True)
    print("--- clean issue ids (for扩展 spike) ---",flush=True)
    print(json.dumps([r["iid"] for r in clean]),flush=True)
    print("CLEAN DONE",flush=True)

if __name__=="__main__": main()
