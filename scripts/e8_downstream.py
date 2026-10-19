from civic.paths import OUTPUT_DIR
pass

import os
import json,numpy as np
d=json.load(open(os.environ.get('CIVIC_PROJECT_ROOT', '.') + '/outputs/spike_repair.json'))
valid=[it for it in d if it.get('base_fail') and not it.get('skip')]
N=len(valid)
print(f"E8 Downstream (repair harness executable, N={N} valid base=fail issues)",flush=True)
res={}
for arm in ['true','false','nocrit']:
    resolved=sum(1 for it in valid if it.get(f'{arm}_resolution')=='pass')
    flips=sum(1 for it in valid if it.get(f'{arm}_flip'))
    applied=sum(1 for it in valid if it.get(f'{arm}_repair_applied'))

    harmful=sum(1 for it in valid if it.get(f'{arm}_outside_critregion'))
    pass_deltas=[it.get(f'{arm}_pass_delta',0) or 0 for it in valid]
    res[arm]={"resolve_rate":round(resolved/N,4),"flip_rate":round(flips/N,4),
              "repair_applied":applied,"harmful_edit_rate":round(harmful/N,4),
              "mean_pass_delta":round(float(np.mean(pass_deltas)),4),"resolved_n":resolved}
    print(f"  [{arm}] resolve_rate={resolved/N:.4f} ({resolved}/{N}) flip={flips/N:.4f} harmful_edit_rate={harmful/N:.4f} mean_pass_delta={np.mean(pass_deltas):.3f}",flush=True)

res['_mapping']={"CIVIC-CR (targeted/true critregion)":"true","FP finding (misplaced)":"false","No-review (no critique)":"nocrit",
                 "note":"true arm = 精确 finding 引导 repair (CIVIC-CR 低 FP 理想态); false arm = 误报 finding 引导 (高 FP baseline); resolve rate 差 = 减少 FP 的下游价值"}
json.dump(res,open(os.environ.get('CIVIC_PROJECT_ROOT', '.') + '/outputs/e8_result.json','w'),indent=2)
print(f"\nE8 核心: true(targeted) resolve {res['true']['resolve_rate']} >> false(FP) {res['false']['resolve_rate']} / nocrit {res['nocrit']['resolve_rate']}",flush=True)
print(f"harmful edit: true {res['true']['harmful_edit_rate']} vs false {res['false']['harmful_edit_rate']} vs nocrit {res['nocrit']['harmful_edit_rate']}",flush=True)
print("E8 DONE",flush=True)
