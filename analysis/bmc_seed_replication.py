"""Same module label-null generator and budgets; only seed base is changed."""
from pathlib import Path
import json
import pandas as pd
from bmc_module_model import module_replicate
from enhance_common import M
from _common import AN

o=Path(AN)/'bmc_results';rows=[]
for rep in range(2000):
    rows.append(module_replicate('label_null',rep,seed_base=70261031))
    if (rep+1)%100==0:
        pd.DataFrame(rows).to_csv(o/'same_mechanism_seed_replicates.csv',index=False)
        print(rep+1,flush=True)
summ=[]
for method in ['oracle','size','matched','label']:
    k=sum(r['p_'+method]<=.05 for r in rows);lo,hi=M.clopper_pearson(k,2000)
    summ.append(dict(method=method,n=2000,hits=k,rate=k/2000,ci_lo=lo,ci_hi=hi))
pd.DataFrame(summ).to_csv(o/'same_mechanism_seed_summary.csv',index=False)
(o/'same_mechanism_seed_settings.json').write_text(json.dumps(dict(
    scenario='label_null',generator='bmc_module_model.module_replicate',
    original_seed_base=20261031,replication_seed_base=70261031,reps=2000,
    n=52,genes=400,modules=40,module_size=10,rho=.6,
    inner_sets=399,inner_probes=20,changed_parameter='seed_base only'),indent=2))
