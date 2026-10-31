"""Known module reference and selection-chain experiments; distinct nulls.

The oracle reference enumerates 40 exchangeable modules. Production reference
uses reduced inner budgets (20 probes/399 sets), explicitly a sensitivity model.
"""
import argparse, json, time
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import rankdata
from sklearn.cluster import KMeans
from enhance_common import M,auc
from concordance_rules import right_tail
from _common import AN
from bmc_module_model import module_replicate

def main():
    p=argparse.ArgumentParser();p.add_argument('--reps',type=int,default=2000);a=p.parse_args()
    out=Path(AN)/'bmc_results';rows=[];sel=[];start=time.monotonic()
    for rep in range(a.reps):
        for j,scenario in enumerate(['label_null','competitive_null','target_enriched']):
            rows.append(module_replicate(scenario,rep))
        # Pure-noise discovery: labels and chosen programme are selected in A.
        rng=np.random.default_rng(30261031+rep);A=rng.normal(size=(52,40));B=rng.normal(size=(45,40))
        km=KMeans(4,n_init=5,random_state=1).fit(A)
        ya=km.labels_==0;yb=km.predict(B)==0
        au=np.array([auc(ya,v) for v in A.T]);idx=int(np.argmax(np.maximum(au,1-au)));sign=1 if au[idx]>=.5 else -1
        independent=rng.permutation(np.arange(45)%2==0)
        sel.append(dict(replicate=rep,internal_auc=auc(ya,sign*A[:,idx]),transferred_expression_label_auc=auc(yb,sign*B[:,idx]),independent_random_label_auc=auc(independent,sign*B[:,idx])))
        if (rep+1)%50==0:
            pd.DataFrame(rows).to_csv(out/'benchmark_replicates.csv',index=False)
            pd.DataFrame(sel).to_csv(out/'selection_chain_replicates.csv',index=False)
            print(rep+1,'elapsed',round(time.monotonic()-start,1),flush=True)
    summary=[]
    for scenario,d in pd.DataFrame(rows).groupby('scenario'):
        for method in ['oracle','size','matched','label']:
            hits=int((d[f'p_{method}']<=.05).sum());lo,hi=M.clopper_pearson(hits,len(d))
            summary.append(dict(scenario=scenario,method=method,n=len(d),hits=hits,rate=hits/len(d),ci_lo=lo,ci_hi=hi))
    pd.DataFrame(summary).to_csv(out/'benchmark_summary.csv',index=False)
    (out/'benchmark_settings.json').write_text(json.dumps(dict(reps=a.reps,n=52,genes=400,module_size=10,modules=40,rho=.6,inner_sets=399,inner_probes=20,seed=20261031,selection_seed=30261031,selection_null='pure noise with independent random evaluation labels'),indent=2))

if __name__=='__main__':main()
