"""Standalone label-permutation check and paired low/full inner-budget sensitivity."""
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import rankdata
from enhance_common import M,auc
from concordance_rules import right_tail
from _common import AN
o=Path(AN)/'bmc_results';rows=[]
for rep in range(2000):
    rng=np.random.default_rng(50261031+rep);y=rng.permutation(np.repeat([False,True],26))
    score=rng.normal(size=52);r=rankdata(score);obs=auc(y,score)
    perms=np.array([rng.permutation(y) for _ in range(399)])
    pa=(perms@r-26*27/2)/(26*26)
    rows.append(dict(replicate=rep,p=right_tail(obs,pa)))
d=pd.DataFrame(rows);d.to_csv(o/'standalone_label_permutation_replicates.csv',index=False)
k=int((d.p<=.05).sum());lo,hi=M.clopper_pearson(k,2000)
pd.DataFrame([dict(n=2000,hits=k,rate=k/2000,ci_lo=lo,ci_hi=hi,seed=50261031)]).to_csv(o/'standalone_label_permutation_summary.csv',index=False)
rows=[]
for rep in range(50):
    rng=np.random.default_rng(60261031+rep);y=rng.permutation(np.repeat([False,True],26));shared=rng.normal(size=(40,52))
    x=np.sqrt(.6)*np.repeat(shared,10,axis=0)+np.sqrt(.4)*rng.normal(size=(400,52));x[:10]+=y
    z=(x-x.mean(1)[:,None])/x.std(1,ddof=1)[:,None];score=z[:10].mean(0);obs=auc(y,score)
    row={'replicate':rep}
    for label,probes,sets in [('reduced',20,399),('full',300,2000)]:
        _,_,r=M.matched_reference(z,10,score.std(ddof=1),np.random.default_rng(rep+71),np.random.default_rng(rep+81),n_probe=probes,n_null=sets)
        row[label+'_p']=M.l1_pvalue(M.auc_from_ranks(r,y),obs)
    rows.append(row)
pd.DataFrame(rows).to_csv(o/'inner_budget_sensitivity.csv',index=False)
print('Standalone permutation control and 50 paired inner-budget comparisons completed')
