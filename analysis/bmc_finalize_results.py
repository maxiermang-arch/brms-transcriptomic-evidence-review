"""Validate completed runs and emit canonical tabular results."""
from pathlib import Path
import json,hashlib
import pandas as pd,numpy as np
from enhance_common import M
from _common import AN

o=Path(AN)/'bmc_results';e=Path(AN)/'enhancement_results'
files=sorted(e.glob('simulation_*.csv'))
frames=[pd.read_csv(f) for f in files if f.stem not in ['simulation_all_replicates','simulation_summary']]
d=pd.concat(frames,ignore_index=True)
assert d.screen1_rho.notna().all() and d.screen2_rho.notna().all()
assert len(d)==7200 and d[['scenario','replicate']].drop_duplicates().shape[0]==1800
assert d.groupby(['scenario','subtype']).size().eq(200).all()
summ=[]
metrics=['L1_size','L1_match','L2','L2_crossfit','L3','Hist','Concordance','Concordance_any','Drug','complete','complete_any','complete_size','complete_crossfit','without_L1_match','without_L2','without_L3','without_Hist','without_Concordance','without_Drug']
for (scenario,s),g in d.groupby(['scenario','subtype']):
    gates=['L1_match','L2','L3','Hist','Concordance']+(['Drug'] if s in [3,4] else [])
    assert np.array_equal(g.complete,g[gates].all(axis=1))
    assert (g.complete<=g.complete_any).all()
    for gate in gates:
        assert np.array_equal(g['without_'+gate],g[[v for v in gates if v!=gate]].all(axis=1))
    for metric in metrics:
        v=g[metric].dropna()
        if len(v)==0:continue
        k=int(v.sum());lo,hi=M.clopper_pearson(k,len(v))
        summ.append(dict(scenario=scenario,subtype=s,metric=metric,n=len(v),hits=k,rate=k/len(v),ci_lo=lo,ci_hi=hi))
d.to_csv(o/'joint_replicates.csv',index=False);pd.DataFrame(summ).to_csv(o/'joint_summary.csv',index=False)
b=pd.read_csv(o/'benchmark_summary.csv');assert b.n.eq(2000).all()
old=pd.read_csv(Path(AN)/'legacy_results/enhancement_results/simulation_summary.csv')
new=pd.DataFrame(summ);diff=new[new.metric=='complete'].merge(old[old.metric=='complete'][['scenario','subtype','rate']],on=['scenario','subtype'],suffixes=('_new','_old'))
diff['difference']=diff.rate_new-diff.rate_old;diff.to_csv(o/'joint_old_new.csv',index=False)
seed=pd.read_csv(o/'same_mechanism_seed_replicates.csv')
assert len(seed)==2000 and seed.replicate.nunique()==2000
assert seed.scenario.eq('label_null').all()
assert seed[['p_oracle','p_size','p_matched','p_label']].notna().all().all()
rows=[]
for experiment,frame in [('original_module',b[b.scenario=='label_null']),('same_mechanism_new_seed',pd.read_csv(o/'same_mechanism_seed_summary.csv')),('standalone_normal_score',pd.read_csv(o/'standalone_label_permutation_summary.csv'))]:
    for _,row in frame.iterrows():
        v=row.to_dict();v['experiment']=experiment
        if 'method' not in v:v['method']='label'
        rows.append(v)
pd.DataFrame(rows).to_csv(o/'null_control_comparison.csv',index=False)
registry=[]
for f in sorted(o.glob('*.csv')):
    if f.name=='result_registry.csv':continue
    registry.append(dict(file=f.name,rows=len(pd.read_csv(f)),sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
pd.DataFrame(registry).to_csv(o/'result_registry.csv',index=False)
(o/'validation.json').write_text(json.dumps(dict(same_mechanism_seed_repeats=2000,standalone_normal_score_repeats=2000,integrated_transfer_rows=len(pd.read_csv(o/'transfer_integrated.csv')),joint_pairs=1800,joint_rows=7200,benchmark_per_scenario=2000,L2_max_difference=float(pd.read_csv(o/'L2_reconstruction.csv').difference.max()),simulated_concordance_tail='positive_right',primary_aggregation='both',sensitivity_aggregation='any',empirical_permutations=20000),indent=2))
print('Validated all joint replicates, benchmark runs and empirical registry')
