"""Release-time consistency checks; does not regenerate scientific estimates."""
from pathlib import Path
import json, sys, subprocess
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parents[1]
O=R/'analysis/enhancement_results'
p=pd.read_csv(O/'proxy_upgrade.csv');cf=pd.read_csv(O/'proxy_crossfit_repeats.csv')
assert len(p)==24 and len(cf)==480
assert not p.duplicated(['cohort','subtype','proxy']).any()
assert ((p.null_ci_lo<=p.null_false_pass)&(p.null_false_pass<=p.null_ci_hi)).all()
assert ((p.crossfit_min<=p.crossfit_median)&(p.crossfit_median<=p.crossfit_max)).all()
s=pd.read_csv(O/'simulation_all_replicates.csv')
assert len(s)==7200 and len(s[['scenario','replicate']].drop_duplicates())==1800
assert (s.groupby(['scenario','subtype']).size()==200).all()
assert not s.duplicated(['scenario','replicate','subtype']).any()
for subtype,d in s.groupby('subtype'):
    gates=['L1_match','L2','L3','Hist','Concordance']+(['Drug'] if subtype in [3,4] else [])
    vals=d[gates].astype(bool)
    assert (d.complete==vals.all(axis=1)).all()
    for g in gates:
        assert (d['without_'+g]==vals.drop(columns=g).all(axis=1)).all()
t=pd.read_csv(O/'transfer_predictions.csv')
probs=t[['p_1','p_2','p_3','p_4']].to_numpy()
assert np.allclose(probs.sum(axis=1),1,atol=1e-12)
assert ((probs>=0)&(probs<=1)).all()
assert np.array_equal(t.prediction.to_numpy(),probs.argmax(axis=1)+1)
v=pd.read_csv(O/'verification_checks.csv')
assert (v.max_abs_difference<1e-10).all()
checks=dict(proxy_rows=len(p),crossfit_runs=len(cf),simulation_pairs=1800,
 subtype_rows=len(s),all_conjunctions_and_ablations_match=True,
 probabilities_normalized=True,input_and_R_numeric_checks_pass=True)
(O/'release_checks.json').write_text(json.dumps(checks,indent=2))
(O/'python_environment.txt').write_text(sys.version+'\n'+subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
print(json.dumps(checks,indent=2))
