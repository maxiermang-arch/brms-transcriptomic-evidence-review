"""Read-only checks of the reviewer snapshot; no scientific results are overwritten."""
from pathlib import Path
import csv,hashlib,importlib,json,sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
R=Path(__file__).resolve().parents[1];O=R/'analysis/bmc_results'
sys.path.insert(0,str(R/'analysis'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=R/'SHA256SUMS.txt'
assert manifest.exists(),'Snapshot manifest missing'
n=0
for line in manifest.read_text().splitlines():
    expected,name=line.split('  ',1);p=R/name
    assert p.is_file() and sha(p)==expected,name
    n+=1
snapshot=json.loads((R/'review_snapshot.json').read_text())
for row in snapshot['included_original_files']:
    assert sha(R/row['path'])==row['sha256'],row['path']
registry=pd.read_csv(O/'result_registry.csv')
for row in registry.itertuples():
    assert sha(O/row.file)==row.sha256,row.file
from enhance_common import L2,raw_cohorts
co=L2.load_cohorts();raw=raw_cohorts()
for c,size in [('A',52),('B',45)]:
    retained=pd.read_csv(O/f'samples_{c}.csv')
    assert len(co[c]['lab'])==len(raw[c][1])==size
    assert list(raw[c][0].columns)==list(raw[c][1].index)==retained['sample'].tolist()
    assert list(raw[c][1].astype(str))==retained.subtype.astype(str).tolist()
    assert co[c]['Z'].shape==({'A':22125,'B':16435}[c],size)
proxies=importlib.import_module('42_proxy_sensitivity').proxy_sets()
for row in json.loads((O/'proxy_catalogue.json').read_text()):
    actual=next(x for x in proxies[str(row['subtype'])] if x[0]==row['proxy'])[2]
    assert sorted(set(actual))==sorted(row['genes'])
metrics=pd.read_csv(O/'transfer_integrated.csv');pred=pd.read_csv(R/'analysis/enhancement_results/transfer_predictions.csv')
assert len(metrics)==16
for row in metrics.itertuples():
    d=pred[(pred.train==row.train)&(pred.test==row.test)&(pred['transform']==row.transform)]
    assert abs(roc_auc_score(d.label==row.subtype,d[f'p_{row.subtype}'])-row.auc)<1e-12
    assert abs(np.mean([(d.prediction[d.label==s]==s).mean() for s in [1,2,3,4]])-row.balanced_accuracy)<1e-12
joint=pd.read_csv(O/'joint_replicates.csv');assert len(joint)==7200
assert joint[['scenario','replicate']].drop_duplicates().shape[0]==1800
for subtype,d in joint.groupby('subtype'):
    gates=['L1_match','L2','L3','Hist','Concordance']+(['Drug'] if subtype in [3,4] else [])
    assert np.array_equal(d.complete,d[gates].all(axis=1))
assert len(pd.read_csv(O/'benchmark_replicates.csv'))==6000
assert len(pd.read_csv(O/'same_mechanism_seed_replicates.csv'))==2000
assert len(pd.read_csv(O/'standalone_label_permutation_replicates.csv'))==2000
assert len(pd.read_csv(O/'selection_chain_replicates.csv'))==2000
print(json.dumps(dict(status='passed',hashed_files=n,unchanged_source_files=len(snapshot['included_original_files']),canonical_csv_files=len(registry),labelled_samples=dict(A=52,B=45),probability_auc_checks=16,paired_cohort_simulations=1800,scientific_outputs_overwritten=False),indent=2))
