"""Trace extension inputs and numerically verify rerun R results."""
from pathlib import Path
import ast,hashlib,importlib,json
import numpy as np
import pandas as pd
from enhance_common import *
from _common import PU,ROOT,load_dataframe,COMPOSITION_PANELS

def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--preserved-root',type=Path,required=True)
    args=parser.parse_args()
    checks=[]
    rawroot=args.preserved_root.resolve()
    assert rawroot != Path(ROOT).resolve() and (rawroot/'data').is_dir()
    parse=importlib.import_module('01_load_and_qc')
    src=Path(ROOT)/'data/geo/GSE245467_Brainmets_raw_counts.xlsx'
    x,_,bad=parse._load_counts_xlsx(str(src));stored=load_dataframe(f'{AN}/X_counts.pkl')
    assert x.index.equals(stored.index) and x.columns.equals(stored.columns)
    checks.append(dict(check='A_raw_xlsx_to_counts',max_abs_difference=float(np.abs(x.values-stored.values).max()),n=int(x.size)))
    assert np.array_equal(x.values,stored.values)
    # Validate input source hash against preserved original; never modify original.
    inputs=[]
    for f in sorted((rawroot/'data').rglob('*')):
        if f.is_file():inputs.append(dict(path=str(f.relative_to(rawroot)),sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
    (OUT/'preserved_input_sha256.json').write_text(json.dumps(inputs,indent=2))
    for co in 'AB':
        for suffix in ['ESTIMATE_raw.txt','ABIS.txt','ABIS_TPM.txt']:
            filename=f'cohort{co}_{suffix}'
            a=pd.read_csv(Path(PU)/filename,sep='\t',index_col=0)
            b=pd.read_csv(rawroot/'data/purity'/filename,sep='\t',index_col=0)
            assert a.shape==b.shape and a.index.equals(b.index) and a.columns.equals(b.columns)
            av=a.apply(pd.to_numeric,errors='coerce').to_numpy();bv=b.apply(pd.to_numeric,errors='coerce').to_numpy()
            assert np.array_equal(np.isnan(av),np.isnan(bv))
            diff=float(np.nanmax(abs(av-bv)));checks.append(dict(check='R_rerun_'+filename,max_abs_difference=diff,n=int(av.size)))
            assert diff == 0, (filename, diff)
    panels=composition_panels();(OUT/'composition_panels_actual.json').write_text(json.dumps(panels,ensure_ascii=False,indent=2))
    deltas={k:{'producer_only':sorted(set(v)-set(COMPOSITION_PANELS[k])),
        'common_only':sorted(set(COMPOSITION_PANELS[k])-set(v))} for k,v in panels.items() if set(v)!=set(COMPOSITION_PANELS[k])}
    record=OUT/'historical_panel_definition_differences.json'
    if deltas or not record.exists():record.write_text(json.dumps(deltas,ensure_ascii=False,indent=2))
    checks.append(dict(check='shared_panels_match_producer',max_abs_difference=len(deltas),n=len(panels)))
    assert not deltas
    C=L2.load_cohorts();raw=raw_cohorts()
    for co in 'AB':
        # A producer standardized all 54 expression samples (2 lacked labels).
        L=np.log2(x.div(x.sum(0),axis=1)*1e6+1) if co=='A' else raw[co][0]
        Z=L2._z(L); comp=C[co]['C'];ds=[]
        for name,genes in panels.items():
            score=Z.loc[[g for g in genes if g in Z.index]].mean(0).reindex(comp.index)
            ds.append(float(abs(score-comp[name]).max()))
        checks.append(dict(check=f'{co}_historical_composition_reconstruction',max_abs_difference=max(ds),n=len(ds)))
        assert max(ds)<1e-10
    # New helper AUC is checked against a separate library including ties.
    rng=np.random.default_rng(51)
    for n in [12,45,52]:
        y=np.arange(n)%3==0;v=rng.integers(0,5,n)
        from sklearn.metrics import roc_auc_score
        assert abs(auc(y,v)-roc_auc_score(y,v))<1e-12
    # Training feature construction cannot change when only held-out data change.
    L,y=raw['B'];C0=C['B']['C'].loc[y.index];tr=np.arange(30);te=np.arange(30,len(y))
    gl=L2.program_genes(C['B']['Z'],'1');cols=L2.adjustment_sets('1')['leaveown_strict']
    s,c=fold_features(L,gl,C0,cols,tr,tr);pert=L.copy();pert.iloc[:,te]+=500
    s2,c2=fold_features(pert,gl,C0,cols,tr,tr)
    assert np.array_equal(s,s2) and np.array_equal(c,c2)
    checks.append(dict(check='training_features_holdout_invariance',max_abs_difference=0,n=len(tr)))
    save(checks,'verification_checks.csv');print(pd.DataFrame(checks).to_string(index=False))

if __name__=='__main__':main()
