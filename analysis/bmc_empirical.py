"""Reconstruct current empirical evidence without changing retained inputs."""
import ast, json, importlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from concordance_rules import screen, aggregate
from enhance_common import raw_cohorts, L2, strict_residual, fold_features, auc
from _common import AN, SUBS, SEED

OUT=Path(AN)/'bmc_results'
OUT.mkdir(exist_ok=True)

def main():
    raw=raw_cohorts();checks=[]
    for c,(x,y) in raw.items():
        assert x.columns.equals(y.index) and x.columns.is_unique
        assert len(y)=={'A':52,'B':45}[c]
        pd.DataFrame({'sample':y.index,'subtype':y.values,'cohort':c}).to_csv(OUT/f'samples_{c}.csv',index=False)
    # Frozen panels extracted as data from the original producer, not reselected.
    tree=ast.parse((Path(AN)/'07_immune_two_cohort.py').read_text())
    node=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PANELS' for t in n.targets))
    immune=ast.literal_eval(node.value)
    hm=json.loads((Path(AN)/'genesets/hallmark.json').read_text())
    immune['IFN-gamma (HM)']=hm['Interferon Gamma Response']
    broad=hm|json.loads((Path(AN)/'genesets/neural.json').read_text())
    z={c:x.sub(x.mean(1),axis=0).div(x.std(1),axis=0).dropna(how='all') for c,(x,y) in raw.items()}
    rho=[];pv=[];rows=[]
    for j,panel in enumerate([immune,broad]):
        sets={}
        for k,g in panel.items():
            gg={c:[v for v in g if v in z[c].index] for c in 'AB'}
            eligible=all(len(v)>=5 for v in gg.values()) if j==0 else len(g)>=15 and all(len(v)/len(g)>=.8 for v in gg.values())
            if eligible:sets[k]=gg
        r={c:rankdata(np.array([z[c].loc[gg[c]].mean(0) for gg in sets.values()]),axis=1) for c in 'AB'}
        rr,pp=screen(r['A'],r['B'],raw['A'][1].to_numpy(),raw['B'][1].to_numpy(),np.random.default_rng(SEED+6100+j),20000)
        rho.extend(rr);pv.extend(pp)
        for s,v,p in zip(SUBS,rr,pp):rows.append(dict(screen=j+1,subtype=s,n_signatures=len(sets),rho=v,p=p))
    both,any_support,q=aggregate(rho,pv)
    for k,row in enumerate(rows):row['q_8']=q.ravel()[k]
    pd.DataFrame(rows).to_csv(OUT/'concordance.csv',index=False)
    pd.DataFrame({'subtype':SUBS,'both':both,'any':any_support}).to_csv(OUT/'concordance_aggregation.csv',index=False)
    old=pd.read_csv(Path(AN)/'enhancement_results/proxy_upgrade.csv')
    proxies=importlib.import_module('42_proxy_sensitivity').proxy_sets()
    cohorts=L2.load_cohorts()
    for c,pack in cohorts.items():
        C=pack['C'].loc[raw[c][1].index];L=raw[c][0]
        for s,sets in proxies.items():
            cols=L2.usable(L2.adjustment_sets(s)['leaveown_strict'],C)
            for code,name,g in sets:
                gg=sorted(set(g)&set(pack['Z'].index));ix=np.arange(L.shape[1])
                scores,cov=fold_features(L,gg,C,cols,ix,ix)
                value=auc(raw[c][1].to_numpy()==s,strict_residual(scores,cov))
                retained=old[(old.cohort==c)&(old.subtype==int(s))&(old.proxy==code)].iloc[0]
                checks.append(dict(cohort=c,subtype=s,proxy=code,reconstructed=value,retained=retained.recomputed_strict_auc,difference=abs(value-retained.recomputed_strict_auc)))
    checks=pd.DataFrame(checks);checks.to_csv(OUT/'L2_reconstruction.csv',index=False)
    assert checks.difference.max()<1e-9, checks.to_string()
    old.to_csv(OUT/'proxy_analysis_verified.csv',index=False)
    pred=pd.read_csv(Path(AN)/'enhancement_results/transfer_predictions.csv');tr=[];cms=[]
    rng=np.random.default_rng(SEED+6200)
    for (train,test,transform),d in pred.groupby(['train','test','transform']):
        y=d.label.to_numpy();yp=d.prediction.to_numpy()
        recall=np.array([(yp[y==s]==s).mean() for s in [1,2,3,4]])
        values=[]
        for _ in range(2000):values.append(np.mean([(yp[rng.choice(np.flatnonzero(y==s),sum(y==s),replace=True)]==s).mean() for s in [1,2,3,4]]))
        lo,hi=np.quantile(values,[.025,.975])
        for i,s in enumerate([1,2,3,4]):
            tr.append(dict(train=train,test=test,transform=transform,subtype=s,recall=recall[i],balanced_accuracy=recall.mean(),ba_ci_lo=lo,ba_ci_hi=hi))
            for t in [1,2,3,4]:cms.append(dict(train=train,test=test,transform=transform,true=s,predicted=t,n=int(((y==s)&(yp==t)).sum())))
    pd.DataFrame(tr).to_csv(OUT/'transfer_summary.csv',index=False)
    pd.DataFrame(cms).to_csv(OUT/'transfer_confusion.csv',index=False)
    print('Verified 24 L2 reconstructions; regenerated positive concordance; summarized frozen transfer',flush=True)

if __name__=='__main__':main()
