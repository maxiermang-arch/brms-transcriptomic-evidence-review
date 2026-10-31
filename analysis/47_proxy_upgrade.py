"""Uniform modern reference, calibration and refitted adjustment for 24 proxies."""
import importlib,time
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
from enhance_common import M,L2,OUT,SEED,auc,bh,save,raw_cohorts,strict_residual,fold_features

def main():
    proxies=importlib.import_module('42_proxy_sensitivity').proxy_sets()
    cohorts=L2.load_cohorts(); raw=raw_cohorts();rows=[]; crows=[]
    for co,pack in cohorts.items():
        Z=pack['Z'].dropna(how='all');Zv=Z.to_numpy();genes=Z.index.tolist()
        assert np.isfinite(Zv).all() and Z.columns.is_unique and Z.index.is_unique
        lab=pack['lab'].reindex(Z.columns).values; C=pack['C'].loc[Z.columns]
        L=raw[co][0].reindex(index=Z.index,columns=Z.columns)
        for sub,sets in proxies.items():
            y=lab==sub;cv=L2.usable(L2.adjustment_sets(sub)['leaveown_strict'],C)
            cov=C[cv].to_numpy()
            for code,name,gl in sets:
                tick=time.monotonic();base=SEED+470000+(0 if co=='A' else 10000)+int(sub)*100+['P','A1','A2'].index(code)*10
                rng=np.random.default_rng(base);gg=sorted(set(gl)&set(genes));k=len(gg)
                obs=Z.loc[gg].mean(0).values;sd=obs.std(ddof=1);ao=auc(y,obs)
                best,ds,R=M.matched_reference(Zv,k,sd,np.random.default_rng(base+1),np.random.default_rng(base+2))
                pm=M.l1_pvalue(M.auc_from_ranks(R,y),ao)
                idx=np.array([rng.choice(len(genes),k,replace=False) for _ in range(2000)])
                Ru=rankdata(M.set_scores(Zv,idx),axis=1)
                pu=M.l1_pvalue(M.auc_from_ranks(Ru,y),ao)
                # Same permutations for matched and size-only calibration.
                hits=hits_u=0;ro=rankdata(obs)
                for _ in range(2000):
                    yp=rng.permutation(y);ap=float(M.auc_from_ranks(ro[None,:],yp)[0])
                    hits+=M.l1_pvalue(M.auc_from_ranks(R,yp),ap)<=.05
                    hits_u+=M.l1_pvalue(M.auc_from_ranks(Ru,yp),ap)<=.05
                lo,hi=M.clopper_pearson(hits,2000)
                l2=auc(y,strict_residual(obs,cov))
                full=np.arange(len(y))
                fresh,freshcov=fold_features(L,gg,C,cv,full,full)
                fresh_auc=auc(y,strict_residual(fresh,freshcov))
                # Refit both gene scaling and composition regression per resample.
                lv=L.loc[gg].to_numpy();boot=[]
                ip=np.flatnonzero(y);inn=np.flatnonzero(~y)
                for _ in range(2000):
                    ids=np.r_[rng.choice(ip,len(ip)),rng.choice(inn,len(inn))]
                    sc,cc=fold_features(L,gg,C,cv,ids,ids)
                    boot.append(auc(y[ids],strict_residual(sc,cc)))
                blo,bhi=np.quantile(boot,[.025,.975]);cross=[]
                for rep in range(20):
                    rr=np.empty(len(y))
                    for tr,te in StratifiedKFold(5,shuffle=True,random_state=base+rep).split(cov,y):
                        st,ct=fold_features(L,gg,C,cv,tr,tr)
                        se,ce=fold_features(L,gg,C,cv,tr,te)
                        beta=np.linalg.lstsq(L2._design(ct),st,rcond=None)[0]
                        rr[te]=se-np.c_[np.ones(len(te)),(ce-ct.mean(0))/L2._sd(ct)]@beta
                    cross.append(auc(y,rr))
                    crows.append(dict(cohort=co,subtype=sub,proxy=code,repeat=rep,crossfit_auc=cross[-1]))
                rows.append(dict(cohort=co,subtype=sub,proxy=code,name=name,n=len(y),n_pos=int(y.sum()),genes=k,
                    raw_auc=ao,size_p=pu,matched_p=pm,neighbourhood_M=best,dispersion_ratio=ds.mean()/sd,
                    null_false_pass=hits/2000,null_ci_lo=lo,null_ci_hi=hi,size_false_pass=hits_u/2000,
                    strict_auc=l2,recomputed_strict_auc=fresh_auc,strict_ci_lo=blo,strict_ci_hi=bhi,crossfit_median=np.median(cross),
                    crossfit_min=min(cross),crossfit_max=max(cross),crossfit_pass_fraction=np.mean(np.array(cross)>=.7)))
                frame=pd.DataFrame(rows);frame['matched_q_24']=bh(frame.matched_p)
                save(frame,'proxy_upgrade.csv');save(crows,'proxy_crossfit_repeats.csv')
                print(f'{co} BrMS{sub} {code}: p={pm:.4f}; calibration={hits/2000:.3f}; L2={l2:.3f}; CF={np.median(cross):.3f}; {time.monotonic()-tick:.1f}s',flush=True)
    # q values above were interim; final file now contains complete family of 24.
    assert len(rows)==24

if __name__=='__main__':main()
