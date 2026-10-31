"""Frozen four-feature transfer and histology-conditional association."""
import re
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score,confusion_matrix
from _common import BRMS_PROGRAMS,SUBS
from enhance_common import *

def main():
    raw=raw_cohorts();common=sorted(set(raw['A'][0].index)&set(raw['B'][0].index))
    common=[g for g in common if all(np.isfinite(raw[c][0].loc[g]).all() for c in 'AB')]
    gidx={s:np.array([i for i,g in enumerate(common) if g in BRMS_PROGRAMS[s]]) for s in SUBS}
    rows=[];pred=[];cm=[];rng=np.random.default_rng(SEED+480000)
    for train,test in [('A','B'),('B','A')]:
        X=raw[train][0].loc[common].to_numpy();T=raw[test][0].loc[common].to_numpy()
        yt=raw[train][1].values;ye=raw[test][1].values
        for transform in ['within_sample_rank','training_gene_z']:
            if transform=='within_sample_rank':
                xx=rankdata(X,axis=0)/len(common);tt=rankdata(T,axis=0)/len(common)
            else:
                mu=X.mean(1,keepdims=True);sd=X.std(1,ddof=1,keepdims=True);sd=np.where(sd>0,sd,1)
                xx=(X-mu)/sd;tt=(T-mu)/sd
            a=np.column_stack([xx[gidx[s]].mean(0) for s in SUBS]);b=np.column_stack([tt[gidx[s]].mean(0) for s in SUBS])
            model=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=3000,solver='lbfgs'))
            model.fit(a,yt);p=model.predict_proba(b);yp=model.predict(b);classes=model[-1].classes_
            subsets={'all':np.ones(len(ye),bool)}
            if test=='A':subsets['breast']=np.array([x.startswith('BRE') for x in raw[test][1].index])
            for subset,mask in subsets.items():
                for j,s in enumerate(classes):
                    y=ye[mask]==s;v=p[mask,j];lo,hi=auc_ci(y,v,rng)
                    rows.append(dict(train=train,test=test,transform=transform,subset=subset,subtype=s,n=int(mask.sum()),
                        n_pos=int(y.sum()),auc=auc(y,v) if min(y.sum(),(~y).sum())>=2 else np.nan,
                        ci_lo=lo,ci_hi=hi,balanced_accuracy=balanced_accuracy_score(ye[mask],yp[mask])))
            mat=confusion_matrix(ye,yp,labels=classes)
            for i,s in enumerate(classes):
                for j,t in enumerate(classes):cm.append(dict(train=train,test=test,transform=transform,true=s,predicted=t,n=int(mat[i,j])))
            for i,sample in enumerate(raw[test][1].index):
                pred.append(dict(train=train,test=test,transform=transform,sample=sample,label=ye[i],prediction=yp[i],**{f'p_{s}':p[i,j] for j,s in enumerate(classes)}))
    save(rows,'transfer_metrics.csv');save(pred,'transfer_predictions.csv');save(cm,'transfer_confusion.csv')
    co=L2.load_cohorts()['A'];Z=co['Z'];lab=co['lab'].values
    hist=np.array([re.match(r'^[A-Z]+',str(s))[0] for s in Z.columns]);strata=[np.flatnonzero(hist==h) for h in np.unique(hist)]
    hr=[];sr=[]
    for s in SUBS:
        score=L2.program_score(Z,s);y=lab==s;obs=auc(y,score);vals=[]
        for _ in range(10000):
            yp=y.copy()
            for ix in strata:yp[ix]=rng.permutation(y[ix])
            vals.append(auc(yp,score))
        vals=np.array(vals)
        # Two tail probabilities relative to the conditional distribution; not centred at .5.
        pp=min(1,2*min((1+(vals>=obs-1e-12).sum())/10001,(1+(vals<=obs+1e-12).sum())/10001))
        hr.append(dict(subtype=s,auc=obs,conditional_null_mean=vals.mean(),conditional_two_sided_p=pp))
        for h,ix in zip(np.unique(hist),strata):
            yy=y[ix];lo,hi=auc_ci(yy,score[ix],rng)
            sr.append(dict(subtype=s,histology=h,n=len(ix),n_pos=int(yy.sum()),n_neg=int((~yy).sum()),
                auc=auc(yy,score[ix]) if min(yy.sum(),(~yy).sum())>=2 else np.nan,ci_lo=lo,ci_hi=hi))
    h=pd.DataFrame(hr);h['q_4']=bh(h.conditional_two_sided_p);save(h,'histology_conditional.csv');save(sr,'histology_strata.csv')
    print(pd.DataFrame(rows).to_string(index=False));print(h.to_string(index=False))
    # Export exactly aligned survival inputs for R proportional-hazards diagnostics.
    b=L2.load_cohorts()['B'];os=b['os'].reindex(b['Z'].columns);c=b['C'].loc[b['Z'].columns]
    surv=pd.DataFrame({'sample':b['Z'].columns,'time':pd.to_numeric(os['os']).values,
        'event':os['status'].astype(str).str.lower().str.startswith('dead').astype(int).values,
        'subtype':b['lab'].values,'purity':c['Purity_ESTIMATE'].values,'brain':c['脑组织合计'].values})
    for s in SUBS:
        v=L2.program_score(b['Z'],s);surv[f'score{s}']=(v-v.mean())/v.std()
    save(surv,'survival_inputs.csv')

if __name__=='__main__':main()
