"""Paired-cohort, dependent-rule operating characteristics; fixed-label simulation.

Reduced inner Monte Carlo for feasible outer replication. This is not a
simulation of the original atlas's clustering/selection or a power bound.
"""
import argparse,json,time
import numpy as np
import pandas as pd
from scipy.stats import rankdata,mannwhitneyu,norm
from sklearn.model_selection import StratifiedKFold
from enhance_common import *

SCENARIOS=[
 dict(name='null_small',signal=0.,conf=0.,nA=52,nB=45,rho=.35,imb=False,shift=0.),
 dict(name='signal_small',signal=1.2,conf=0.,nA=52,nB=45,rho=.35,imb=False,shift=0.),
 dict(name='histology_only',signal=0.,conf=1.5,nA=52,nB=45,rho=.35,imb=False,shift=0.),
 dict(name='signal_histology',signal=1.2,conf=1.5,nA=52,nB=45,rho=.35,imb=False,shift=0.),
 dict(name='signal_n100',signal=1.2,conf=0.,nA=100,nB=100,rho=.35,imb=False,shift=0.),
 dict(name='signal_n200',signal=1.2,conf=0.,nA=200,nB=200,rho=.35,imb=False,shift=0.),
 dict(name='signal_imbalanced',signal=1.2,conf=0.,nA=52,nB=45,rho=.35,imb=True,shift=0.),
 dict(name='signal_highcorr',signal=1.2,conf=0.,nA=52,nB=45,rho=.75,imb=False,shift=0.),
 dict(name='signal_shift',signal=1.2,conf=0.,nA=52,nB=45,rho=.35,imb=False,shift=1.),
]

def cached_neighbourhood(Z,k,pool,n,rng):
    # Cache ONLY expression-dependent sorted neighbours; preserve production RNG.
    if cached_neighbourhood.last is not Z:
        # Row-wise matvec retains the production operation and tie ordering.
        cached_neighbourhood.orders=np.array([np.argsort(-(Z@Z[a])) for a in range(len(Z))])
        cached_neighbourhood.last=Z
    orders=cached_neighbourhood.orders
    out=np.empty((n,k),int)
    for i in range(n):
        a=int(rng.integers(len(Z)));out[i]=rng.choice(orders[a,:pool],k,replace=False)
    return out
cached_neighbourhood.last=None

def labels(n,imb,rng):
    p=np.array([.08,.15,.32,.45] if imb else [.25]*4)
    counts=np.maximum(2,np.floor(n*p).astype(int))
    while counts.sum()<n: counts[np.argmax(n*p-counts)]+=1
    while counts.sum()>n: counts[np.argmax(counts-n*p)]-=1
    return rng.permutation(np.repeat(np.arange(4),counts))

def dataset(sc,n,co,rng):
    y=labels(n,sc['imb'],rng)
    hist=np.where(rng.random(n)<.8,y,rng.integers(4,size=n)) if sc['conf'] else rng.integers(4,size=n)
    nuisance=rng.normal(size=(n,4));rho=sc['rho'];raw=np.empty((400,n))
    for j in range(20):
        ix=slice(j*20,(j+1)*20);shared=rng.normal(size=n)
        signal=sc['signal']*(y==j) if j<4 else np.zeros(n)
        cf=sc['conf']*(hist==j) if j<4 else np.zeros(n)
        raw[ix]=np.sqrt(rho)*shared+np.sqrt(1-rho)*rng.normal(size=(20,n))+signal+cf+.35*nuisance[:,j%4]
        if co=='B' and sc['shift']:
            # Label-independent, sample-varying batch disturbance, not constant shift.
            raw[ix]+=sc['shift']*rng.normal(size=n)
    sd=raw.std(1,ddof=1);Z=(raw-raw.mean(1)[:,None])/sd[:,None]
    c=pd.DataFrame(nuisance,columns=L2.PANELS)
    c['ESTIMATE_Stromal']=nuisance[:,2];c['ESTIMATE_Immune']=nuisance[:,1]
    c['ESTIMATE_score']=c.ESTIMATE_Stromal+c.ESTIMATE_Immune
    c['Purity_ESTIMATE']=1/(1+np.exp(c.ESTIMATE_score/2))
    return Z,y,hist,c,sd

def histology_oof(hist,y):
    # Fixed Laplace-smoothed multinomial histology baseline, unlike empirical L1-logit.
    out=np.empty((len(y),4))
    for i in range(len(y)):
        tr=(hist==hist[i]);tr[i]=False
        out[i]=(np.bincount(y[tr],minlength=4)+1)/(tr.sum()+4)
    return out

def cohort_rules(Z,y,hist,c,sc,co,sd,rng):
    allr=[];signature=np.array([Z[j*20:(j+1)*20].mean(0) for j in range(20)])
    signature2=np.array([Z[np.r_[np.arange(j*20,j*20+20),np.arange(((j+1)%20)*20,((j+1)%20)*20+20)]].mean(0) for j in range(16)])
    ranks=[rankdata(signature,axis=1),rankdata(signature2,axis=1)]
    ho=histology_oof(hist,y)
    for s in range(4):
        yy=y==s;obs=signature[s];ao=auc(yy,obs)
        _,_,R=M.matched_reference(Z,20,obs.std(ddof=1),rng,rng,n_probe=20,n_null=399)
        pm=M.l1_pvalue(M.auc_from_ranks(R,yy),ao)
        ids=np.array([rng.choice(400,20,replace=False) for _ in range(399)])
        pu=M.l1_pvalue(M.auc_from_ranks(rankdata(M.set_scores(Z,ids),axis=1),yy),ao)
        cov=c[L2.adjustment_sets(str(s+1))['leaveown_strict']].to_numpy()
        adj=auc(yy,strict_residual(obs,cov));lo,hi=auc_ci(yy,obs,rng,n=399)
        # Proper refitted L2 bootstrap (conditional on gene scaling in simulated cohort).
        vals=[];ip=np.flatnonzero(yy);im=np.flatnonzero(~yy)
        for _ in range(99):
            ix=np.r_[rng.choice(ip,len(ip)),rng.choice(im,len(im))]
            vals.append(auc(yy[ix],strict_residual(obs[ix],cov[ix])))
        alo,ahi=np.quantile(vals,[.025,.975])
        # Class minority may be <5; choose valid fold count.
        folds=min(5,int(yy.sum()),int((~yy).sum()));rr=np.empty(len(y))
        for tr,te in StratifiedKFold(folds,shuffle=True,random_state=SEED).split(cov,yy):
            beta=np.linalg.lstsq(L2._design(cov[tr]),obs[tr],rcond=None)[0]
            rr[te]=obs[te]-np.c_[np.ones(len(te)),(cov[te]-cov[tr].mean(0))/L2._sd(cov[tr])]@beta
        # Analytic AUC for the fixed scaled score in no-histology scenarios.
        w=1/sd[s*20:(s+1)*20]/20
        var=sc['rho']*w.sum()**2+(1-sc['rho'])*(w*w).sum()+.35**2*w.sum()**2
        if co=='B':var+=sc['shift']**2*w.sum()**2
        pop_auc=float(norm.cdf(sc['signal']*w.sum()/np.sqrt(2*var))) if sc['conf']==0 else np.nan
        allr.append(dict(L1_size=pu<=.05,L1_match=pm<=.05,L2=adj>=.7,L3=lo>.5,Hist=auc(yy,ho[:,s])<.8,
            raw_auc=ao,adjusted_auc=adj,crossfit_auc=auc(yy,rr),L2_crossfit=auc(yy,rr)>=.7,
            L2_ci_lo=alo,L2_ci_hi=ahi,raw_ci_coverage=(lo<=pop_auc<=hi) if np.isfinite(pop_auc) else np.nan))
    return allr,ranks

def concordance(ra,rb,ya,yb,rng):
    from concordance_rules import screen, aggregate
    rho=[]; p=[]
    for a,b in zip(ra,rb):
        rr,pp=screen(a,b,ya,yb,rng,nperm=399)
        rho.extend(rr);p.extend(pp)
    both,any_screen,q=aggregate(rho,p)
    return both,any_screen,np.array(rho).reshape(2,4),np.array(p).reshape(2,4),q

def drug_rules(sc,rng):
    out={}
    for s,n in [(2,12),(3,20)]:
        y=labels(n,False,rng);v=rng.normal(0,.35,n)
        if sc['signal']>0:v-=.45*(y==s)
        p=[mannwhitneyu(v[y==j],v[y!=j],alternative='less',method='asymptotic').pvalue for j in range(4)]
        out[s]=bh(p)[s]<=.05
    return out

def run(sc,rep):
    rng=np.random.default_rng(SEED+490000+SCENARIOS.index(sc)*100000+rep)
    A=dataset(sc,sc['nA'],'A',rng);B=dataset(sc,sc['nB'],'B',rng)
    a,ra=cohort_rules(*A[:4],sc,'A',A[4],rng);b,rb=cohort_rules(*B[:4],sc,'B',B[4],rng)
    con,con_any,con_rho,con_p,con_q=concordance(ra,rb,A[1],B[1],rng);dr=drug_rules(sc,rng);rows=[]
    for s in range(4):
        rules={k:a[s][k] and b[s][k] for k in ['L1_size','L1_match','L2','L3','L2_crossfit']}
        rules['Hist']=a[s]['Hist'] and b[s]['Hist'];rules['Concordance']=bool(con[s]);rules['Concordance_any']=bool(con_any[s])
        if s in dr:rules['Drug']=bool(dr[s])
        gates=['L1_match','L2','L3','Hist','Concordance']+(['Drug'] if s in dr else [])
        rr=dict(scenario=sc['name'],replicate=rep,subtype=s+1,**rules,complete=all(rules[k] for k in gates),
            complete_size=all(rules['L1_size'] if k=='L1_match' else rules[k] for k in gates),
            complete_crossfit=all(rules['L2_crossfit'] if k=='L2' else rules[k] for k in gates),
            raw_auc_A=a[s]['raw_auc'],raw_auc_B=b[s]['raw_auc'],adjusted_auc_A=a[s]['adjusted_auc'],adjusted_auc_B=b[s]['adjusted_auc'],
            coverage_A=a[s]['raw_ci_coverage'],coverage_B=b[s]['raw_ci_coverage'])
        for gate in ['L1_match','L2','L3','Hist','Concordance','Drug']:
            rr['without_'+gate]=all(rules[k] for k in gates if k!=gate) if gate in gates else np.nan
        rr['complete_any']=all(rules['Concordance_any'] if k=='Concordance' else rules[k] for k in gates)
        for j in range(2):
            rr[f'screen{j+1}_rho']=con_rho[j,s];rr[f'screen{j+1}_p']=con_p[j,s];rr[f'screen{j+1}_q']=con_q[j,s]
        rows.append(rr)
    return rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--reps',type=int,default=200);p.add_argument('--start',type=int,default=0);p.add_argument('--stop',type=int,default=len(SCENARIOS));args=p.parse_args()
    # Equivalence test of cached production neighbourhood selection.
    z=np.random.default_rng(31).normal(size=(80,20));orig=M.neighbourhood_sets
    for pool in [8,20,80]:
        assert np.array_equal(orig(z,8,pool,30,np.random.default_rng(32)),cached_neighbourhood(z,8,pool,30,np.random.default_rng(32)))
    M.neighbourhood_sets=cached_neighbourhood
    for sc in SCENARIOS[args.start:args.stop]:
        rows=[];t=time.monotonic()
        for rep in range(args.reps):
            rows.extend(run(sc,rep))
            if (rep+1)%10==0:
                save(rows,f'simulation_{sc["name"]}.csv')
                print(sc['name'],rep+1,'elapsed',round(time.monotonic()-t,1),flush=True)
        save(rows,f'simulation_{sc["name"]}.csv')
    (OUT/'simulation_settings.json').write_text(json.dumps(dict(concordance_tail='positive_right',primary_aggregation='both',sensitivity_aggregation='any',scenarios=SCENARIOS,reps=args.reps,inner_reference=399,inner_probes=20,seed=SEED+490000),indent=2))

if __name__=='__main__':main()
