"""Shared extension helpers; no mutation on import."""
from pathlib import Path
import importlib
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from _common import AN, SEED

OUT = Path(AN) / 'enhancement_results'
M = importlib.import_module('44_dispersion_matched_null')
L2 = importlib.import_module('_l2_common')

def auc(y, x):
    y = np.asarray(y, bool); x = np.asarray(x)
    if y.sum() == 0 or y.sum() == len(y): return np.nan
    r = rankdata(x)
    return float((r[y].sum()-y.sum()*(y.sum()+1)/2)/(y.sum()*(len(y)-y.sum())))

def auc_ci(y, x, rng, n=2000):
    y = np.asarray(y, bool); x = np.asarray(x)
    a, b = x[y], x[~y]
    if min(len(a),len(b)) < 2: return (np.nan,np.nan)
    aa = a[rng.integers(len(a),size=(n,len(a)))]; bb = b[rng.integers(len(b),size=(n,len(b)))]
    ranks = rankdata(np.concatenate([aa,bb],axis=1),axis=1)
    vals = (ranks[:,:len(a)].sum(axis=1)-len(a)*(len(a)+1)/2)/(len(a)*len(b))
    return tuple(np.quantile(vals,[.025,.975]))

def bh(p):
    p = np.asarray(p,float); q = np.full(len(p),np.nan); ok=np.flatnonzero(np.isfinite(p))
    order=ok[np.argsort(p[ok])]
    q[order] = np.minimum(1,np.minimum.accumulate((p[order]*len(ok)/np.arange(1,len(ok)+1))[::-1])[::-1])
    return q

def save(rows, name):
    OUT.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT/name,index=False)

def strict_residual(score,cov):
    return L2._resid(score,L2._design(cov))[0]

def cross_residual(score,cov,y,seed):
    from sklearn.model_selection import StratifiedKFold
    res=np.empty(len(y)); cov=np.asarray(cov,float)
    for tr,te in StratifiedKFold(5,shuffle=True,random_state=seed).split(cov,y):
        beta=np.linalg.lstsq(L2._design(cov[tr]),score[tr],rcond=None)[0]
        dt=np.c_[np.ones(len(te)),(cov[te]-cov[tr].mean(0))/L2._sd(cov[tr])]
        res[te]=score[te]-dt@beta
    return res

def raw_cohorts():
    from _common import load_dataframe,load_pkl_tuple,SUBS
    XA=load_dataframe(f'{AN}/X_counts.pkl')
    la=load_dataframe(f'{AN}/labels_used.pkl')['brms'].astype(str)
    A=np.log2(XA[la.index].div(XA[la.index].sum(0),axis=1)*1e6+1)
    B,lb,osb=load_pkl_tuple(f'{AN}/gse184869_symbol_matrix.pkl')
    lb=lb.iloc[:,0] if isinstance(lb,pd.DataFrame) else lb
    lb=lb.astype(str);lb=lb[lb.isin(SUBS)];B=B[lb.index]
    return {'A':(A,la),'B':(B,lb)}

def composition_panels():
    """Actual historical producer definitions, not stale _common alternatives."""
    import ast
    tree=ast.parse((Path(AN)/'14_purity_adjustment.py').read_text())
    node=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign) and
        any(isinstance(t,ast.Name) and t.id=='PANEL' for t in n.targets))
    return ast.literal_eval(node.value)

def fold_features(L, programme, C, columns, training, testing):
    """Gene scaling and all expression-derived panel means fitted in training."""
    panels=composition_panels()
    genes=sorted((set(programme)|set(g for gl in panels.values() for g in gl))&set(L.index))
    v=L.loc[genes].to_numpy(); mu=v[:,training].mean(1);sd=v[:,training].std(1,ddof=1);sd=np.where(sd>0,sd,1)
    z=(v[:,testing]-mu[:,None])/sd[:,None]
    gm={g:i for i,g in enumerate(genes)}
    scores={name:z[[gm[g] for g in gl if g in gm]].mean(0) for name,gl in panels.items()}
    scores['脑组织合计']=np.mean([scores[k] for k in panels if k.startswith('脑组织')],axis=0)
    scores['免疫合计']=np.mean([scores[k] for k in panels if k.startswith('免疫')],axis=0)
    cov=np.column_stack([scores[c] if c in scores else C[c].to_numpy()[testing] for c in columns])
    return z[[gm[g] for g in programme if g in gm]].mean(0),cov
