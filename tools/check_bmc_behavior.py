"""Small independent behavioral checks for revision-critical numerical rules."""
from pathlib import Path
import sys,importlib,json
import numpy as np
from scipy.stats import rankdata
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'analysis'))
from concordance_rules import screen,aggregate,right_tail
M=importlib.import_module('44_dispersion_matched_null')
assert M.l1_pvalue(np.array([.3333333333]),1/3)==1.0
assert right_tail(.5,[.2,.5,.9])==.75
rho=np.ones((2,4));p=np.full((2,4),.001)
both,any_screen,q=aggregate(rho,p);assert both.all() and any_screen.all()
both,any_screen,q=aggregate(-rho,p);assert not both.any() and not any_screen.any()
rho[1]=0;both,any_screen,q=aggregate(rho,p);assert not both.any() and any_screen.all()
y=np.repeat(np.arange(4),6);x=np.random.default_rng(3).normal(size=(12,24))
r,p=screen(rankdata(x,axis=1),rankdata(-x,axis=1),y,y,np.random.default_rng(4),399)
assert np.max(r)<-.99 and np.min(p)>.9
report={'rounded_ties_counted_in_tail':True,'add_one_tail':True,'negative_concordance_rejected':True,'both_any_distinguished':True,'opposite_spectrum_right_tail':True}
out=Path(__file__).resolve().parents[1]/'analysis/bmc_results/behavior_checks.json';out.write_text(json.dumps(report,indent=2));print(report)
