"""Export technical definitions and integrated metrics from existing inputs."""
from pathlib import Path
import json
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score
from enhance_common import L2,composition_panels
from _common import AN

o=Path(AN)/'bmc_results';e=Path(AN)/'enhancement_results';co=L2.load_cohorts()
aliases={'脑组织合计':'Brain mean','免疫合计':'Immune mean','基质-成纤维':'Fibroblast','上皮-肿瘤':'Epithelial',
         'ESTIMATE_Stromal':'ESTIMATE stromal','ESTIMATE_Immune':'ESTIMATE immune','ESTIMATE_score':'ESTIMATE total','Purity_ESTIMATE':'ESTIMATE purity'}
names={'脑组织-星形胶质':'Astrocytic','脑组织-神经元':'Neuronal','脑组织-少突胶质':'Oligodendrocytic','脑组织-小胶质':'Microglial',
       '免疫-泛白系':'Pan-leukocyte','免疫-T/NK':'T/NK','免疫-髓系':'Myeloid','基质-成纤维':'Fibroblast','上皮-肿瘤':'Epithelial'}
cov=[];samples=[];panels=[]
for c,pack in co.items():
    for s in ['1','2','3','4']:
        requested=L2.adjustment_sets(s)['leaveown_strict'];used=L2.usable(requested,pack['C'])
        cov.append(dict(cohort=c,subtype=s,covariates='; '.join(aliases[k] for k in used),
                        code_columns='; '.join(used),excluded_unavailable='; '.join(k for k in requested if k not in used)))
        samples.append(dict(cohort=c,subtype=s,n=len(pack['lab']),positive_n=int((pack['lab']==s).sum()),genes=int(pack['Z'].shape[0])))
for key,genes in composition_panels().items():
    panels.append(dict(panel=names[key],code_name=key,defined_n=len(genes),genes=genes,
                       used_A=len(set(genes)&set(co['A']['Z'].index)),used_B=len(set(genes)&set(co['B']['Z'].index))))
pd.DataFrame(cov).to_csv(o/'strict_adjustment_covariates.csv',index=False)
pd.DataFrame(samples).to_csv(o/'sample_counts_by_subtype.csv',index=False)
(o/'composition_panel_definitions.json').write_text(json.dumps(panels,indent=2))
pd.DataFrame([{k:v for k,v in row.items() if k!='genes'} for row in panels]).to_csv(o/'composition_panel_coverage.csv',index=False)

tr=pd.read_csv(o/'transfer_summary.csv')
metrics=pd.read_csv(e/'transfer_metrics.csv');metrics=metrics[metrics.subset=='all']
merged=tr.merge(metrics[['train','test','transform','subtype','n','n_pos','auc','ci_lo','ci_hi','balanced_accuracy']],
    on=['train','test','transform','subtype'],validate='one_to_one',suffixes=('','_original'))
assert np.max(np.abs(merged.balanced_accuracy-merged.balanced_accuracy_original))<1e-12
pred=pd.read_csv(e/'transfer_predictions.csv')
for row in merged.itertuples():
    d=pred[(pred.train==row.train)&(pred.test==row.test)&(pred['transform']==row.transform)]
    assert abs(roc_auc_score(d.label==row.subtype,d[f'p_{row.subtype}'])-row.auc)<1e-12
merged.drop(columns='balanced_accuracy_original').to_csv(o/'transfer_integrated.csv',index=False)
print('Exported actual covariates, gene coverage and integrated frozen transfer; all 16 AUCs verified')
