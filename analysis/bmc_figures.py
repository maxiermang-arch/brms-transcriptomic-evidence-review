"""Six evidence-led figures from canonical results; Python backend only."""
from pathlib import Path
import re,json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from _common import AN
o=Path(AN)/'bmc_results';e=Path(AN)/'enhancement_results';f=Path(AN).parents[1]/'submission/figures';f.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none','axes.labelsize':8})
blue='#235789';orange='#B3541E'
def save(fig,n):
    for ext in ['pdf','svg','png']:fig.savefig(f/f'Figure{n}.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)

fig,ax=plt.subplots(figsize=(6.7,4.8));ax.axis('off')
boxes=[('Released labels and processed expression','A: 52 mixed-origin brain metastases; B: 45 breast-origin brain metastases'),('References and conditional evidence','Competitive gene-set references; fixed-label randomization; calibration'),('Sensitivity and cohort context','12 proxies in two cohorts; refitted composition adjustment; histology'),('Frozen transfer','Train A / evaluate B and reverse; probability ranks and class decisions'),('Operating characteristics','Nine paired-cohort scenarios; module-reference and noise-selection benchmarks'),('Output','Dimension-specific evidence and uncertainty; operational joint pass rates')]
for i,(title,desc) in enumerate(boxes):
    y=.96-i*.16;ax.text(.03,y,title+'\n'+desc,ha='left',va='top',fontsize=8,bbox=dict(boxstyle='round,pad=.6',facecolor='#EEF3F7',edgecolor=blue),transform=ax.transAxes)
save(fig,1)
p=pd.read_csv(o/'proxy_analysis_verified.csv')
fig,axs=plt.subplots(1,2,figsize=(6.7,5.8),layout='constrained',sharex=True)
for ax,c in zip(axs,'AB'):
    d=p[p.cohort==c];y=np.arange(len(d));v=d.null_false_pass
    ax.errorbar(v,y,xerr=[v-d.null_ci_lo,d.null_ci_hi-v],fmt='o',color=blue,capsize=2)
    ax.axvline(.05,color='gray',ls='--');ax.set(yticks=y,yticklabels=[f'BrMS{s} {q}' for s,q in zip(d.subtype,d.proxy)],xlabel='Conditional false-pass fraction',title=f'Cohort {c}');ax.invert_yaxis()
save(fig,2)
fig,axs=plt.subplots(1,2,figsize=(6.7,5.8),layout='constrained',sharex=True)
for ax,c in zip(axs,'AB'):
    d=p[p.cohort==c];y=np.arange(len(d));v=d.recomputed_strict_auc
    ax.errorbar(v,y,xerr=[v-d.strict_ci_lo,d.strict_ci_hi-v],fmt='o',color=blue,capsize=2,label='Full-cohort / refitted 95% interval')
    ax.scatter(d.crossfit_median,y,marker='x',color=orange,label='Median cross-fit AUC');ax.axvline(.7,color='gray',ls='--')
    ax.set(yticks=y,yticklabels=[f'BrMS{s} {q}' for s,q in zip(d.subtype,d.proxy)],xlabel='Composition-adjusted AUC',title=f'Cohort {c}');ax.invert_yaxis()
axs[0].legend(fontsize=6,loc='lower left');save(fig,3)
samples=pd.read_csv(o/'samples_A.csv');samples['origin']=samples['sample'].str.extract(r'^([A-Z]+)')[0].str.replace('BRM','',regex=False)
tab=pd.crosstab(samples.origin,samples.subtype);h=pd.read_csv(e/'histology_conditional.csv')
fig,axs=plt.subplots(1,2,figsize=(6.7,3.0),layout='constrained')
axs[0].imshow(tab,aspect='auto',cmap='Blues');axs[0].set(xticks=range(4),xticklabels=[f'BrMS{s}' for s in range(1,5)],yticks=range(len(tab)),yticklabels=tab.index,title='Cohort A: origin and released labels')
for i in range(len(tab)):
    for j in range(4):axs[0].text(j,i,str(tab.iloc[i,j]),ha='center',va='center',color='black')
x=np.arange(4);axs[1].bar(x-.18,h.auc,.36,color=blue,label='Observed');axs[1].bar(x+.18,h.conditional_null_mean,.36,color='#B9CAD8',label='Conditional null mean')
axs[1].set(xticks=x,xticklabels=[f'BrMS{s}' for s in range(1,5)],ylabel='Programme AUC',ylim=(0,1.06),title='Within-origin label randomization')
for i,q in enumerate(h.q_4):axs[1].text(i,1.01,f'q={q:.3g}',ha='center',fontsize=6)
axs[1].legend(fontsize=6,loc='upper center',bbox_to_anchor=(.5,-.14),ncol=2,frameon=False);save(fig,4)
t=pd.read_csv(o/'transfer_summary.csv');cm=pd.read_csv(o/'transfer_confusion.csv')
fig,axs=plt.subplots(1,2,figsize=(6.7,3.0),layout='constrained')
for ax,(train,test) in zip(axs,[('A','B'),('B','A')]):
    d=cm[(cm.train==train)&(cm.test==test)&(cm['transform']=='within_sample_rank')].pivot(index='true',columns='predicted',values='n')
    v=d.div(d.sum(1),axis=0);ax.imshow(v,cmap='Blues',vmin=0,vmax=1)
    r=t[(t.train==train)&(t.test==test)&(t['transform']=='within_sample_rank')].iloc[0]
    ax.set(xticks=range(4),xticklabels=[f'BrMS{s}' for s in range(1,5)],yticks=range(4),yticklabels=[f'BrMS{s}' for s in range(1,5)],xlabel='Predicted label',ylabel='Released label',title=f'{train} to {test}: balanced accuracy {r.balanced_accuracy:.3f}\n95% interval {r.ba_ci_lo:.3f} to {r.ba_ci_hi:.3f}')
    for i in range(4):
        for j in range(4):ax.text(j,i,str(d.iloc[i,j]),ha='center',va='center',color='white' if v.iloc[i,j]>.6 else 'black')
save(fig,5)
b=pd.read_csv(o/'benchmark_summary.csv');s=pd.read_csv(o/'joint_summary.csv');sel=pd.read_csv(o/'selection_chain_replicates.csv')
fig=plt.figure(figsize=(6.7,5.2),layout='constrained');gs=fig.add_gridspec(2,2);axs=[fig.add_subplot(gs[0,0]),fig.add_subplot(gs[:,1]),fig.add_subplot(gs[1,0])]
for j,m in enumerate(['oracle','size','matched','label']):
    d=b[b.method==m].set_index('scenario').loc[['label_null','competitive_null','target_enriched']];v=d.rate
    axs[0].errorbar(np.arange(3)+(j-1.5)*.12,v,yerr=[v-d.ci_lo,d.ci_hi-v],fmt='o',capsize=2,label=m)
axs[0].axhline(.05,color='gray',ls='--');axs[0].set(xticks=range(3),xticklabels=['Label null','Competitive\nnull','Target\nenriched'],ylabel='Rejection fraction',title='Distinct inferential targets');axs[0].legend(fontsize=6)
d=s[s.metric=='complete'].pivot(index='scenario',columns='subtype',values='rate');axs[1].imshow(d,cmap='Blues',vmin=0,vmax=1,aspect='auto');axs[1].set(xticks=range(4),xticklabels=range(1,5),yticks=range(len(d)),yticklabels=d.index,xlabel='BrMS index',title='Joint conditional pass fractions')
for i in range(len(d)):
    for j in range(4):axs[1].text(j,i,f'{d.iloc[i,j]:.2f}',ha='center',va='center',fontsize=6)
for key,label in [('internal_auc','Selected internal'),('transferred_expression_label_auc','Expression-derived test labels'),('independent_random_label_auc','Independent random labels')]:axs[2].hist(sel[key].dropna(),bins=np.linspace(0,1,26),histtype='step',density=True,label=label)
axs[2].set(xlabel='Selected programme AUC',ylabel='Density',title='Pure-noise discovery and selection');axs[2].legend(fontsize=6)
for ax,label in zip(axs,['a','b','c']):ax.text(-.10,1.03,label,transform=ax.transAxes,fontweight='bold',fontsize=10,ha='right',va='bottom')
save(fig,6)
(f/'figure_contracts.json').write_text(json.dumps({'backend':'Python matplotlib','exports':['PDF','SVG','PNG'],'figures':{'1':'Evidence and data provenance','2':'Conditional calibration is definition dependent','3':'Adjustment and proxy sensitivity','4':'Histology-conditional evidence','5':'Ranking differs from class assignment','6':'Distinct nulls and joint-rule limitations'},'source_directory':'repository/analysis/bmc_results'},indent=2))
print('Six figures exported')
