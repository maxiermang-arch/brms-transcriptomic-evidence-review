"""Export log expression for the official MCPcounter R implementation."""
from pathlib import Path
import subprocess,itertools,json
import numpy as np,pandas as pd
from scipy.stats import mannwhitneyu
from enhance_common import raw_cohorts,bh
from _common import AN

o=Path(AN)/'bmc_results';o.mkdir(exist_ok=True)
g=Path(AN)/'genesets'
signature=pd.read_csv(g/'mcp_genes.txt',sep='\t')
raw=raw_cohorts();coverage=[]
for c,(x,y) in raw.items():
    x.to_csv(o/f'mcp_expression_{c}.csv')
    for ct,d in signature.groupby('Cell population'):
        coverage.append(dict(cohort=c,celltype=ct,signature_genes=d['HUGO symbols'].nunique(),measured=d['HUGO symbols'].isin(x.index).sum()))
script='''args<-commandArgs(TRUE)
source(args[1])
genes<-read.delim(args[2],check.names=FALSE,stringsAsFactors=FALSE)
for(co in c('A','B')) {
 x<-as.matrix(read.csv(file.path(args[3],paste0('mcp_expression_',co,'.csv')),row.names=1,check.names=FALSE))
 s<-MCPcounter.estimate(x,featuresType='HUGO_symbols',genes=genes)
 write.csv(t(s),file.path(args[3],paste0('mcp_official_',co,'.csv')))
}
capture.output(sessionInfo(),file=file.path(args[3],'MCP_R_sessionInfo.txt'))
'''
(o/'run_mcp_official.R').write_text(script)
subprocess.run(['Rscript',str(o/'run_mcp_official.R'),str(g/'MCPcounter.R'),str(g/'mcp_genes.txt'),str(o)],check=True)
pd.DataFrame(coverage).to_csv(o/'mcp_coverage.csv',index=False)
rows=[]
for c,(x,y) in raw.items():
    scores=pd.read_csv(o/f'mcp_official_{c}.csv',index_col=0).reindex(y.index)
    for a,b in itertools.combinations(['1','2','3','4'],2):
        for ct in scores:
            aa=scores.loc[y==a,ct].dropna();bb=scores.loc[y==b,ct].dropna()
            if min(len(aa),len(bb))<2:continue
            u,p=mannwhitneyu(aa,bb,alternative='two-sided',method='asymptotic')
            rows.append(dict(cohort=c,subtype_a=a,subtype_b=b,celltype=ct,n_a=len(aa),n_b=len(bb),rank_biserial=2*u/(len(aa)*len(bb))-1,p=p))
d=pd.DataFrame(rows)
d['q_cohort_all']=d.groupby('cohort').p.transform(lambda v:bh(v.to_numpy()))
d.to_csv(o/'mcp_official_contrasts.csv',index=False)
print('Official MCPcounter completed; all contrasts and cohort-wide BH retained')
