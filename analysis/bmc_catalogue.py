from pathlib import Path
import importlib,json,pandas as pd
from enhance_common import L2
from _common import AN
co=L2.load_cohorts();out=Path(AN)/'bmc_results';rows=[]
for s,sets in importlib.import_module('42_proxy_sensitivity').proxy_sets().items():
 for code,name,genes in sets:
  rows.append(dict(subtype=s,proxy=code,name=name,genes=sorted(set(genes)),used_A=len(set(genes)&set(co['A']['Z'].index)),used_B=len(set(genes)&set(co['B']['Z'].index))))
(out/'proxy_catalogue.json').write_text(json.dumps(rows,indent=2))
pd.DataFrame([{k:v for k,v in row.items() if k!='genes'} for row in rows]).to_csv(out/'proxy_catalogue.csv',index=False)
