#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
阶段 2 · 步骤 6：第二队列独立验证（GSE184869）
议题 A · 2026-09-13

GSE184869：中山大学肿瘤防治中心，RNA-seq，log2 TMM-CPM（batch corrected），
           protein-coding；90 样本中 45 例有 BrMS 标签 + 45 例有效 OS。
用途：① 独立队列复现四亚型审计（不同中心/不同批次/已 batch-correct）
     ② 若一致 → 证明 GSE245467 的结论不是单队列偶然
"""
import json, pickle, os, warnings
import numpy as np, pandas as pd
import openpyxl, mygene
from scipy import stats
from sklearn.metrics import roc_auc_score
# —— 共享配置（P0-2 去硬编码路径）——
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ROOT, AN, GEO, SUPP, RNG  # noqa
warnings.filterwarnings("ignore")


def main():
    CACHE=f"{AN}/gse184869_symbol_matrix.pkl"
    
    # ---------------------------------------------------------------- 载入 + 映射
    if os.path.exists(CACHE):
        X2, lab2, os2 = pickle.load(open(CACHE,"rb"))
        print("从缓存载入")
    else:
        wb=openpyxl.load_workbook(f"{ROOT}/data/geo/GSE184869_matrix.xlsx", read_only=True)
        ws=wb[wb.sheetnames[0]]
        it=ws.iter_rows(values_only=True); h=next(it)
        cols=[str(c).strip() for c in h[1:]]
        ids=[]; vals=[]
        for row in it:
            if row[0] is None: continue
            ids.append(str(row[0]).strip())
            vals.append([np.nan if v is None else float(v) for v in row[1:1+len(cols)]])
        E=pd.DataFrame(np.array(vals), index=ids, columns=cols)
        print(f"Ensembl 矩阵: {E.shape}")
    
        # Ensembl → Symbol
        mg=mygene.MyGeneInfo()
        q=mg.querymany(E.index.tolist(), scopes="ensembl.gene", fields="symbol",
                       species="human", as_dataframe=True, verbose=False)
        q=q[q["symbol"].notna()].copy()
        q=q[~q.index.duplicated(keep="first")]
        print(f"映射成功: {len(q)} / {len(E)}")
        m=E.loc[q.index].copy()
        m.index=q["symbol"].values
        m=m[~m.index.duplicated(keep="first")]
        X2=m
        # 标签
        wb2=openpyxl.load_workbook(f"{ROOT}/data/supplementary/41467_2026_68748_MOESM3_ESM.xlsx",read_only=True)
        rr=[x for x in wb2["Supplementary Data 2"].iter_rows(values_only=True)]
        g=[x for x in rr[2:] if x[0] and str(x[3]).strip()=="GSE184869"]
        lab_map={str(x[2]).strip():str(x[9]).strip() for x in g}
        os_map={str(x[2]).strip():(x[6],x[7]) for x in g}
        used=[c for c in X2.columns if lab_map.get(c) in list("1234")]
        X2=X2[used]
        lab2=pd.Series({c:lab_map[c] for c in used}, name="brms")
        os2=pd.DataFrame({c:{"os":os_map[c][0],"status":os_map[c][1]} for c in used}).T
        pickle.dump((X2,lab2,os2), open(CACHE,"wb"))
        # P1-1 双写：解包为 3 个 .parquet + manifest.json
        import json as _json
        stem = CACHE[:-4]
        X2.to_parquet(stem + ".X.parquet", engine="pyarrow")
        lab2.to_frame("brms").to_parquet(stem + ".labels.parquet", engine="pyarrow")
        os2.to_parquet(stem + ".os.parquet", engine="pyarrow")
        _json.dump({"src": CACHE, "format": "tuple(X=DataFrame, labels=Series, os=DataFrame)",
                    "pieces": [stem + ".X.parquet", stem + ".labels.parquet",
                               stem + ".os.parquet"]},
                   open(stem + ".manifest.json", "w"), ensure_ascii=False, indent=2)
        print(f"标签样本: {len(used)}（双写 .parquet）")
    
    print("="*80); print("GSE184869（第二队列，中山肿瘤）"); print("="*80)
    print(f"  基因 × 样本: {X2.shape}")
    print(f"  亚型分布: {dict(lab2.value_counts().sort_index())}")
    
    # 已是 log2 TMM-CPM，不再 CPM 归一化
    L=X2.copy()
    Z=L.sub(L.mean(axis=1),axis=0).div(L.std(axis=1).replace(0,np.nan),axis=0).dropna(how="all")
    
    gs_h=json.load(open(f"{AN}/genesets/hallmark.json"))
    gs_n=json.load(open(f"{AN}/genesets/neural.json"))
    DIMS={
     "Neural-like (BrMS1)": [("GO",k,v) for k,v in gs_n.items()],
     "Immune / EMT (BrMS2)": [("HM","Interferon Gamma Response",gs_h["Interferon Gamma Response"]),
                              ("HM","Epithelial Mesenchymal Transition",gs_h["Epithelial Mesenchymal Transition"]),
                              ("HM","Allograft Rejection",gs_h["Allograft Rejection"]),
                              ("HM","Inflammatory Response",gs_h["Inflammatory Response"])],
     "Metabolic (BrMS3)": [("HM","Oxidative Phosphorylation",gs_h["Oxidative Phosphorylation"]),
                           ("HM","Fatty Acid Metabolism",gs_h["Fatty Acid Metabolism"]),
                           ("HM","Glycolysis",gs_h["Glycolysis"]),
                           ("HM","mTORC1 Signaling",gs_h["mTORC1 Signaling"])],
     "Proliferative (BrMS4)": [("HM","E2F Targets",gs_h["E2F Targets"]),
                               ("HM","G2-M Checkpoint",gs_h["G2-M Checkpoint"]),
                               ("HM","Myc Targets V1",gs_h["Myc Targets V1"])],
    }
    SUB={"Neural-like (BrMS1)":"1","Immune / EMT (BrMS2)":"2",
         "Metabolic (BrMS3)":"3","Proliferative (BrMS4)":"4"}
    
    rows=[]
    for dim,items in DIMS.items():
        sub=SUB[dim]; y=(lab2==sub).astype(int).values
        print(f"\n### {dim}  (n={y.sum()} vs {len(y)-y.sum()})")
        print(f"  {'基因集':<40}{'覆盖':>6}{'AUC':>8}{'δ':>8}{'p(perm)':>10}")
        for src,name,gl in items:
            g=[x for x in gl if x in Z.index]
            if len(g)<10: print(f"  {name[:38]:<40} 覆盖不足"); continue
            s=Z.loc[g].mean(axis=0); auc=roc_auc_score(y,s.values)
            a=s[lab2==sub].values; b=s[lab2!=sub].values
            gt=np.sum(a[:,None]>b[None,:]); lt=np.sum(a[:,None]<b[None,:]); cliff=(gt-lt)/(len(a)*len(b))
            cnt=0
            for _ in range(1000):
                if roc_auc_score(RNG.permutation(y),s.values)>=auc: cnt+=1
            pp=(cnt+1)/1001
            star="***" if pp<0.001 else "**" if pp<0.01 else "*" if pp<0.05 else "ns"
            print(f"  {name[:38]:<40}{len(g):>6}{auc:>8.3f}{cliff:>8.3f}{pp:>10.4f} {star}")
            rows.append({"cohort":"GSE184869","dimension":dim,"geneset":name,"n_genes":len(g),
                         "auc":auc,"cliff_delta":cliff,"p_perm":pp})
    r2=pd.DataFrame(rows); r2.to_csv(f"{AN}/audit_GSE184869.csv",index=False)
    
    print("\n"+"="*80); print("★ 汇总对照"); print("="*80)
    sm=r2.groupby("dimension").agg(best_auc=("auc","max"),mean_auc=("auc","mean"),min_p=("p_perm","min"))
    print(sm.round(3).to_string())
    sm.to_csv(f"{AN}/audit_summary_GSE184869.csv")
    print(f"\n  ✅ 已存 audit_GSE184869.csv / audit_summary_GSE184869.csv")
    
    # 与队列 1 对比
    try:
        r1=pd.read_csv(f"{AN}/audit_four_subtypes.csv")
        s1=r1.groupby("dimension").agg(best_auc_1=("auc","max"))
        s1.index=[i.replace("  "," ") for i in s1.index]
        cmp=pd.DataFrame({"GSE245467_bestAUC":s1["best_auc_1"],"GSE184869_bestAUC":sm["best_auc"]})
        cmp["Δ"]=cmp["GSE184869_bestAUC"]-cmp["GSE245467_bestAUC"]
        print("\n  两队列对照:")
        print(cmp.round(3).to_string())
        cmp.to_csv(f"{AN}/two_cohort_comparison.csv")
    except Exception as e:
        print("  (队列1 对照跳过:",e,")")
    

if __name__ == "__main__":
    main()
