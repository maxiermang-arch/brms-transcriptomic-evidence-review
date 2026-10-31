#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
阶段 3 · 步骤 12：为 ESTIMATE / ABIS 导出输入矩阵
议题 A · 2026-09-13

导出两份：
  A) log2 尺度（给 ESTIMATE，rank-based ssGSEA，log 尺度更稳）
  B) 线性尺度（给 ABIS，rlm 稳健线性回归，配方要求 TPM/线性）
     - GSE184869：已有 log2 TMM-CPM → 2^x − 1 还原为线性 TMM-CPM
     - GSE245467：raw counts → CPM（线性）
     ⚠️ 局限：ABIS 官方要求 TPM；GSE245467 无基因长度 → 用 CPM 近似，须在文档中声明
"""
import numpy as np, pandas as pd, pickle, os
# —— 共享配置（P0-2 去硬编码路径）——
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ROOT, AN, PU, SUBS, safe_makedirs  , load_pkl_tuple  , load_dataframe  # noqa
os.makedirs(PU,exist_ok=True)


def main():
    # ---------------- 队列 A：GSE245467（raw counts）
    X1=load_dataframe(f"{AN}/X_counts.pkl")
    lab1=load_dataframe(f"{AN}/labels_gse245467.pkl")
    k=lab1[lab1.isin(SUBS)].index; X1=X1[k]; lab1=lab1[k]
    cpm1=(X1/X1.sum(axis=0)*1e6)
    lg1=np.log2(cpm1+1)
    lg1.to_csv(f"{PU}/cohortA_log2CPM.tsv",sep="\t")
    cpm1.to_csv(f"{PU}/cohortA_linearCPM.tsv",sep="\t")
    print(f"队列 A  GSE245467: {lg1.shape}")
    
    # ---------------- 队列 B：GSE184869（log2 TMM-CPM）
    X2,lab2,_=load_pkl_tuple(f"{AN}/gse184869_symbol_matrix.pkl")
    k2=lab2[lab2.isin(SUBS)].index; X2=X2[k2]; lab2=lab2[k2]
    lin2=np.power(2.0,X2)-1.0
    lin2[lin2<0]=0
    X2.to_csv(f"{PU}/cohortB_log2TMM.tsv",sep="\t")
    lin2.to_csv(f"{PU}/cohortB_linearTMM.tsv",sep="\t")
    print(f"队列 B  GSE184869: {X2.shape}")
    
    # ---------------- 标签
    lab=pd.DataFrame({"cohort":["A"]*len(lab1)+["B"]*len(lab2),
                      "sample":list(lab1.index)+list(lab2.index),
                      "brms":list(lab1.values)+list(lab2.values)})
    lab.to_csv(f"{PU}/labels.tsv",sep="\t",index=False)
    print(f"标签: {dict(lab.groupby('cohort')['brms'].value_counts())}")
    print("✅ 导出完成")
    

if __name__ == "__main__":
    main()
