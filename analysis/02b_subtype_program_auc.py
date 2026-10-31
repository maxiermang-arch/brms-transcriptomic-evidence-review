#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""02b · 亚型×队列×"defining program" AUC 计算（P0-4 唯一真相源）

用途：把 11_fig6_triangulation.py A panel 所需的 c1 (GSE245467) 与
c2 (GSE184869) 从硬编码数字迁移为可复现计算 + CSV 落盘。

策略：
  按 _common.FIG6_REP_PROGRAM 映射，每个亚型选一个"代表程序"，
  取 meanZ → one-vs-rest AUC。

依赖：X_counts.pkl (01), labels_gse245467.pkl (01),
       gse184869_symbol_matrix.pkl (06)

独立可跑：python 02b_subtype_program_auc.py
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# —— 共享配置（P0-2 去硬编码路径）——
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (  # noqa: E402
    AN, FIG6_REP_PROGRAM, GENESETS, SUBS, NM,
    load_dataframe, load_pkl_tuple,
)

# 输入
X_COUNTS_PKL = os.path.join(AN, "X_counts.pkl")
LABELS_GSE245467_PKL = os.path.join(AN, "labels_gse245467.pkl")
GSE184869_PKL = os.path.join(AN, "gse184869_symbol_matrix.pkl")
OUT_CSV = os.path.join(AN, "subtype_program_auc.csv")


def _zscore_log_cpm(X: pd.DataFrame) -> pd.DataFrame:
    """log2(CPM+1) → row-wise Z。（GSE245467 是 counts 输入）"""
    L = np.log2(X.div(X.sum(axis=0), axis=1) * 1e6 + 1)
    Z = L.sub(L.mean(axis=1), axis=0).div(
        L.std(axis=1).replace(0, np.nan), axis=0)
    return Z.dropna(how="all")


def _zscore_log_already(X: pd.DataFrame) -> pd.DataFrame:
    """Row-wise Z（GSE184869 已是 log2 TMM-CPM）。"""
    Z = X.sub(X.mean(axis=1), axis=0).div(
        X.std(axis=1).replace(0, np.nan), axis=0)
    return Z.dropna(how="all")


def _load_program(geneset_file: str, key: str) -> List[str]:
    path = os.path.join(GENESETS, geneset_file)
    return json.load(open(path))[key]


def _program_auc(Z: pd.DataFrame, lab: pd.Series,
                 program: List[str], sub: str) -> float:
    g = [x for x in program if x in Z.index]
    if len(g) < 5 or (lab == sub).sum() < 2:
        return float("nan")
    s = Z.loc[g].mean(axis=0)
    y = (lab == sub).astype(int)
    return float(roc_auc_score(y, s))


def main():
    # ---------- 队列 A (GSE245467, melanoma, n=52) ----------
    print("=" * 78)
    print("Cohort A · GSE245467")
    print("=" * 78)
    Xa = load_dataframe(X_COUNTS_PKL)
    laba = load_dataframe(LABELS_GSE245467_PKL).iloc[:, 0]  # Series→DataFrame 后取回 Series
    laba = laba[laba.isin(SUBS)]
    Xa = Xa[laba.index]
    Za = _zscore_log_cpm(Xa)
    print(f"  Z 矩阵: {Za.shape}  亚型: {dict(laba.value_counts().sort_index())}")

    # ---------- 队列 B (GSE184869, breast, n=45) ----------
    print("\n" + "=" * 78)
    print("Cohort B · GSE184869")
    print("=" * 78)
    Xb, labb, _ = load_pkl_tuple(GSE184869_PKL)
    labb = labb.iloc[:, 0] if isinstance(labb, pd.DataFrame) else labb  # Series 形态
    labb = labb[labb.isin(SUBS)]
    Xb = Xb[labb.index]
    Zb = _zscore_log_already(Xb)
    print(f"  Z 矩阵: {Zb.shape}  亚型: {dict(labb.value_counts().sort_index())}")

    # ---------- 亚型×队列×FIG6 代表程序 AUC ----------
    rows = []
    print("\n" + "=" * 78)
    print(f"图 6 代表程序映射（图 6 A panel 单一真相源 _common.FIG6_REP_PROGRAM）")
    print("=" * 78)
    for cohort, (Z, lab) in [("A", (Za, laba)), ("B", (Zb, labb))]:
        for sub in SUBS:
            gs_file, gs_key = FIG6_REP_PROGRAM[sub]
            program = _load_program(gs_file, gs_key)
            auc = _program_auc(Z, lab, program, sub)
            rows.append({
                "cohort": cohort,
                "subtype": NM[sub],
                "subtype_code": sub,
                "program": f"{gs_file}::{gs_key}",
                "n_genes_used": sum(1 for x in program if x in Z.index),
                "n_samples": int((lab == sub).sum()),
                "auc_one_vs_rest": round(auc, 3),
            })
    R = pd.DataFrame(rows)
    R.to_csv(OUT_CSV, index=False)
    print(R.to_string(index=False))
    print(f"\n✅ 已写 {OUT_CSV}")


if __name__ == "__main__":
    main()
