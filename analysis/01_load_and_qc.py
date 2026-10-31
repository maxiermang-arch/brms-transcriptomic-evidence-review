#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
阶段 2 · 步骤 1：数据装载 + 质量核查 + 独立终点（生存）可用性
议题 A（脑转移 BrMS 亚型代谢打分复现审计） · 2026-09-13

关键设计说明：
  用代谢打分预测 BrMS3（定义=代谢型）本身是**循环论证**，不构成独立验证。
  唯一出路是预测**不依赖代谢定义的外部终点** → 生存（OS）或 PDO 药敏 IC50。
  本步骤先确认 GSE245467 / 公共数据集中生存数据的可用性。
"""
from __future__ import annotations
import json
import os
import re
import sys
from collections import Counter

import numpy as np
import openpyxl
import pandas as pd

# —— 共享配置（P0-2 去硬编码路径）——
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ROOT, DATA, SUPP, AN  # noqa


# 01 自身产物 → 下游装载目标
X_COUNTS_PKL = os.path.join(AN, "X_counts.pkl")
LABELS_ALL_PKL = os.path.join(AN, "labels_all.pkl")
LABELS_GSE245467_PKL = os.path.join(AN, "labels_gse245467.pkl")


def _num(v):
    """字符串数值化（容忍 None / 空 / 异常）。"""
    try:
        return float(str(v).strip())
    except Exception:
        return np.nan


def main():
    """所有副作用集中在本函数（import 时无副作用，PMR-safe）。"""
    # ----------------------------------------------------------- Data 2 全表
    wb = openpyxl.load_workbook(SUPP, read_only=True)
    r2 = [x for x in wb["Supplementary Data 2"].iter_rows(values_only=True)]
    hdr = r2[1]
    cols = ["type", "cohort", "sample", "acc", "batch", "age",
            "os", "os_status", "primary", "brms", "tissue", "pheno",
            "ic50_gami", "ic50_eve", "ic50_abe"]
    df2 = pd.DataFrame([x[:15] for x in r2[2:] if x[0]], columns=cols)

    print("=" * 78)
    print("Data 2 · 全表")
    print("=" * 78)
    print(f"  总行数: {len(df2)}")
    for c in ["os", "os_status", "ic50_gami", "ic50_eve", "ic50_abe"]:
        nn = df2[c].notna() & (df2[c].astype(str).str.strip() != "")
        print(f"  {c:12s} 非空 {nn.sum():4d} / {len(df2)}  ({nn.mean():.1%})")

    print("\n  各数据集生存数据可用性:")
    print(f"  {'数据集':<16}{'n':>5}{'有OS':>7}{'有status':>9}{'有效OS':>8}")
    surv_rows = []
    for acc, g in df2.groupby(df2["acc"].astype(str).str.strip()):
        n = len(g)
        hasos = g["os"].notna().sum()
        hasst = g["os_status"].notna().sum()
        oss = g["os"].map(_num)
        valid = ((oss.notna()) & (oss > 0)).sum()
        print(f"  {acc:<16}{n:>5}{hasos:>7}{hasst:>9}{valid:>8}")
        if valid >= 20:
            surv_rows.append(acc)

    print(f"\n  ★ 有效生存数据 ≥20 的数据集: {surv_rows}")

    # ----------------------------------------------------------- GSE245467 · counts
    print("\n" + "=" * 78)
    print("GSE245467 · counts 矩阵")
    print("=" * 78)

    # 加载 counts .xlsx（与 X_counts.pkl 同源；01 首次运行时不存在 pickle，
    # 这里按需跑一次解析并落到 X_COUNTS_PKL；后续运行可直接复用）
    counts_xlsx = os.path.join(ROOT, "data", "geo",
                               "GSE245467_raw_counts.xlsx")
    X, samples, bad_genes = _load_counts_xlsx(counts_xlsx)

    # 标签映射
    lab = df2[df2["acc"].astype(str).str.strip() == "GSE245467"]
    lab = lab.set_index(lab["sample"].astype(str).str.strip())[
        "brms"].astype(str).str.strip()
    common = [s for s in samples if s in lab.index]
    print(f"  保留基因: {X.shape[0]}   丢弃(Excel 日期化): {len(bad_genes)}")
    print(f"  样本: {X.shape[1]}")
    print(f"  总 counts: {X.values.sum():,.0f}")
    print(f"  零表达基因: {(X.sum(axis=1) == 0).sum()}")
    print(f"\n  ★ 标签可用样本: {len(common)} / {len(samples)}")
    print(f"  亚型分布: {dict(Counter(lab[s] for s in common))}")

    # 生存（GSE245467）
    g467 = df2[df2["acc"].astype(str).str.strip() == "GSE245467"].copy()
    g467["sample"] = g467["sample"].astype(str).str.strip()
    g467 = g467.set_index("sample")
    os_ = g467["os"].map(_num)
    st_ = g467["os_status"].map(_num)
    print(f"  GSE245467 生存数据: OS 有效 {((os_.notna()) & (os_ > 0)).sum()} "
          f"/ {len(g467)}; status 有效 {st_.notna().sum()}")

    # ----------------------------------------------------------- 落盘（P1-1 双写 pickle + parquet）
    X.to_pickle(X_COUNTS_PKL)
    X.to_parquet(X_COUNTS_PKL[:-4] + ".parquet", engine="pyarrow")
    lab.to_frame("brms").to_pickle(LABELS_ALL_PKL)
    lab.to_frame("brms").to_parquet(LABELS_ALL_PKL[:-4] + ".parquet",
                                    engine="pyarrow")
    pd.Series({s: lab[s] for s in common}, name="brms"
              ).to_pickle(LABELS_GSE245467_PKL)
    pd.Series({s: lab[s] for s in common}, name="brms"
              ).to_parquet(LABELS_GSE245467_PKL[:-4] + ".parquet",
                           engine="pyarrow")
    json.dump({"bad_genes": bad_genes,
               "n_genes": int(X.shape[0]),
               "n_samples": int(X.shape[1]),
               "surv_accs": surv_rows},
              open(os.path.join(AN, "step1_meta.json"), "w"),
              ensure_ascii=False, indent=2)
    print(f"\n  ✅ 已存 {X_COUNTS_PKL}（双写 .parquet）等")


def _load_counts_xlsx(path: str):
    """从 GSE245467 raw counts xlsx 还原 X / samples / bad_genes。"""
    wb3 = openpyxl.load_workbook(path, read_only=True)
    ws = wb3[wb3.sheetnames[0]]
    it = ws.iter_rows(values_only=True)
    h = next(it)
    samples = [str(c).strip() for c in h[1:] if c is not None]
    genes_raw, mat = [], []
    BAD = re.compile(r'^[0-9/:\- ]+[a-z]?$|^\d{4}-\d{2}-\d{2}')
    bad_genes = []
    for row in it:
        if row[0] is None:
            continue
        g = str(row[0]).strip()
        if BAD.match(g):
            bad_genes.append(g)
            continue
        genes_raw.append(g)
        mat.append([0 if v is None else int(v)
                    for v in row[1:1 + len(samples)]])
    X = pd.DataFrame(np.array(mat, dtype=np.float64),
                     index=genes_raw, columns=samples)
    return X, samples, bad_genes


if __name__ == "__main__":
    main()
