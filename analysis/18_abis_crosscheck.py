#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
阶段 3 · 步骤 18：ABIS 交叉验证 + CPM/TPM 尺度敏感性
议题 A · 2026-09-13

回答三个问题：
 Q1  ABIS（17 类免疫细胞绝对构成）与我们自建的免疫面板、ESTIMATE ImmuneScore 是否一致？
     → 若是，说明"免疫含量"这个轴不是我们面板的自造产物，第三方工具也看得到。
 Q2  CPM 版 vs TPM 版 ABIS 的秩相关有多高？
     → 若 ρ≈1，则"用哪个尺度"不影响任何结论，只需在局限里声明绝对值不可解释。
 Q3  用 ABIS 总量判别 BrMS2 的 AUC，与用自建免疫面板的 AUC 是否接近？
     → 若接近，则第八章 8.4 的"自身成分域替代"结论在第三方工具下同样成立。
"""
import os, numpy as np, pandas as pd
from scipy.stats import spearmanr, mannwhitneyu
from sklearn.metrics import roc_auc_score

# —— 共享配置（P0-2 去硬编码路径）——
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import PU, SUBS  # noqa
rows = []


def rd(f):
    d = pd.read_csv(f"{PU}/{f}", sep="\t", index_col=0)
    return d


def auc(y, s):
    y = np.asarray(y).astype(int)
    if y.sum() == 0 or y.sum() == len(y):
        return np.nan
    return roc_auc_score(y, np.asarray(s, dtype=float))



def main():
    print("=" * 78)
    print("Q1/Q2  ABIS 与自建免疫面板的一致性、CPM vs TPM 尺度敏感性")
    print("=" * 78)
    for co in ("A", "B"):
        tpm = rd(f"cohort{co}_ABIS_TPM.txt")
        cpm = rd(f"cohort{co}_ABIS.txt")
        comp = rd(f"cohort{co}_composition.txt")
        g = tpm.columns.intersection(comp.index)
        tpm, cpm, comp = tpm[g], cpm[g], comp.loc[g]
    
        # Q2 尺度敏感性
        rho_cell = {c: spearmanr(tpm.loc[c], cpm.loc[c]).statistic for c in tpm.index}
        rho_all = spearmanr(tpm.values.ravel(), cpm.values.ravel()).statistic
        print(f"\n--- 队列 {co}（n={len(g)}）---")
        print(f"  逐细胞类型 Spearman(TPM vs CPM): 中位={np.median(list(rho_cell.values())):.3f} "
              f"最小={min(rho_cell.values()):.3f}")
        print(f"  全局 Spearman = {rho_all:.3f}")
        lo = sorted(rho_cell.items(), key=lambda kv: kv[1])[:3]
        print("  最不一致的 3 类:", ", ".join(f"{k} {v:.2f}" for k, v in lo))
    
        # Q1 与自建面板 / ESTIMATE 的一致性
        tot_t = tpm.sum(axis=0)
        tot_c = cpm.sum(axis=0)
        targets = {"自建免疫合计": comp["免疫合计"], "自建免疫-髓系": comp["免疫-髓系"],
                   "自建免疫-T/NK": comp["免疫-T/NK"], "ESTIMATE_Immune": comp["ESTIMATE_Immune"],
                   "ESTIMATE_Purity": comp["Purity_ESTIMATE"]}
        print("  ABIS 总量 vs 各参照（Spearman）:")
        for nm, v in targets.items():
            rt = spearmanr(tot_t, v).statistic
            rc = spearmanr(tot_c, v).statistic
            print(f"    {nm:18s}  CPM 版 ρ={rc:+.3f}   TPM 版 ρ={rt:+.3f}")
            rows.append({"cohort": co, "参照": nm, "rho_CPM": round(rc, 3), "rho_TPM": round(rt, 3)})
    
        # Q3 用 ABIS 总量判别 BrMS2
        y2 = (comp["brms"].astype(str) == "2").astype(int).values
        print("  BrMS2 判别 AUC:")
        for nm, s in [("ABIS 总量(CPM)", tot_c), ("ABIS 总量(TPM)", tot_t),
                      ("自建免疫合计", comp["免疫合计"]), ("ESTIMATE_Immune", comp["ESTIMATE_Immune"])]:
            a = auc(y2, s)
            print(f"    {nm:18s} AUC={a:.3f}")
            rows.append({"cohort": co, "参照": f"BrMS2_AUC::{nm}", "rho_CPM": round(a, 3), "rho_TPM": ""})
    
        # 髓系 / T-NK 子集对齐
        print("  子集对齐（ABIS 分群 vs 自建面板，Spearman）:")
        abis_mye = tpm.loc[[c for c in tpm.index if "Mono" in c or "Neutro" in c or "DC" in c]].sum(axis=0)
        abis_tnk = tpm.loc[[c for c in tpm.index if c.startswith("T ") or c in ("NK", "MAIT")]].sum(axis=0)
        print(f"    ABIS 髓系群 vs 自建免疫-髓系 : ρ={spearmanr(abis_mye, comp['免疫-髓系']).statistic:+.3f}")
        print(f"    ABIS T/NK 群 vs 自建免疫-T/NK: ρ={spearmanr(abis_tnk, comp['免疫-T/NK']).statistic:+.3f}")
    
    pd.DataFrame(rows).to_csv(f"{PU}/abis_crosscheck.csv", index=False)
    print(f"\n✅ 已写出 {PU}/abis_crosscheck.csv")
    

if __name__ == "__main__":
    main()
