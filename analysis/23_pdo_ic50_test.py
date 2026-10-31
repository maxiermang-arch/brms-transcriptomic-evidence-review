"""analysis/23_pdo_ic50_test.py — PDO 药敏 IC50 × BrMS 亚型真实检验（审核修复 S1）

背景：paper_v1.md §3.6 原稿引用 palbociclib p<0.05 属虚构（管线中不存在该药与该检验）。
本脚本用 Supplementary Data 2 中 20 个带 BrMS 标签的 PDO 真实运行：
  - abemaciclib（CDK4/6 抑制剂，BrMS4 增殖轴假说：预测 BrMS4 IC50 更低）
  - everolimus（mTOR 抑制剂，对照）
  - gamitrinib（线粒体 Hsp90/TRAP1 抑制剂，BrMS3 代谢轴假说，n=12）
检验：Mann-Whitney U（one-vs-rest，双侧 + 预设方向单侧），附 effect size (rank-biserial)。
多重检验：3 药 × 4 亚型 = 12 个 primary one-vs-rest 检验做 BH-FDR（探索性，如实报告）。
RNG/种子：无随机成分（纯秩检验），完全确定。
"""
from __future__ import annotations
import os
import sys

import numpy as np
import openpyxl
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ROOT, SUPP, AN, SUBS, NM  # noqa: E402

OUT_CSV = os.path.join(AN, "pdo_ic50_by_brms.csv")


def _nonempty(v) -> bool:
    return str(v).strip() not in ("", "None", "nan")


def _f(v):
    try:
        return float(str(v).strip())
    except Exception:
        return np.nan


def load_pdo() -> pd.DataFrame:
    wb = openpyxl.load_workbook(SUPP, read_only=True)
    rows = [x for x in wb["Supplementary Data 2"].iter_rows(values_only=True)]
    cols = ["type", "cohort", "sample", "acc", "batch", "age", "os",
            "os_status", "primary", "brms", "tissue", "pheno",
            "ic50_gami", "ic50_eve", "ic50_abe"]
    df = pd.DataFrame([x[:15] for x in rows[2:] if x[0]], columns=cols)
    pdo = df[df["type"].astype(str) == "PDO"].copy()
    pdo["brms"] = pdo["brms"].astype(str).str.strip()
    for c, new in [("ic50_gami", "gamitrinib"), ("ic50_eve", "everolimus"),
                   ("ic50_abe", "abemaciclib")]:
        pdo[new] = pdo[c].map(_f)
    return pdo.set_index("sample")[["primary", "brms", "gamitrinib",
                                    "everolimus", "abemaciclib"]]


def bh_fdr(p: np.ndarray) -> np.ndarray:
    """BH-FDR（与 21 脚本同实现）。"""
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n + 1)
    adj = np.empty(n, dtype=float)
    cummin = 1.0
    for i in order[::-1]:
        cummin = min(cummin, p[i] * n / ranks[i])
        adj[i] = min(cummin, 1.0)
    return adj


def main():
    pdo = load_pdo()
    print("=" * 78)
    print("PDO 药敏 IC50 × BrMS（Supplementary Data 2，n=%d PDO）" % len(pdo))
    print("=" * 78)
    print("  BrMS 分布:", pdo["brms"].value_counts().sort_index().to_dict())
    for d in ["gamitrinib", "everolimus", "abemaciclib"]:
        print(f"  {d:12s} n={pdo[d].notna().sum()}")

    drugs = ["abemaciclib", "everolimus", "gamitrinib"]
    # 预设方向：abemaciclib → BrMS4 更敏感（lower IC50）；
    #          gamitrinib → BrMS3 更敏感；everolimus → 无方向假设（对照）
    hypo_dir = {"abemaciclib": "4", "gamitrinib": "3", "everolimus": None}

    rows = []
    for drug in drugs:
        for sub in SUBS:
            v = pdo[drug].dropna()
            lab = pdo.loc[v.index, "brms"]
            in_g = v[lab == sub].values
            out_g = v[lab != sub].values
            if len(in_g) < 2 or len(out_g) < 2:
                rows.append({"drug": drug, "subtype": sub,
                             "n_in": len(in_g), "n_out": len(out_g),
                             "median_in": np.nan, "median_out": np.nan,
                             "U": np.nan, "p_two": np.nan, "p_dir": np.nan,
                             "rank_biserial": np.nan, "preset_dir": False})
                continue
            U, p_two = stats.mannwhitneyu(in_g, out_g,
                                          alternative="two-sided")
            # 方向性单侧：in 组更低（更敏感）
            _, p_low = stats.mannwhitneyu(in_g, out_g,
                                          alternative="less")
            rb = 1 - 2 * U / (len(in_g) * len(out_g))  # rank-biserial
            rows.append({
                "drug": drug, "subtype": sub,
                "n_in": len(in_g), "n_out": len(out_g),
                "median_in": np.median(in_g), "median_out": np.median(out_g),
                "U": U, "p_two": p_two, "p_dir": p_low,
                "rank_biserial": rb,
                "preset_dir": hypo_dir[drug] == sub,
            })
    res = pd.DataFrame(rows)
    # 多重检验校正：
    #   1) 全局 12 检验（3 药 × 4 亚型）—— 用于描述性汇报，会把预设方向、
    #      探索性对照混在一起，可能稀释功效；
    #   2) 按药物分族（每族 m = 4）—— 主分析：同一药物内不同亚型间比较
    #      共享同一对照组，分族更合理；
    #   3) 对单侧 p 也做每药 m = 4 的 BH，用于判定预设方向。
    res["p_bh_global"] = bh_fdr(res["p_two"].fillna(1.0).values)
    res["p_bh_drug"] = np.nan
    res["q_dir_drug"] = np.nan
    for drug in drugs:
        m = res["drug"] == drug
        res.loc[m, "p_bh_drug"] = bh_fdr(res.loc[m, "p_two"].fillna(1.0).values)
        res.loc[m, "q_dir_drug"] = bh_fdr(res.loc[m, "p_dir"].fillna(1.0).values)
    res.to_csv(OUT_CSV, index=False)
    print("\n" + res.round(4).to_string(index=False))
    print(f"\n  ✅ Wrote {OUT_CSV}")

    # 关键读数
    print("\n" + "=" * 78)
    print("关键读数（预设方向检验：* 标记，q 为每药 m=4 BH-FDR）")
    print("=" * 78)
    for _, r in res.iterrows():
        if r["preset_dir"]:
            sig = "*" if (r["p_dir"] < 0.05 and r["q_dir_drug"] < 0.05) else "(n.s.)"
            print(f"  [{r['drug']} → BrMS{r['subtype']}] median "
                  f"{r['median_in']:.3f} vs rest {r['median_out']:.3f} | "
                  f"one-sided p={r['p_dir']:.4f}, q={r['q_dir_drug']:.3f} {sig}")


if __name__ == "__main__":
    main()
