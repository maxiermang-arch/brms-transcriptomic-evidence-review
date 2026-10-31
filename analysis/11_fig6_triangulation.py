#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""11 · 三角验证总图（Fig6） · 全部数字从 CSV 读取（P0-4）

P0-4 后所有 panel 数字均从 CSV 读取，不再使用模块级 magic numbers：

  A panel (c1/c2)
    source: subtype_program_auc.csv (02b_subtype_program_auc.py 写盘)
    元数据：cohort × subtype × auc_one_vs_rest，按 FIG6_REP_PROGRAM 重算

  B panel (met / met_p / n)
    source: metabolomics_vs_BrMS.csv (10_gold_standard_collision.py 写盘)

  C panel (vals, #免疫细胞型 FDR<0.05)
    source: paper_MCPcounter.csv (从原始 ipynb 导出)，本脚本按 6 个 pair 重算

任何数字若与论文 Fig6 不一致，**以 CSV 为真相**；写入 fig6_provenance.json。
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# —— 共享配置（P0-2 去硬编码路径）——
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import AN, FG, NM, SUBS  # noqa: E402


SUBTYPE_AUC_CSV = os.path.join(AN, "subtype_program_auc.csv")      # 02b 写盘
METABOLOMICS_CSV = os.path.join(AN, "metabolomics_vs_BrMS.csv")     # 10 写盘
MCP_COUNTER_CSV = os.path.join(AN, "paper_MCPcounter.csv")         # 数据来源
PROVENANCE_JSON = os.path.join(AN, "fig6_provenance.json")

# C panel 的 6 个 pair（顺序敏感）
C_PANEL_PAIRS = [("1", "2"), ("1", "3"), ("1", "4"),
                 ("2", "3"), ("2", "4"), ("3", "4")]


def _load_a_panel() -> tuple[list[float], list[float]]:
    """从 subtype_program_auc.csv 取 cohort × subtype → c1 (A), c2 (B)。"""
    df = pd.read_csv(SUBTYPE_AUC_CSV)
    c1, c2 = [], []
    for sub in SUBS:
        # subtype_code 在 CSV 里是 int（如 1,2,3,4），SUBS 是 str 比较需统一
        sub_int = int(sub)
        row_a = df[(df.cohort == "A") & (df.subtype_code == sub_int)]
        row_b = df[(df.cohort == "B") & (df.subtype_code == sub_int)]
        if row_a.empty or row_b.empty:
            raise RuntimeError(
                f"subtype_program_auc.csv 缺少 cohort×subtype 行（{sub}）")
        c1.append(float(row_a.iloc[0]["auc_one_vs_rest"]))
        c2.append(float(row_b.iloc[0]["auc_one_vs_rest"]))
    return c1, c2


def _load_b_panel() -> tuple[list[float], list[float], list[int]]:
    """从 metabolomics_vs_BrMS.csv 取 met / met_p / n，按 BrMS1–4 顺序。"""
    df = pd.read_csv(METABOLOMICS_CSV)
    df["code"] = df["subtype"].str.extract(r"BrMS(\d)").astype(int)
    df = df.set_index("code").reindex([1, 2, 3, 4])
    if df["PCA_best_auc"].isna().any():
        raise RuntimeError("metabolomics_vs_BrMS.csv 缺少 BrMS1–4 全部 4 行")
    met = df["PCA_best_auc"].astype(float).round(3).tolist()
    met_p = df["PCA_perm_p"].astype(float).tolist()
    n = df["n"].astype(int).tolist()
    return met, met_p, n


def _compute_c_panel() -> list[int]:
    """从 paper_MCPcounter.csv 重算：6 个两两比较对中 FDR<0.05 的 celltype 数。"""
    df = pd.read_csv(MCP_COUNTER_CSV)
    df = df[df["group1"].astype(str).isin(["1", "2", "3", "4"])
            & df["group2"].astype(str).isin(["1", "2", "3", "4"])]
    df["p.adj"] = pd.to_numeric(df["p.adj"], errors="coerce")
    vals = []
    for g1, g2 in C_PANEL_PAIRS:
        mask = ((df["group1"].astype(str) == g1)
                & (df["group2"].astype(str) == g2))
        n_sig = int(df.loc[mask, "p.adj"].lt(0.05).sum())
        vals.append(n_sig)
    return vals


def main():
    # ---------- 读所有数字 ----------
    c1, c2 = _load_a_panel()
    met, met_p, n = _load_b_panel()
    vals = _compute_c_panel()
    mcp = [14, 2, 6, 7]  # C panel 中标注"1vs3 + 1vs4 ..."的合计（仅文本标注）

    # ---------- 落 provenance ----------
    provenance = {
        "A_panel_c1": {"source": os.path.basename(SUBTYPE_AUC_CSV),
                       "values": c1, "semantics": "Best defining program AUC, cohort A (GSE245467, melanoma, n=52)"},
        "A_panel_c2": {"source": os.path.basename(SUBTYPE_AUC_CSV),
                       "values": c2, "semantics": "Best defining program AUC, cohort B (GSE184869, breast, n=45)"},
        "B_panel_met": {"source": os.path.basename(METABOLOMICS_CSV),
                        "values": met, "semantics": "PCA AUC of 75-metabolite panel, QMH (one batch)"},
        "B_panel_met_p": {"source": os.path.basename(METABOLOMICS_CSV),
                          "values": met_p, "semantics": "perm p-value for B panel AUC"},
        "B_panel_n": {"source": os.path.basename(METABOLOMICS_CSV),
                      "values": n, "semantics": "n samples per BrMS subtype in metabolomics"},
        "C_panel_vals": {"source": os.path.basename(MCP_COUNTER_CSV),
                         "values": vals, "semantics": "n immune cell types with FDR<0.05 per pairwise comparison"},
    }
    with open(PROVENANCE_JSON, "w") as f:
        json.dump(provenance, f, ensure_ascii=False, indent=2)

    # ---------- 画图 ----------
    cols = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    fig = plt.figure(figsize=(15, 4.6), dpi=300)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1, 1.05], wspace=0.30)
    x = np.arange(4)
    w = 0.36

    # A 两队列转录组
    ax = fig.add_subplot(gs[0])
    ax.bar(x - w / 2, c1, w, color=[c for c in cols], alpha=.95,
           edgecolor="white", lw=1.2, label="GSE245467 (melanoma, n=52)")
    ax.bar(x + w / 2, c2, w, color=[c for c in cols], alpha=.45,
           edgecolor="white", lw=1.2, hatch="///",
           label="GSE184869 (breast, n=45)")
    ax.axhline(.5, color="k", ls=":", lw=1.2)
    ax.text(3.42, .515, "chance", fontsize=8, style="italic")
    for xi, (a, b) in enumerate(zip(c1, c2)):
        ax.text(xi - w / 2, a + .015, f"{a:.2f}",
                ha="center", fontsize=8.5, fontweight="bold")
        ax.text(xi + w / 2, b + .015, f"{b:.2f}",
                ha="center", fontsize=8.5)
    ax.set_xticks(x); ax.set_xticklabels(SUBS, fontsize=9); ax.set_ylim(0, 1.08)
    ax.set_ylabel("Best AUC of the subtype's defining program", fontsize=9.5)
    ax.set_title("A  Transcriptomic reproducibility\nacross two independent cohorts",
                 fontsize=10.5, fontweight="bold")
    ax.legend(fontsize=7.4, frameon=False, loc="upper center", ncol=1,
              bbox_to_anchor=(.5, 1.0))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=.22, ls=":")

    # B 代谢组
    ax = fig.add_subplot(gs[1])
    ax.bar(x, met, .62, color=cols, alpha=.9, edgecolor="white", lw=1.2)
    ax.axhline(.5, color="k", ls=":", lw=1.2)
    for xi, (v, p) in enumerate(zip(met, met_p)):
        s = "***" if p < 0.01 else "**" if p < 0.05 else "" if p < 0.1 else "n.s."
        ax.text(xi, v + .018,
                f"{v:.2f}\n{s if s else 'n.s.'}",
                ha="center", fontsize=8.5,
                fontweight="bold" if p < 0.05 else "normal",
                color=cols[xi] if p < 0.05 else "#555")
        ax.text(xi, .035, f"n={n[xi]}",
                ha="center", fontsize=8, color="white", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(SUBS, fontsize=9); ax.set_ylim(0, 1.08)
    ax.set_ylabel("PCA AUC of measured energy metabolome", fontsize=9.5)
    ax.set_title("B  Gold-standard collision\n(75-metabolite panel, QMH, one batch)",
                 fontsize=10.5, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=.22, ls=":")

    # C 论文自己的 MCP-counter
    ax = fig.add_subplot(gs[2])
    labs = ["1 vs 2", "1 vs 3", "1 vs 4", "2 vs 3", "2 vs 4", "3 vs 4"]
    bc = ["#BBBBBB" if v == 0 else ("#C44E52" if v >= 6 else "#DD8452")
          for v in vals]
    ax.barh(np.arange(6), vals, .62, color=bc, edgecolor="white", lw=1.2)
    for i, v in enumerate(vals):
        ax.text(v + .12, i, f"{v}/9", va="center", fontsize=9,
                fontweight="bold" if v else "normal",
                color="#C44E52" if v >= 6 else "#444")
    ax.set_yticks(np.arange(6)); ax.set_yticklabels(labs, fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 9)
    ax.set_xlabel("# immune/stromal cell types with FDR < 0.05", fontsize=9.5)
    ax.set_title("C  The paper's own MCP-counter\nBrMS1, not BrMS2, separates the subtypes",
                 fontsize=10.5, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=.22, ls=":")

    fig.text(.5, .005,
             "Independent audit of the BrMS stratification system  ·  "
             "GSE245467 / GSE184869 / Supplementary Data 15 & 19  ·  "
             "2026-09-13",
             ha="center", fontsize=8, style="italic", color="#555")
    fig.tight_layout(rect=[0, 0.035, 1, 1])
    for ext in ("png", "pdf"):
        fig.savefig(f"{FG}/Fig6_triangulation.{ext}",
                    dpi=400, bbox_inches="tight")
    print(f"✅ Fig6_triangulation.png/.pdf")
    print(f"✅ 已写 provenance: {PROVENANCE_JSON}")


if __name__ == "__main__":
    main()
