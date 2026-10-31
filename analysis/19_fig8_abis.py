#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
阶段 3 · 步骤 19：Fig8 —— ABIS 外部交叉验证与尺度敏感性（含负面结果）
议题 A · 2026-09-13
"""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings("ignore")

# —— 共享配置（P0-2 去硬编码路径）——
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ROOT, PU, FG, COHORT_META  # noqa
rd = lambda f: pd.read_csv(f"{PU}/{f}", sep="\t", index_col=0)


def main():
    COL = {"A": "#4878CF", "B": "#EE854A"}
    LBL = {"A": "cohort A (melanoma, n=52)", "B": "cohort B (breast, n=45)"}
    
    fig = plt.figure(figsize=(13.6, 4.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.25, 1.0], wspace=0.30)
    
    # ---------------- Panel A: ABIS 总量 vs 自建免疫面板 ----------------
    ax = fig.add_subplot(gs[0, 0])
    for co in ("A", "B"):
        t = rd(f"cohort{co}_ABIS.txt").sum(axis=0)
        c = rd(f"cohort{co}_composition.txt")
        g = t.index.intersection(c.index)
        x, y = t[g].values, c.loc[g, "免疫合计"].values
        rho = spearmanr(x, y).statistic
        ax.scatter(x, y, s=26, alpha=.82, color=COL[co], edgecolor="white", linewidth=.5,
                   label=f"{LBL[co]}   $\\rho$={rho:+.2f}")
    ax.set_xlabel("ABIS total immune content (%)", fontsize=9)
    ax.set_ylabel("own immune panel score (z)", fontsize=9)
    ax.set_title("A  Third-party ABIS vs own immune panel", fontsize=9.6, loc="left", fontweight="bold")
    ax.legend(fontsize=7.6, loc="upper left", framealpha=.9)
    ax.tick_params(labelsize=8); ax.grid(alpha=.18, lw=.6)
    
    # ---------------- Panel B: 逐细胞类型 CPM vs TPM 秩相关 ----------------
    ax = fig.add_subplot(gs[0, 1])
    rho = {}
    for co in ("A", "B"):
        t, c = rd(f"cohort{co}_ABIS_TPM.txt"), rd(f"cohort{co}_ABIS.txt")
        rho[co] = {k: spearmanr(t.loc[k], c.loc[k]).statistic for k in t.index}
    order = sorted(rho["A"], key=lambda k: (rho["A"][k] + rho["B"][k]) / 2, reverse=True)
    ypos = np.arange(len(order))
    h = .39
    ax.barh(ypos + h / 2, [rho["A"][k] for k in order], height=h, color=COL["A"], label="cohort A")
    ax.barh(ypos - h / 2, [rho["B"][k] for k in order], height=h, color=COL["B"], label="cohort B")
    ax.axvline(0.8, color="#B03A2E", ls="--", lw=1, alpha=.8)
    ax.text(0.805, len(order) - .3, " ρ=0.80", color="#B03A2E", fontsize=7.4, va="top")
    ax.set_yticks(ypos); ax.set_yticklabels(order, fontsize=7.4)
    ax.set_xlim(0, 1.05); ax.set_xlabel(r"Spearman $\rho$  (same cells, TPM vs CPM input)", fontsize=8.6)
    ax.set_title("B  Scale sensitivity of ABIS per cell type", fontsize=9.6, loc="left", fontweight="bold")
    ax.legend(fontsize=7.6, loc="lower right", framealpha=.9)
    ax.tick_params(labelsize=8); ax.grid(alpha=.18, lw=.6, axis="x")
    
    # ---------------- Panel C: BrMS2 判别 AUC 四方对比 ----------------
    ax = fig.add_subplot(gs[0, 2])
    meth = ["ABIS\n(CPM)", "ABIS\n(TPM)", "own immune\npanel", "ESTIMATE\nImmune"]
    vals = {co: [] for co in ("A", "B")}
    for co in ("A", "B"):
        t, c, comp = rd(f"cohort{co}_ABIS_TPM.txt"), rd(f"cohort{co}_ABIS.txt"), rd(f"cohort{co}_composition.txt")
        g = t.columns.intersection(comp.index); comp = comp.loc[g]
        # baseline 兼容：composition.txt 的 brms 列在 14 端可能写成 str 或 float
        # （取决于 lA 是否有 NaN）。统一归一化为 string 以便 == "2" 比较稳定。
        br = comp["brms"].astype(str).str.rstrip(".0").replace({"nan": ""})
        y2 = (br == "2").astype(int).values
        for s in (c[g].sum(axis=0), t[g].sum(axis=0), comp["免疫合计"], comp["ESTIMATE_Immune"]):
            vals[co].append(roc_auc_score(y2, np.asarray(s, dtype=float)))
    xp = np.arange(len(meth)); w = .36
    ax.bar(xp - w / 2, vals["A"], w, color=COL["A"], label="cohort A")
    ax.bar(xp + w / 2, vals["B"], w, color=COL["B"], label="cohort B")
    ax.axhline(0.5, color="#666666", ls=":", lw=1)
    for i, (a, b) in enumerate(zip(vals["A"], vals["B"])):
        ax.text(i - w / 2, a + .015, f"{a:.2f}", ha="center", fontsize=7.4)
        ax.text(i + w / 2, b + .015, f"{b:.2f}", ha="center", fontsize=7.4)
    ax.set_xticks(xp); ax.set_xticklabels(meth, fontsize=7.8)
    ax.set_ylim(0, 1.12); ax.set_ylabel("AUC for BrMS2", fontsize=9)
    ax.set_title("C  Who can call BrMS2?", fontsize=9.6, loc="left", fontweight="bold")
    ax.legend(fontsize=7.4, loc="upper left", ncol=2, framealpha=.9)
    ax.tick_params(labelsize=8); ax.grid(alpha=.18, lw=.6, axis="y")
    
    fig.savefig(f"{FG}/Fig8_abis_crosscheck.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(f"{FG}/Fig8_abis_crosscheck.pdf", bbox_inches="tight", facecolor="white")
    print("✅ Fig8 已输出")
    print("Panel C values:", {k: [round(v, 3) for v in vv] for k, vv in vals.items()})
    

if __name__ == "__main__":
    main()
