"""analysis/24_fig1_formal.py — 正式 Figure 1（出版级, ≥300 dpi PNG + PDF）

内容（对应 paper_v1.md Figure Legend）:
  (A) Cohort A (GSE245467, n=52): 4 个 programme score 按 BrMS 真值分组的 boxplot
      —— 展示各 programme 的组内判别力（标注 raw AUC, 16-system L1 数字）
  (B) Cohort C (E-MTAB-8659, n=63, unlabelled): programme score 密度分布
      与 cohort A 叠加 —— 展示跨平台分布一致性（BrMS4 最一致, BrMS3 差异最大）

规范: 画布宽 174 mm (Springer 双栏全宽), 英文标注 (无 CJK), Arial,
      PNG 300 dpi + 矢量 PDF, 无网格少墨。
数字来源: AUC 来自 auc_matrix/02b L1 (raw): 1=0.931, 2=0.708, 3=0.884, 4=0.888
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import AN, FG, SUBS, NM, BRMS_PROGRAMS

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 7,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
})

NM_EN = {"1": "neural", "2": "immune/EMT", "3": "epithelial/metabolic", "4": "proliferative"}
AUC_L1 = {"1": 0.931, "2": 0.708, "3": 0.884, "4": 0.888}
COL = {"1": "#7B68AE", "2": "#59A14F", "3": "#F1A83B", "4": "#D1615D"}
COL_C = "#4C72B0"  # cohort C overlay

MM = 1 / 25.4
FIGW, FIGH = 174 * MM, 128 * MM


def main():
    # ---- 1) Cohort A per-sample programme scores (与 22 脚本同一算法) ----
    lab = pd.read_parquet(f"{AN}/labels_used.parquet")
    Xc = pd.read_parquet(f"{AN}/X_counts.parquet")
    X = Xc[lab.index]
    cpm = X.div(X.sum(axis=0), axis=1) * 1e6
    L = np.log2(cpm + 1)
    ZA = L.sub(L.mean(axis=1), axis=0).div(L.std(axis=1).replace(0, np.nan), axis=0)
    REF = pd.DataFrame(index=lab.index)
    for sub in SUBS:
        gset = [g for g in BRMS_PROGRAMS[sub] if g in ZA.index]
        REF[sub] = ZA.loc[gset].mean(axis=0)
    REF_z = (REF - REF.mean(axis=0)) / REF.std(axis=0).replace(0, np.nan)
    y = lab["brms"].astype(str).values

    # ---- 2) Cohort C z-scores ----
    S_z = pd.read_csv(f"{AN}/emtab8659_br_projection_zscores.csv", index_col=0)
    S_z.columns = S_z.columns.astype(str)

    # ---- 3) Figure: 2 rows x 4 cols ----
    fig, axes = plt.subplots(2, 4, figsize=(FIGW, FIGH))

    # Panel row A: cohort A boxplots by label
    for j, sub in enumerate(SUBS):
        ax = axes[0, j]
        data = [REF_z.loc[y == s, sub].dropna().values for s in SUBS]
        bp = ax.boxplot(
            data, widths=0.55, patch_artist=True, showfliers=True,
            flierprops=dict(marker="o", ms=2, mfc="none", mec="0.3", mew=0.5),
            medianprops=dict(color="black", lw=0.8),
            whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6),
        )
        for patch, s in zip(bp["boxes"], SUBS):
            patch.set_facecolor(COL[s]); patch.set_alpha(0.75)
            patch.set_linewidth(0.6)
        ax.set_xticks(range(1, 5))
        ax.set_xticklabels([f"BrMS{s}" for s in SUBS], fontsize=6)
        ax.set_title(f"BrMS{sub} score ({NM_EN[sub]})\nAUC = {AUC_L1[sub]:.2f}", fontsize=7)
        if j == 0:
            ax.set_ylabel("z-score (cohort A)", fontsize=7)
        ax.tick_params(labelsize=6, length=2.5, width=0.5)

    # Panel row B: density overlay A (grey) vs C (blue)
    from scipy.stats import gaussian_kde
    for j, sub in enumerate(SUBS):
        ax = axes[1, j]
        a = REF_z[sub].dropna().values
        c = S_z[sub].dropna().values
        grid = np.linspace(min(a.min(), c.min()) - 0.3, max(a.max(), c.max()) + 0.3, 300)
        ax.plot(grid, gaussian_kde(a)(grid), color="0.55", lw=1.0,
                label=f"GSE245467 (n=52)")
        ax.fill_between(grid, gaussian_kde(a)(grid), color="0.55", alpha=0.15)
        ax.plot(grid, gaussian_kde(c)(grid), color=COL_C, lw=1.0,
                label=f"E-MTAB-8659 (n=63)")
        ax.fill_between(grid, gaussian_kde(c)(grid), color=COL_C, alpha=0.15)
        ax.set_xlabel(f"BrMS{sub} programme z-score", fontsize=7)
        if j == 0:
            ax.set_ylabel("density", fontsize=7)
            ax.legend(fontsize=5.5, frameon=False, loc="upper left")
        ax.tick_params(labelsize=6, length=2.5, width=0.5)

    fig.text(0.012, 0.975, "A", fontsize=10, fontweight="bold", va="top")
    fig.text(0.012, 0.485, "B", fontsize=10, fontweight="bold", va="top")

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
    fig.subplots_adjust(hspace=0.55, wspace=0.32)

    out_png = f"{FG}/Figure1_programme_scores.png"
    out_pdf = f"{FG}/Figure1_programme_scores.pdf"
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)
    for p in (out_png, out_pdf):
        print(f"OK {p} ({os.path.getsize(p)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
