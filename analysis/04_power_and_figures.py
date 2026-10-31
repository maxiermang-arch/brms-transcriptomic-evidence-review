import json, warnings
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import os, sys
from _common import ROOT, AN, FG, SEED, RNG, load_dataframe
import os

def main():
    """
阶段 2 · 步骤 4：统计功效 + 出版级图件
议题 A · 2026-09-13

功效问题：n=54（BrMS3=17）能检出多大的 AUC？若真实效应在本设计下"检不出"，
          阴性结论就不能说成"不存在"，只能写成"本设计未检出"。
"""
    matplotlib.use('Agg')
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    warnings.filterwarnings('ignore')
    os.makedirs(FG, exist_ok=True)
    plt.rcParams.update({'font.size': 8, 'axes.linewidth': 0.7, 'font.family': 'DejaVu Sans', 'xtick.major.width': 0.7, 'ytick.major.width': 0.7})
    X = load_dataframe(f'{AN}/X_counts.pkl')
    lab = load_dataframe(f'{AN}/labels_all.pkl')['brms']
    lab = lab[lab.index.isin(X.columns)]
    lab = lab[lab.isin(list('1234'))]
    S = load_dataframe(f'{AN}/scores_meanZ.pkl')
    A = load_dataframe(f'{AN}/scores_aucell.pkl')
    auc = load_dataframe(f'{AN}/auc_matrix.pkl')
    gs = json.load(open(f'{AN}/genesets/hallmark.json'))
    METAB = ['Oxidative Phosphorylation', 'Fatty Acid Metabolism', 'Glycolysis', 'ROS', 'mTORC1 Signaling']
    METAB = ['Oxidative Phosphorylation', 'Fatty Acid Metabolism', 'Glycolysis', 'mTORC1 Signaling']
    n1, n2 = ((lab == '3').sum(), (lab != '3').sum())
    print('=' * 78)
    print(f'统计功效模拟（n1={n1} BrMS3 vs n2={n2} 其他，2000 次/点）')
    print('=' * 78)

    def sim_power(n1, n2, d, alpha=0.05, B=2000):
        rej = 0
        for _ in range(B):
            a = RNG.normal(0, 1, n1)
            b = RNG.normal(d, 1, n2)
            if stats.mannwhitneyu(a, b, alternative='two-sided').pvalue < alpha:
                rej += 1
        return rej / B
    print(f"  {'真实 Cohen d':>14}{'对应 AUC':>11}{'功效':>9}")
    for d in [0.3, 0.5, 0.8, 1.0, 1.2, 1.5]:
        auroc = stats.norm.cdf(d / np.sqrt(2))
        print(f'  {d:>14.2f}{auroc:>11.3f}{sim_power(n1, n2, d):>9.2f}')
    print(f'\n  → 本设计（n1={n1}/n2={n2}）能达 80% 功效的最小真实 AUC ≈ {stats.norm.cdf(1.0 / np.sqrt(2)):.2f}–{stats.norm.cdf(1.2 / np.sqrt(2)):.2f}')
    print('  → 观察到的代谢通路 AUC 0.556–0.600 落在『检不出』区间：')
    print('     这是**功效不足**吗？需与增殖通路的 0.883 对比——同样的 n，增殖通路轻松显著，')
    print('     说明问题不在功效，而在**代谢信号本身弱**。')
    order = ['Oxidative Phosphorylation', 'Fatty Acid Metabolism', 'Glycolysis', 'mTORC1 Signaling', 'Epithelial Mesenchymal Transition', 'Interferon Gamma Response', 'Allograft Rejection', 'E2F Targets', 'G2-M Checkpoint']
    M = auc.loc[[p for p in order if p in auc.index]]
    fig, ax = plt.subplots(figsize=(3.4, 3.2), dpi=400)
    im = ax.imshow(M.values, cmap='RdBu_r', norm=TwoSlopeNorm(vmin=0.35, vcenter=0.5, vmax=0.95), aspect='auto')
    ax.set_xticks(range(M.shape[1]))
    ax.set_xticklabels(M.columns, fontsize=7.5)
    ax.set_yticks(range(M.shape[0]))
    short = [p.replace('Oxidative Phosphorylation', 'OXPHOS').replace('Fatty Acid Metabolism', 'FAO').replace('Epithelial Mesenchymal Transition', 'EMT').replace('Interferon Gamma Response', 'IFN-γ').replace('Allograft Rejection', 'Allograft').replace('G2-M Checkpoint', 'G2M') for p in M.index]
    ax.set_yticklabels(short, fontsize=7.5)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M.values[i, j]
            ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=6.2, color='white' if v > 0.8 or v < 0.36 else 'black')
    ax.set_title('Cross-specificity matrix (one-vs-rest AUC)', fontsize=8, pad=6)
    ax.axhline(3.5, color='k', lw=0.9)
    ax.axhline(6.5, color='k', lw=0.9)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=6.5)
    cb.set_label('AUC', fontsize=7)
    fig.tight_layout()
    fig.savefig(f'{FG}/Fig1_cross_specificity.png', bbox_inches='tight')
    fig.savefig(f'{FG}/Fig1_cross_specificity.pdf', bbox_inches='tight')
    print(f'\n  ✅ Fig1 → figures/Fig1_cross_specificity.png/.pdf')
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9), dpi=400)
    labels = ['OXPHOS', 'FAO', 'Glycolysis', 'mTORC1', 'E2F', 'G2M']
    mz = [roc_auc_score((lab == '3').astype(int), S[p]) if p in METAB else roc_auc_score((lab == '4').astype(int), S[p]) for p in ['Oxidative Phosphorylation', 'Fatty Acid Metabolism', 'Glycolysis', 'mTORC1 Signaling', 'E2F Targets', 'G2-M Checkpoint']]
    ac = [roc_auc_score((lab == '3').astype(int), A[p]) if p in METAB else roc_auc_score((lab == '4').astype(int), A[p]) for p in ['Oxidative Phosphorylation', 'Fatty Acid Metabolism', 'Glycolysis', 'mTORC1 Signaling', 'E2F Targets', 'G2-M Checkpoint']]
    tgt = ['BrMS3'] * 4 + ['BrMS4'] * 2
    x = np.arange(len(labels))
    w = 0.36
    for ax, name, vals in [(axes[0], 'mean Z-score', mz), (axes[1], 'AUCell (rank-based)', ac)]:
        cols = ['#c0392b' if t == 'BrMS3' else '#2471a3' for t in tgt]
        ax.bar(x, vals, w, color=cols, edgecolor='k', linewidth=0.5)
        ax.axhline(0.5, color='gray', ls='--', lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=6.8, rotation=30, ha='right')
        ax.set_ylim(0.3, 1.0)
        ax.set_title(name, fontsize=8)
        ax.set_ylabel('AUC (one-vs-rest)' if ax is axes[0] else '', fontsize=7.5)
        for xi, v in zip(x, vals):
            ax.text(xi, v + 0.012, f'{v:.2f}', ha='center', fontsize=5.8)
    from matplotlib.patches import Patch
    axes[1].legend(handles=[Patch(facecolor='#c0392b', edgecolor='k', label='metabolic → BrMS3'), Patch(facecolor='#2471a3', edgecolor='k', label='proliferative → BrMS4')], fontsize=6.2, loc='upper left', frameon=False, bbox_to_anchor=(0.0, 0.99))
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(f'{FG}/Fig2_method_dependence.png', bbox_inches='tight')
    fig.savefig(f'{FG}/Fig2_method_dependence.pdf', bbox_inches='tight')
    print(f'  ✅ Fig2 → figures/Fig2_method_dependence.png/.pdf')
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.4), dpi=400)
    show = [('Oxidative Phosphorylation', '3'), ('Fatty Acid Metabolism', '3'), ('E2F Targets', '4'), ('G2-M Checkpoint', '4')]
    for ax, (p, sub) in zip(axes, show):
        a = S[p][lab == sub].values
        b = S[p][lab != sub].values
        parts = ax.violinplot([a, b], showextrema=False, widths=0.8)
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor('#c0392b' if sub == '3' else '#2471a3')
            pc.set_alpha(0.55)
            pc.set_edgecolor('k')
        bp = ax.boxplot([a, b], widths=0.18, showfliers=False, patch_artist=True)
        for i, box in enumerate(bp['boxes']):
            box.set_facecolor('white')
            box.set_edgecolor('k')
            box.set_linewidth(0.6)
        for el in ['whiskers', 'caps', 'medians']:
            for it in bp[el]:
                it.set_color('k')
                it.set_linewidth(0.6)
        u, pv = stats.mannwhitneyu(a, b, alternative='two-sided')
        ax.set_xticks([1, 2])
        ax.set_xticklabels([f'BrMS{sub}', 'Other'], fontsize=7)
        ax.set_title(p.replace('Oxidative Phosphorylation', 'OXPHOS').replace('Fatty Acid Metabolism', 'FAO').replace('G2-M Checkpoint', 'G2M'), fontsize=7.5)
        ptxt = 'p < 0.001' if pv < 0.001 else f'p = {pv:.3f}'
        ax.text(0.5, 0.94, ptxt, transform=ax.transAxes, ha='center', fontsize=6.5, color='#7b241c' if pv < 0.001 else 'black')
        ax.axhline(0, color='gray', ls=':', lw=0.6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(f'{FG}/Fig3_score_distributions.png', bbox_inches='tight')
    fig.savefig(f'{FG}/Fig3_score_distributions.pdf', bbox_inches='tight')
    print(f'  ✅ Fig3 → figures/Fig3_score_distributions.png/.pdf')
    print('\n  图件已全部生成')
if __name__ == '__main__':
    main()
