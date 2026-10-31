import json, pickle, numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
import warnings
import os, sys
from _common import ROOT, AN, SUBS, NAMES, RNG, load_pkl_tuple, load_dataframe

def main():
    from _common import SEED
    RNG = np.random.default_rng(SEED)
    """
阶段 2 · 步骤 7：跨队列「标签↔生物学映射」一致性检验
议题 A · 2026-09-13

背景：两队列四亚型审计出现分野
  GSE245467 : BrMS1 0.931 / BrMS2 0.529 / BrMS3 0.600 / BrMS4 0.883
  GSE184869 : BrMS1 0.872 / BrMS2 0.925 / BrMS3 0.653 / BrMS4 0.790
→ BrMS2（免疫/EMT）在两队列间剧烈翻转（Δ=+0.40）。必须回答：
  (a) 这是"通路打分方法"的假象吗？→ 换用**具体细胞类型标记物**（不依赖 Hallmark 通路）
  (b) 还是"标签在该队列里本来就没对应那套生物学"？→ 用「标签↔面板效应量向量」的跨队列相关

核心统计量：对每个亚型，把「N 个免疫/基质面板 → 该亚型的 AUC」组成向量，
            在两队列间做 Spearman ρ。ρ 高 = 该标签在两队列指向同一套生物学。
            并用标签置换构造 ρ 的零分布。
"""
    warnings.filterwarnings('ignore')
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    PANELS = {'T cell (CD3)': ['CD3D', 'CD3E', 'CD3G', 'CD2', 'CD247', 'TRAC', 'LCK', 'ZAP70', 'ITK', 'CD28', 'IL7R', 'TRBC2'], 'CD8 T cell': ['CD8A', 'CD8B', 'CD3D', 'CD3E', 'CD3G', 'GZMK', 'CCL5', 'CD27'], 'Treg': ['FOXP3', 'IL2RA', 'CTLA4', 'IKZF2', 'TNFRSF18', 'TNFRSF4', 'CCR8'], 'NK cell': ['KLRD1', 'NKG7', 'GNLY', 'PRF1', 'KLRF1', 'NCR1', 'KIR2DL3'], 'B cell': ['MS4A1', 'CD19', 'CD79A', 'CD79B', 'PAX5', 'BLK', 'TCL1A'], 'Plasma cell': ['MZB1', 'JCHAIN', 'IGHG1', 'IGKC', 'SDC1', 'TNFRSF17'], 'Myeloid/Macrophage': ['CD68', 'CD14', 'CSF1R', 'AIF1', 'LYZ', 'ITGAM', 'FCGR1A', 'MSR1', 'C1QA', 'C1QB', 'CD163', 'TYROBP', 'FCER1G'], 'Monocyte': ['S100A8', 'S100A9', 'S100A12', 'FCN1', 'VCAN', 'CD14', 'LILRB1'], 'Dendritic cell': ['ITGAX', 'CD1C', 'CLEC9A', 'LAMP3', 'CCR7', 'BATF3', 'FLT3'], 'Neutrophil': ['FCGR3B', 'CSF3R', 'CXCR2', 'S100A8', 'S100A9', 'MNDA', 'FPR1'], 'Mast cell': ['TPSAB1', 'TPSB2', 'CPA3', 'MS4A2', 'KIT', 'HDC'], 'Cytolytic activity': ['GZMA', 'GZMB', 'GZMH', 'PRF1', 'GNLY', 'NKG7', 'KLRD1'], 'T-cell exhaustion': ['PDCD1', 'CTLA4', 'LAG3', 'HAVCR2', 'TIGIT', 'TOX', 'ENTPD1'], 'Leukocyte (PTPRC)': ['PTPRC', 'LCP2', 'HCLS1', 'ARHGDIB', 'CORO1A'], 'EMT mesenchymal(+)': ['CDH2', 'VIM', 'FN1', 'SNAI1', 'SNAI2', 'TWIST1', 'ZEB1', 'ZEB2', 'MMP2', 'MMP9', 'TGFB1', 'SPARC', 'COL1A1', 'COL1A2', 'FBN1', 'ACTA2'], 'EMT epithelial(-)': ['CDH1', 'EPCAM', 'KRT8', 'KRT18', 'KRT19', 'CLDN3', 'CLDN4', 'OCLN', 'TJP1']}
    gs_h = json.load(open(f'{AN}/genesets/hallmark.json'))
    PANELS['IFN-gamma (HM)'] = gs_h['Interferon Gamma Response']
    X1 = load_dataframe(f'{AN}/X_counts.pkl')
    lab1 = load_dataframe(f'{AN}/labels_gse245467.pkl')
    X2, lab2, _ = load_pkl_tuple(f'{AN}/gse184869_symbol_matrix.pkl')
    keep1 = lab1[lab1.isin(SUBS)].index
    X1 = X1[keep1]
    lab1 = lab1[keep1]
    keep2 = lab2[lab2.isin(SUBS)].index
    X2 = X2[keep2]
    lab2 = lab2[keep2]

    def norm(X, is_counts):
        L = np.log2(X / X.sum(axis=0) * 1000000.0 + 1) if is_counts else X.copy()
        Z = L.sub(L.mean(axis=1), axis=0).div(L.std(axis=1).replace(0, np.nan), axis=0)
        return Z.dropna(how='all')
    Z1 = norm(X1, True)
    Z2 = norm(X2, False)
    print('=' * 84)
    print(f'队列 1 GSE245467 : {Z1.shape}  亚型 {dict(lab1.value_counts().sort_index())}')
    print(f'队列 2 GSE184869 : {Z2.shape}  亚型 {dict(lab2.value_counts().sort_index())}')
    print('=' * 84)

    def auc_table(Z, lab):
        rows = {}
        for pname, gl in PANELS.items():
            g = [x for x in gl if x in Z.index]
            if len(g) < 5:
                continue
            s = Z.loc[g].mean(axis=0).values
            rows[pname] = {sub: roc_auc_score((lab == sub).astype(int).values, s) for sub in SUBS if (lab == sub).sum() > 0}
            rows[pname]['n_genes'] = len(g)
        return pd.DataFrame(rows).T
    A1 = auc_table(Z1, lab1)
    A2 = auc_table(Z2, lab2)
    common = [p for p in A1.index if p in A2.index]
    nG = {p: int(A1.loc[p, 'n_genes']) for p in common}
    print(f'\n可用面板: {len(common)} 个\n')

    def rho_and_p(a, b, B=2000):
        """Spearman ρ 及其置换零分布（在队列 B 上置换标签后重算 AUC 向量）"""
        rho = stats.spearmanr(a, b).statistic
        cnt = 0
        for _ in range(B):
            perm = lab2.sample(frac=1, random_state=int(RNG.integers(1000000000.0))).values
            bb = []
            for p in common:
                g = [x for x in PANELS[p] if x in Z2.index]
                s = Z2.loc[g].mean(axis=0).values
                bb.append(roc_auc_score((perm == sub).astype(int), s))
            if stats.spearmanr(a, bb).statistic >= rho:
                cnt += 1
        return (rho, (cnt + 1) / (B + 1))
    reps = []
    for sub in SUBS:
        a = A1[sub][common].values
        b = A2[sub][common].values
        rho, p = rho_and_p(a, b)
        reps.append({'subtype': NAMES[sub], 'cohort1_bestAUC': round(float(a.max()), 3), 'cohort2_bestAUC': round(float(b.max()), 3), 'spearman_rho': round(float(rho), 3), 'p_perm': round(float(p), 4), 'rho_star': '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'})
    R = pd.DataFrame(reps)
    print('★ 标签↔生物学映射的跨队列一致性（16 个免疫/基质面板）')
    print('-' * 84)
    print(R.to_string(index=False))
    print('-' * 84)
    print('\n★ BrMS2：逐面板 AUC 对照')
    print(f"  {'面板':<24}{'队列1':>8}{'队列2':>8}{'Δ':>8}   {'方向'}")
    det = []
    for p in common:
        a = float(A1.loc[p, '2'])
        b = float(A2.loc[p, '2'])
        d = b - a
        arrow = '↑↑' if d > 0.15 else '↑' if d > 0.05 else '↓↓' if d < -0.15 else '↓' if d < -0.05 else '='
        print(f'  {p:<24}{a:>8.3f}{b:>8.3f}{d:>+8.3f}   {arrow}')
        det.append({'panel': p, 'n_genes': nG[p], 'auc_c1': round(a, 3), 'auc_c2': round(b, 3), 'delta': round(d, 3)})
    pd.DataFrame(det).to_csv(f'{AN}/brms2_panel_detail.csv', index=False)
    print('\n★ BrMS3：逐面板 AUC 对照（对照用）')
    for p in common:
        a = float(A1.loc[p, '3'])
        b = float(A2.loc[p, '3'])
        print(f'  {p:<24}{a:>8.3f}{b:>8.3f}{b - a:>+8.3f}')
    pd.DataFrame({'cohort1': A1[SUBS].astype(float).mean(axis=1).round(3), 'cohort2': A2[SUBS].astype(float).mean(axis=1).round(3), 'n_genes': [nG[p] for p in A1.index]}).to_csv(f'{AN}/panel_auc_both_cohorts.csv')
    R.to_csv(f'{AN}/cross_cohort_consistency.csv', index=False)
    print(f'\n  ✅ 已存 cross_cohort_consistency.csv / brms2_panel_detail.csv / panel_auc_both_cohorts.csv')
if __name__ == '__main__':
    main()
