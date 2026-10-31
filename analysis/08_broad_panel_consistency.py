import json, pickle, numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
import warnings
import os, sys
from _common import ROOT, AN, SUBS, NM, NAMES, RNG, load_pkl_tuple, load_dataframe

def main():
    from _common import SEED
    RNG = np.random.default_rng(SEED)
    """
阶段 2 · 步骤 8：无偏全通路「标签↔生物学映射」跨队列一致性（★核心判据）
议题 A · 2026-09-13

为什么要做：步骤 7 用的 16 个面板偏免疫/基质，对 BrMS4（增殖型）天然不利
           → ρ(BrMS4) 不显著可能是"面板集选择偏倚"，不是标签问题。
方法：改用 **全部 50 条 Hallmark + 8 条神经 GO = 58 条通路**（覆盖度 ≥80% 才入）
      对每个亚型，把「58 条通路 → 该亚型的 AUC」组成向量，在两队列间求 Spearman ρ。
      零模型：置换队列 2 的标签重算 ρ。
用途：**一个数字回答"这个亚型的标签在另一个队列里还指同一套生物学吗"。**
"""
    warnings.filterwarnings('ignore')
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    X1 = load_dataframe(f'{AN}/X_counts.pkl')
    lab1 = load_dataframe(f'{AN}/labels_gse245467.pkl')
    X2, lab2, _ = load_pkl_tuple(f'{AN}/gse184869_symbol_matrix.pkl')
    k1 = lab1[lab1.isin(SUBS)].index
    X1 = X1[k1]
    lab1 = lab1[k1]
    k2 = lab2[lab2.isin(SUBS)].index
    X2 = X2[k2]
    lab2 = lab2[k2]

    def norm(X, is_counts):
        L = np.log2(X / X.sum(axis=0) * 1000000.0 + 1) if is_counts else X.copy()
        return L.sub(L.mean(axis=1), axis=0).div(L.std(axis=1).replace(0, np.nan), axis=0).dropna(how='all')
    Z1 = norm(X1, True)
    Z2 = norm(X2, False)
    gs = dict(json.load(open(f'{AN}/genesets/hallmark.json')))
    gs.update(json.load(open(f'{AN}/genesets/neural.json')))
    panels = {}
    for k, v in gs.items():
        g1 = [x for x in v if x in Z1.index]
        g2 = [x for x in v if x in Z2.index]
        if len(v) >= 15 and len(g1) / len(v) >= 0.8 and (len(g2) / len(v) >= 0.8):
            panels[k] = (g1, g2)
    print(f'入组通路: {len(panels)} / {len(gs)}（覆盖度门槛 80%，每集 ≥15 基因）')

    def aucs(Z, lab, sub, gl):
        s = Z.loc[gl].mean(axis=0).values
        y = np.asarray(lab) == sub
        return roc_auc_score(y.astype(int), s)
    P = list(panels)
    A1 = {}
    A2 = {}
    for p in P:
        g1, g2 = panels[p]
        for sub in SUBS:
            A1.setdefault(sub, []).append(aucs(Z1, lab1, sub, g1))
            A2.setdefault(sub, []).append(aucs(Z2, lab2, sub, g2))
    D1 = pd.DataFrame(A1, index=P)
    D2 = pd.DataFrame(A2, index=P)
    D1.to_csv(f'{AN}/panel_auc_broad_cohort1.csv')
    D2.to_csv(f'{AN}/panel_auc_broad_cohort2.csv')
    print('\n' + '=' * 88)
    print('★ 标签↔生物学映射：58 条通路的跨队列一致性（置换零模型 2000 次）')
    print('=' * 88)
    rows = []
    for sub in SUBS:
        a = D1[sub].values
        b = D2[sub].values
        rho = stats.spearmanr(a, b).statistic
        cnt = 0
        for _ in range(2000):
            perm = RNG.permutation(lab2.values)
            bb = [aucs(Z2, perm, sub, panels[p][1]) for p in P]
            if stats.spearmanr(a, bb).statistic >= rho:
                cnt += 1
        pp = (cnt + 1) / 2001
        t1 = set(D1[sub].nlargest(10).index)
        t2 = set(D2[sub].nlargest(10).index)
        jac = len(t1 & t2) / len(t1 | t2)
        med = float(np.median(np.abs(a - b)))
        mx = float(np.max(a))
        mx2 = float(np.max(b))
        star = '***' if pp < 0.001 else '**' if pp < 0.01 else '*' if pp < 0.05 else 'ns'
        rows.append({'subtype': NAMES[sub], 'n': f'{(lab1 == sub).sum()}v{(lab2 == sub).sum()}', 'c1_bestAUC': round(mx, 3), 'c2_bestAUC': round(mx2, 3), 'spearman_rho': round(float(rho), 3), 'p_perm': round(pp, 4), 'sig': star, 'top10_Jaccard': round(jac, 3), 'median|dAUC|': round(med, 3)})
    R = pd.DataFrame(rows)
    print(R.to_string(index=False))
    print('-' * 88)
    print('\n★ 各亚型在两队列的 top-8 通路')
    for sub in SUBS:
        print(f'\n  [{NAMES[sub]}]')
        t1 = D1[sub].nlargest(8)
        t2 = D2[sub].nlargest(8)
        for i, (a, b) in enumerate(zip(t1.items(), t2.items())):
            print(f'    c1 {a[0][:44]:<46}{a[1]:.3f}   |   c2 {b[0][:44]:<46}{b[1]:.3f}')
    R.to_csv(f'{AN}/broad_panel_consistency.csv', index=False)
    print(f'\n  ✅ 已存 broad_panel_consistency.csv / panel_auc_broad_cohort{{1,2}}.csv')
if __name__ == '__main__':
    main()
