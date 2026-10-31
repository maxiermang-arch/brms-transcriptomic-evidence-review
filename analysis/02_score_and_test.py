import json, os, sys
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import warnings
import os, sys
from _common import ROOT, AN, SEED

def main():
    """
阶段 2 · 步骤 2：代谢通路打分 + 亚型判别力 + 交叉特异性
议题 A（脑转移 BrMS 亚型代谢打分复现审计） · 2026-09-13

设计要点（重要）：
  用代谢打分预测 BrMS3（定义本身就是"代谢型"）是**循环论证**，不构成独立验证。
  所以本脚本的输出**不是**"复现成功"的宣告，而是：
    ① 交叉特异性矩阵 —— 代谢通路是否**只**打中 BrMS3、增殖通路是否**只**打中 BrMS4
    ② 标签置换零模型 —— 真实 AUC 是否落在随机分布尾部
  真正的独立终点（OS）在步骤 3。
"""
    warnings.filterwarnings('ignore')
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    gs = json.load(open(f'{AN}/genesets/hallmark.json'))
    METAB = ['Oxidative Phosphorylation', 'Fatty Acid Metabolism', 'Glycolysis', 'Reactive Oxygen Species Pathway', 'mTORC1 Signaling']
    CTRL = ['E2F Targets', 'G2-M Checkpoint', 'Epithelial Mesenchymal Transition', 'Interferon Gamma Response', 'Allograft Rejection']
    ALL = METAB + CTRL
    X = pd.read_pickle(f'{AN}/X_counts.pkl')
    lab = pd.read_pickle(f'{AN}/labels_all.pkl')['brms']
    lab = lab[lab.index.isin(X.columns)]
    lab = lab[lab.isin(['1', '2', '3', '4'])]
    Xs = X[lab.index]
    print('=' * 78)
    print('输入')
    print('=' * 78)
    print(f'  基因 × 样本: {Xs.shape}')
    print(f'  亚型分布: {dict(lab.value_counts().sort_index())}')
    lib = Xs.sum(axis=0)
    cpm = Xs.div(lib, axis=1) * 1000000.0
    L = np.log2(cpm + 1)
    print(f'  log2(CPM+1) 完成；中位文库 {lib.median():,.0f}')
    Z = L.sub(L.mean(axis=1), axis=0).div(L.std(axis=1).replace(0, np.nan), axis=0)
    Z = Z.dropna(how='all')

    def score_meanZ(genes):
        g = [x for x in genes if x in Z.index]
        if len(g) < 10:
            return (None, 0)
        return (Z.loc[g].mean(axis=0), len(g))
    S, ncov = ({}, {})
    for name in ALL:
        s, n = score_meanZ(gs[name])
        if s is not None:
            S[name] = s
            ncov[name] = n
    S = pd.DataFrame(S)
    print(f'\n  打分矩阵: {S.shape}')
    print('  通路覆盖基因数:', ncov)

    def ovr_auc(score, y):
        yb = (y == '3').astype(int)
        return roc_auc_score(yb, score)
    print('\n' + '=' * 78)
    print('交叉特异性矩阵（one-vs-rest AUC，行=通路，列=亚型）')
    print('=' * 78)
    auc = pd.DataFrame(index=ALL, columns=['BrMS1', 'BrMS2', 'BrMS3', 'BrMS4'], dtype=float)
    for p in ALL:
        for k, sub in zip(['BrMS1', 'BrMS2', 'BrMS3', 'BrMS4'], ['1', '2', '3', '4']):
            y = (lab == sub).astype(int)
            auc.loc[p, k] = roc_auc_score(y, S[p])
    print(auc.round(3).to_string())
    print('\n  各通路的最强靶亚型:')
    for p in ALL:
        kmax = auc.loc[p].idxmax()
        vmax = auc.loc[p].max()
        kind = '代谢' if p in METAB else '对照'
        print(f'    [{kind}] {p:38s} → {kmax}  AUC={vmax:.3f}')
    print('\n' + '=' * 78)
    print('标签置换零模型（1000 次，置换 BrMS 标签）')
    print('=' * 78)
    NPERM = 1000
    perm = {p: [] for p in ALL}
    y0 = lab.values
    for i in range(NPERM):
        yp = pd.Series(RNG.permutation(y0), index=lab.index)
        for p in ALL:
            yb = (yp == '3').astype(int)
            perm[p].append(roc_auc_score(yb, S[p].values))
    perm = pd.DataFrame(perm)
    print(f"  {'通路':<38}{'真实AUC':>9}{'零分布均值':>11}{'PNULL':>9}{'经验p':>9}")
    rows = []
    for p in ALL:
        real = auc.loc[p, 'BrMS3']
        null = perm[p]
        emp_p = (np.sum(null >= real) + 1) / (NPERM + 1)
        print(f"  {p:<38}{real:>9.3f}{null.mean():>11.3f}{'':>9}{emp_p:>9.4f}")
        rows.append({'pathway': p, 'kind': 'metab' if p in METAB else 'ctrl', 'auc_BrMS3': real, 'null_mean': null.mean(), 'emp_p': emp_p})
    pd.DataFrame(rows).to_csv(f'{AN}/perm_test_BrMS3.csv', index=False)
    print('\n' + '=' * 78)
    print('★ 交叉阴性对照（破除循环论证的关键）')
    print('=' * 78)
    print('  逻辑：若代谢通路真对 BrMS3 特异，则它预测 BrMS4 的 AUC 应显著低于')
    print('        增殖通路预测 BrMS4 的 AUC；反之亦然（双向特异性）')
    sub = pd.DataFrame({'代谢→BrMS3': [auc.loc[p, 'BrMS3'] for p in METAB], '代谢→BrMS4': [auc.loc[p, 'BrMS4'] for p in METAB]}, index=METAB)
    sub2 = pd.DataFrame({'增殖→BrMS4': [auc.loc[p, 'BrMS4'] for p in ['E2F Targets', 'G2-M Checkpoint']], '增殖→BrMS3': [auc.loc[p, 'BrMS3'] for p in ['E2F Targets', 'G2-M Checkpoint']]}, index=['E2F Targets', 'G2-M Checkpoint'])
    print('\n  代谢通路组:')
    print(sub.round(3).to_string())
    print('\n  增殖通路组:')
    print(sub2.round(3).to_string())
    print(f"\n  代谢→BrMS3 均值 vs 代谢→BrMS4 均值: {sub['代谢→BrMS3'].mean():.3f} vs {sub['代谢→BrMS4'].mean():.3f}")
    print(f"  增殖→BrMS4 均值 vs 增殖→BrMS3 均值: {sub2['增殖→BrMS4'].mean():.3f} vs {sub2['增殖→BrMS3'].mean():.3f}")
    S.to_pickle(f'{AN}/scores_gse245467.pkl')
    S.to_parquet(f'{AN}/scores_gse245467.parquet', engine='pyarrow')
    auc.to_pickle(f'{AN}/auc_matrix.pkl')
    auc.to_parquet(f'{AN}/auc_matrix.parquet', engine='pyarrow')
    lab.to_frame('brms').to_pickle(f'{AN}/labels_used.pkl')
    lab.to_frame('brms').to_parquet(f'{AN}/labels_used.parquet', engine='pyarrow')
    print(f'\n  ✅ 已存 scores / auc / labels（双写 .parquet）')
if __name__ == '__main__':
    main()
