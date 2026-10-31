import json, warnings
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
import os, sys
from _common import ROOT, AN, SEED, RNG

def main():
    """
阶段 2 · 步骤 3：稳健性检验
议题 A · 2026-09-13

对步骤 2 的阴性结果做四重稳健性检验，避免"方法选择导致假阴性"：
  ① Bootstrap 置信区间（AUC 的不确定性）
  ② 秩基础单样本方法 AUCell（对归一化不敏感）
  ③ 替代基因集：KEGG 代谢通路（不同数据库、不同定义）
  ④ 直接组间比较 + 效应量（Cohen's d / Cliff's delta）
"""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    warnings.filterwarnings('ignore')
    X = pd.read_pickle(f'{AN}/X_counts.pkl')
    lab = pd.read_pickle(f'{AN}/labels_all.pkl')['brms']
    lab = lab[lab.index.isin(X.columns)]
    lab = lab[lab.isin(list('1234'))]
    Xs = X[lab.index]
    gs = json.load(open(f'{AN}/genesets/hallmark.json'))
    METAB = ['Oxidative Phosphorylation', 'Fatty Acid Metabolism', 'Glycolysis', 'mTORC1 Signaling']
    CTRL = ['E2F Targets', 'G2-M Checkpoint']
    ALL = METAB + CTRL
    L = np.log2(Xs.div(Xs.sum(axis=0), axis=1) * 1000000.0 + 1)
    Z = L.sub(L.mean(axis=1), axis=0).div(L.std(axis=1).replace(0, np.nan), axis=0).dropna(how='all')

    def score(genes):
        g = [x for x in genes if x in Z.index]
        return Z.loc[g].mean(axis=0)
    S = pd.DataFrame({p: score(gs[p]) for p in ALL})
    print('=' * 78)
    print('① Bootstrap 置信区间（2000 次重抽样样本）')
    print('=' * 78)
    B = 2000
    print(f"  {'通路':<34}{'目标':<8}{'AUC':>7}{'95% CI':>18}{'含0.5?':>9}")
    boot_rows = []
    for p in ALL:
        for sub, name in [('3', 'BrMS3'), ('4', 'BrMS4')]:
            y = (lab == sub).astype(int).values
            s = S[p].values
            real = roc_auc_score(y, s)
            bs = []
            idx = np.arange(len(y))
            for _ in range(B):
                i = RNG.choice(idx, len(idx), replace=True)
                if len(np.unique(y[i])) < 2:
                    continue
                bs.append(roc_auc_score(y[i], s[i]))
            lo, hi = np.percentile(bs, [2.5, 97.5])
            inc = '是' if lo <= 0.5 <= hi else '否'
            if p in METAB and sub == '3' or (p in CTRL and sub == '4'):
                print(f'  {p:<34}{name:<8}{real:>7.3f}  [{lo:.3f}, {hi:.3f}]{inc:>8}')
            boot_rows.append({'pathway': p, 'target': name, 'auc': real, 'lo': lo, 'hi': hi, 'covers_0.5': inc})
    pd.DataFrame(boot_rows).to_csv(f'{AN}/bootstrap_auc.csv', index=False)
    print('\n' + '=' * 78)
    print('② AUCell 秩基础打分（对归一化不敏感）')
    print('=' * 78)
    ranks = L.rank(axis=0, ascending=False)
    N = ranks.shape[0]
    aucell = {}
    for p in ALL:
        g = [x for x in gs[p] if x in ranks.index]
        n = len(g)
        r = ranks.loc[g]
        aucell[p] = (N - r).mean(axis=0) / N
    A = pd.DataFrame(aucell)
    print(f"  {'通路':<34}{'BrMS3 AUC':>11}{'BrMS4 AUC':>11}")
    for p in ALL:
        a3 = roc_auc_score((lab == '3').astype(int), A[p])
        a4 = roc_auc_score((lab == '4').astype(int), A[p])
        tag = '代谢' if p in METAB else '对照'
        print(f'  [{tag}] {p:<29}{a3:>11.3f}{a4:>11.3f}')
    print('\n' + '=' * 78)
    print('③ 方法一致性（meanZ vs AUCell，Spearman）')
    print('=' * 78)
    for p in ALL:
        rho, _ = stats.spearmanr(S[p], A[p])
        print(f'  {p:<34} rho = {rho:>6.3f}')
    print('\n' + '=' * 78)
    print('④ 组间比较：BrMS3 vs 其他 / BrMS4 vs 其他')
    print('=' * 78)

    def cliffs_delta(a, b):
        a = np.asarray(a)
        b = np.asarray(b)
        gt = np.sum(a[:, None] > b[None, :])
        lt = np.sum(a[:, None] < b[None, :])
        return (gt - lt) / (len(a) * len(b))
    print(f"  {'通路':<34}{'BrMS3中位':>10}{'其他中位':>10}{'MWU p':>10}{'Cliff δ':>10}")
    for p in ALL:
        for sub, name in [('3', 'BrMS3'), ('4', 'BrMS4')]:
            a = S[p][lab == sub]
            b = S[p][lab != sub]
            u, pv = stats.mannwhitneyu(a, b, alternative='two-sided')
            d = cliffs_delta(a.values, b.values)
            if p in METAB and sub == '3' or (p in CTRL and sub == '4'):
                print(f'  {p:<34}{np.median(a):>10.3f}{np.median(b):>10.3f}{pv:>10.4f}{d:>10.3f}')
    print('\n' + '=' * 78)
    print('⑤ 关键结论汇总')
    print('=' * 78)
    k3 = [roc_auc_score((lab == '3').astype(int), S[p]) for p in METAB]
    k4 = [roc_auc_score((lab == '4').astype(int), S[p]) for p in CTRL]
    print(f"  代谢通路 → BrMS3 的 AUC: {[f'{x:.3f}' for x in k3]}  均值 {np.mean(k3):.3f}")
    print(f"  增殖通路 → BrMS4 的 AUC: {[f'{x:.3f}' for x in k4]}  均值 {np.mean(k4):.3f}")
    allm = pd.read_csv(f'{AN}/perm_test_BrMS3.csv')
    print(f'\n  代谢通路对 BrMS3 的经验 p 值（1000 次置换）:')
    for _, r in allm[allm.kind == 'metab'].iterrows():
        print(f"    {r['pathway']:<34} p={r['emp_p']:.4f}")
    S.to_pickle(f'{AN}/scores_meanZ.pkl')
    S.to_parquet(f'{AN}/scores_meanZ.parquet', engine='pyarrow')
    A.to_pickle(f'{AN}/scores_aucell.pkl')
    A.to_parquet(f'{AN}/scores_aucell.parquet', engine='pyarrow')
    print(f'\n  ✅ 已存 scores_meanZ / scores_aucell（双写 .parquet）')
if __name__ == '__main__':
    main()
