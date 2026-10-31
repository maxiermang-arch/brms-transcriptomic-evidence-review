import json, pickle, numpy as np, pandas as pd
from scipy import stats
from statsmodels.duration.hazard_regression import PHReg
import matplotlib
import matplotlib.pyplot as plt
import warnings
import os, sys
from _common import ROOT, AN, FG, SUBS, NM, NAMES, SEED, load_pkl_tuple, load_dataframe

def main():
    """
阶段 2 · 步骤 9：独立临床终点检验（GSE184869，45 例带术后生存）
议题 A · 2026-09-13

为什么做：前面全部是"标签自我一致性"检验（打分 ↔ 标签）。
        生存是**唯一真正外部的终点**——亚型/打分若真有生物学意义，应体现在生存上。
数据：Supplementary Data 2 第 7 列 Post-operative survival、第 8 列 status（DOD / censored）
      实测 45 例：41 Dead / 4 Alive（事件率 91%，信息量尚可）
方法：① 各亚型 vs 其余 的 log-rank ② 单变量 Cox（亚型，BrMS4 为参照）
      ③ 连续打分（免疫 / 代谢 / 增殖）的 Cox ④ KM 曲线
"""
    matplotlib.use('Agg')
    warnings.filterwarnings('ignore')
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    X2, lab2, os2 = load_pkl_tuple(f'{AN}/gse184869_symbol_matrix.pkl')
    k = lab2[lab2.isin(SUBS)].index
    X2 = X2[k]
    lab2 = lab2[k]
    os2 = os2.loc[k]
    T = pd.to_numeric(os2['os'], errors='coerce').values.astype(float)
    E = os2['status'].astype(str).str.lower().str.startswith('dead').astype(int).values
    print(f'n={len(T)}  事件={E.sum()}/{len(E)}  随访中位={np.median(T):.1f} 月  范围 {T.min():.0f}–{T.max():.0f}')
    Z = np.log2(X2 + 0) / 1.0
    Z = Z.sub(Z.mean(axis=1), axis=0).div(Z.std(axis=1).replace(0, np.nan), axis=0).dropna(how='all')

    def logrank(t, e, g):
        o = np.argsort(t)
        t, e, g = (t[o], e[o], g[o])
        times = np.unique(t[e == 1])
        O1 = E1 = V = 0.0
        for tt in times:
            at = t >= tt
            n = at.sum()
            n1 = (at & (g == 1)).sum()
            d = ((t == tt) & (e == 1)).sum()
            d1 = ((t == tt) & (e == 1) & (g == 1)).sum()
            if n < 2:
                continue
            O1 += d1
            E1 += d * n1 / n
            V += d * (n1 / n) * (1 - n1 / n) * (n - d) / (n - 1) if n > 1 else 0
        chi = (O1 - E1) ** 2 / V if V > 0 else 0
        return (chi, stats.chi2.sf(chi, 1), O1, E1)
    print('\n' + '=' * 90)
    print('★ 各亚型 vs 其余：log-rank（术后生存，GSE184869）')
    print('=' * 90)
    rows = []
    for s in SUBS:
        g = (lab2.values == s).astype(int)
        chi, p, O1, E1 = logrank(T, E, g)
        try:
            m = PHReg(T, g.reshape(-1, 1), status=E).fit(disp=0)
            hr = np.exp(m.params[0])
            lo, hi = np.exp(m.conf_int()[0])
            cp = m.pvalues[0]
        except Exception:
            hr = lo = hi = cp = np.nan
        med = np.median(T[g == 1])
        med0 = np.median(T[g == 0])
        rows.append({'subtype': NAMES[s], 'n': int(g.sum()), 'deaths': int(E[g == 1].sum()), 'medianOS_months': round(float(med), 1), 'medianOS_rest': round(float(med0), 1), 'logrank_chi2': round(float(chi), 3), 'logrank_p': round(float(p), 4), 'HR': round(float(hr), 3), 'HR_lo': round(float(lo), 3), 'HR_hi': round(float(hi), 3), 'cox_p': round(float(cp), 4)})
        print(f'  {NAMES[s]:<22} n={int(g.sum()):2d} 中位OS {med:6.1f} vs {med0:6.1f}月  log-rank p={p:.4f}  HR={hr:.2f} [{lo:.2f},{hi:.2f}] p={cp:.4f}')
    S = pd.DataFrame(rows)
    S.to_csv(f'{AN}/survival_by_subtype.csv', index=False)
    PANELS = {'immune (PTPRC leukocyte)': ['PTPRC', 'LCP2', 'HCLS1', 'ARHGDIB', 'CORO1A'], 'immune (T+myeloid)': ['CD3D', 'CD3E', 'CD2', 'CD68', 'CD14', 'AIF1', 'LYZ', 'TYROBP', 'FCER1G', 'C1QA'], 'metabolic (OXPHOS+FAO)': None, 'proliferative (E2F+G2M)': None, 'neural (proj guidance)': None, 'EMT epithelial(-)': ['CDH1', 'EPCAM', 'KRT8', 'KRT18', 'KRT19', 'CLDN3', 'CLDN4', 'OCLN', 'TJP1']}
    gs = dict(json.load(open(f'{AN}/genesets/hallmark.json')))
    PANELS['metabolic (OXPHOS+FAO)'] = list(set(gs['Oxidative Phosphorylation']) | set(gs['Fatty Acid Metabolism']))
    PANELS['proliferative (E2F+G2M)'] = list(set(gs['E2F Targets']) | set(gs['G2-M Checkpoint']))
    PANELS['neural (proj guidance)'] = json.load(open(f'{AN}/genesets/neural.json'))['Neuron Projection Guidance (GO:0097485)']
    print('\n' + '=' * 90)
    print('★ 连续打分的单变量 Cox（每 +1 SD）')
    print('=' * 90)
    crows = []
    for name, gl in PANELS.items():
        g = [x for x in gl if x in Z.index]
        if len(g) < 5:
            print(f'  {name:<28} 覆盖不足')
            continue
        s = Z.loc[g].mean(axis=0).values
        s = (s - s.mean()) / s.std()
        m = PHReg(T, s.reshape(-1, 1), status=E).fit(disp=0)
        hr = np.exp(m.params[0])
        lo, hi = np.exp(m.conf_int()[0])
        pp = m.pvalues[0]
        gb = (s > np.median(s)).astype(int)
        chi, lp, _, _ = logrank(T, E, gb)
        st = '***' if pp < 0.001 else '**' if pp < 0.01 else '*' if pp < 0.05 else 'ns'
        print(f'  {name:<28} n_genes={len(g):3d}  HR={hr:5.2f} [{lo:.2f},{hi:.2f}]  Cox p={pp:.4f} {st:3s} | log-rank p={lp:.4f}')
        crows.append({'panel': name, 'n_genes': len(g), 'HR_per_SD': round(float(hr), 3), 'HR_lo': round(float(lo), 3), 'HR_hi': round(float(hi), 3), 'cox_p': round(float(pp), 4), 'logrank_p': round(float(lp), 4)})
    C = pd.DataFrame(crows)
    C.to_csv(f'{AN}/survival_by_score.csv', index=False)
    cols = {'1': '#4C72B0', '2': '#DD8452', '3': '#55A868', '4': '#C44E52'}

    def km(ax, groups, title):
        for s, g in groups.items():
            t = np.sort(T[g])
            e = E[g][np.argsort(T[g])]
            n = len(t)
            surv = np.cumprod(1 - e / np.arange(n, 0, -1))
            ax.step(np.r_[0, t], np.r_[1, surv], where='post', lw=2, color=cols[s], label=f'{NAMES[s]} (n={g.sum()})')
        ax.set_xlabel('Post-operative survival (months)', fontsize=11)
        ax.set_ylabel('Overall survival', fontsize=11)
        ax.set_ylim(0, 1.02)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=8.5, frameon=False)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(alpha=0.25, ls=':')
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), dpi=300)
    ax = axes[0]
    for s in SUBS:
        g = lab2.values == s
        t = np.sort(T[g])
        e = E[g][np.argsort(T[g])]
        n = len(t)
        surv = np.cumprod(1 - e / np.arange(n, 0, -1))
        ax.step(np.r_[0, t], np.r_[1, surv], where='post', lw=2, color=cols[s], label=f'{NAMES[s]} (n={g.sum()})')
    ax.set_xlabel('Post-operative survival (months)', fontsize=11)
    ax.set_ylabel('Overall survival', fontsize=11)
    ax.set_ylim(0, 1.02)
    ax.set_title('A  BrMS subtypes (GSE184869, n=45)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8.5, frameon=False, loc='upper right')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(alpha=0.25, ls=':')
    ax = axes[1]
    focus = ['immune (T+myeloid)', 'proliferative (E2F+G2M)', 'metabolic (OXPHOS+FAO)']
    cmap = {'immune (T+myeloid)': '#DD8452', 'proliferative (E2F+G2M)': '#C44E52', 'metabolic (OXPHOS+FAO)': '#55A868'}
    for name in focus:
        g = [x for x in PANELS[name] if x in Z.index]
        s = Z.loc[g].mean(axis=0).values
        hi = s > np.median(s)
        for lab_, mask, ls_, al in [(f"{name.split(' ')[0]} high", hi, '-', 1.0), (f"{name.split(' ')[0]} low", ~hi, '--', 0.55)]:
            t = np.sort(T[mask])
            e = E[mask][np.argsort(T[mask])]
            n = len(t)
            surv = np.cumprod(1 - e / np.arange(n, 0, -1))
            ax.step(np.r_[0, t], np.r_[1, surv], where='post', lw=2, color=cmap[name], ls=ls_, alpha=al, label=lab_)
    ax.set_xlabel('Post-operative survival (months)', fontsize=11)
    ax.set_ylabel('Overall survival', fontsize=11)
    ax.set_ylim(0, 1.02)
    ax.set_title('B  Pathway scores, median split', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(alpha=0.25, ls=':')
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(f'{FG}/Fig5_survival_GSE184869.{ext}', dpi=400, bbox_inches='tight')
    print(f'\n  ✅ 已存 Fig5_survival_GSE184869.png/.pdf + survival_by_subtype.csv + survival_by_score.csv')
if __name__ == '__main__':
    main()
