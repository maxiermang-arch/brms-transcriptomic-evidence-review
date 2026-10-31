# -*- coding: utf-8 -*-
import json, numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
from statsmodels.duration.hazard_regression import PHReg
import statsmodels.api as sm
import matplotlib
import matplotlib.pyplot as plt
import warnings
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ROOT, AN, PU, FG, SUBS, NM, BRMS_PROGRAMS, SEED
from _l2_common import (load_cohorts, adjustment_sets, usable, program_genes,
                        program_score, residual_auc, crossfit_auc, boot_ci,
                        fit_diag, rank_info, PANELS, ESTIMATE, OWN_PANEL,
                        OWN_ESTIMATE, SPECS)

# 程序名称改为英文（R5 指出沉积物里混中文标签；括号里的基因数由代码实时算，
# 不再硬编码，避免"113 基因"与 n_genes=123 自相矛盾）
PROGNAME = {
    "1": "Neuron projection guidance (GO:0097485)",
    "2": "Immune markers CD3/CD8/myeloid/C1Q",
    "3": "Epithelial junction CDH1/EPCAM/KRT",
    "4": "E2F targets + G2-M checkpoint",
}
COHORT_NAME = {"A": "melanoma GSE245467", "B": "breast GSE184869"}


def main():
    """
阶段 3 · 步骤 16：非循环纯度校正 + 生存终点纯度校正（★最终判据）
议题 A · 2026-09-13；2026-09-15 大修

为什么重做（第二轮，2026-09-15）：
  第一版（步骤 14）的协变量集包含"与该亚型程序同源"的成分面板 → 循环校正。
  第二版（本脚本初版）改为留出自身域，但有两个缺陷，被预审与外部意见点出：
    (1) `C['brms'].astype(str)` 在队列 A 上得到 `'1.0'`，与 SUBS 恒不匹配，
        队列 A 的 y 全为 0，AUC 静默失败 → CSV 里队列 A 四行 AUC 空白；
        稿件据此误判为"队列 A 纯度估计不完整"。真实原因是标签解析 bug。
    (2) 留出自身域只排除**以自身域命名的成分面板**，五个 ESTIMATE 派生列从未
        排除 → 对 BrMS2（免疫域）模型里仍有一个独立的免疫含量测量。
  本版：标签归一为整数字符串、显式剔除计数为空的样本，并把 ESTIMATE 同源列
  一并留出（strict），同时报告五种规格 + 交叉拟合 + 嵌套 bootstrap CI。

规格说明（五种）：
  full              四个面板 + 四个 ESTIMATE 列（不做任何留出）
  leaveown_current  留出自身面板，ESTIMATE 全保留（= 第二版的行为）
  leaveown_strict   留出自身面板 **和** 同域 ESTIMATE 列（= 本版主判据）
  purity_only       只校正 Purity_ESTIMATE
  no_estimate       留出自身面板，完全不用 ESTIMATE
"""
    matplotlib.use('Agg')
    warnings.filterwarnings('ignore')
    D = load_cohorts()
    for co in ('A', 'B'):
        if D[co]['dropped']:
            print(f"  ⚠ 队列 {co} 剔除无计数样本 {len(D[co]['dropped'])} 个："
                  f"{', '.join(D[co]['dropped'])}")

    rows = []
    print('=' * 118)
    print('★★ 非循环校正（留出自身域，strict 版）：原始 AUC → 五种规格的残差 AUC')
    print('=' * 118)
    for co in ('A', 'B'):
        C, lab, Lz = D[co]['C'], D[co]['lab'], D[co]['Z']
        print(f"\n  --- 队列 {co}（{COHORT_NAME[co]}，n={len(lab)}）---")
        print(f"  {'亚型':<20}{'基因':>5}{'原始':>7}"
              f"{'full':>7}{'current':>9}{'strict':>8}{'纯度':>7}{'无EST':>7}"
              f"{'strictΔ':>9}{'交叉拟合':>9}{'95%CI':>18}")
        for s in SUBS:
            g = program_genes(Lz, s)
            prog = Lz.loc[g].mean(axis=0).values
            y = (lab.values == s).astype(int)
            raw = float(roc_auc_score(y, prog))
            sets = {k: usable(v, C) for k, v in adjustment_sets(s).items()}
            auc, r2, cond = {}, {}, {}
            for k in SPECS:
                cols = sets[k]
                if not cols:
                    auc[k] = r2[k] = cond[k] = np.nan
                    continue
                d = fit_diag(prog, C[cols].values.astype(float), y)
                auc[k], r2[k], cond[k] = d['auc'], d['r2'], d['cond']
            # 主判据 = strict；交叉拟合与嵌套 bootstrap 只针对 strict 与 current
            cov_s = C[sets['leaveown_strict']].values.astype(float)
            cf_s = crossfit_auc(prog, cov_s, y)
            lo, hi = boot_ci(prog, cov_s, y)
            cov_c = C[sets['leaveown_current']].values.astype(float)
            cf_c = crossfit_auc(prog, cov_c, y)
            lo_c, hi_c = boot_ci(prog, cov_c, y)

            owndom = OWN_PANEL[s]
            owndom_auc = np.nan
            if owndom:
                v = C[owndom].mean(axis=1).values
                owndom_auc = float(roc_auc_score(y, v))
            print(f'  {NM[s]:<20}{len(g):>5}{raw:>7.3f}'
                  f"{auc['full']:>7.3f}{auc['leaveown_current']:>9.3f}"
                  f"{auc['leaveown_strict']:>8.3f}{auc['purity_only']:>7.3f}"
                  f"{auc['no_estimate']:>7.3f}{auc['leaveown_strict'] - raw:>+9.3f}"
                  f"{cf_s:>9.3f}   [{lo:.3f}, {hi:.3f}]")
            rows.append({
                'cohort': co, 'subtype': NM[s], 'program': PROGNAME[s],
                'n_genes': len(g), 'raw_auc': round(raw, 3),
                'adj_full_auc': round(float(auc['full']), 3),
                'adj_leaveown_current_auc': round(float(auc['leaveown_current']), 3),
                'adj_leaveown_strict_auc': round(float(auc['leaveown_strict']), 3),
                'adj_purity_auc': round(float(auc['purity_only']), 3),
                'adj_no_estimate_auc': round(float(auc['no_estimate']), 3),
                # 向后兼容列名：下游 40_figure_QA_redraw 读 adj_leaveown_auc
                'adj_leaveown_auc': round(float(auc['leaveown_strict']), 3),
                'delta': round(float(auc['leaveown_strict']) - raw, 3),
                'crossfit_strict': round(float(cf_s), 3),
                'ci_lo': round(float(lo), 3), 'ci_hi': round(float(hi), 3),
                'crossfit_current': round(float(cf_c), 3),
                'ci_lo_current': round(float(lo_c), 3),
                'ci_hi_current': round(float(hi_c), 3),
                'R2_on_covars': round(float(r2['leaveown_strict']), 3),
                'R2_full': round(float(r2['full']), 3),
                'cond_strict': round(float(cond['leaveown_strict']), 1),
                'cond_full': round(float(cond['full']), 1),
                'rank_strict': rank_info(C[sets['leaveown_strict']].values.astype(float))[0],
                'ncol_strict': rank_info(C[sets['leaveown_strict']].values.astype(float))[1],
                'rank_full': rank_info(C[sets['full']].values.astype(float))[0],
                'ncol_full': rank_info(C[sets['full']].values.astype(float))[1],
                'own_domain': (owndom[0] if owndom else '-'),
                'own_domain_auc': (round(owndom_auc, 3) if owndom else None),
                'covariates': '|'.join(sets['leaveown_strict']),
                'covariates_full': '|'.join(sets['full']),
                'covariates_leaveown_current': '|'.join(sets['leaveown_current']),
            })
    R = pd.DataFrame(rows)
    R.to_csv(f'{AN}/purity_adjustment_leaveown.csv', index=False)

    # 长表版本（补充材料直接用）
    long_rows = []
    for co in ('A', 'B'):
        C, lab, Lz = D[co]['C'], D[co]['lab'], D[co]['Z']
        for s in SUBS:
            g = program_genes(Lz, s)
            prog = Lz.loc[g].mean(axis=0).values
            y = (lab.values == s).astype(int)
            raw = float(roc_auc_score(y, prog))
            for k in SPECS:
                cols = usable(adjustment_sets(s)[k], C)
                if not cols:
                    continue
                cov = C[cols].values.astype(float)
                d = fit_diag(prog, cov, y)
                a_in, r2v = d['auc'], d['r2']
                a_cf = crossfit_auc(prog, cov, y)
                lo, hi = boot_ci(prog, cov, y)
                long_rows.append({
                    'cohort': co, 'subtype': f'BrMS{s}', 'program': PROGNAME[s],
                    'n_genes': len(g), 'adjustment': k,
                    'k_covariates': len(cols), 'covariates': '|'.join(cols),
                    'raw_auc': round(raw, 3),
                    'auc_in_sample': round(a_in, 3),
                    'auc_crossfit': round(a_cf, 3),
                    'ci_lo': round(lo, 3), 'ci_hi': round(hi, 3),
                    'delta_vs_raw': round(float(a_in) - raw, 3),
                    'r2': round(float(r2v), 3), 'cond': round(d['cond'], 1),
                })
    pd.DataFrame(long_rows).to_csv(f'{AN}/l2_adjustment_specs.csv', index=False)

    print('\n' + '=' * 118)
    print('★ ESTIMATE 纯度与脑组织含量按亚型（中位数）')
    print('=' * 118)
    tab = []
    for co in ('A', 'B'):
        C, lab = D[co]['C'], D[co]['lab']
        print(f'\n  --- 队列 {co} ---')
        print(f"  {'亚型':<20}{'n':>4}{'纯度':>9}{'脑组织':>9}{'免疫':>9}{'上皮':>9}")
        for s in SUBS:
            m = lab.values == s
            row = {'cohort': co, 'subtype': NM[s], 'n': int(m.sum()),
                   'purity_median': round(float(C.loc[m, 'Purity_ESTIMATE'].median()), 3),
                   'brain_median': round(float(C.loc[m, '脑组织合计'].median()), 3),
                   'immune_median': round(float(C.loc[m, '免疫合计'].median()), 3),
                   'epithelial_median': round(float(C.loc[m, '上皮-肿瘤'].median()), 3)}
            tab.append(row)
            print(f"  {NM[s]:<20}{int(m.sum()):>4}{row['purity_median']:>9.3f}"
                  f"{row['brain_median']:>9.2f}{row['immune_median']:>9.2f}"
                  f"{row['epithelial_median']:>9.2f}")
    P = pd.DataFrame(tab)
    P.to_csv(f'{AN}/purity_by_subtype.csv', index=False)
    for co in ('A', 'B'):
        C, lab = D[co]['C'], D[co]['lab']
        a = C.loc[lab.values == '1', 'Purity_ESTIMATE']
        b = C.loc[lab.values != '1', 'Purity_ESTIMATE']
        u = stats.mannwhitneyu(a, b)
        print(f'  队列 {co}: BrMS1 纯度 {a.median():.3f} vs 其余 {b.median():.3f}  MWU p={u.pvalue:.4f}')
        ab = C.loc[lab.values == '1', '脑组织合计']
        bb = C.loc[lab.values != '1', '脑组织合计']
        u2 = stats.mannwhitneyu(ab, bb)
        print(f'            BrMS1 脑组织 {ab.median():+.2f} vs 其余 {bb.median():+.2f}  MWU p={u2.pvalue:.4f}')

    print('\n' + '=' * 118)
    print('★ 生存终点的纯度校正（GSE184869）')
    print('=' * 118)
    lab2 = D['B']['lab']
    os2 = D['B']['os'].reindex(lab2.index)
    ok = pd.to_numeric(os2['os'], errors='coerce').notna().values
    lab2, os2 = lab2[ok], os2[ok]
    Lz = D['B']['Z']
    C = D['B']['C'].loc[lab2.index]
    Lz = Lz.reindex(columns=lab2.index)
    T = pd.to_numeric(os2['os'], errors='coerce').values.astype(float)
    E = os2['status'].astype(str).str.lower().str.startswith('dead').astype(int).values
    print(f'  n={len(T)} 事件={E.sum()}')
    print(f"  {'模型':<46}{'HR':>7}{'95%CI':>16}{'p':>9}")
    res = []

    def coxfit(X, label, names):
        X = np.asarray(X, float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        m = PHReg(T, X, status=E).fit(disp=0)
        ci = np.exp(m.conf_int())
        for i, nm_ in enumerate(names):
            hr = np.exp(m.params[i])
            lo, hi = ci[i]
            print(f"  {label + ' · ' + nm_:<46}{hr:>7.2f}   [{lo:.2f},{hi:.2f}]{'':>3}{m.pvalues[i]:>9.4f}")
            res.append({'model': label, 'term': nm_, 'HR': round(float(hr), 3),
                        'lo': round(float(lo), 3), 'hi': round(float(hi), 3),
                        'p': round(float(m.pvalues[i]), 4)})
    for s in SUBS:
        y = (lab2.values == s).astype(float)
        coxfit(y, f'BrMS{s} unadjusted', [f'BrMS{s}'])
        X = np.column_stack([y, C['Purity_ESTIMATE'].values, C['脑组织合计'].values])
        coxfit(X, f'BrMS{s} adjusted purity+brain', [f'BrMS{s}', '+purity', '+brain'])
    for s in SUBS:
        g = program_genes(Lz, s)
        prog = Lz.loc[g].mean(axis=0).values
        prog = (prog - prog.mean()) / prog.std()
        coxfit(prog, f'{PROGNAME[s]} unadj., per +1 SD', [PROGNAME[s]])
        X = np.column_stack([prog, C['Purity_ESTIMATE'].values])
        coxfit(X, f'{PROGNAME[s]} adj. purity', [PROGNAME[s], '+purity'])
    coxfit(C['Purity_ESTIMATE'].values, 'ESTIMATE purity, per +1 SD', ['Purity'])
    coxfit(C['脑组织合计'].values, 'Brain content, per +1 SD', ['Brain'])
    coxfit(C['免疫合计'].values, 'Immune content, per +1 SD', ['Immune'])
    S = pd.DataFrame(res)
    S.to_csv(f'{AN}/survival_purity_adjusted.csv', index=False)

    # ---------------------------------------------------------------- 图
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.7), dpi=300)
    cols = {'1': '#4C72B0', '2': '#DD8452', '3': '#55A868', '4': '#C44E52'}
    ax = axes[0]
    data, labs = [], []
    for co in ('A', 'B'):
        C, lab = D[co]['C'], D[co]['lab']
        for s in SUBS:
            data.append(C.loc[lab.values == s, 'Purity_ESTIMATE'].values)
            labs.append((s, co))
    for i, (s, co) in enumerate(labs):
        ax.scatter(np.full(len(data[i]), i) + np.random.default_rng(0).normal(0, 0.06, len(data[i])),
                   data[i], s=26, color=cols[s], alpha=0.75, edgecolor='white', lw=0.6, zorder=3)
        ax.plot([i - 0.28, i + 0.28], [np.median(data[i])] * 2, color='k', lw=2.4, zorder=4)
    ax.set_xticks(range(8))
    ax.set_xticklabels([f"{s}\n{('Mel' if co == 'A' else 'Bre')}" for s, co in labs], fontsize=8.5)
    ax.axvline(3.5, color='grey', ls='--', lw=1)
    ax.set_ylabel('ESTIMATE tumour purity', fontsize=9.5)
    ax.set_ylim(0.35, 1.02)
    ax.set_title('A  ESTIMATE tumour purity by subtype', fontsize=10.5, fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=0.22, ls=':')

    ax = axes[1]
    for ci, co in enumerate(('A', 'B')):
        sub = R[R.cohort == co]
        ax.plot(sub['raw_auc'], sub['adj_leaveown_auc'], ('o' if co == 'A' else 's'),
                ms=10, color=['#4C72B0', '#DD8452'][ci], label=f'cohort {co}', zorder=3, alpha=0.9)
    for s in SUBS:
        for _, r in R[R.subtype == NM[s]].iterrows():
            ax.annotate('', xy=(r['raw_auc'], r['adj_leaveown_auc']),
                        xytext=(r['raw_auc'], r['raw_auc']),
                        arrowprops=dict(arrowstyle='->', color=cols[s], lw=1.3, alpha=0.75))
            ax.plot([r['crossfit_strict']], [r['adj_leaveown_auc']], marker='x', ms=6,
                    color=cols[s], alpha=0.85, zorder=4)
    ax.plot([0.4, 1], [0.4, 1], 'k:', lw=1.2)
    ax.text(0.93, 0.9, 'no change', fontsize=8, style='italic', rotation=32)
    ax.axhline(0.5, color='grey', ls=':', lw=1)
    ax.set_xlabel('Raw AUC of the defining programme (cross = cross-fitted)', fontsize=9)
    ax.set_ylabel('AUC after strict leave-own-domain-out', fontsize=9)
    ax.set_xlim(0.42, 0.97)
    ax.set_ylim(0.35, 0.97)
    ax.set_title("B  Raw AUC -> composition-adjusted AUC\n(own panel AND same-domain ESTIMATE excluded)",
                 fontsize=10, fontweight='bold')
    ax.legend(fontsize=8, frameon=False, loc='lower left')
    for s, dx, dy in [('4', 0.005, 0.015), ('1', -0.1, 0.03), ('2', 0.01, -0.055), ('3', -0.07, -0.055)]:
        r = R[R.subtype == NM[s]].iloc[0]
        ax.text(r['raw_auc'] + dx, r['adj_leaveown_auc'] + dy, NM[s].split()[0],
                fontsize=9, fontweight='bold', color=cols[s])
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(alpha=0.22, ls=':')

    ax = axes[2]
    lbl, vals, cs = [], [], []
    for co in ('A', 'B'):
        for s in SUBS:
            r = R[(R.cohort == co) & (R.subtype == NM[s])].iloc[0]
            lbl.append(f"{NM[s].split()[0]}\n{('Mel' if co == 'A' else 'Bre')}")
            vals.append(r['own_domain_auc'] or 0)
            cs.append(cols[s])
    ax.bar(range(8), vals, 0.62, color=cs, alpha=0.9, edgecolor='white', lw=1.2)
    ax.axhline(0.5, color='grey', ls=':', lw=1.2)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.015, f'{v:.2f}', ha='center', fontsize=8.5, fontweight='bold')
    ax.set_xticks(range(8))
    ax.set_xticklabels(lbl, fontsize=8.5)
    ax.set_ylim(0, 1.08)
    ax.axvline(3.5, color='grey', ls='--', lw=1)
    ax.set_ylabel('AUC of the matching composition domain', fontsize=9)
    ax.set_title("C  Composition alone vs the\nsubtype's defining programme", fontsize=10.5, fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=0.22, ls=':')
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(f'{FG}/Fig7_purity_correction.{ext}', dpi=400, bbox_inches='tight')
    print(f'\n  OK Fig7_purity_correction.png/.pdf + purity_adjustment_leaveown.csv'
          f' + l2_adjustment_specs.csv + purity_by_subtype.csv + survival_purity_adjusted.csv')


if __name__ == '__main__':
    main()
