import json, os, pickle, numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm
import warnings
import os, sys
from _common import ROOT, AN, PU, SUBS, NM, SEED, COMPOSITION_PANELS, load_pkl_tuple, load_dataframe

def main():
    """
阶段 3 · 步骤 14：肿瘤纯度 / 组织成分显式校正（★核心判据）
议题 A · 2026-09-13

问题：BrMS1 的"神经样"判别力最高（AUC 0.93），但其 top 基因是 GFAP/AQP4/ERMN
      → 疑为"正常脑组织含量（肿瘤纯度）"驱动，而非肿瘤内在程序。
      脑转移特有的混杂轴是 **正常脑组织含量**（胶质+神经元+髓鞘），
      ESTIMATE 的 stromal/immune（TCGA 实体瘤校准）覆盖不了它 → 必须自建。

方法：
  ① 成分打分：脑组织（星形胶质+神经元+少突胶质+小胶质）/ 免疫 / 基质 / 上皮
  ② ESTIMATE 官方分数 + 官方纯度公式
  ③ **校正三法**：
     (i)  成分分数的原始 AUC（成分本身能否区分亚型）
     (ii) **残差法**：把程序分数对成分分数回归，取残差 → 残差 AUC（程序是否含"成分之外"的信息）
     (iii) 逻辑回归：program vs program + 成分 → ΔAUC 与 program 项显著性
  ④ 生存终点的纯度校正
"""
    warnings.filterwarnings('ignore')
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    PANEL = {'脑组织-星形胶质': ['GFAP', 'AQP4', 'SLC1A2', 'SLC1A3', 'ALDH1L1', 'GJA1', 'S100B', 'SOX9', 'NFIA', 'SLC4A4'], '脑组织-神经元': ['RBFOX3', 'SYP', 'SNAP25', 'NEFL', 'NEFM', 'NEFH', 'MAP2', 'SCN2A', 'RELN', 'GABRA1', 'GRIN1', 'SYN1', 'CAMK2A', 'SYT1', 'GRIA2', 'KCNJ4', 'NECAB1', 'ERMN'], '脑组织-少突胶质': ['PLP1', 'MBP', 'MOBP', 'MAG', 'CNP', 'MOG', 'CLDN11', 'OPALIN', 'KLK6'], '脑组织-小胶质': ['P2RY12', 'TMEM119', 'CX3CR1', 'SALL1', 'GPR34'], '免疫-泛白系': ['PTPRC', 'LCP2', 'HCLS1', 'ARHGDIB', 'CORO1A', 'CD48', 'CD52', 'CD53'], '免疫-T/NK': ['CD3D', 'CD3E', 'CD2', 'CD8A', 'CD8B', 'NKG7', 'GNLY', 'KLRD1', 'IL2RB', 'TRAC'], '免疫-髓系': ['CD68', 'CD14', 'AIF1', 'LYZ', 'TYROBP', 'FCER1G', 'C1QA', 'C1QB', 'ITGAM', 'CSF1R'], '基质-成纤维': ['COL1A1', 'COL1A2', 'COL3A1', 'DCN', 'LUM', 'FAP', 'PDGFRB', 'ACTA2', 'FBN1', 'POSTN', 'THY1'], '上皮-肿瘤': ['EPCAM', 'KRT8', 'KRT18', 'KRT19', 'CDH1', 'CLDN3', 'CLDN4', 'MUC1', 'KRT7', 'ELF3']}

    def load_cohort(which):
        if which == 'A':
            X = load_dataframe(f'{AN}/X_counts.pkl')
            lab = load_dataframe(f'{AN}/labels_gse245467.pkl')
            k = lab[lab.isin(SUBS)].index
            X = X[k]
            lab = lab.loc[k]  # 行选择（lab 是 DataFrame，列=brms）
            L = np.log2(X / X.sum(axis=0) * 1000000.0 + 1)
        else:
            X, lab, _ = load_pkl_tuple(f'{AN}/gse184869_symbol_matrix.pkl')
            k = lab[lab.isin(SUBS)].index
            X = X[k]
            lab = lab.loc[k] if isinstance(lab, pd.DataFrame) else lab[k]
            L = X.copy()
        return (L, lab)

    def z(L):
        return L.sub(L.mean(axis=1), axis=0).div(L.std(axis=1).replace(0, np.nan), axis=0).dropna(how='all')

    def score(Lz, genes):
        g = [x for x in genes if x in Lz.index]
        return (Lz.loc[g].mean(axis=0), len(g))

    def read_estimate(co):
        f = f'{PU}/cohort{co}_ESTIMATE_raw.txt'
        if not os.path.exists(f):
            return None
        d = pd.read_csv(f, sep='\t', index_col=0)
        d.index = [str(i).strip() for i in d.index]
        want = ['StromalScore', 'ImmuneScore', 'ESTIMATEScore']
        if not all((w in d.index for w in want)):
            return None
        num = d.loc[want].apply(pd.to_numeric, errors='coerce')
        num = num[[c for c in num.columns if not str(c).startswith('Description')]]
        mp = f'{PU}/cohort{co}_samplemap.txt'
        if os.path.exists(mp):
            m = pd.read_csv(mp, sep='\t')
            num = num.rename(columns=dict(zip(m['safe'], m['original'])))
        return num
    rows = []
    detail = {}
    for co in ('A', 'B'):
        L, lab = load_cohort(co)
        Lz = z(L)
        print('\n' + '=' * 104)
        print(f"★ 队列 {co}  ({('GSE245467 melanoma' if co == 'A' else 'GSE184869 breast')})   n={len(lab)}")
        print('=' * 104)
        comp = {}
        print(f"  {'成分面板':<20}{'n基因':>6}", end='')
        for s in SUBS:
            print(f"{'AUC→' + s:>10}", end='')
        print()
        for name, gl in PANEL.items():
            s, n = score(Lz, gl)
            if n < 4:
                print(f'  {name:<20} 覆盖不足')
                continue
            comp[name] = s.values
            a = [roc_auc_score((lab == t).astype(int).values, s.values) for t in SUBS]
            print(f'  {name:<20}{n:>6}' + ''.join((f'{v:>10.3f}' for v in a)))
        C = pd.DataFrame(comp, index=lab.index)
        brain = [c for c in C.columns if c.startswith('脑组织')]
        brainTC = C[brain].mean(axis=1)
        C['脑组织合计'] = brainTC
        imm = C[[c for c in C.columns if c.startswith('免疫')]].mean(axis=1)
        C['免疫合计'] = imm
        est = read_estimate(co)
        if est is not None:
            est = est.reindex(columns=lab.index)
            C['ESTIMATE_Stromal'] = est.loc['StromalScore'].values
            C['ESTIMATE_Immune'] = est.loc['ImmuneScore'].values
            C['ESTIMATE_score'] = est.loc['ESTIMATEScore'].values
            C['Purity_ESTIMATE'] = np.cos(0.6049872018 + 0.0001467884 * est.loc['ESTIMATEScore'].values)
            print(f"  ESTIMATE 已载入：纯度范围 {C['Purity_ESTIMATE'].min():.3f} – {C['Purity_ESTIMATE'].max():.3f}（中位 {C['Purity_ESTIMATE'].median():.3f}）")
        else:
            print('  ⚠️ 未找到 ESTIMATE 输出')
        for c in [x for x in C.columns if x.startswith('ESTIMATE') or x.startswith('Purity')]:
            if C[c].isna().any():
                print(f'    ⚠️ {c}: {int(C[c].isna().sum())} 个缺失 → 中位数填补')
                C[c] = C[c].fillna(C[c].median())
        detail[co] = {'C': C, 'lab': lab, 'Lz': Lz}
    NMAP = {'1': 'BrMS1', '2': 'BrMS2', '3': 'BrMS3', '4': 'BrMS4'}
    PROG = {'1': 'Neuron Projection Guidance (GO:0097485)', '2': '免疫标记 (CD3/CD8/髓系/C1Q)', '3': '上皮连接 (CDH1/EPCAM/KRT8-19/CLDN)', '4': 'E2F Targets + G2-M Checkpoint'}
    neural = json.load(open(f'{AN}/genesets/neural.json'))
    hm = json.load(open(f'{AN}/genesets/hallmark.json'))
    GSETS = {'1': neural['Neuron Projection Guidance (GO:0097485)'], '2': ['CD3D', 'CD3E', 'CD3G', 'CD2', 'CD247', 'TRAC', 'LCK', 'ZAP70', 'CD8A', 'CD8B', 'CD68', 'CD14', 'AIF1', 'LYZ', 'TYROBP', 'FCER1G', 'C1QA', 'C1QB'], '3': ['CDH1', 'EPCAM', 'KRT8', 'KRT18', 'KRT19', 'CLDN3', 'CLDN4', 'OCLN', 'TJP1'], '4': list(set(hm['E2F Targets']) | set(hm['G2-M Checkpoint']))}

    def residual_auc(prog, covars, y):
        """把 prog 对 covars 回归，取残差，算残差对 y 的 AUC"""
        Xd = sm.add_constant(covars)
        m = sm.OLS(prog, Xd).fit()
        r = np.asarray(m.resid)
        return (roc_auc_score(y, r), float(m.rsquared))
    out = []
    for co in ('A', 'B'):
        d = detail[co]
        C = d['C']
        lab = d['lab']
        Lz = d['Lz']
        print('\n' + '=' * 104)
        print(f'★★ 队列 {co}：纯度/组织成分校正规正（原始 AUC → 校正后残差 AUC）')
        print('=' * 104)
        covcols = [c for c in ['脑组织合计', '免疫合计', '上皮-肿瘤', 'ESTIMATE_Stromal', 'ESTIMATE_Immune', 'Purity_ESTIMATE'] if c in C.columns]
        cov1 = ['脑组织合计']
        cov2 = covcols
        print(f'  协变量① 脑组织合计 | 协变量② {covcols}')
        print(f"  {'亚型':<20}{'程序':<40}{'原始AUC':>9}{'校正①AUC':>10}{'校正②AUC':>10}{'Δ②':>8}{'R²(②)':>8}")
        for s in SUBS:
            g = [x for x in GSETS[s] if x in Lz.index]
            prog = Lz.loc[g].mean(axis=0)
            y = (lab == s).astype(int).values
            raw = roc_auc_score(y, prog.values)
            a1, r1 = residual_auc(prog.values, C[cov1].values, y)
            if len(cov2) >= 2:
                a2, r2 = residual_auc(prog.values, C[cov2].values, y)
            else:
                a2, r2 = (a1, r1)
            print(f'  {NM[s]:<20}{PROG[s][:38]:<40}{raw:>9.3f}{a1:>10.3f}{a2:>10.3f}{a2 - raw:>+8.3f}{r2:>8.3f}')
            out.append({'cohort': co, 'subtype': NM[s], 'n_genes': len(g), 'program': PROG[s], 'raw_auc': round(raw, 3), 'adj_brain_auc': round(a1, 3), 'adj_full_auc': round(a2, 3), 'delta_auc': round(a2 - raw, 3), 'R2_on_covars': round(r2, 3)})
        print(f'\n  参照：**脑组织合计分数**本身对各亚型的 AUC')
        for s in SUBS:
            y = (lab == s).astype(int).values
            print(f"    {NM[s]:<20} AUC={roc_auc_score(y, C['脑组织合计'].values):.3f}   免疫合计 AUC={roc_auc_score(y, C['免疫合计'].values):.3f}")
    R = pd.DataFrame(out)
    R.to_csv(f'{AN}/purity_adjustment.csv', index=False)
    print('\n' + '=' * 104)
    print('★ 程序分数与组织成分分数的相关（Spearman）')
    print('=' * 104)
    for co in ('A', 'B'):
        d = detail[co]
        C = d['C']
        Lz = d['Lz']
        print(f'\n  --- 队列 {co} ---')
        print(f"  {'程序':<30}{'脑组织':>9}{'免疫':>9}{'基质':>9}{'上皮':>9}")
        for s in SUBS:
            g = [x for x in GSETS[s] if x in Lz.index]
            prog = Lz.loc[g].mean(axis=0).values
            vals = []
            for c in ['脑组织合计', '免疫合计']:
                vals.append(stats.spearmanr(prog, C[c].values).statistic)
            st = [k for k in C.columns if k.startswith('基质')]
            ep = [k for k in C.columns if k.startswith('上皮')]
            vals.append(stats.spearmanr(prog, C[st].mean(axis=1).values).statistic if st else np.nan)
            vals.append(stats.spearmanr(prog, C[ep].mean(axis=1).values).statistic if ep else np.nan)
            print(f'  {NM[s]:<30}' + ''.join((f'{v:>9.3f}' for v in vals)))
    for co in ('A', 'B'):
        d = detail[co]
        df = d['C'].copy()
        # lab 在 P1-1 后是 DataFrame（Series pickle 经 load_dataframe 落为单列 DataFrame）
        lab_ser = d['lab'].iloc[:, 0] if isinstance(d['lab'], pd.DataFrame) else d['lab']
        # baseline 兼容：brms 必须以**字符串**落盘，否则下游如 19 用
        # `(brms.astype(str) == "2")` 时会因 "2.0" != "2" 全 False → Panel C NaN。
        # 关键：把 NaN 留空（to_csv na_rep=''）而不是 'nan'——否则 read_csv 时 pandas
        # 会把字符串 "nan" 又翻成 NaN，brms 整列变 float。
        df['brms'] = lab_ser.astype(str).where(lab_ser.notna(), '')
        df.to_csv(f'{PU}/cohort{co}_composition.txt', sep='\t', na_rep='')
    print(f'\n  ✅ 已存 purity_adjustment.csv + cohort{{A,B}}_composition.txt')
if __name__ == '__main__':
    main()
