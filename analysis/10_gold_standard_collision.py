import openpyxl, numpy as np, pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
import warnings
import os, sys
from _common import ROOT, AN, SUPP, FG, SUBS, NM, SEED

# 置换检验的随机源固定在 _common 的 SEED，与 26_/33_ 同一约定。
# 2026-09-19 修复：此前 RNG 未导入，脚本执行到 PCA 置换段会 NameError。
RNG = np.random.default_rng(SEED)


def main():
    """
阶段 2 · 步骤 10：金标准对撞 —— 实测能量代谢组 vs "代谢型"标签（★决定性检验）
议题 A · 2026-09-13

逻辑链：如果 BrMS3 真是"代谢型"，那么
        (a) 独立的通路打分应能在两个转录组队列里找回它 → 步骤 5/6 结果：❌ 代谢通路两队列都弱
        (b) **它的实测能量代谢组应与其余亚型有系统差异** → 本步检验（最硬，不用转录组推断）
数据：Supplementary Data 15 = 74 例 BrM 的 75 个能量代谢物定量（列 = Sample_ID）
      用 Supplementary Data 1 的 Sample_ID → BrMS 映射
附带：Supplementary Data 19 = 论文自己的 MCP-counter 免疫去卷积（对撞我的 BrMS2 结论）
      Supplementary Data 23 = 论文自己的 EMT 相关性分析（对撞我的 EMT 结论）
"""
    warnings.filterwarnings('ignore')
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    XL = f'{ROOT}/data/supplementary/41467_2026_68748_MOESM3_ESM.xlsx'
    wb = openpyxl.load_workbook(XL, read_only=True)

    def sheet(sn, header_row=1):
        ws = wb[sn]
        rows = list(ws.iter_rows(values_only=True))
        head = [str(x).strip() if x is not None else f'c{i}' for i, x in enumerate(rows[header_row])]
        return pd.DataFrame(rows[header_row + 1:], columns=head)
    d1 = sheet('Supplementary Data 1', 1)
    print(f'Data 1: {d1.shape}  列示例 {list(d1.columns)[:12]}')
    d1 = d1[d1['Sample_ID'].notna()]
    idmap = {}
    for _, r in d1.iterrows():
        b = str(r['BrMS']).strip()
        if b in SUBS:
            idmap[str(r['Sample_ID']).strip()] = {'brms': b, 'tumor': str(r['Tumor Type']).strip(), 'primary': str(r['Primary Tumor Type']).strip(), 'batch': str(r['Batch']).strip()}
    print(f'Data 1 可映射样本（有 BrMS 标签）: {len(idmap)}')
    d15 = sheet('Supplementary Data 15', 1)
    d15 = d15.rename(columns={d15.columns[0]: 'metabolite'})
    metab = d15.set_index('metabolite')
    metab.columns = [str(c).strip() for c in metab.columns]
    metab = metab.apply(pd.to_numeric, errors='coerce')
    print(f'\nData 15 代谢组: {metab.shape[0]} 个代谢物 × {metab.shape[1]} 个样本')
    hit = [c for c in metab.columns if c in idmap]
    print(f'  ★ 代谢组样本可映射到 BrMS 标签: {len(hit)} / {metab.shape[1]}')
    if len(hit) < 20:
        print('  映射率过低，改用 Data 1 的 Patient ID 再试')
    for key in ['Patient ID', 'Sample_ID']:
        m2 = {}
        for _, r in d1.iterrows():
            b = str(r['BrMS']).strip()
            if b in SUBS and pd.notna(r[key]):
                m2[str(r[key]).strip()] = b
        hit2 = [c for c in metab.columns if c in m2]
        print(f'  用 {key} 映射命中: {len(hit2)}')
        if len(hit2) > len(hit):
            hit = hit2
            idmap = {k: {'brms': v} for k, v in m2.items()}
    if len(hit) >= 20:
        X = metab[hit]
        lab = pd.Series({c: idmap[c]['brms'] for c in hit})
        print(f'\n  亚型分布: {dict(lab.value_counts().sort_index())}')
        print(f'  代谢物缺失率: {(X.isna().mean(axis=1) > 0.5).sum()}/{X.shape[0]} 行 >50% 缺失')
        X = X[X.isna().mean(axis=1) <= 0.3]
        X = X.fillna(X.median(axis=1), axis=0)
        Z = X.sub(X.mean(axis=1), axis=0).div(X.std(axis=1).replace(0, np.nan), axis=0).dropna(how='all')
        print(f'  可用代谢物: {Z.shape[0]}')
        print('\n' + '=' * 96)
        print('★ 决定性检验：实测能量代谢组能否区分 BrMS3（代谢型）？')
        print('=' * 96)
        rows = []
        for s in SUBS:
            y = (lab == s).astype(int).values
            ps = [stats.mannwhitneyu(Z.values[i][y == 1], Z.values[i][y == 0]).pvalue for i in range(Z.shape[0])]
            ps = np.array(ps)
            order = np.argsort(ps)
            ranked = ps[order] * len(ps) / np.arange(1, len(ps) + 1)
            padj = np.empty_like(ranked)
            padj[order] = np.minimum.accumulate(ranked[::-1])[::-1]
            padj = np.clip(padj, 0, 1)
            nsig = (padj < 0.05).sum()
            pca = PCA(n_components=3, random_state=0).fit(Z.T.values)
            sc = pca.transform(Z.T.values)
            aucs = [max(roc_auc_score(y, sc[:, i]), 1 - roc_auc_score(y, sc[:, i])) for i in range(3)]
            cnt = 0
            for _ in range(1000):
                yp = RNG.permutation(y)
                # 零分布必须使用与观测统计量**完全相同**的变换：3 个主成分各自
                # 折叠到 ≥0.5 再取最大（= 在 3×2 个方向上取最优）。2026-09-19 修复：
                # 此前零分布只取 3 个方向的最大原始 AUC，搜索空间比观测小一半，
                # 会把观测统计量系统性抬高、p 偏小（与本文批判的 C1 反保守同类）。
                a = max(max(roc_auc_score(yp, sc[:, i]), 1 - roc_auc_score(yp, sc[:, i]))
                        for i in range(3))
                if a >= max(aucs):
                    cnt += 1
            pp = (cnt + 1) / 1001
            star = '***' if pp < 0.01 else '**' if pp < 0.05 else '*' if pp < 0.1 else 'ns'
            print(f'  {NM[s]:<20} n={y.sum():2d}  显著代谢物(FDR<0.05)={nsig:2d}/{Z.shape[0]}  | PCA 最佳AUC={max(aucs):.3f} (置换p={pp:.3f} {star})')
            rows.append({'subtype': NM[s], 'n': int(y.sum()), 'n_sig_metabolites_FDR05': int(nsig), 'n_metabolites': Z.shape[0], 'PCA_best_auc': round(float(max(aucs)), 3), 'PCA_perm_p': round(float(pp), 4)})
        R = pd.DataFrame(rows)
        R.to_csv(f'{AN}/metabolomics_vs_BrMS.csv', index=False)
        y3 = (lab == '3').astype(int).values
        ps3 = [stats.mannwhitneyu(Z.values[i][y3 == 1], Z.values[i][y3 == 0]).pvalue for i in range(Z.shape[0])]
        top = pd.DataFrame({'metabolite': Z.index, 'p': ps3}).sort_values('p').head(15)
        print('\n  BrMS3 差异最显著的 15 个代谢物:')
        for _, r in top.iterrows():
            v = Z.loc[r['metabolite']]
            print(f"    {r['metabolite'][:42]:<44} p={r['p']:.4f}  BrMS3均值={v[y3 == 1].mean():+.2f}  其余={v[y3 == 0].mean():+.2f}")
        top.to_csv(f'{AN}/brms3_top_metabolites.csv', index=False)
    else:
        print('\n  ⚠️ 代谢组样本与 BrMS 标签无法映射 → 记录为局限，不做推断')
    print('\n' + '=' * 96)
    print('★ 对撞：论文自己的 MCP-counter 免疫去卷积（Data 19）')
    print('=' * 96)
    d19 = sheet('Supplementary Data 19', 1)
    d19 = d19.rename(columns={d19.columns[0]: 'celltype'})
    d19['p.adj'] = pd.to_numeric(d19['p.adj'], errors='coerce')
    sig = d19[d19['p.adj'] < 0.05]
    print(f'  总比较 {len(d19)} 条；FDR<0.05 的 {len(sig)} 条')
    tab = sig.groupby(['celltype']).size().sort_values(ascending=False)
    print('  显著差异最多的细胞类型:')
    print(tab.head(12).to_string())
    print('\n  按亚型对统计（group1=BrMS1）:')
    print(sig.groupby(['group1', 'group2']).size().to_string())
    pair = sig.groupby(['group1', 'group2']).size()
    print('\n  ★ 读法：若论文数据中 BrMS2 确实是最免疫浸润的亚型，则含 2 的比较应对 T/NK/髓系细胞显示最多显著项')
    d19.to_csv(f'{AN}/paper_MCPcounter.csv', index=False)
    print('\n' + '=' * 96)
    print('★ 对撞：论文自己的 EMT 相关性分析（Data 23）')
    print('=' * 96)
    d23 = sheet('Supplementary Data 23', 1)
    print(d23.head(25).to_string(index=False))
    d23.to_csv(f'{AN}/paper_EMT_correlation.csv', index=False)
    print(f'\n  ✅ 已存 metabolomics_vs_BrMS.csv / brms3_top_metabolites.csv / paper_MCPcounter.csv / paper_EMT_correlation.csv')
if __name__ == '__main__':
    main()
