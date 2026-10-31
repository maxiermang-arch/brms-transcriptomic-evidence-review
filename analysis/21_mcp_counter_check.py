#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Historical MCP-counter-signature mean-rank sensitivity analysis.
This is not the MCPcounter estimator or standard ssGSEA. Scores average
within-sample gene ranks divided by the measured gene count. Cross-study
significance counts do not test replication or biological equivalence.
"""
import os
import sys
import json
import urllib.request
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (
    AN, GENESETS, SUBS, NM, SEED,
    load_dataframe, load_pkl_tuple, PU,
)

# ============ MCP-counter 签名（来自 ebecht/MCPcounter v1.2.0, master）============
SIG_URL = "https://cdn.jsdelivr.net/gh/ebecht/MCPcounter@master/Signatures/genes.txt"
SIG_LOCAL = os.path.join(GENESETS, "mcp_genes.txt")


def _load_signature() -> dict:
    """返回 {celltype: [gene, ...]} — 11 celltypes，去掉 'CD8 T cells'（与 'T cells' 重叠且只 1 基因）"""
    sig = pd.read_csv(SIG_LOCAL, sep="\t")
    # 列名: 'HUGO symbols', 'Cell population', 'ENTREZID', 'ENSEMBL ID'
    sig.columns = ["hugo", "celltype", "entrez", "ensembl"]
    out = {}
    for ct, sub in sig.groupby("celltype"):
        out[ct] = sub["hugo"].dropna().astype(str).str.strip().tolist()
    # 论文 Data 19 只报 9 种（去 'CD8 T cells'）；我们也去，避免双计 T cells
    out.pop("CD8 T cells", None)
    return out


def signature_mean_rank(X: pd.DataFrame, gene_set: list) -> pd.Series:
    """签名平均秩评分：对每个样本，把 X.columns 按表达值升序排 rank，
    然后 mean_rank(G_h) / N = celltype 丰度 score."""
    # X: 基因 × 样本
    n_genes = X.shape[0]
    # 对每个样本独立 rank（从小到大）
    ranks = X.rank(axis=0, method="average", na_option="bottom")  # 基因×样本
    present = [g for g in gene_set if g in X.index]
    if len(present) < 2:
        return pd.Series(np.nan, index=X.columns)
    mean_rank = ranks.loc[present].mean(axis=0)
    return mean_rank / n_genes


def _load_cohorts() -> dict:
    out = {}
    # A: log2(CPM+1) → row-zscore (与 02b 一致)
    Xa = load_dataframe(f"{AN}/X_counts.pkl")
    laba = load_dataframe(f"{AN}/labels_gse245467.pkl").iloc[:, 0]
    laba = laba[laba.isin(SUBS)]
    Xa = Xa[laba.index]
    Xa_cpm = np.log2(Xa.div(Xa.sum(axis=0), axis=1) * 1e6 + 1)
    out["A"] = (Xa_cpm, laba.astype(str))

    # B: 已是 log2-TMM, 直接用
    Xb, labb, _ = load_pkl_tuple(f"{AN}/gse184869_symbol_matrix.pkl")
    labb = labb.iloc[:, 0] if isinstance(labb, pd.DataFrame) else labb
    labb = labb[labb.isin(SUBS)]
    Xb = Xb[labb.index]
    out["B"] = (Xb, labb.astype(str))
    return out


def main():
    sig = _load_signature()
    print(f"MCP-counter 签名: {len(sig)} celltypes; 总基因数 = {sum(len(v) for v in sig.values())}")
    cohorts = _load_cohorts()

    # ----- 1) 计算两队列 9 celltype × n_samples 矩阵 -----
    scores = {}
    for co, (X, lab) in cohorts.items():
        print(f"\n队列 {co}: n={len(lab)}, X={X.shape}")
        df = pd.DataFrame(index=lab.index)
        for ct, genes in sig.items():
            df[ct] = signature_mean_rank(X, genes).reindex(lab.index)
        scores[co] = df
        out_csv = f"{AN}/mcp_counter_scores_{co}.csv"
        df.to_csv(out_csv)
        print(f"  → {out_csv}")
        # BrMS 亚型中位数
        print(f"  BrMS × celltype 中位数:")
        med = df.assign(brms=lab.values).groupby("brms").median(numeric_only=True)
        print(med.round(3).to_string())

    # ----- 2) 两队列各自 BrMS 4×4 两两 Mann-Whitney U + Bonferroni -----
    pairwise_rows = []
    for co, df in scores.items():
        lab = cohorts[co][1]
        celltypes = list(df.columns)
        for s1 in SUBS:
            for s2 in SUBS:
                if s1 >= s2:
                    continue
                for ct in celltypes:
                    a = df.loc[lab == s1, ct].dropna().values
                    b = df.loc[lab == s2, ct].dropna().values
                    if len(a) < 2 or len(b) < 2:
                        continue
                    try:
                        u, p = mannwhitneyu(a, b, alternative="two-sided")
                    except Exception:
                        continue
                    pairwise_rows.append({
                        "cohort": co, "subtype1": s1, "subtype2": s2,
                        "celltype": ct, "U": float(u),
                        "n1": int(len(a)), "n2": int(len(b)),
                        "median1": float(np.median(a)),
                        "median2": float(np.median(b)),
                        "p_value": float(p),
                    })
    pw_df = pd.DataFrame(pairwise_rows)
    # Bonferroni within (cohort, celltype)
    pw_df["p_adj_bonf"] = pw_df.groupby(["cohort", "celltype"])["p_value"].transform(
        lambda p: np.minimum(p * pw_df.loc[p.index, "celltype"].groupby(pw_df.loc[p.index, "celltype"]).transform("size").values, 1.0))
    # FDR (Benjamini-Hochberg) within (cohort, celltype) — 与 paper 校正方式更接近
    def _fdr(p_series):
        p = np.asarray(p_series, dtype=float)
        n = len(p)
        order = np.argsort(p)
        ranks = np.empty(n, dtype=int)
        ranks[order] = np.arange(1, n + 1)
        adj = np.empty(n, dtype=float)
        cummin = 1.0
        for i in order[::-1]:
            v = p[i] * n / ranks[i]
            cummin = min(cummin, v)
            adj[i] = min(cummin, 1.0)
        return pd.Series(adj, index=p_series.index)
    pw_df["p_adj_fdr"] = pw_df.groupby(["cohort", "celltype"])["p_value"].transform(_fdr)
    pw_df.to_csv(f"{AN}/mcp_counter_pairwise.csv", index=False)
    print(f"\n✅ mcp_counter_pairwise.csv  ({len(pw_df)} rows)")

    # ----- 3) 与 paper Data 19 的"1 vs 4 高显著 / 2 vs 4 低显著"模式对比 -----
    paper = pd.read_csv(f"{AN}/paper_MCPcounter.csv")
    paper = paper[paper["celltype"] != "CD8 T cells"] if "CD8 T cells" in paper["celltype"].unique() else paper
    paper_grp = paper.groupby(["group1", "group2"]).agg(
        paper_sig=("p.adj", lambda p: int((p < 0.05).sum())),
        paper_n=("p.adj", "size"),
    ).reset_index()

    ours_rows = []
    for co in ["A", "B"]:
        sub = pw_df[pw_df["cohort"] == co]
        for (s1, s2) in [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]:
            for ct in celltypes:
                row = sub[(sub["subtype1"] == str(s1)) & (sub["subtype2"] == str(s2)) & (sub["celltype"] == ct)]
                if len(row):
                    ours_rows.append({
                        "cohort": co, "comparison": f"{s1}_vs_{s2}",
                        "celltype": ct,
                        "our_padj_bonf": float(row["p_adj_bonf"].iloc[0]),
                        "our_padj_fdr": float(row["p_adj_fdr"].iloc[0]),
                        "our_sig_bonf": int(row["p_adj_bonf"].iloc[0] < 0.05),
                        "our_sig_fdr": int(row["p_adj_fdr"].iloc[0] < 0.05),
                    })
    ours_df = pd.DataFrame(ours_rows)

    # 按 comparison 统计"显著 celltype 数"
    ours_grp = ours_df.groupby(["cohort", "comparison"]).agg(
        ours_sig_bonf=("our_sig_bonf", "sum"),
        ours_sig_fdr=("our_sig_fdr", "sum"),
        ours_n=("our_sig_bonf", "size"),
    ).reset_index()

    # 与 paper 对齐（paper 的 group1='1' vs group2 in ['2','3','4']; 但 paper 缺 1 vs 2，所以比 1 vs 3/4）
    print("\n" + "=" * 80)
    print("★ 与 paper Data 19 对齐（BrMS 两两比较：显著 celltype 数 / 总数）")
    print("    Bonferroni 与 FDR 两种多重校正并列")
    print("=" * 80)
    print(f"{'comparison':<14} {'paper (sig/total)':<22} {'A Bonf (sig/total)':<22} {'A FDR (sig/total)':<22} {'B Bonf (sig/total)':<22} {'B FDR (sig/total)':<22}")
    print("-" * 130)
    for comp in ["1_vs_3", "1_vs_4", "2_vs_4", "3_vs_4"]:
        g1, g2 = comp.split("_vs_")
        paper_row = paper[(paper["group1"] == g1) & (paper["group2"] == g2)]
        paper_str = "（paper 未报）" if len(paper_row) == 0 else f"{int((paper_row['p.adj']<0.05).sum())}/{len(paper_row)}"
        a_row = ours_grp[(ours_grp["cohort"] == "A") & (ours_grp["comparison"] == comp)]
        b_row = ours_grp[(ours_grp["cohort"] == "B") & (ours_grp["comparison"] == comp)]
        if len(a_row):
            a_bonf = f"{int(a_row['ours_sig_bonf'].iloc[0])}/{int(a_row['ours_n'].iloc[0])}"
            a_fdr = f"{int(a_row['ours_sig_fdr'].iloc[0])}/{int(a_row['ours_n'].iloc[0])}"
        else:
            a_bonf, a_fdr = "n/a", "n/a"
        if len(b_row):
            b_bonf = f"{int(b_row['ours_sig_bonf'].iloc[0])}/{int(b_row['ours_n'].iloc[0])}"
            b_fdr = f"{int(b_row['ours_sig_fdr'].iloc[0])}/{int(b_row['ours_n'].iloc[0])}"
        else:
            b_bonf, b_fdr = "n/a", "n/a"
        print(f"{comp:<14} {paper_str:<22} {a_bonf:<22} {a_fdr:<22} {b_bonf:<22} {b_fdr:<22}")

    # 写 vs paper 表
    comparison_summary = []
    for comp in ["1_vs_2", "1_vs_3", "1_vs_4", "2_vs_3", "2_vs_4", "3_vs_4"]:
        g1, g2 = comp.split("_vs_")
        paper_row = paper[(paper["group1"] == g1) & (paper["group2"] == g2)]
        paper_str = "n/a (paper 未报)" if len(paper_row) == 0 else f"{int((paper_row['p.adj']<0.05).sum())}/{len(paper_row)}"
        for co in ["A", "B"]:
            row = ours_grp[(ours_grp["cohort"] == co) & (ours_grp["comparison"] == comp)]
            if len(row):
                ours_bonf = f"{int(row['ours_sig_bonf'].iloc[0])}/{int(row['ours_n'].iloc[0])}"
                ours_fdr = f"{int(row['ours_sig_fdr'].iloc[0])}/{int(row['ours_n'].iloc[0])}"
            else:
                ours_bonf, ours_fdr = "n/a", "n/a"
            comparison_summary.append({
                "comparison": comp, "cohort": co,
                "paper_sig_total": paper_str,
                "ours_bonf_sig_total": ours_bonf,
                "ours_fdr_sig_total": ours_fdr,
            })
    pd.DataFrame(comparison_summary).to_csv(f"{AN}/mcp_counter_vs_paper.csv", index=False)
    print(f"\n✅ mcp_counter_vs_paper.csv  ({len(comparison_summary)} rows)")

    # ----- 4) 一句话结论 -----
    print("\n" + "=" * 80)
    print("★ 复算结论（与论文 Data 19 模式对比）")
    print("=" * 80)
    # 找 1 vs 4 显著比例
    p_1v4 = paper[(paper["group1"] == "1") & (paper["group2"] == "4")]
    paper_1v4 = (p_1v4["p.adj"] < 0.05).sum() if len(p_1v4) else 0
    paper_1v4_n = len(p_1v4) if len(p_1v4) else 1
    a_1v4 = ours_grp[(ours_grp["cohort"] == "A") & (ours_grp["comparison"] == "1_vs_4")]
    b_1v4 = ours_grp[(ours_grp["cohort"] == "B") & (ours_grp["comparison"] == "1_vs_4")]
    if len(a_1v4):
        a_1v4_bonf = f"{int(a_1v4['ours_sig_bonf'].iloc[0])}/{int(a_1v4['ours_n'].iloc[0])}"
        a_1v4_fdr = f"{int(a_1v4['ours_sig_fdr'].iloc[0])}/{int(a_1v4['ours_n'].iloc[0])}"
    else:
        a_1v4_bonf, a_1v4_fdr = "n/a", "n/a"
    if len(b_1v4):
        b_1v4_bonf = f"{int(b_1v4['ours_sig_bonf'].iloc[0])}/{int(b_1v4['ours_n'].iloc[0])}"
        b_1v4_fdr = f"{int(b_1v4['ours_sig_fdr'].iloc[0])}/{int(b_1v4['ours_n'].iloc[0])}"
    else:
        b_1v4_bonf, b_1v4_fdr = "n/a", "n/a"
    print(f"  BrMS1 vs BrMS4: paper {paper_1v4}/{paper_1v4_n}, A 队 Bonf {a_1v4_bonf} FDR {a_1v4_fdr}, B 队 Bonf {b_1v4_bonf} FDR {b_1v4_fdr}")
    # BrMS2 vs BrMS4
    p_2v4 = paper[(paper["group1"] == "2") & (paper["group2"] == "4")]
    paper_2v4 = (p_2v4["p.adj"] < 0.05).sum() if len(p_2v4) else 0
    paper_2v4_n = len(p_2v4) if len(p_2v4) else 1
    a_2v4 = ours_grp[(ours_grp["cohort"] == "A") & (ours_grp["comparison"] == "2_vs_4")]
    b_2v4 = ours_grp[(ours_grp["cohort"] == "B") & (ours_grp["comparison"] == "2_vs_4")]
    if len(a_2v4):
        a_2v4_bonf = f"{int(a_2v4['ours_sig_bonf'].iloc[0])}/{int(a_2v4['ours_n'].iloc[0])}"
    else:
        a_2v4_bonf = "n/a"
    if len(b_2v4):
        b_2v4_bonf = f"{int(b_2v4['ours_sig_bonf'].iloc[0])}/{int(b_2v4['ours_n'].iloc[0])}"
    else:
        b_2v4_bonf = "n/a"
    print(f"  BrMS2 vs BrMS4: paper {paper_2v4}/{paper_2v4_n}, A 队 Bonf {a_2v4_bonf}, B 队 Bonf {b_2v4_bonf}")


if __name__ == "__main__":
    main()