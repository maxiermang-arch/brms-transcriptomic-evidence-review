# -*- coding: utf-8 -*-
"""analysis/42_proxy_sensitivity.py — 代理基因集敏感性分析（Supplementary Table S8）

为什么需要它
------------
稿件用的四套"程序基因集"是**我们自己挑的代理**：脑转移图谱发布的亚型标签
并不附带完整的程序基因列表，已发布的只有每个亚型的少数 top markers。因此
每个亚型的判定都可能被质疑为"你选的基因集决定了你的结论"。

本脚本把每个亚型换成另外两套独立来源的代理基因集，重跑 L1（size-matched
零分布）与 L2（strict leave-own-domain-out），回答：
    判定是"这套基因集的性质"，还是"这个标签的性质"？

三套代理的选取原则：**同源不重复**。同一亚型的三套应尽量来自不同来源，
而不是同一通路的三个近义变体，否则敏感性分析没有信息量。

    BrMS1  P = Neuron Projection Guidance (GO)   —— 稿件主程序
           A1 = Neuron Differentiation (GO)      —— 同一谱系、更宽的神经发生
           A2 = Gliogenesis (GO)                 —— 胶质而非神经元，不同生物学家系
    BrMS2  P = 手工整理的 CD3/CD8/髓系/C1Q 17 基因
           A1 = Hallmark Allograft Rejection     —— 独立来源的免疫排斥特征
           A2 = T Cell Activation (GO)           —— GO 来源的 T 细胞活化
    BrMS3  P = 上皮连接 9 基因（CDH1/EPCAM/KRT…）—— 我们的代理
           A1 = 图谱发布的 top markers（CYP3A5/AKR7A3/CYP2C9/UGT1A4/SULT1E1）
           A2 = Hallmark Fatty Acid Metabolism   —— "代谢型"读法
    BrMS4  P = Hallmark E2F ∪ G2-M（315 基因）
           A1 = E2F Targets 单通路
           A2 = G2-M Checkpoint 单通路

输出: analysis/proxy_sensitivity.csv
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import AN, SEED, SUBS, BRMS_PROGRAMS  # noqa: E402
import _l2_common as L2                            # noqa: E402

N_NULL = 2000                       # 与 26_size_matched_null.py 保持一致
L2_THRESHOLD = 0.70
HERE = os.path.dirname(os.path.abspath(__file__))
GS = os.path.join(HERE, "genesets")


def _primary_null() -> dict:
    """主表 L1 的数值（Table 2），按 (cohort, subtype) 索引。

    P 行必须与 Table 2 逐位一致，所以直接读 26 号脚本的输出而**不重算**：
    两条独立计算路径（即使种子相同）会因随机流消耗顺序不同而产生蒙特卡洛
    差异，那会让"P 行复现主表"的声明落空。
    """
    df = pd.read_csv(os.path.join(HERE, "size_matched_null.csv"))
    return {
        (str(r.cohort), str(r.subtype)):
        (float(r.auc_obs), float(r.null_mean), float(r.null_sd),
         float(r.emp_p), float(r.z), int(r.n_genes))
        for r in df.itertuples()
    }


def _primary_strict() -> dict:
    """主表 L2 strict 的样本内 AUC（Table 3），按 (cohort, BrMSx) 索引。"""
    df = pd.read_csv(os.path.join(HERE, "l2_sensitivity.csv"))
    df = df[df.adjustment == "leaveown_strict"]
    return {(str(r.cohort), str(r.subtype)): float(r.auc_in_sample)
            for r in df.itertuples()}


def _load(name: str) -> dict:
    with open(os.path.join(GS, name), encoding="utf8") as fh:
        return json.load(fh)


def proxy_sets() -> dict[str, list[tuple[str, str, list[str]]]]:
    """subtype -> [(code, label, gene list), ...]，第一项即稿件主程序。"""
    neural = _load("neural.json")
    immune = _load("immune.json")
    hall = _load("hallmark.json")

    # 图谱发布的 BrMS3 top markers（稿件 §3.10 引用的五个基因）
    published_3 = ["CYP3A5", "AKR7A3", "CYP2C9", "UGT1A4", "SULT1E1"]

    return {
        "1": [
            ("P", "Neuron projection guidance (primary)",
             sorted(BRMS_PROGRAMS["1"])),
            ("A1", "Neuron differentiation (GO:0030182)",
             sorted(neural["Neuron Differentiation (GO:0030182)"])),
            ("A2", "Gliogenesis (GO:0042063)",
             sorted(neural["Gliogenesis (GO:0042063)"])),
        ],
        "2": [
            ("P", "Curated CD3/CD8/myeloid/C1Q (primary)",
             sorted(BRMS_PROGRAMS["2"])),
            ("A1", "Hallmark allograft rejection",
             sorted(hall["Allograft Rejection"])),
            ("A2", "T cell activation (GO:0042110)",
             sorted(immune["T Cell Activation (GO:0042110)"])),
        ],
        "3": [
            ("P", "Epithelial junction CDH1/EPCAM/KRT (primary)",
             sorted(BRMS_PROGRAMS["3"])),
            ("A1", "Published BrMS3 top markers",
             sorted(published_3)),
            ("A2", "Hallmark fatty acid metabolism",
             sorted(hall["Fatty Acid Metabolism"])),
        ],
        "4": [
            ("P", "Hallmark E2F + G2-M (primary)",
             sorted(BRMS_PROGRAMS["4"])),
            ("A1", "Hallmark E2F targets only",
             sorted(hall["E2F Targets"])),
            ("A2", "Hallmark G2-M checkpoint only",
             sorted(hall["G2-M Checkpoint"])),
        ],
    }


def size_matched_null(Zv: np.ndarray, genes: list[str], gset: list[str],
                      ybin: np.ndarray, rng: np.random.Generator,
                      n_null: int = N_NULL):
    """观测 AUC + size-matched 随机基因集零分布（与 26 号脚本同一口径）。"""
    present = [g for g in gset if g in genes]
    k = len(present)
    idx_g = [genes.index(g) for g in present]
    obs = float(roc_auc_score(ybin, Zv[idx_g].mean(axis=0)))
    nulls = np.empty(n_null)
    for i in range(n_null):
        rs = rng.choice(len(genes), size=k, replace=False)
        nulls[i] = roc_auc_score(ybin, Zv[rs].mean(axis=0))
    mu, sd = float(nulls.mean()), float(nulls.std())
    p = (1 + int((nulls >= obs).sum())) / (n_null + 1)
    z = (obs - mu) / sd if sd > 0 else float("nan")
    return obs, mu, sd, p, z, present


def main() -> None:
    print("=" * 78)
    print("代理基因集敏感性分析（Supplementary Table S8）")
    print("=" * 78)

    coh = L2.load_cohorts()
    proxies = proxy_sets()
    rng = np.random.default_rng(SEED)
    rows = []

    prim_null = _primary_null()
    prim_strict = _primary_strict()
    for tag in ("A", "B"):
        d = coh[tag]
        Z, C, lab = d["Z"], d["C"], d["lab"]
        genes = Z.index.tolist()
        Zv = Z.to_numpy()
        for s in SUBS:
            ybin = (lab.values == s).astype(int)
            if ybin.sum() < 3:
                continue
            for code, label, gset in proxies[s]:
                if code == "P":
                    # 复现主表：数值直接取自 Table 2 / Table 3，不重算。
                    obs, mu, sd, p, z, n_prim = prim_null[(tag, s)]
                    adj = prim_strict[(tag, f"BrMS{s}")]
                    present = [g for g in gset if g in genes]
                else:
                    obs, mu, sd, p, z, present = size_matched_null(
                        Zv, genes, gset, ybin, rng)
                    n_prim = len(present)
                    cov = L2.usable(L2.adjustment_sets(s)["leaveown_strict"], C)
                    score = Z.loc[present].mean(axis=0).values
                    adj = L2.residual_auc(score, C[cov].values.astype(float),
                                          ybin)
                rows.append({
                    "cohort": tag, "subtype": f"BrMS{s}", "proxy": code,
                    "proxy_label": label, "n_genes": n_prim,
                    "raw_auc": round(obs, 4), "null_mean": round(mu, 4),
                    "null_sd": round(sd, 4), "emp_p": round(p, 4),
                    "z": round(float(z), 4),
                    # L1 的公开规则只有 empirical p ≤ 0.05（§2.4、Table 1）。
                    # 早期版本在此额外要求 z > 2，与公开规则不符，已删除。
                    "passes_L1": bool(p <= 0.05),
                    "auc_strict": round(float(adj), 4),
                    "delta_vs_raw": round(float(adj - obs), 4),
                    "passes_L2_strict": bool(adj >= L2_THRESHOLD),
                })
                print(f"  {tag} BrMS{s} {code:2s} n={len(present):4d} "
                      f"AUC={obs:.3f} z={z:5.2f} strict={adj:.3f} "
                      f"{'pass' if adj >= L2_THRESHOLD else 'fail'}")

    df = pd.DataFrame(rows)
    out = os.path.join(AN, "proxy_sensitivity.csv")
    df.to_csv(out, index=False)
    print(f"\nwrote {out}  ({len(df)} rows)")

    print("\n[结论] 每个亚型三套代理的 L2-strict 判定是否一致")
    for s in SUBS:
        g = df[df.subtype == f"BrMS{s}"]
        for tag in ("A", "B"):
            gg = g[g.cohort == tag]
            verdicts = "".join("P" if v else "F" for v in gg.passes_L2_strict)
            print(f"  {tag} BrMS{s}: {verdicts}  "
                  f"strict AUC " + " / ".join(f"{v:.3f}" for v in gg.auc_strict))


if __name__ == "__main__":
    main()
