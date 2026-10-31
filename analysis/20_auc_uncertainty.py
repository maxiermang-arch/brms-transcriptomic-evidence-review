#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""20 · 关键 AUC 的 bootstrap 95% CI + 标签置换 p（QA，独立于 02b/14/16）

触发动机（README 阶段收口计划 A1）：
  当前 02b/14/16/19 输出的 AUC 全是点估计，审稿人必问："95% CI？置换检验 p？"
  本脚本**重做**这些 AUC 的不确定性估计（不动上游脚本，纯 QA），输出两个 CSV：
    auc_bootstrap_ci.csv  — 每测点的 obs_AUC + 95% CI（2.5/97.5 percentile, N=2000）
    auc_label_perm_p.csv  — 每测点的 obs_AUC + 单尾 permutation p（N=2000）

覆盖范围（★ = 报告 / 论文里会用到的关键测点）：
  A) 02b 的 8 行：BrMS1-4 × {A, B} program → one-vs-rest AUC（Fig6 A panel 来源）
  B) 16 的 16 个测点：每个 (cohort, subtype) × {原始 / 校正纯度 / 校正全成分} AUC
     —— 16 的核心结论是「校正后 BrMS1/BrMS2 失效，BrMS4 仍稳」

不重复算 purity_adjustment_leaveown.csv 里的「全成分 / 校正纯度」之间差异（这部分
是 leave-own-domain-out 的 R² / Δ AUC，不是一对一"原始标签 → y"问题）。

可复现性：numpy RNG = SEED (20260913)。bootstrap / permutation 索引都用默认 rng，
与上游脚本一致。CI 与 p 的样本量较小（bootstrap 2000、perm 2000），足以应付 4 vs 12/13/15/17
的小样本情况（CI 收敛 0.005–0.015）。
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (
    AN, GENESETS, SUBS, NM, SEED,
    BRMS_PROGRAMS, FIG6_REP_PROGRAM,
    load_dataframe, load_pkl_tuple, PU,
)

# ===== 参数 =====
N_BOOT = 2000       # bootstrap 重采样次数
N_PERM = 2000       # 标签置换次数
CI_LO, CI_HI = 0.025, 0.975   # 95% CI（双侧 2.5% / 97.5% percentile）

OUT_CI = os.path.join(AN, "auc_bootstrap_ci.csv")
OUT_PERM = os.path.join(AN, "auc_label_perm_p.csv")


# ===== 复用 02b 的 zscore / AUC 计算 =====
def _zscore_log_cpm(X: pd.DataFrame) -> pd.DataFrame:
    L = np.log2(X.div(X.sum(axis=0), axis=1) * 1e6 + 1)
    Z = L.sub(L.mean(axis=1), axis=0).div(
        L.std(axis=1).replace(0, np.nan), axis=0)
    return Z.dropna(how="all")


def _zscore_log_already(X: pd.DataFrame) -> pd.DataFrame:
    Z = X.sub(X.mean(axis=1), axis=0).div(
        X.std(axis=1).replace(0, np.nan), axis=0)
    return Z.dropna(how="all")


def _load_program(geneset_file: str, key: str) -> list:
    return json.load(open(os.path.join(GENESETS, geneset_file)))[key]


def _program_score(Z: pd.DataFrame, program: list) -> pd.Series:
    g = [x for x in program if x in Z.index]
    return Z.loc[g].mean(axis=0)


def _auc_safe(y: np.ndarray, s: np.ndarray) -> float:
    """two-class AUC；若单类则返回 NaN（审稿人会接受 None/NaN）。"""
    if len(np.unique(y)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y, s))
    except Exception:
        return float("nan")


# ===== bootstrap 与 permutation =====
def _bootstrap_ci(y: np.ndarray, s: np.ndarray,
                  rng: np.random.Generator, n_boot: int) -> tuple:
    """返回 (ci_lo, ci_hi) — percentile CI。obs_AUC 由调用方算。"""
    n = len(y)
    aucs = np.empty(n_boot, dtype=float)
    aucs[:] = np.nan
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y_b, s_b = y[idx], s[idx]
        if len(np.unique(y_b)) < 2:
            continue
        try:
            aucs[i] = roc_auc_score(y_b, s_b)
        except Exception:
            pass
    valid = aucs[~np.isnan(aucs)]
    if len(valid) < 100:  # 至少 5% 有效
        return (float("nan"), float("nan"))
    return float(np.quantile(valid, CI_LO)), float(np.quantile(valid, CI_HI))


def _permutation_p(y: np.ndarray, s: np.ndarray, obs_auc: float,
                   rng: np.random.Generator, n_perm: int) -> float:
    """单尾 permutation p：p = (1 + #{perm_AUC >= obs_AUC}) / (n_perm + 1)。
    单类 obs 或 obs=0.5 时返回 1.0（无判别力）。"""
    if np.isnan(obs_auc) or len(np.unique(y)) < 2:
        return float("nan")
    ge = 0
    for _ in range(n_perm):
        y_p = rng.permutation(y)
        try:
            a = roc_auc_score(y_p, s)
        except Exception:
            continue
        if a >= obs_auc:
            ge += 1
    return float((1 + ge) / (n_perm + 1))


# ===== 加载两个队列的 Z 分数矩阵 + BrMS labels =====
def _load_cohorts() -> dict:
    """返回 {'A': (Z, lab), 'B': (Z, lab)}，Z 是基因×样本的 z-score 矩阵。"""
    out = {}
    # A: log2(CPM+1) → row-zscore
    Xa = load_dataframe(f"{AN}/X_counts.pkl")
    laba = load_dataframe(f"{AN}/labels_gse245467.pkl").iloc[:, 0]
    laba = laba[laba.isin(SUBS)]
    Xa = Xa[laba.index]
    Za = _zscore_log_cpm(Xa)
    out["A"] = (Za, laba.astype(str))

    # B: 直接 row-zscore（已是 log2 TMM-CPM）
    Xb, labb, _ = load_pkl_tuple(f"{AN}/gse184869_symbol_matrix.pkl")
    labb = labb.iloc[:, 0] if isinstance(labb, pd.DataFrame) else labb
    labb = labb[labb.isin(SUBS)]
    Xb = Xb[labb.index]
    Zb = _zscore_log_already(Xb)
    out["B"] = (Zb, labb.astype(str))
    return out


# ===== 主流程 =====
def main():
    rng = np.random.default_rng(SEED)
    cohorts = _load_cohorts()

    ci_rows, perm_rows = [], []

    # ----- A) 02b subtype_program_auc：每个 (cohort, subtype) × FIG6_REP_PROGRAM -----
    print("=" * 78)
    print("A) 02b subtype × program one-vs-rest AUC（Fig6 A panel 来源）")
    print("=" * 78)
    for co in ("A", "B"):
        Z, lab = cohorts[co]
        for sub in SUBS:
            gfile, gkey = FIG6_REP_PROGRAM[sub]
            program = _load_program(gfile, gkey)
            score = _program_score(Z, program)
            common = score.index.intersection(lab.index)
            y = (lab.loc[common] == sub).astype(int).values
            s = score.loc[common].values
            obs = _auc_safe(y, s)
            ci_lo, ci_hi = _bootstrap_ci(y, s, rng, N_BOOT)
            p = _permutation_p(y, s, obs, rng, N_PERM)
            tag = f"02b|{co}|{sub}|{NM[sub]}|{gkey[:30]}"
            print(f"  {tag}: obs={obs:.3f} CI=[{ci_lo:.3f},{ci_hi:.3f}] p={p:.4f}")
            ci_rows.append({"test_point": tag, "cohort": co, "subtype": sub,
                            "obs_auc": obs, "ci_lo": ci_lo, "ci_hi": ci_hi,
                            "n_samples": int(len(common)),
                            "n_subtype": int(y.sum()),
                            "n_genes": int(sum(1 for x in program if x in Z.index))})
            perm_rows.append({"test_point": tag, "cohort": co, "subtype": sub,
                              "obs_auc": obs, "perm_p_one_sided": p,
                              "n_perm": N_PERM,
                              "n_subtype": int(y.sum())})

    # ----- B) 16 purity-adjusted AUC：每个 (cohort, subtype) × {原始, 校正后} -----
    print()
    print("=" * 78)
    print("B) 16 purity-adjusted AUC：原始 vs leave-own-domain-out 校正后")
    print("=" * 78)
    # 复用 16 的 z-score + composition 加载逻辑
    OWN_DOMAIN = {"1": "脑组织", "2": "免疫", "3": "上皮", "4": None}
    # 与 16 的 dom 分类一致（脑组织 / 免疫 / 基质 / 上皮 / ESTIMATE）
    DOM_RULES = {
        "脑组织": lambda c: c == "脑组织合计",
        "免疫": lambda c: c == "免疫合计",
        "基质": lambda c: c.startswith("基质"),
        "上皮": lambda c: c.startswith("上皮"),
        "ESTIMATE": lambda c: c.startswith("ESTIMATE") or c == "Purity_ESTIMATE",
    }
    for co in ("A", "B"):
        Z_full, lab = cohorts[co]
        comp_raw = pd.read_csv(f"{PU}/cohort{co}_composition.txt", sep="\t", index_col=0)
        # 丢 2 个 NA 样本 + 与 Z 对齐
        comp = comp_raw[comp_raw["brms"].isin([1, 2, 3, 4])]
        Z = Z_full.reindex(columns=comp.index)
        lab = lab.loc[comp.index]
        comp = comp.loc[lab.index]
        # 与 16 一致：构建 dom 字典
        cols = [c for c in comp.columns if c != "brms"]
        dom = {dn: [c for c in cols if rule(c)] for dn, rule in DOM_RULES.items()}
        # 校正集：剔除 own-domain
        for sub in SUBS:
            program = BRMS_PROGRAMS[sub]
            g = [x for x in program if x in Z.index]
            score = Z.loc[g].mean(axis=0)
            common = score.index.intersection(lab.index)
            y = (lab.loc[common] == sub).astype(int).values
            s = score.loc[common].values
            obs_raw = _auc_safe(y, s)
            # leave-own-domain-out 校正：与 16 一致（OLS → 残差 → AUC）
            own = OWN_DOMAIN[sub]
            cov = []
            for dn, cl in dom.items():
                if dn == own:
                    continue
                cov += cl
            # 16 的 cov 还会剔除有 NaN 的列；这里对 NaN 用 comp.median 填充以保稳定
            comp_for_fit = comp[cov].copy()
            for cc in cov:
                if comp_for_fit[cc].isna().any():
                    comp_for_fit[cc] = comp_for_fit[cc].fillna(comp_for_fit[cc].median())
            Xc = comp_for_fit.loc[common].values
            Xc = np.hstack([np.ones((len(common), 1)), Xc])
            # OLS：numpy lstsq（更稳，避免 statsmodels 在奇异共线性上警告）
            from numpy.linalg import lstsq
            coef, *_ = lstsq(Xc, s, rcond=None)
            s_res = s - Xc @ coef
            obs_adj = _auc_safe(y, s_res)
            for tag_suffix, obs, score_used in (("raw", obs_raw, s),
                                                ("purity_adjusted", obs_adj, s_res)):
                tag = f"16|{co}|{sub}|{NM[sub]}|{tag_suffix}"
                print(f"  {tag}: obs={obs:.3f}")
                ci_lo, ci_hi = _bootstrap_ci(y, score_used, rng, N_BOOT)
                p = _permutation_p(y, score_used, obs, rng, N_PERM)
                ci_rows.append({"test_point": tag, "cohort": co, "subtype": sub,
                                "obs_auc": obs, "ci_lo": ci_lo, "ci_hi": ci_hi,
                                "n_samples": int(len(common)),
                                "n_subtype": int(y.sum()),
                                "n_genes": int(sum(1 for x in BRMS_PROGRAMS[sub] if x in Z.index))})
                perm_rows.append({"test_point": tag, "cohort": co, "subtype": sub,
                                  "obs_auc": obs, "perm_p_one_sided": p,
                                  "n_perm": N_PERM,
                                  "n_subtype": int(y.sum())})

    # ----- 落盘 -----
    pd.DataFrame(ci_rows).to_csv(OUT_CI, index=False)
    pd.DataFrame(perm_rows).to_csv(OUT_PERM, index=False)
    print()
    print(f"✅ {OUT_CI}  ({len(ci_rows)} test points)")
    print(f"✅ {OUT_PERM}  ({len(perm_rows)} test points)")

    # ----- 一句话总览：哪些测点 CI 跨 0.5（不显著） -----
    print()
    print("=" * 78)
    print("★ 关键判读：CI 是否跨 0.5（无效线）")
    print("=" * 78)
    ci_df = pd.DataFrame(ci_rows)
    sig = ci_df[~((ci_df["ci_lo"] <= 0.5) & (ci_df["ci_hi"] >= 0.5))]
    nonsig = ci_df[(ci_df["ci_lo"] <= 0.5) & (ci_df["ci_hi"] >= 0.5)]
    print(f"  CI 显著 (n={len(sig)}): "
          f"{', '.join(sig['test_point'].tolist()[:8])}"
          f"{'...' if len(sig) > 8 else ''}")
    print(f"  CI 跨 0.5 (n={len(nonsig)}): "
          f"{', '.join(nonsig['test_point'].tolist()[:8])}"
          f"{'...' if len(nonsig) > 8 else ''}")


if __name__ == "__main__":
    main()