"""analysis/32_calibration.py — 管线自身的校准与回收检验

两位审稿人（R1-M7 / R2-M3）要求：管线自身从未被校准，无法判断
"通过"意味着亚型真的稳健，还是本审计不敏感。本脚本做两个对照：

对照 C1（空分类器 / 阴性对照）
    把 cohort A 的 BrMS 标签随机置换 N_PERM 次。**每次置换都必须重建
    L1 的参照分布**：L1 的比较对象是"同样大小的随机基因集"，而随机
    基因集的 AUC 分布本身取决于标签 —— 标签被置换后，随机基因集的
    AUC 分布会向 0.5 收缩。因此本脚本在每一次置换下：
        1. 用置换标签 y_p 重算目标程序的 AUC；
        2. 用**同一套置换标签** y_p 重算全部 size-matched 随机基因集的
           AUC，得到这一次置换自己的经验零分布；
        3. 由该零分布求经验尾概率 p 与 z。
    随机基因集的**成员**是固定的（一次抽定后复用），但它们的 AUC 每次
    都重算。沿用原始标签下的零分布是错的：那种做法的 p 不是对 L1 检验
    程序的重复，而得出的"假通过率 0%"只是因为分子用 ≈0.5、分母用 0.758。

对照 C2（植入的组织学混杂伪亚型 / 回收检验）
    伪标签 = 黑色素瘤 vs 其余（按构造 100% 由组织学决定）。零分布同样
    在该伪标签下重建。该伪分类器应当：L1 通过（有真实表达差异），而
    混杂筛查报警（histology-alone AUC = 1.000）。

对照 C3（基因集规模虚高）
    引用 size_matched_null.csv 报告的零分布均值，说明"大基因集本身就能
    拿到约 0.6 的 AUC"。这是描述性引用，不参与判定。

输出: analysis/calibration_controls.csv
"""
from __future__ import annotations

import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import beta, rankdata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import AN, SUBS, BRMS_PROGRAMS, SEED  # noqa: E402

N_PERM = 1000        # 置换标签次数
N_NULL = 2000        # 每次置换下重建零分布所用的随机基因集数（与 L1 一致）


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """二项比例的 Clopper-Pearson 精确区间（k 次成功 / n 次试验）。"""
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def load_A():
    lab = pd.read_parquet(f"{AN}/labels_used.parquet")
    lab["tag"] = [re.match(r"^([A-Z]+)", s).group(1) for s in lab.index]
    hist_map = {"LUNBRM": "Lung", "BREBRM": "Breast",
                "MELBRM": "Melanoma", "RCCBRM": "RCC"}
    lab["histo"] = lab["tag"].map(hist_map)
    Xc = pd.read_parquet(f"{AN}/X_counts.parquet")
    X = Xc[lab.index]
    cpm = X.div(X.sum(axis=0), axis=1) * 1e6
    L = np.log2(cpm + 1)
    ZA = L.sub(L.mean(axis=1), axis=0).div(L.std(axis=1).replace(0, np.nan), axis=0)
    ZA = ZA.dropna(how="all")
    y = lab["brms"].astype(str).values
    return ZA, y, lab


def auc_vec(R: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """由平均秩矩阵批量算 one-vs-rest AUC（Mann-Whitney U 的秩和形式）。

    R 的每一行是一个打过分数的平均秩；pos 是正类掩码。
    """
    n1 = int(pos.sum())
    n0 = R.shape[1] - n1
    a = (R[:, pos].sum(axis=1) - n1 * (n1 + 1) / 2.0) / (n1 * n0)
    return np.round(a, 10)          # 让真正的平局并列，保证跨机一致


def draw_null_sets(rng: np.random.Generator, n_genes: int, k: int,
                   n_null: int) -> np.ndarray:
    """一次抽定 n_null 个大小为 k 的随机基因集（成员固定，AUC 每次重算）。"""
    return np.array([rng.choice(n_genes, size=k, replace=False)
                     for _ in range(n_null)])


def null_scores(Zv: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """(n_null, k) 的基因索引 -> (n_null, n_samples) 的随机集分数。"""
    out = np.empty((idx.shape[0], Zv.shape[1]))
    for i, rows in enumerate(idx):
        out[i] = Zv[rows].mean(axis=0)
    return out


def l1_from_null(obs: np.ndarray, S_null: np.ndarray, pos: np.ndarray):
    """在给定标签下算 L1 的观测 AUC、经验 p 与 z。

    obs    目标程序分数（长度 n）
    S_null 随机基因集分数（n_null × n），在同一套标签下评分
    pos    正类掩码（长度 n）
    """
    n1 = int(pos.sum())
    n0 = len(pos) - n1
    r_obs = rankdata(obs)
    auc_obs = float((r_obs[pos].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))
    R = rankdata(S_null, axis=1)
    nulls = auc_vec(R, pos)
    mu = float(nulls.mean())
    sd = float(nulls.std())
    p = float((1 + int((nulls >= auc_obs).sum())) / (len(nulls) + 1))
    z = float((auc_obs - mu) / sd) if sd > 0 else float("nan")
    return auc_obs, mu, sd, p, z


def main():
    ZA, y, lab = load_A()
    genes = ZA.index.tolist()
    Zv = ZA.to_numpy()

    print("=" * 78)
    print("校准：对照 C1（空分类器）与 C2（植入的组织学混杂）")
    print("=" * 78)
    print(f"  每次置换下重建 L1 零分布：随机基因集 {N_NULL} 个（成员固定，"
          f"AUC 每次重算）")

    rng_null = np.random.default_rng(SEED)          # 只用于抽定随机集成员
    RNG = np.random.default_rng(SEED + 7)           # 只用于置换标签

    scores, null_idx, S_null = {}, {}, {}
    for sub in SUBS:
        gset = [g for g in BRMS_PROGRAMS[sub] if g in ZA.index]
        scores[sub] = ZA.loc[gset].mean(axis=0).values
        null_idx[sub] = draw_null_sets(rng_null, len(genes), len(gset), N_NULL)
        S_null[sub] = null_scores(Zv, null_idx[sub])

    rows = []

    # ---------- C1：置换标签 ----------
    print(f"\n[C1] 随机置换标签 {N_PERM} 次，每次在置换标签下重建零分布")
    zz_by_sub = {s: [] for s in SUBS}
    pp_by_sub = {s: [] for s in SUBS}
    mu_by_sub = {s: [] for s in SUBS}
    sd_by_sub = {s: [] for s in SUBS}
    for _ in range(N_PERM):
        yp = RNG.permutation(y)
        for sub in SUBS:
            pos = (yp == sub)
            auc_obs, mu, sd, p, z = l1_from_null(scores[sub], S_null[sub], pos)
            zz_by_sub[sub].append(z)
            pp_by_sub[sub].append(p)
            mu_by_sub[sub].append(mu)
            sd_by_sub[sub].append(sd)

    print(f"  {'亚型':6s} {'null均值':>9s} {'null标准差':>10s} "
          f"{'z均值':>7s} {'z>=1.96比例 [95%CI]':>26s} "
          f"{'经验p<=0.05比例 [95%CI]':>30s}")
    for sub in SUBS:
        zz = np.asarray(zz_by_sub[sub])
        pp = np.asarray(pp_by_sub[sub])
        nz = int((zz >= 1.96).sum())
        npx = int((pp <= 0.05).sum())
        f_z, f_p = nz / N_PERM, npx / N_PERM
        lo_z, hi_z = clopper_pearson(nz, N_PERM)
        lo_p, hi_p = clopper_pearson(npx, N_PERM)
        print(f"  BrMS{sub:5s} {np.mean(mu_by_sub[sub]):9.3f} "
              f"{np.mean(sd_by_sub[sub]):10.3f} {zz.mean():7.3f} "
              f"{f_z:11.3f} [{lo_z:.3f}, {hi_z:.3f}] "
              f"{f_p:13.3f} [{lo_p:.3f}, {hi_p:.3f}]")
        for stat, val in (
            ("n_perm", N_PERM), ("n_null_per_perm", N_NULL),
            ("null_mean_of_means", round(float(np.mean(mu_by_sub[sub])), 3)),
            ("null_sd_of_means", round(float(np.mean(sd_by_sub[sub])), 3)),
            ("z_mean", round(float(zz.mean()), 3)),
            ("z_sd", round(float(zz.std()), 3)),
            ("false_pass_rate_z1.96", round(f_z, 3)),
            ("false_pass_rate_z1.96_ci_lo", round(lo_z, 4)),
            ("false_pass_rate_z1.96_ci_hi", round(hi_z, 4)),
            ("false_pass_rate_p0.05", round(f_p, 3)),
            ("false_pass_rate_p0.05_ci_lo", round(lo_p, 4)),
            ("false_pass_rate_p0.05_ci_hi", round(hi_p, 4)),
        ):
            rows.append({"control": "C1_permuted_labels", "subtype": sub,
                         "stat": stat, "value": val})

    # ---------- C1b：零分布自洽性诊断 ----------
    # 若把"目标程序"换成一个**随机抽定的、本身就在参照集合里的基因集**，
    # 置换标签下的经验 p 必须近似均匀、假通过率必须回到 5%。如果回到 5%，
    # 说明参照分布本身没问题，真实程序偏高的假通过率来自程序自身的性质
    # （共表达使模块均值的方差大于等大小随机集）；如果也偏高，说明是参照
    # 分布自身的缺陷。这个对照把两种解释分开。
    print("\n[C1b] 诊断：以随机基因集自身作为‘伪目标程序’重跑 C1")
    pseudo_rate = {}
    for sub in SUBS:
        pp = []
        for _ in range(N_PERM):
            yp = RNG.permutation(y)
            _, _, _, p, _ = l1_from_null(S_null[sub][0], S_null[sub], yp == sub)
            pp.append(p)
        pp = np.asarray(pp)
        k = int((pp <= 0.05).sum())
        lo, hi = clopper_pearson(k, N_PERM)
        pseudo_rate[sub] = k / N_PERM
        print(f"  BrMS{sub:5s} 伪目标程序假通过率 = {k / N_PERM:.3f} "
              f"[{lo:.3f}, {hi:.3f}]  （名义 0.050）")
        rows.append({"control": "C1b_null_self_consistency", "subtype": sub,
                     "stat": "false_pass_rate_p0.05",
                     "value": round(k / N_PERM, 3)})
        rows.append({"control": "C1b_null_self_consistency", "subtype": sub,
                     "stat": "false_pass_rate_p0.05_ci_lo", "value": round(lo, 4)})
        rows.append({"control": "C1b_null_self_consistency", "subtype": sub,
                     "stat": "false_pass_rate_p0.05_ci_hi", "value": round(hi, 4)})

    # ---------- C2：植入的组织学混杂伪分类器 ----------
    print("\n[C2] 伪分类器 = 黑色素瘤 vs 其余（构造上 100% 由组织学决定）")
    y_pseudo = (lab["histo"] == "Melanoma").values
    H = pd.get_dummies(lab["histo"], drop_first=False).astype(float)
    sub = "2"
    auc_pseudo, mu_p, sd_p, p_pseudo, z_pseudo = l1_from_null(
        scores[sub], S_null[sub], y_pseudo)
    print(f"  L1：免疫程序对该伪标签 AUC = {auc_pseudo:.3f}, z = {z_pseudo:.2f}, "
          f"经验 p = {p_pseudo:.4f}  -> "
          f"{'通过' if p_pseudo <= 0.05 else '不通过'}")
    print(f"     （该伪标签下的零分布：均值 {mu_p:.3f}, 标准差 {sd_p:.3f}）")

    # 组织学对该伪标签是**完全分离**的（melanoma 哑变量本身即可完美预测），
    # 逻辑回归的 Hessian 因此奇异、无法拟合；这不影响 AUC，因为它按构造就是
    # 1.000。直接以该哑变量作为分数计算，不再拟合模型。
    auc_hist = float(np.round(
        auc_vec(rankdata(np.asarray(H["Melanoma"].values, dtype=float)[None, :],
                         axis=1), y_pseudo)[0], 10))
    print(f"  混杂筛查：组织学单独预测该伪标签 AUC = {auc_hist:.3f}  -> 报警"
          f"（完全分离，AUC 由构造给出，未拟合模型）")
    for stat, val in (("L1_auc", round(float(auc_pseudo), 3)),
                      ("L1_z", round(float(z_pseudo), 2)),
                      ("L1_p", round(float(p_pseudo), 4)),
                      ("null_mean", round(float(mu_p), 3)),
                      ("null_sd", round(float(sd_p), 3)),
                      ("histology_alone_auc", round(auc_hist, 3))):
        rows.append({"control": "C2_planted_histology_confound",
                     "subtype": "pseudo", "stat": stat, "value": val})

    # ---------- C3：规模虚高（引用既有零分布） ----------
    null = pd.read_csv(f"{AN}/size_matched_null.csv")
    null = null[null["cohort"] == "A"]
    for r in null.itertuples():
        rows.append({"control": "C3_size_inflation", "subtype": str(r.subtype),
                     "stat": f"null_mean_{r.n_genes}genes",
                     "value": float(r.null_mean)})

    out = pd.DataFrame(rows)
    out.to_csv(f"{AN}/calibration_controls.csv", index=False)
    print(f"\n  Wrote calibration_controls.csv ({len(out)} rows)")

    print("\n[读法] C1 的假通过率应当接近名义 5%。若明显低于 5%，说明 L1 "
          "保守；若高于 5%，说明反保守。C1 只覆盖“标签与表达无关”这一种"
          "零假设，不等于竞争性基因集检验在所有情景下的错误率。")


if __name__ == "__main__":
    main()
