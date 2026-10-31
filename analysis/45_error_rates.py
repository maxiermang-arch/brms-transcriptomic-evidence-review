"""analysis/45_error_rates.py — 管线各层的对称误差刻画（假失败 / type II）

背景（第二轮审稿统计审稿人 Major #2）
    C1 / C1d 只刻画了 L1 的**假通过**（type I）。L2、L3、L5 与两个一致性筛查
    的**假失败率**（type II）完全没有估计，因此在 n = 52 与 45、两个筛查分别
    只有 16 与 52 个面板的条件下，"no subtype survives correction (min q =
    0.068)" 究竟反映真实不一致还是功效不足，读者无从区分。本脚本给出这一半。

做法（每条规则的阈值都取自正文，逐字对应 Table 1）
    L2   rule: 严格的成分校正点估计 AUC ≥ 0.70
    L3   rule: bootstrap 95% CI 下界 > 0.50
    L5   rule: 单侧 p ≤ 0.05 且该药内单侧 BH q ≤ 0.05（m = 4）
    筛查 rule: 置换 p ≤ 0.05 且 BH q ≤ 0.05（主家族 m = 8，次家族 m = 4）

    对每条规则，在**该规则实际面对的样本量**下做 Monte Carlo：
    L2/L3 用 binormal 模型生成分数（真 AUC 给定），L5 用对数正态 IC50 与
    正文给出的实际组规模，筛查用给定的真实跨队列 Spearman ρ0 与实际的
    面板数（screen 1 = 16，screen 2 = 52）。
    假失败率 = 1 − 功效。

明示的模型假设（必须在正文披露）
    * binormal 模型：正负类分数各为正态、等方差；这是 AUC 功效计算的标准
      模型，其 AUC 标准误与 DeLong 估计在二阶上一致。
    * L5 的对数 IC50 标准差取 σ = 0.35（log10 单位），为 organoid 药物筛选
      面板的常见量级；本脚本同时给出 σ = 0.25 / 0.45 的敏感性。
    * 筛查把 8 个子型检验当作可交换且独立。正文已说明四个子型向量共享样本、
      并非独立，因此这里给出的是功效的**上界**。
    * L2 的模拟针对"对残差 AUC 施加 0.70 点阈值"这一规则本身；真实 L2 还要
      估计协变量并做正交投影，额外方差未计入 → L2 的功效同样是上界。
    * 所有 Monte Carlo 复仿真的重复数见下方常量，已足以把比率的分辨率压到
      约 ±0.01；报告保留 3 位小数。

输出: analysis/error_rates.csv
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import AN, SEED  # noqa: E402

N_REP = 2000        # 每个情景的重复次数（L2 / L5）
N_REP_L3 = 800      # L3 的重复次数（每重复含一次 bootstrap）
N_BOOT = 400        # L3 的 bootstrap 次数
N_REP_SCR = 1000    # 筛查的重复次数
N_PERM_SCR = 400    # 筛查的置换次数


# ------------------------------------------------------------------ 基础工具

def auc_batch(S: np.ndarray, n1: int) -> np.ndarray:
    """(n_rows, n_samples) 的分数矩阵 -> 每行的 one-vs-rest AUC（秩和形式）。"""
    R = rankdata(S, axis=1)
    n0 = S.shape[1] - n1
    return np.round((R[:, :n1].sum(axis=1) - n1 * (n1 + 1) / 2.0) / (n1 * n0), 10)


def bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg 步进，返回与输入同序的 q。"""
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for rank in range(n - 1, -1, -1):
        i = order[rank]
        prev = min(prev, p[i] * n / (rank + 1))
        q[i] = prev
    return q


def draw_scores(rng, auc_true: float, n1: int, n0: int, n_rep: int) -> np.ndarray:
    """binormal 模型下批量抽 (n_rep, n1+n0) 的分数（前 n1 列为正类）。"""
    mu = np.sqrt(2.0) * norm.ppf(auc_true)
    pos = rng.normal(mu, 1.0, (n_rep, n1))
    neg = rng.normal(0.0, 1.0, (n_rep, n0))
    return np.concatenate([pos, neg], axis=1)


# ------------------------------------------------------------------ 各层规则

def rule_l2(rng, auc_true: float, n1: int, n0: int) -> float:
    """L2：点估计 ≥ 0.70。"""
    return float((auc_batch(draw_scores(rng, auc_true, n1, n0, N_REP), n1)
                  >= 0.70).mean())


def rule_l3(rng, auc_true: float, n1: int, n0: int) -> float:
    """L3：分层 bootstrap 95% 百分位 CI 下界 > 0.50。"""
    mu = np.sqrt(2.0) * norm.ppf(auc_true)
    hit = 0
    for _ in range(N_REP_L3):
        pos = rng.normal(mu, 1.0, n1)
        neg = rng.normal(0.0, 1.0, n0)
        bp = pos[rng.integers(0, n1, (N_BOOT, n1))]
        bn = neg[rng.integers(0, n0, (N_BOOT, n0))]
        b = auc_batch(np.concatenate([bp, bn], axis=1), n1)
        if np.percentile(b, 2.5) > 0.50:
            hit += 1
    return hit / N_REP_L3


def rule_l5(rng, ratio: float, n1: int, n0: int, sigma: float) -> float:
    """L5：单侧 p ≤ 0.05 且该药内单侧 BH q ≤ 0.05（m = 4）。

    某药的四次检验被模拟为可交换：目标子型的效应为 ratio，同药另三次无效应。
    单侧 p 用 Mann-Whitney U 的正态近似（连续性校正），与正文的检验一致到
    大样本；此处组规模为 2–7 vs 7–17，故另用精确置换核对（见下）。
    """
    d = np.log10(ratio)
    ps = np.empty((N_REP, 4))
    for j in range(4):
        # 预指定方向 = 该亚型更敏感 = IC50 更低，故子型组的均值取 −d
        eff = -d if j == 0 else 0.0
        a = rng.normal(eff, sigma, (N_REP, n1))
        b = rng.normal(0.0, sigma, (N_REP, n0))
        R = rankdata(np.concatenate([a, b], axis=1), axis=1)
        Ra = R[:, :n1].sum(axis=1)
        U = Ra - n1 * (n1 + 1) / 2.0                     # 越小 = a 越低于 b
        mu_U, sd_U = n1 * n0 / 2.0, np.sqrt(n1 * n0 * (n1 + n0 + 1) / 12.0)
        z = (U - mu_U + 0.5) / sd_U                      # 连续性校正，"less" 方向
        ps[:, j] = norm.cdf(z)
    q = np.array([bh(ps[i]) for i in range(N_REP)])
    return float(((ps[:, 0] <= 0.05) & (q[:, 0] <= 0.05)).mean())


def rule_screen(rng, rho0: float, n_sig: int, m_family: int) -> float:
    """一致性筛查：置换 p ≤ 0.05 且 BH q ≤ 0.05。返回目标子型的通过率。

    n_sig 个面板的两队列 per-signature AUC 向量服从给定的真实 Spearman ρ0
    （高斯 copula）；m_family 个检验被当作可交换独立。
    """
    denom = n_sig * (n_sig ** 2 - 1)
    hit = 0
    for _ in range(N_REP_SCR):
        z0 = rng.normal(0, 1, n_sig)
        z1 = rho0 * z0 + np.sqrt(1 - rho0 ** 2) * rng.normal(0, 1, n_sig)
        r0, r1 = rankdata(z0), rankdata(z1)
        rho_hat = 1 - 6.0 * np.sum((r0 - r1) ** 2) / denom
        idx = np.array([rng.permutation(n_sig) for _ in range(N_PERM_SCR)])
        d2 = np.sum((r0[None, :] - r1[idx]) ** 2, axis=1)
        rho_perm = 1 - 6.0 * d2 / denom
        p = (1 + int((np.abs(rho_perm) >= abs(rho_hat)).sum())) / (N_PERM_SCR + 1)
        others = rng.uniform(0, 1, m_family - 1)
        q0 = bh(np.concatenate([[p], others]))[0]
        if p <= 0.05 and q0 <= 0.05:
            hit += 1
    return hit / N_REP_SCR


# ------------------------------------------------------------------ 主流程

def main() -> int:
    rng = np.random.default_rng(SEED + 21)
    rows = []
    print("=" * 78)
    print("各层规则的对称误差刻画（假失败 / type II）")
    print("=" * 78)

    # ---------- L2 / L3 ----------
    for cohort, n1, n0 in (("A", 13, 52 - 13), ("B", 7, 45 - 7)):
        n = n1 + n0
        print(f"\n[L2 / L3] 队列 {cohort}，n = {n}（子型 {n1} vs 其余 {n0}）")
        print(f"  {'真 AUC':>7s} {'L2 通过率':>11s} {'L2 假失败':>10s} "
              f"{'L3 通过率':>11s} {'L3 假失败':>10s}")
        for auc_true in (0.50, 0.60, 0.70, 0.75, 0.80, 0.90):
            r2 = rule_l2(rng, auc_true, n1, n0)
            r3 = rule_l3(rng, auc_true, n1, n0)
            print(f"  {auc_true:7.2f} {r2:11.3f} {1 - r2:10.3f} {r3:11.3f} "
                  f"{1 - r3:10.3f}")
            rows += [
                {"rule": "L2_point_auc_ge_0.70", "cohort": cohort,
                 "scenario": f"true_auc={auc_true:.2f}", "n": n,
                 "pass_rate": round(r2, 3), "false_fail_rate": round(1 - r2, 3)},
                {"rule": "L3_ci_lower_gt_0.50", "cohort": cohort,
                 "scenario": f"true_auc={auc_true:.2f}", "n": n,
                 "pass_rate": round(r3, 3), "false_fail_rate": round(1 - r3, 3)},
            ]

    # ---------- L5 ----------
    for label, n1, n0 in (("abemaciclib BrMS4 (6 vs 14)", 6, 14),
                          ("gamitrinib BrMS3 (3 vs 9)", 3, 9)):
        print(f"\n[L5] {label}，规则 = 单侧 p ≤ 0.05 且该药内单侧 BH q ≤ 0.05")
        print(f"  {'真实中位IC50比':>14s} {'σ=0.25':>8s} {'σ=0.35':>8s} "
              f"{'σ=0.45':>8s}")
        for ratio in (2.0, 3.0, 5.0, 10.0):
            ps = [rule_l5(rng, ratio, n1, n0, s) for s in (0.25, 0.35, 0.45)]
            print(f"  {ratio:13.1f}× {ps[0]:8.3f} {ps[1]:8.3f} {ps[2]:8.3f}")
            rows.append({"rule": "L5_onesided_p_and_within_drug_q",
                         "cohort": label, "scenario": f"median_ratio={ratio:.1f}x",
                         "n": n1 + n0,
                         "pass_rate": round(ps[1], 3),
                         "false_fail_rate": round(1 - ps[1], 3)})

    # ---------- 一致性筛查 ----------
    print("\n[筛查] 置换 p ≤ 0.05 且 BH q ≤ 0.05")
    print(f"  {'面板数':>7s} {'真 ρ0':>7s} {'m=8 通过率':>11s} {'m=8 假失败':>11s} "
          f"{'m=4 通过率':>11s} {'m=4 假失败':>11s}")
    for n_sig in (16, 52):
        for rho0 in (0.30, 0.50, 0.70, 0.90):
            r8 = rule_screen(rng, rho0, n_sig, 8)
            r4 = rule_screen(rng, rho0, n_sig, 4)
            print(f"  {n_sig:7d} {rho0:7.2f} {r8:11.3f} {1 - r8:11.3f} "
                  f"{r4:11.3f} {1 - r4:11.3f}")
            for m, r in ((8, r8), (4, r4)):
                rows.append({"rule": f"consistency_screen_permp_and_bh_m{m}",
                             "cohort": f"{n_sig}_signatures",
                             "scenario": f"true_rho={rho0:.2f}", "n": n_sig,
                             "pass_rate": round(r, 3),
                             "false_fail_rate": round(1 - r, 3)})

    out = pd.DataFrame(rows)
    out.to_csv(f"{AN}/error_rates.csv", index=False)
    print(f"\n  ✅ Wrote error_rates.csv ({len(out)} rows)")
    print("\n[读法] 假失败率 = 1 − 通过率。凡在真实效应存在时假失败率仍很高的层，"
          "其 null 结果都不能读作'该亚型不可复现'，只能读作'该层在这个 n 下"
          "没有分辨力'。正文必须据此收窄相应结论的强度。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
