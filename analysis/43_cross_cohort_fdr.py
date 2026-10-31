"""analysis/43_cross_cohort_fdr.py — Table 4 的多重检验校正（可复现）

背景
----
Table 4 的 *q* 值此前是手算的，没有落盘，无法回溯。本脚本读 07/08 号脚本
的输出（两个跨队列一致性筛查的 Spearman ρ 与置换 p），把两套 BH 校正
重新算一遍并存档：

  * 主要规则：两个筛查合计 8 个检验（m = 8）合并做 BH。
    理由：亚型层面的"跨队列一致"声称必须在两套 panel 下都成立，
    因此一个亚型贡献两个检验，多重性应当按 8 计。
  * 次要规则：单个筛查内部 4 个检验（m = 4）做 BH，仅作对照报告。

同时把"每个亚型 m = 8 下的两个 q 中较小者"单独成列，供 Table 4 的
*q* (m = 8) 列使用，并显式暴露此前表内 BrMS4 一格用的是较大值。

输出: analysis/cross_cohort_fdr.csv
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _common import AN


def bh(p):
    """Benjamini-Hochberg step-up q 值。p 为 1 维数组，返回同长度 q。"""
    p = np.asarray(p, dtype=float)
    m = len(p)
    order = np.argsort(p, kind="mergesort")     # 稳定排序，平局可复现
    ps = p[order]
    q = ps * m / (np.arange(m) + 1.0)
    q = np.minimum.accumulate(q[::-1])[::-1]     # 从大到小取累计最小
    q = np.minimum(q, 1.0)
    out = np.empty(m, dtype=float)
    out[order] = q
    return out


def main():
    s1 = pd.read_csv(f"{AN}/cross_cohort_consistency.csv")     # 16-panel 免疫/EMT
    s2 = pd.read_csv(f"{AN}/broad_panel_consistency.csv")      # 52-pathway

    s1["subtype"] = s1["subtype"].astype(str).str.strip()
    # broad_panel 的 subtype 列形如 "BrMS1 neural"，取第一段
    s2["subtype"] = s2["subtype"].astype(str).str.split().str[0].str.strip()

    subs = ["BrMS1", "BrMS2", "BrMS3", "BrMS4"]
    recs = []
    for s in subs:
        r1 = s1.loc[s1.subtype == s].iloc[0]
        r2 = s2.loc[s2.subtype == s].iloc[0]
        recs.append({"subtype": s,
                     "screen1_rho": round(float(r1.spearman_rho), 3),
                     "screen1_p": float(r1.p_perm),
                     "screen2_rho": round(float(r2.spearman_rho), 3),
                     "screen2_p": float(r2.p_perm)})
    df = pd.DataFrame(recs)

    # ---- 次要规则：筛查内 m = 4 ----
    df["q1_m4"] = bh(df["screen1_p"].values)
    df["q2_m4"] = bh(df["screen2_p"].values)

    # ---- 主要规则：两筛查合并 m = 8 ----
    pooled = np.concatenate([df["screen1_p"].values, df["screen2_p"].values])
    q8 = bh(pooled)
    df["q1_m8"] = q8[: len(df)]
    df["q2_m8"] = q8[len(df):]
    df["q_m8_min"] = np.minimum(df["q1_m8"], df["q2_m8"])
    df["q_m8_max"] = np.maximum(df["q1_m8"], df["q2_m8"])

    for c in [c for c in df.columns if c.startswith(("screen", "q"))]:
        df[c] = df[c].round(4)

    print("=" * 78)
    print("Table 4 的 BH 校正（主要 m = 8 / 次要 m = 4）")
    print("=" * 78)
    hdr = (f"{'subtype':8s} {'ρ1':>7s} {'p1':>8s} {'q1(m4)':>7s} "
           f"{'ρ2':>7s} {'p2':>8s} {'q2(m4)':>7s} "
           f"{'q1(m8)':>7s} {'q2(m8)':>7s} {'min':>7s} {'max':>7s}")
    print(hdr)
    for r in df.itertuples():
        print(f"{r.subtype:8s} {r.screen1_rho:7.3f} {r.screen1_p:8.4f} "
              f"{r.q1_m4:7.3f} {r.screen2_rho:7.3f} {r.screen2_p:8.4f} "
              f"{r.q2_m4:7.3f} {r.q1_m8:7.3f} {r.q2_m8:7.3f} "
              f"{r.q_m8_min:7.3f} {r.q_m8_max:7.3f}")

    print(f"\n  主要规则 m = 8：最小 q = {df.q_m8_min.min():.3f}"
          f"（{df.loc[df.q_m8_min.idxmin(), 'subtype']}）"
          f"；q ≤ 0.05 的检验数 = {int((q8 <= 0.05).sum())} / 8")
    print(f"  次要规则 m = 4：screen1 q ≤ 0.05 的亚型 = "
          f"{list(df.loc[df.q1_m4 <= 0.05, 'subtype'])}；"
          f"screen2 = {list(df.loc[df.q2_m4 <= 0.05, 'subtype'])}")

    df.to_csv(f"{AN}/cross_cohort_fdr.csv", index=False)
    print("\n  ✅ Wrote cross_cohort_fdr.csv")


if __name__ == "__main__":
    main()
