"""analysis/33_l3_uncertainty.py — 审稿意见 S8：L3 一直缺少自己的输出

现有的 auc_bootstrap_ci.csv / auc_label_perm_p.csv 是**旧版程序**
（BrMS2 = Allograft Rejection、BrMS3 = Fatty Acid Metabolism，192/156 基因）
算出来的，与 Table 1 的 17/9/315 基因程序不是同一套，不能直接引用。
本脚本对当前 BRMS_PROGRAMS 重新计算：
  - bootstrap 95% CI（N = 2,000，百分位法，按亚型分层重抽样）
  - one-sided label-permutation p（N = 2,000）
cohort A 与 cohort B 各算一次。

2026-09-16 修订（一致性修复）：
  原实现自带 zscore_expr()，对两个队列都做 CPM + log2。队列 B 的矩阵本身
  已经是 log2 尺度，被再取一次对数，导致本脚本的程序分与 L2 的
  `_l2_common.program_score()` 不是同一个量（队列 B BrMS1 AUC 0.853 对
  L2 的 0.872）。现改为直接复用 _l2_common 的装载与打分，保证
  L2 / L3 两层的程序分逐位一致；两者的原始 AUC 因此可以直接互相核对。
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import AN, SUBS, SEED  # noqa: E402
import _l2_common as L2  # noqa: E402

from sklearn.metrics import roc_auc_score  # noqa: E402

RNG = np.random.default_rng(SEED)
N_BOOT = 2000
N_PERM = 2000


def run(Z: pd.DataFrame, y: np.ndarray, cohort: str) -> list:
    rows = []
    for sub in SUBS:
        gset = L2.program_genes(Z, sub)
        if len(gset) < 3:
            continue
        # 与 L2 完全相同的打分口径（mean z-score over 可用基因）
        s = L2.program_score(Z, sub)
        ybin = (y == sub).astype(int)
        if ybin.sum() < 3 or len(ybin) - ybin.sum() < 3:
            continue
        obs = roc_auc_score(ybin, s)

        # bootstrap（分层重抽样）
        idx_pos = np.where(ybin == 1)[0]
        idx_neg = np.where(ybin == 0)[0]
        boots = np.empty(N_BOOT)
        for i in range(N_BOOT):
            ip = RNG.choice(idx_pos, size=len(idx_pos), replace=True)
            inn = RNG.choice(idx_neg, size=len(idx_neg), replace=True)
            ii = np.concatenate([ip, inn])
            boots[i] = roc_auc_score(ybin[ii], s[ii])
        lo, hi = np.percentile(boots, [2.5, 97.5])

        # label permutation（one-sided）
        perms = np.empty(N_PERM)
        for i in range(N_PERM):
            yp = RNG.permutation(ybin)
            perms[i] = roc_auc_score(yp, s)
        p_perm = (1 + (perms >= obs).sum()) / (N_PERM + 1)

        rows.append({"cohort": cohort, "subtype": sub, "n_genes": len(gset),
                     "n_pos": int(ybin.sum()), "auc_obs": round(float(obs), 3),
                     "ci_lo": round(float(lo), 3), "ci_hi": round(float(hi), 3),
                     "perm_p_one_sided": float(p_perm),
                     "n_perm_exceed": int((perms >= obs).sum())})
        print(f"  [{cohort}] BrMS{sub}  {len(gset):4d} genes  "
              f"AUC {obs:.3f}  95%CI [{lo:.3f}, {hi:.3f}]  "
              f"perm p {p_perm:.4f}")
    return rows


def main():
    print("=" * 78)
    print("S8：L3 不确定性（当前 BRMS_PROGRAMS，bootstrap 2,000 + 置换 2,000）")
    print("=" * 78)
    cohorts = L2.load_cohorts()
    rows = []
    for co in ("A", "B"):
        d = cohorts[co]
        Z = d["Z"]
        lab = d["lab"].reindex(Z.columns)
        keep = lab.isin(SUBS).values
        Z = Z.loc[:, keep]
        y = lab[keep].values
        if d["dropped"]:
            print(f"\n[Cohort {co}] 剔除 {len(d['dropped'])} 个全空样本：{d['dropped']}")
        print(f"\n[Cohort {co}]  n = {Z.shape[1]}")
        rows += run(Z, y, co)
    pd.DataFrame(rows).to_csv(f"{AN}/l3_uncertainty.csv", index=False)
    print(f"\n  ✅ Wrote l3_uncertainty.csv ({len(rows)} rows)")


if __name__ == "__main__":
    main()
