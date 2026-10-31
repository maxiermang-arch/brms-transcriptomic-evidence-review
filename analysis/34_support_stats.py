"""analysis/34_support_stats.py — 审稿意见 S10 / S20 的支撑统计

S10  Table 2 的 4x4 交叉表有多格期望频数 < 5，渐近卡方 p 不可靠。
     本脚本用 Monte Carlo（置换组织学标签，保持边际）重算 p。
S20  §2.4 预设"两个最大单一组织学层"，Table 2 中实为 Lung (17) 与
     Breast (16)；§3.2 却只报了第三大的 Melanoma (14)。本脚本把四个
     层内的 programme AUC 全部算出并落盘，供正文如实报告。

输出: analysis/chisq_montecarlo.csv, analysis/stratified_auc.csv
"""
import os
import re
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import AN, SUBS, BRMS_PROGRAMS, SEED

from sklearn.metrics import roc_auc_score
from scipy.stats import chi2_contingency

RNG = np.random.default_rng(SEED)
N_MC = 20000


def main():
    lab = pd.read_parquet(f"{AN}/labels_used.parquet")
    lab["tag"] = [re.match(r"^([A-Z]+)", s).group(1) for s in lab.index]
    hist_map = {"LUNBRM": "Lung", "BREBRM": "Breast",
                "MELBRM": "Melanoma", "RCCBRM": "RCC"}
    lab["histo"] = lab["tag"].map(hist_map)

    print("=" * 78)
    print("S10 / S20：交叉表 Monte Carlo p + 各组织学层内 programme AUC")
    print("=" * 78)

    # ---------- S10 ----------
    ct = pd.crosstab(lab["histo"], lab["brms"])
    chi2, p_asym, dof, exp = chi2_contingency(ct.values)
    n = ct.values.sum()
    v = (chi2 / (n * (min(ct.shape) - 1))) ** 0.5
    # Monte Carlo：置换 histo 标签，边际自动保持
    obs = chi2
    cnt = 0
    for _ in range(N_MC):
        perm = RNG.permutation(lab["histo"].values)
        ct2 = pd.crosstab(pd.Series(perm, index=lab.index), lab["brms"].values)
        try:
            c2, _, _, _ = chi2_contingency(ct2.values)
        except ValueError:
            continue
        if c2 >= obs:
            cnt += 1
    p_mc = (1 + cnt) / (N_MC + 1)
    min_exp = float(exp.min())
    n_lt5 = int((exp < 5).sum())
    print(f"\n[S10] chi2 = {chi2:.2f}, dof = {dof}, 渐近 p = {p_asym:.3e}")
    print(f"      期望频数最小 = {min_exp:.2f}，< 5 的格子数 = {n_lt5}/16")
    print(f"      Monte Carlo p ({N_MC} 次置换) = {p_mc:.2e}")
    print(f"      Cramers V = {v:.3f}")
    pd.DataFrame([{"chi2": round(float(chi2), 3), "dof": int(dof),
                   "p_asymptotic": float(p_asym), "p_montecarlo": float(p_mc),
                   "n_mc": N_MC, "min_expected": round(min_exp, 3),
                   "n_cells_lt5": n_lt5,
                   "cramers_v": round(float(v), 3)}]).to_csv(
        f"{AN}/chisq_montecarlo.csv", index=False)

    # ---------- S20 ----------
    Xc = pd.read_parquet(f"{AN}/X_counts.parquet")
    X = Xc[lab.index]
    cpm = X.div(X.sum(axis=0), axis=1) * 1e6
    L = np.log2(cpm + 1)
    ZA = L.sub(L.mean(axis=1), axis=0).div(L.std(axis=1).replace(0, np.nan), axis=0)
    ZA = ZA.dropna(how="all")

    print("\n[S20] 各组织学层内的 programme AUC（one-vs-rest）")
    print(f"  {'层':10s} {'n':>4s} " + " ".join(f"{'BrMS'+s:>8s}" for s in SUBS))
    rows = []
    for h in ["Lung", "Breast", "Melanoma", "RCC"]:
        m = (lab["histo"] == h).values
        yy = lab["brms"].astype(str).values[m]
        line = f"  {h:10s} {m.sum():4d} "
        for s in SUBS:
            gset = [g for g in BRMS_PROGRAMS[s] if g in ZA.index]
            sc = ZA.loc[gset].mean(axis=0).values[m]
            yb = (yy == s).astype(int)
            if yb.sum() == 0 or yb.sum() == len(yb):
                line += f"{'—':>9s}"
                rows.append({"stratum": h, "n": int(m.sum()), "subtype": s,
                             "n_pos": int(yb.sum()), "auc": None})
                continue
            auc = roc_auc_score(yb, sc)
            line += f"{auc:9.3f}"
            rows.append({"stratum": h, "n": int(m.sum()), "subtype": s,
                         "n_pos": int(yb.sum()), "auc": round(float(auc), 3)})
        print(line)
    pd.DataFrame(rows).to_csv(f"{AN}/stratified_auc.csv", index=False)
    print(f"\n  ✅ Wrote chisq_montecarlo.csv, stratified_auc.csv")


if __name__ == "__main__":
    main()
