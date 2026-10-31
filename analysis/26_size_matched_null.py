"""analysis/26_size_matched_null.py — R-8 / R-9 / R-11

R-8  基因集规模零分布：BrMS4 = 315 基因，BrMS2 = 17，BrMS3 = 9。
     "BrMS4 最稳健" 是否只是"基因集大"？用 size-matched 随机基因集
     构造零分布，看各亚型观测 AUC 相对同等规模随机集的百分位。
R-9  离散性检验：四亚型是离散的还是一条连续增殖轴？
     用 silhouette（ programme-score 空间） + gap statistic 近似。
R-11 交叉验证 AUC（分层 5-fold，按标签分层），替代全部样本内 AUC。

输出: analysis/size_matched_null.csv, analysis/discreteness.csv,
      analysis/cv_auc.csv
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import AN, SUBS, BRMS_PROGRAMS, SEED
import _l2_common as L2

from sklearn.metrics import roc_auc_score, silhouette_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

RNG = np.random.default_rng(SEED)
N_NULL = 2000


def main():
    lab = pd.read_parquet(f"{AN}/labels_used.parquet")
    Xc = pd.read_parquet(f"{AN}/X_counts.parquet")
    X = Xc[lab.index]
    cpm = X.div(X.sum(axis=0), axis=1) * 1e6
    L = np.log2(cpm + 1)
    ZA = L.sub(L.mean(axis=1), axis=0).div(L.std(axis=1).replace(0, np.nan), axis=0)
    ZA = ZA.dropna(how="all")
    genes = ZA.index.tolist()
    y = lab["brms"].astype(str).values

    print("=" * 78)
    print("R-8 / R-9 / R-11")
    print("=" * 78)

    # ---------- R-8: size-matched null ----------
    print("\n[R-8] size-matched 随机基因集零分布 (N=%d)" % N_NULL)
    print(f"  {'亚型':6s} {'n基因':>6s} {'观测AUC':>8s} {'零分布均值':>10s} "
          f"{'P95':>7s} {'经验p':>9s} {'z':>7s}")
    null_rows = []
    scores = {}
    for sub in SUBS:
        gset = [g for g in BRMS_PROGRAMS[sub] if g in ZA.index]
        k = len(gset)
        obs = ZA.loc[gset].mean(axis=0).values
        scores[sub] = obs
        auc_obs = roc_auc_score((y == sub).astype(int), obs)

        nulls = np.empty(N_NULL)
        for i in range(N_NULL):
            rs = RNG.choice(len(genes), size=k, replace=False)
            nulls[i] = roc_auc_score((y == sub).astype(int),
                                     ZA.iloc[rs].mean(axis=0).values)
        mu, sd = nulls.mean(), nulls.std()
        p_emp = (1 + (nulls >= auc_obs).sum()) / (N_NULL + 1)
        z = (auc_obs - mu) / sd if sd > 0 else np.nan
        print(f"  BrMS{sub:5s} {k:6d} {auc_obs:8.3f} {mu:10.3f} "
              f"{np.percentile(nulls,95):7.3f} {p_emp:9.4f} {z:7.2f}")
        null_rows.append({"subtype": sub, "n_genes": k, "auc_obs": round(auc_obs, 3),
                          "null_mean": round(mu, 3), "null_sd": round(sd, 3),
                          "null_p95": round(float(np.percentile(nulls, 95)), 3),
                          "emp_p": round(float(p_emp), 4), "z": round(float(z), 2),
                          "cohort": "A"})

    # ---------- R-8b: 队列 B 的同规模零分布（2026-09-16 补） ----------
    # 用**独立**随机流，保证上面队列 A 的随机序列与历史结果逐位一致，
    # 从而 R-9 / R-11 的输出不因本次补充而改变。
    # 打分口径复用 _l2_common，与 L2 / L3 完全一致。
    print("\n[R-8b] size-matched 随机基因集零分布 — 队列 B (N=%d)" % N_NULL)
    RNG_B = np.random.default_rng(SEED + 1)
    _co = L2.load_cohorts()
    ZB = _co["B"]["Z"]
    labB = _co["B"]["lab"].reindex(ZB.columns)
    keepB = labB.isin(SUBS).values
    ZB, yB = ZB.loc[:, keepB], labB[keepB].values
    genesB = ZB.index.tolist()
    print(f"  队列 B n = {ZB.shape[1]}，可用基因 {len(genesB)}")
    print(f"  {'亚型':6s} {'n基因':>6s} {'观测AUC':>8s} {'零分布均值':>10s} "
          f"{'P95':>7s} {'经验p':>9s} {'z':>7s}")
    for sub in SUBS:
        gset = L2.program_genes(ZB, sub)
        k = len(gset)
        obs = L2.program_score(ZB, sub)
        auc_obs = roc_auc_score((yB == sub).astype(int), obs)
        nulls = np.empty(N_NULL)
        for i in range(N_NULL):
            rs = RNG_B.choice(len(genesB), size=k, replace=False)
            nulls[i] = roc_auc_score((yB == sub).astype(int),
                                     ZB.iloc[rs].mean(axis=0).values)
        mu, sd = nulls.mean(), nulls.std()
        p_emp = (1 + (nulls >= auc_obs).sum()) / (N_NULL + 1)
        z = (auc_obs - mu) / sd if sd > 0 else np.nan
        print(f"  BrMS{sub:5s} {k:6d} {auc_obs:8.3f} {mu:10.3f} "
              f"{np.percentile(nulls, 95):7.3f} {p_emp:9.4f} {z:7.2f}")
        null_rows.append({"subtype": sub, "n_genes": k, "auc_obs": round(auc_obs, 3),
                          "null_mean": round(mu, 3), "null_sd": round(sd, 3),
                          "null_p95": round(float(np.percentile(nulls, 95)), 3),
                          "emp_p": round(float(p_emp), 4), "z": round(float(z), 2),
                          "cohort": "B"})

    pd.DataFrame(null_rows)[
        ["cohort", "subtype", "n_genes", "auc_obs", "null_mean", "null_sd",
         "null_p95", "emp_p", "z"]
    ].to_csv(f"{AN}/size_matched_null.csv", index=False)

    # ---------- R-9: 离散性 ----------
    print("\n[R-9] 离散性检验 (programme-score 空间)")
    S = np.column_stack([scores[s] for s in SUBS])
    Sz = (S - S.mean(0)) / S.std(0)
    sil = silhouette_score(Sz, y)
    # 随机标签下的 silhouette 零分布
    sil_null = np.empty(500)
    for i in range(500):
        yp = RNG.permutation(y)
        sil_null[i] = silhouette_score(Sz, yp)
    p_sil = (1 + (sil_null >= sil).sum()) / 501
    print(f"  silhouette (观测标签) = {sil:.4f}")
    print(f"  silhouette (随机标签) = {sil_null.mean():.4f} ± {sil_null.std():.4f}")
    print(f"  经验 p = {p_sil:.4f}  → {'结构弱/不显著' if p_sil>0.05 else '存在聚类结构'}")
    pd.DataFrame([{"silhouette_obs": round(sil, 4),
                   "silhouette_null_mean": round(float(sil_null.mean()), 4),
                   "silhouette_null_sd": round(float(sil_null.std()), 4),
                   "emp_p": round(float(p_sil), 4)}]).to_csv(
        f"{AN}/discreteness.csv", index=False)

    # ---------- R-11: 交叉验证 AUC ----------
    print("\n[R-11] 分层 5-fold 交叉验证 AUC (vs 样本内 AUC)")
    print(f"  {'亚型':6s} {'样本内AUC':>10s} {'CV AUC':>9s}")
    cv_rows = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for sub in SUBS:
        ybin = (y == sub).astype(int)
        auc_in = roc_auc_score(ybin, scores[sub])
        oof = np.empty(len(ybin))
        for tr, te in skf.split(Sz, ybin):
            clf = LogisticRegression(max_iter=2000, C=1.0)
            clf.fit(Sz[tr], ybin[tr])
            oof[te] = clf.predict_proba(Sz[te])[:, 1]
        auc_cv = roc_auc_score(ybin, oof)
        print(f"  BrMS{sub:5s} {auc_in:10.3f} {auc_cv:9.3f}")
        cv_rows.append({"subtype": sub, "auc_insample": round(auc_in, 3),
                        "auc_cv": round(auc_cv, 3),
                        "delta": round(auc_cv - auc_in, 3)})
    pd.DataFrame(cv_rows).to_csv(f"{AN}/cv_auc.csv", index=False)

    print(f"\n  ✅ Wrote size_matched_null.csv, discreteness.csv, cv_auc.csv")


if __name__ == "__main__":
    main()
