"""Size- and dispersion-matched L1 reference and conditional calibration.

Mean-score dispersion is a summary of within-set covariance for standardised
genes. Positive score rescaling cannot alter AUC. Co-expression neighbourhoods
approximate the target dispersion; finite uniform draws neither prove that
matching is impossible nor guarantee a neighbourhood-grid solution. Matching
is expression-only, so rank matrices can be reused under label permutations.
Calibration remains imperfect in cohort B and must be reported as such.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.stats import beta, rankdata
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import AN, SUBS, BRMS_PROGRAMS, SEED  # noqa: E402
import _l2_common as L2  # noqa: E402

N_NULL = 2000          # 匹配零分布的成员数（与 L1 的 N = 2,000 一致）
N_UNIFORM = 20000      # 均匀候选池大小（用于检查均匀抽样覆盖）
N_PROBE = 300          # 每个 M 值下的标定探测集数
N_PERM = 1000          # C1d 的置换标签次数（与 C1 一致）


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def set_scores(Zv: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """(n_sets, k) 的基因索引 -> (n_sets, n_samples) 的模块均值分数（稀疏矩阵乘）。"""
    n_sets, k = idx.shape
    rows = np.repeat(np.arange(n_sets), k)
    A = csr_matrix((np.full(n_sets * k, 1.0 / k), (rows, idx.ravel())),
                   shape=(n_sets, Zv.shape[0]))
    return np.asarray(A @ Zv)


def auc_from_ranks(R: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """由**预先算好的**平均秩矩阵批量算 one-vs-rest AUC（秩和形式）。

    秩只取决于分数向量本身、与标签无关；置换标签时只需换 pos 掩码。这把
    1,000 次置换的代价从 1,000 次 rankdata 降到 1 次。
    """
    n1 = int(pos.sum())
    n0 = R.shape[1] - n1
    return np.round((R[:, pos].sum(axis=1) - n1 * (n1 + 1) / 2.0) / (n1 * n0), 10)


def neighbourhood_sets(Zv: np.ndarray, k: int, pool_M: int, n_sets: int,
                       rng: np.random.Generator) -> np.ndarray:
    """共表达邻域抽样：随机锚基因 -> 取正相关最高的 pool_M 个 -> 随机取 k 个。"""
    n_genes = Zv.shape[0]
    out = np.empty((n_sets, k), dtype=np.int64)
    for i in range(n_sets):
        a = int(rng.integers(n_genes))
        r = Zv @ Zv[a]                       # 基因已按样本标准化 → 点积/n 即相关系数
        order = np.argsort(-r)[:pool_M]      # 正相关最高的 pool_M 个（含锚自己）
        out[i] = rng.choice(order, size=k, replace=False)
    return out


def load_A():
    """与 26_size_matched_null.py 的队列 A 装载逐字一致，保证观测 AUC 可复现。"""
    lab = pd.read_parquet(f"{AN}/labels_used.parquet")
    Xc = pd.read_parquet(f"{AN}/X_counts.parquet")
    X = Xc[lab.index]
    cpm = X.div(X.sum(axis=0), axis=1) * 1e6
    L = np.log2(cpm + 1)
    ZA = L.sub(L.mean(axis=1), axis=0).div(L.std(axis=1).replace(0, np.nan), axis=0)
    ZA = ZA.dropna(how="all")
    return ZA, lab["brms"].astype(str).values


def matched_reference(Zv, k, s_obs, rng_probe, rng_match,
                      n_probe=N_PROBE, n_null=N_NULL):
    """Production L1 reference shared with the synthetic regression tests.

    Matching uses expression only. Matching the mean dispersion is an
    approximation, not a guarantee of calibration or a scale effect on AUC.
    """
    n_genes = Zv.shape[0]
    if not 1 <= k <= n_genes or n_probe < 1 or n_null < 1:
        raise ValueError('invalid gene-set or Monte Carlo size')
    grid = [k]
    while grid[-1] < n_genes:
        grid.append(min(n_genes, max(grid[-1] + 1, int(round(grid[-1] * 1.6)))))
    best_M, best_gap = n_genes, np.inf
    for M in sorted(set(grid)):
        probe = neighbourhood_sets(Zv, k, M, n_probe, rng_probe)
        sd_m = float(set_scores(Zv, probe).std(axis=1, ddof=1).mean())
        if abs(sd_m - s_obs) < best_gap:
            best_M, best_gap = M, abs(sd_m - s_obs)
    idx = neighbourhood_sets(Zv, k, best_M, n_null, rng_match)
    scores = set_scores(Zv, idx)
    return best_M, scores.std(axis=1, ddof=1), rankdata(scores, axis=1)


def l1_pvalue(null_auc, observed_auc):
    return float((1 + int((null_auc >= observed_auc - 1e-10).sum())) / (len(null_auc) + 1))


def l1_pass(p):
    return bool(np.isfinite(p) and p <= 0.05)


def load_B():
    co = L2.load_cohorts()
    ZB = co["B"]["Z"]
    labB = co["B"]["lab"].reindex(ZB.columns)
    keep = labB.isin(SUBS).values
    return ZB.loc[:, keep], labB[keep].values


def main() -> int:
    print("=" * 78)
    print("L1 size-and-dispersion-matched 零分布对照")
    print("=" * 78)

    published = pd.read_csv(f"{AN}/size_matched_null.csv")
    rng_uniform = np.random.default_rng(SEED + 11)
    rng_probe = np.random.default_rng(SEED + 12)
    rng_match = np.random.default_rng(SEED + 14)
    rng_perm = np.random.default_rng(SEED + 13)

    rows, cal_rows, matched_store = [], [], {}

    for tag, Z, y in (("A", *load_A()), ("B", *load_B())):
        genes = Z.index.tolist()
        Zv = Z.to_numpy()
        n_genes = len(genes)
        store = {}
        for sub in SUBS:
            gset = (L2.program_genes(Z, sub) if tag == "B"
                    else [g for g in BRMS_PROGRAMS[sub] if g in Z.index])
            k = len(gset)
            obs = Z.loc[gset].mean(axis=0).values
            auc_obs = float(roc_auc_score((y == sub).astype(int), obs))
            s_obs = float(obs.std(ddof=1))

            # ---- 均匀随机集：证明离散度匹配在均匀抽样下不可行 ----
            idx_u = np.array([rng_uniform.choice(n_genes, size=k, replace=False)
                              for _ in range(N_UNIFORM)])
            s_uni = set_scores(Zv, idx_u).std(axis=1, ddof=1)
            pct = float((s_uni < s_obs).mean() * 100)

            # ---- 邻域池大小 M 的网格标定 ----
            best_M, s_m, R = matched_reference(Zv, k, s_obs, rng_probe, rng_match)
            pos = (y == sub)
            auc_null = auc_from_ranks(R, pos)
            mu_d, sd_d = float(auc_null.mean()), float(auc_null.std())
            p_d = l1_pvalue(auc_null, auc_obs)
            z_d = float((auc_obs - mu_d) / sd_d) if sd_d > 0 else float("nan")

            pub = published[(published["cohort"] == tag) &
                            (published["subtype"].astype(str) == sub)]
            size_p = float(pub["emp_p"].iloc[0]) if len(pub) else float("nan")
            size_mu = float(pub["null_mean"].iloc[0]) if len(pub) else float("nan")
            size_sd = float(pub["null_sd"].iloc[0]) if len(pub) else float("nan")
            size_z = float(pub["z"].iloc[0]) if len(pub) else float("nan")

            rows.append({
                "cohort": tag, "subtype": sub, "n_genes": k, "n_samples": len(y),
                "dispersion_obs": round(s_obs, 4),
                "dispersion_uniform_mean": round(float(s_uni.mean()), 4),
                "dispersion_obs_percentile_uniform": round(pct, 2),
                "implied_mean_corr_obs": round(
                    (s_obs ** 2 * k - 1.0) / (k - 1.0), 4) if k > 1 else np.nan,
                "neighbourhood_M": best_M,
                "neighbourhood_M_over_k": round(best_M / k, 2),
                "dispersion_matched_mean": round(float(s_m.mean()), 4),
                "matched_over_obs": round(float(s_m.mean() / s_obs), 4),
                "auc_obs": round(auc_obs, 3),
                "size_null_mean": round(size_mu, 3), "size_null_sd": round(size_sd, 3),
                "size_p": size_p, "size_z": size_z,
                "disp_null_mean": round(mu_d, 3), "disp_null_sd": round(sd_d, 3),
                "disp_p": round(p_d, 4), "disp_z": round(z_d, 2),
                "n_matched": N_NULL,
            })
            print(f"\n  [{tag}] BrMS{sub}  k={k}  n={len(y)}")
            print(f"      观测 AUC {auc_obs:.3f}   程序离散度 s = {s_obs:.3f} "
                  f"(隐含平均相关 r̄ = {(s_obs**2*k-1)/(k-1):.3f})")
            print(f"      均匀随机集：离散度均值 {s_uni.mean():.4f}，观测位于第 "
                  f"{pct:.1f} 百分位  → 均匀抽样无法匹配")
            print(f"      邻域抽样：M = {best_M} ({best_M/k:.1f}k)，匹配集离散度 "
                  f"{s_m.mean():.4f}（/观测 = {s_m.mean()/s_obs:.3f}）")
            print(f"      size-matched 零分布 : 均值 {size_mu:.3f} SD {size_sd:.3f} "
                  f"p {size_p:.4f} z {size_z:.2f}")
            print(f"      disp-matched 零分布 : 均值 {mu_d:.3f} SD {sd_d:.3f} "
                  f"p {p_d:.4f} z {z_d:.2f}")
            store[sub] = {"R": R, "obs": obs, "k": k, "M": best_M}

        matched_store[tag] = {"y": y, "subs": store}

    pd.DataFrame(rows)[[
        "cohort", "subtype", "n_genes", "n_samples", "dispersion_obs",
        "dispersion_uniform_mean", "dispersion_obs_percentile_uniform",
        "implied_mean_corr_obs", "neighbourhood_M", "neighbourhood_M_over_k",
        "dispersion_matched_mean", "matched_over_obs", "auc_obs", "size_null_mean",
        "size_null_sd", "size_p", "size_z", "disp_null_mean", "disp_null_sd",
        "disp_p", "disp_z", "n_matched",
    ]].to_csv(f"{AN}/dispersion_matched_null.csv", index=False)

    # ------------------------------- C1d：升级后的 L1 在置换标签下的假通过率
    print("\n" + "=" * 78)
    print(f"C1d：置换标签 {N_PERM} 次，零分布改用 dispersion-matched 集")
    print("=" * 78)
    c1 = pd.read_csv(f"{AN}/calibration_controls.csv")
    c1 = c1[(c1["control"] == "C1_permuted_labels") &
            (c1["stat"] == "false_pass_rate_p0.05")]
    for tag, pack in matched_store.items():
        y = pack["y"]
        n_s = len(y)
        for sub in SUBS:
            st = pack["subs"][sub]
            R_obs = rankdata(st["obs"])
            hits = 0
            for _ in range(N_PERM):
                pos = (rng_perm.permutation(y) == sub)
                auc_null = auc_from_ranks(st["R"], pos)
                n1 = int(pos.sum())
                n0 = n_s - n1
                a_obs = float((R_obs[pos].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))
                p = l1_pvalue(auc_null, a_obs)
                if l1_pass(p):
                    hits += 1
            lo, hi = clopper_pearson(hits, N_PERM)
            old = c1[c1["subtype"].astype(str) == sub]
            old_rate = (float(old["value"].iloc[0]) if (tag == "A" and len(old))
                        else float("nan"))
            old_txt = f"{old_rate:.3f}" if tag == "A" else "—"
            print(f"  {tag:4s} BrMS{sub}  size-matched {old_txt:>7s}   "
                  f"disp-matched {hits / N_PERM:6.3f} [{lo:.3f}, {hi:.3f}]")
            cal_rows.append({
                "control": "C1d_dispersion_matched",
                "cohort": tag, "subtype": sub, "n_perm": N_PERM,
                "false_pass_rate_p0.05": round(hits / N_PERM, 4),
                "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                "size_matched_rate": None if tag == "B" else old_rate,
            })

    pd.DataFrame(cal_rows).to_csv(f"{AN}/calibration_dispersion.csv", index=False)
    print("\n  ✅ Wrote dispersion_matched_null.csv, calibration_dispersion.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
