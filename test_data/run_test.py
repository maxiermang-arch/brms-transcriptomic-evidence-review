#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run every layer of the BrMS audit pipeline on the synthetic test data set.

This is the "test data set and results" required by the *Briefings in
Bioinformatics* Problem-solving protocol format. It shares the production L1 reference builder and L2 adjustment module.
Other layers use small reference implementations of the stated decision rules;
this is not a full real-data pipeline rerun.

Layers
------
L1  size- and dispersion-matched gene-set nulls    -> diagnostic z, matched p
L2  composition adjustment, five specifications    -> residual AUC, R2, cond
L3  bootstrap interval + label permutation         -> CI, permutation p
L4  independent immune re-deconvolution            -> reporting only, no verdict
L5  projection onto unlabelled cohort C + drug directions on the PDO panel
A1  confounder screen (categorical histology)      -> in-sample and LOO AUC
A2  cross-cohort consistency, two signature screens-> Spearman rho, p, q

Usage
-----
    python run_test.py                 # writes test_output_*.csv
    python run_test.py --expected      # regenerates expected_output_*.csv
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import importlib.util

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ANALYSIS = os.path.join(ROOT, "analysis")
sys.path.insert(0, ANALYSIS)

SEED = 20260913
SUBS = ["1", "2", "3", "4"]
N_NULL = 500        # L1 size-matched null draws
N_BOOT = 1000       # L3 bootstrap
N_PERM = 1000       # L3 / A2 permutations
L2_THRESHOLD = 0.70
CONFOUND_THRESHOLD = 0.80

try:
    import _l2_common as L2
except ImportError as exc:                                   # pragma: no cover
    raise SystemExit(
        f"cannot import the production Layer 2 module from {ANALYSIS}: {exc}\n"
        "run this script from within the code deposit so that ../analysis "
        "is importable.") from exc


# ------------------------------------------------------------------ helpers
def bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p values."""
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for rank, i in enumerate(order[::-1]):
        k = n - rank
        prev = min(prev, p[i] * n / k)
        q[i] = prev
    return q


def zmat(X: pd.DataFrame) -> pd.DataFrame:
    """counts -> CPM -> log2(x+1) -> row-wise z (same as the real pipeline)."""
    cpm = X.div(X.sum(axis=0), axis=1) * 1e6
    L = np.log2(cpm + 1)
    return (L.sub(L.mean(axis=1), axis=0)
             .div(L.std(axis=1).replace(0, np.nan), axis=0).dropna(how="all"))


def score(Z: pd.DataFrame, genes: list[str]) -> np.ndarray:
    g = [x for x in genes if x in Z.index]
    return Z.loc[g].mean(axis=0).values


def auc(y: np.ndarray, s: np.ndarray) -> float:
    return float(roc_auc_score(y, s))


# ------------------------------------------------------------------ inputs
def load_inputs():
    man = json.load(open(os.path.join(HERE, "test_programs.json"), encoding="utf8"))
    co = {}
    for tag, cf, lf, pf in (("A", "test_counts.csv", "test_labels.csv",
                             "test_composition.csv"),
                            ("B", "test_counts_B.csv", "test_labels_B.csv",
                             "test_composition_B.csv")):
        X = pd.read_csv(os.path.join(HERE, cf), index_col=0)
        lab = pd.read_csv(os.path.join(HERE, lf)).set_index("sample")
        C = pd.read_csv(os.path.join(HERE, pf), index_col=0)
        Z = zmat(X)
        co[tag] = {"Z": Z, "lab": lab["brms"].astype(str),
                   "C": C, "genes": Z.index.tolist()}
        co[tag]["histology"] = lab["histology"] if "histology" in lab else None
    Xc = pd.read_csv(os.path.join(HERE, "test_counts_C.csv"), index_col=0)
    co["C"] = {"Z": zmat(Xc)}
    co["pdo"] = pd.read_csv(os.path.join(HERE, "test_pdo_ic50.csv"))
    return co, man


# ------------------------------------------------------------------ L1
def layer1(co, man, n_null=N_NULL):
    spec = importlib.util.spec_from_file_location('production_l1', os.path.join(ANALYSIS, '44_dispersion_matched_null.py'))
    prod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prod)
    rows = []
    for tag in ("A", "B"):
        d = co[tag]
        Z, y_all, genes = d["Z"], d["lab"].values, d["genes"]
        Zv = Z.to_numpy()          # numpy view: the null loop below is hot
        rng = np.random.default_rng(SEED)
        for name in SUBS + ["neg_control"]:
            gset = man["negative_control"] if name == "neg_control" else man["programs"][name]
            k = len(gset)
            yb = (y_all == ("1" if name == "neg_control" else name)).astype(int)
            obs = auc(yb, score(Z, gset))
            nulls = np.empty(n_null)
            for i in range(n_null):
                idx = rng.choice(len(genes), size=k, replace=False)
                nulls[i] = auc(yb, Zv[idx].mean(axis=0))
            mu, sd = nulls.mean(), nulls.std()
            p = (1 + (nulls >= obs).sum()) / (n_null + 1)
            z = (obs - mu) / sd
            M, dispersions, ranks = prod.matched_reference(
                Zv, k, float(score(Z, gset).std(ddof=1)),
                np.random.default_rng(SEED + 12), np.random.default_rng(SEED + 14),
                n_probe=60, n_null=n_null)
            matched_auc = prod.auc_from_ranks(ranks, yb.astype(bool))
            matched_p = prod.l1_pvalue(matched_auc, obs)
            rows.append({"cohort": tag, "programme": name, "n_genes": k,
                         "auc_obs": round(obs, 4), "null_mean": round(mu, 4),
                         "null_sd": round(sd, 4),
                         "null_p95": round(float(np.percentile(nulls, 95)), 4),
                         "emp_p": round(float(p), 4), "z": round(float(z), 4),
                         "disp_emp_p": matched_p,
                         "matched_mean_dispersion": float(dispersions.mean()),
                         "neighbourhood_M": M,
                         "passes_size_only": bool(p <= 0.05),
                         "passes_L1": prod.l1_pass(matched_p)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ L2
def layer2(co, man):
    rows = []
    for tag in ("A", "B"):
        d = co[tag]
        C, y_all, Z = d["C"], d["lab"].values, d["Z"]
        for s in SUBS:
            sc = score(Z, man["programs"][s])
            yb = (y_all == s).astype(int)
            raw = auc(yb, sc)
            for spec in L2.SPECS:
                cov = L2.usable(L2.adjustment_sets(s)[spec], C)
                diag = L2.fit_diag(sc, C[cov].values.astype(float), yb)
                rank, ncol = L2.rank_info(C[cov].values.astype(float))
                rows.append({"cohort": tag, "subtype": s,
                             "adjustment": spec, "k_covariates": len(cov),
                             "covariates": "|".join(cov),
                             "raw_auc": round(raw, 4),
                             "auc_in_sample": round(diag["auc"], 4),
                             "delta_vs_raw": round(diag["auc"] - raw, 4),
                             "r2": round(diag["r2"], 4),
                             "cond": float(f"{diag['cond']:.6g}"),
                             "numeric_rank": rank, "design_columns": ncol,
                             "passes_070": bool(diag["auc"] >= L2_THRESHOLD)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ L3
def layer3(co, man, n_boot=N_BOOT, n_perm=N_PERM):
    rows = []
    d = co["A"]
    Z, y_all = d["Z"], d["lab"].values
    rng = np.random.default_rng(SEED)
    for name in SUBS + ["neg_control"]:
        gset = man["negative_control"] if name == "neg_control" else man["programs"][name]
        yb = (y_all == ("1" if name == "neg_control" else name)).astype(int)
        sc = score(Z, gset)
        obs = auc(yb, sc)
        boots = np.empty(n_boot)
        n = len(yb)
        for i in range(n_boot):
            pos = np.flatnonzero(yb == 1); neg = np.flatnonzero(yb == 0)
            idx = np.r_[rng.choice(pos,len(pos),replace=True),rng.choice(neg,len(neg),replace=True)]
            boots[i] = auc(yb[idx], sc[idx])
        perms = np.empty(n_perm)
        for i in range(n_perm):
            perms[i] = auc(rng.permutation(yb), sc)
        p_perm = (1 + (perms >= obs).sum()) / (n_perm + 1)
        rows.append({"cohort": "A", "programme": name, "auc_obs": round(obs, 4),
                     "ci_lo": round(float(np.percentile(boots, 2.5)), 4),
                     "ci_hi": round(float(np.percentile(boots, 97.5)), 4),
                     "perm_p": round(float(p_perm), 4),
                     "passes_L3": bool(np.percentile(boots, 2.5) > 0.50)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ L4
def layer4(co, man):
    """Reporting only: re-estimate immune content from an independent panel."""
    rows, ps = [], []
    for tag in ("A", "B"):
        d = co[tag]
        est = score(d["Z"], man["l4_independent_immune_panel"])
        lab = d["lab"].values
        vals = {s: est[lab == s] for s in SUBS}
        for s in SUBS:
            rows.append({"cohort": tag, "subtype": s, "n": int((lab == s).sum()),
                         "mean_estimate": round(float(vals[s].mean()), 4),
                         "sd": round(float(vals[s].std(ddof=1)), 4),
                         "verdict": "reporting only"})
        for i, a in enumerate(SUBS):
            for b in SUBS[i + 1:]:
                u = mannwhitneyu(vals[a], vals[b], alternative="two-sided")
                ps.append({"cohort": tag, "contrast": f"{a} vs {b}",
                           "p": round(float(u.pvalue), 4),
                           "verdict": "reporting only"})
    df = pd.DataFrame(rows)
    con = pd.DataFrame(ps)
    if len(con):
        con["q_bh"] = np.round(bh(con["p"].values), 4)
        con["bonferroni_0.05"] = con["p"] <= 0.05 / len(con)
    return df, con


# ------------------------------------------------------------------ L5
def layer5_projection(co, man):
    ZA, ZC = co["A"]["Z"], co["C"]["Z"]
    lab = co["A"]["lab"].values
    S = np.column_stack([score(ZA, man["programs"][s]) for s in SUBS])
    mu, sd = S.mean(axis=0), S.std(axis=0)
    SA = (S - mu) / sd
    cent = np.array([SA[lab == s].mean(axis=0) for s in SUBS])
    SC = (np.column_stack([score(ZC, man["programs"][s]) for s in SUBS]) - mu) / sd
    dist = ((SC[:, None, :] - cent[None, :, :]) ** 2).sum(axis=2)
    call = np.array([SUBS[i] for i in dist.argmin(axis=1)])
    rows = [{"subtype": s, "n_called": int((call == s).sum()),
             "sd_scores_A": round(float(SA[:, i].std(ddof=1)), 4),
             "sd_scores_C": round(float(SC[:, i].std(ddof=1)), 4)}
            for i, s in enumerate(SUBS)]
    df = pd.DataFrame(rows)
    df["compressed_in_C"] = df.sd_scores_C < df.sd_scores_A
    return df


def layer5_drug(co):
    pdo = co["pdo"]
    lab = pdo["brms"].astype(str).values
    rows = []
    for drug in ("abe", "gam", "eve"):
        v = pdo[drug].values
        for s in SUBS:
            inn, out = v[lab == s], v[lab != s]
            p2 = mannwhitneyu(inn, out, alternative="two-sided").pvalue
            p1 = mannwhitneyu(inn, out, alternative="less").pvalue
            rows.append({"drug": drug, "subtype": s, "n_in": len(inn),
                         "n_out": len(out),
                         "median_in": round(float(np.median(inn)), 4),
                         "median_out": round(float(np.median(out)), 4),
                         "p_two_sided": round(float(p2), 6),
                         "p_one_sided": round(float(p1), 6)})
    df = pd.DataFrame(rows)
    for col, out in (("p_two_sided", "q_per_drug"),
                     ("p_one_sided", "q_directional")):
        df[out] = np.nan
        for drug, g in df.groupby("drug"):
            df.loc[g.index, out] = np.round(bh(g[col].values), 6)
    # no empty strings: an empty cell becomes NaN on the CSV round-trip and
    # the reference comparison in verify.py would then fail on a text column
    df["pre_specified"] = np.where((df.drug == "abe") & (df.subtype == "4"),
                                   "BrMS4 more sensitive", "none")
    df["control"] = np.where(df.drug == "eve", "undirected control", "directed")
    df["passes_L5"] = ((df.p_one_sided <= 0.05) & (df.q_directional <= 0.05)
                       & (df.pre_specified != "none")
                       & (df.control != "undirected control"))
    return df


# ------------------------------------------------------------------ A1
def added1_confounder(co):
    d = co["A"]
    lab = d["lab"].values
    H = pd.get_dummies(d["histology"], prefix="h").astype(float).values
    rows = []
    for s in SUBS:
        y = (lab == s).astype(int)
        if y.sum() < 3:
            continue
        clf = LogisticRegression(max_iter=5000, C=1.0)
        clf.fit(H, y)
        in_sample = auc(y, clf.predict_proba(H)[:, 1])
        loo = np.empty(len(y))
        rng = np.random.default_rng(SEED)
        for i in range(len(y)):
            m = np.ones(len(y), bool)
            m[i] = False
            if len(np.unique(y[m])) < 2:
                loo[i] = np.nan
                continue
            c2 = LogisticRegression(max_iter=5000, C=1.0)
            c2.fit(H[m], y[m])
            loo[i] = c2.predict_proba(H[i:i + 1])[:, 1][0]
        rows.append({"subtype": s, "n": int(y.sum()),
                     "auc_in_sample": round(float(in_sample), 4),
                     "auc_loo": round(float(auc(y, loo)), 4),
                     "flagged": bool(auc(y, loo) >= CONFOUND_THRESHOLD)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ A2
def rank_matrix(S: np.ndarray) -> np.ndarray:
    """Row-wise average ranks of a signatures x samples score matrix."""
    from scipy.stats import rankdata
    return np.apply_along_axis(rankdata, 1, S)


def auc_vec(R: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """One-vs-rest AUC for every row of a rank matrix, from the Mann-Whitney U.

    Identical to `roc_auc_score` (average ranks for ties) but vectorised over
    signatures, which is what makes the permutation loop in `added2_consistency`
    affordable: the ranks depend only on the scores, so they are computed once
    and every permuted labelling is a different column selection.
    """
    n1 = len(pos)
    n0 = R.shape[1] - n1
    a = (R[:, pos].sum(axis=1) - n1 * (n1 + 1) / 2.0) / (n1 * n0)
    # Round before ranking. Two AUCs that differ only by floating-point noise
    # (1e-16) are ranked as distinct values by `rankdata`, which makes the
    # Spearman rho computed from them depend on the BLAS build. Rounding makes
    # genuine ties tie, so the result is stable across machines.
    return np.round(a, 10)


def added2_consistency(co, man, n_perm=N_PERM):
    rows = []
    for screen in ("screen1_immune", "screen2_broad"):
        sigs = man["screens"][screen]
        names = sorted(sigs)
        R = {tag: rank_matrix(np.array([score(co[tag]["Z"], sigs[j])
                                        for j in names]))
             for tag in ("A", "B")}
        lab = {tag: co[tag]["lab"].values for tag in ("A", "B")}
        vecs = {tag: {s: auc_vec(R[tag], np.where(lab[tag] == s)[0])
                      for s in SUBS} for tag in ("A", "B")}
        obs, perm = {}, {}
        rng = np.random.default_rng(SEED)
        nB = len(lab["B"])
        for s in SUBS:
            obs[s] = float(spearmanr(vecs["A"][s], vecs["B"][s]).statistic)
            n1 = int((lab["B"] == s).sum())
            null = np.empty(n_perm)
            for i in range(n_perm):
                pos = rng.permutation(nB)[:n1]
                null[i] = spearmanr(vecs["A"][s], auc_vec(R["B"], pos)).statistic
            perm[s] = (1 + (null >= obs[s]).sum()) / (n_perm + 1)
        p = np.array([perm[s] for s in SUBS])
        q = bh(p)
        for i, s in enumerate(SUBS):
            rows.append({"screen": screen, "subtype": s,
                         "n_signatures": len(names),
                         "rho": round(obs[s], 4), "p": round(float(p[i]), 4),
                         "q": round(float(q[i]), 4),
                         "passes_A2": bool(p[i] <= 0.05 and q[i] <= 0.05)})
    result = pd.DataFrame(rows)
    result['q_primary_m8'] = bh(result.p.to_numpy())
    result['passes_A2'] = (result.p <= 0.05) & (result.q_primary_m8 <= 0.05)
    return result


# ------------------------------------------------------------------ main
def run_all() -> dict[str, pd.DataFrame]:
    co, man = load_inputs()
    l4_main, l4_con = layer4(co, man)
    return {
        "L1": layer1(co, man),
        "L2": layer2(co, man),
        "L3": layer3(co, man),
        "L4": l4_main,
        "L4_contrasts": l4_con,
        "L5_projection": layer5_projection(co, man),
        "L5_drug": layer5_drug(co),
        "A1": added1_confounder(co),
        "A2": added2_consistency(co, man),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--expected", action="store_true",
                    help="regenerate expected_output_*.csv instead of test_output_*.csv")
    args = ap.parse_args()

    out = run_all()
    prefix = "expected" if args.expected else "test"
    for key, df in out.items():
        path = os.path.join(HERE, f"{prefix}_output_{key}.csv")
        df.to_csv(path, index=False)
        print(f"\n=== {key} ===")
        print(df.to_string(index=False))
    print(f"\nwrote {len(out)} files as {prefix}_output_*.csv")


if __name__ == "__main__":
    main()
