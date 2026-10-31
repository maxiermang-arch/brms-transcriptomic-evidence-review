#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Minimal end-to-end test of Layer 1 (size-matched null) on the test data set.

This is the "test data set and results" required by the Briefings in
Bioinformatics protocol format. It runs in a few seconds, needs no network and
no GEO download, and exercises exactly the code path used for Table 2 of the
manuscript:

    counts -> CPM -> log2(x+1) -> row-wise z -> mean programme score
    -> one-vs-rest AUC -> size-matched random gene-set null -> empirical p, z

Acceptance criteria (all four must hold; `verify.py` checks them):
  1. every planted programme (the BrMS1-BrMS4 analogues) is detected, z > 2;
  2. the negative-control programme is NOT detected, z < 2 and p > 0.05;
  3. the four planted AUCs and z values match `expected_output.csv`;
  4. results are identical across repeated runs (fixed seed 20260913).

Usage
-----
    python run_l1_test.py                      # writes test_output.csv
    python run_l1_test.py --expected           # regenerate expected_output.csv
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

SEED = 20260913
HERE = os.path.dirname(os.path.abspath(__file__))
N_NULL = 500
SUBS = ["1", "2", "3", "4"]


def load_inputs():
    X = pd.read_csv(os.path.join(HERE, "test_counts.csv"), index_col=0)
    lab = pd.read_csv(os.path.join(HERE, "test_labels.csv"))
    lab = lab.set_index("sample")["brms"].astype(str)
    with open(os.path.join(HERE, "test_programs.json"), encoding="utf8") as fh:
        programs = json.load(fh)
    return X, lab, programs


def to_z(X: pd.DataFrame) -> pd.DataFrame:
    """counts -> CPM -> log2(x+1) -> row-wise z (identical to analysis/26)."""
    cpm = X.div(X.sum(axis=0), axis=1) * 1e6
    L = np.log2(cpm + 1)
    Z = L.sub(L.mean(axis=1), axis=0).div(L.std(axis=1).replace(0, np.nan), axis=0)
    return Z.dropna(how="all")


def run(n_null: int = N_NULL) -> pd.DataFrame:
    X, lab, programs = load_inputs()
    Z = to_z(X)
    Z = Z[lab.index]
    genes = Z.index.tolist()
    y = lab.values

    rng = np.random.default_rng(SEED)
    rows = []
    for name in SUBS + ["neg_control"]:
        gset = [g for g in programs[name] if g in Z.index]
        k = len(gset)
        obs = Z.loc[gset].mean(axis=0).values

        if name == "neg_control":
            # the control is not tied to any label; test it against subtype 1
            ybin = (y == "1").astype(int)
        else:
            ybin = (y == name).astype(int)

        auc_obs = roc_auc_score(ybin, obs)
        nulls = np.empty(n_null)
        for i in range(n_null):
            idx = rng.choice(len(genes), size=k, replace=False)
            nulls[i] = roc_auc_score(ybin, Z.iloc[idx].mean(axis=0).values)
        mu, sd = nulls.mean(), nulls.std()
        p_emp = (1 + (nulls >= auc_obs).sum()) / (n_null + 1)
        z = (auc_obs - mu) / sd if sd > 0 else np.nan

        rows.append({
            "programme": name,
            "n_genes": k,
            "auc_obs": round(float(auc_obs), 4),
            "null_mean": round(float(mu), 4),
            "null_sd": round(float(sd), 4),
            "null_p95": round(float(np.percentile(nulls, 95)), 4),
            "emp_p": round(float(p_emp), 4),
            "z": round(float(z), 4),
            "passes_L1": bool(z > 2 and p_emp <= 0.05),
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--expected", action="store_true",
                    help="regenerate expected_output.csv instead of test_output.csv")
    ap.add_argument("--n-null", type=int, default=N_NULL)
    args = ap.parse_args()

    df = run(args.n_null)
    out = "expected_output.csv" if args.expected else "test_output.csv"
    df.to_csv(os.path.join(HERE, out), index=False)

    print(df.to_string(index=False))
    print(f"\nwrote {out}")
    planted = df[df.programme.isin(SUBS)]
    ctrl = df[df.programme == "neg_control"].iloc[0]
    ok = bool(planted.passes_L1.all()) and (not bool(ctrl.passes_L1))
    print(f"\nplanted programmes detected: {int(planted.passes_L1.sum())}/4")
    print(f"negative control correctly not detected: {not bool(ctrl.passes_L1)}")
    print(f"LAYER 1 TEST: {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
