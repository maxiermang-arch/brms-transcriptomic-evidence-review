#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""41 · L2 sensitivity: does leave-own-domain-out really exclude the own domain?

Raised by the pre-submission review: `16_final_purity_verdict.py` builds the
adjustment set by excluding the *panel* named after each subtype's own domain,
but the five ESTIMATE-derived columns (StromalScore, ImmuneScore, ESTIMATEScore,
derived purity) are never excluded. For BrMS2 (immune domain) the model
therefore still contains an independent immune-content measurement, so
leave-own-domain-out is not fully implemented for that subtype.

This script compares five adjustment sets, per subtype and cohort:

  full              all four composition panels + all four ESTIMATE columns
  leaveown_current  own panel removed, all ESTIMATE columns retained  (=现行)
  leaveown_strict   own panel AND the same-domain ESTIMATE columns removed
  purity_only       Purity_ESTIMATE only
  no_estimate       own panel removed, no ESTIMATE column at all

Reported for each: in-sample residual AUC, 5-fold cross-fitted residual AUC
(residualisation fitted on the training folds only, so the test-fold AUC is
not contaminated by having seen those samples), and a nested bootstrap 95% CI
in which the OLS is refitted inside every resample.

Output: analysis/l2_sensitivity.csv
        analysis/l2_sensitivity_summary.txt
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# 2026-09-15：装载 / 打分 / 校正改为引用 _l2_common（与 16_final_purity_verdict.py
# 共用同一套实现），本脚本只负责"五种规格并列报告"这一层。
from _l2_common import (load_cohorts, adjustment_sets, usable, program_genes,  # noqa: E402
                        fit_diag, rank_info, crossfit_auc, boot_ci,
                        SPECS, PANELS, ESTIMATE, OWN_PANEL, OWN_ESTIMATE)
from _common import AN, SUBS, NM  # noqa: E402


def max_abs_corr(cov: np.ndarray) -> float:
    """Largest absolute pairwise correlation among the covariates.

    Reported alongside the condition number: the condition number says the
    design is ill-conditioned, this says *which* pair is responsible. Columns
    with zero variance give NaN and are ignored.
    """
    if cov.shape[1] < 2:
        return float("nan")
    sd = cov.std(axis=0)
    keep = sd > 0
    if keep.sum() < 2:
        return float("nan")
    c = np.corrcoef(cov[:, keep], rowvar=False)
    off = ~np.eye(c.shape[0], dtype=bool)
    return float(np.nanmax(np.abs(c[off])))

N_BOOT = 2000


def main() -> None:
    # ---- data -----------------------------------------------------------
    D = load_cohorts()

    rows, log = [], []
    for co in ("A", "B"):
        C, lab, Zc = D[co]["C"], D[co]["lab"], D[co]["Z"]
        y_all = lab.values
        if D[co]["dropped"]:
            log.append(f"  NOTE cohort {co}: dropped {len(D[co]['dropped'])} sample(s) "
                       f"with no usable counts: {', '.join(D[co]['dropped'])}")
        log.append("=" * 100)
        log.append(f"Cohort {co}   n = {len(y_all)}   purity columns present: "
                   f"{[c for c in ESTIMATE if c in C.columns]}")
        log.append("=" * 100)

        for s in SUBS:
            g = program_genes(Zc, s)
            prog = Zc.loc[g].mean(axis=0).values
            y = (y_all == s).astype(int)
            if y.sum() < 3:
                continue
            own_p, own_e = OWN_PANEL[s], OWN_ESTIMATE[s]

            sets = adjustment_sets(s)

            raw = float(roc_auc_score(y, prog))
            log.append(f"\n  BrMS{s} ({NM[s]})   own panel = {own_p or 'none'}   "
                       f"own ESTIMATE = {own_e or 'none'}")
            log.append(f"    program genes used = {len(g)}    raw AUC = {raw:.3f}")
            log.append(f"    {'adjustment set':<18}{'k':>4}{'in-sample':>11}{'cross-fit':>11}"
                       f"{'boot 95% CI':>22}{'delta vs raw':>14}")

            for name in SPECS:
                cols = usable(sets[name], C)
                if len(cols) == 0:
                    continue
                cov = C[cols].values.astype(float)
                diag = fit_diag(prog, cov, y)
                a_in = diag["auc"]
                a_cf = crossfit_auc(prog, cov, y)
                lo, hi = boot_ci(prog, cov, y, n=N_BOOT)
                rank, ncol = rank_info(cov)
                mcorr = max_abs_corr(cov)
                log.append(f"    {name:<18}{len(cols):>4}{a_in:>11.3f}{a_cf:>11.3f}"
                           f"   [{lo:.3f}, {hi:.3f}]{a_in - raw:>+13.3f}"
                           f"   rank {rank}/{ncol}  cond {diag['cond']:.3g}"
                           f"  max|r| {mcorr:.3f}")
                rows.append({
                    "cohort": co, "subtype": f"BrMS{s}", "n_genes": len(g),
                    "adjustment": name, "k_covariates": len(cols),
                    "covariates": "|".join(cols),
                    # --- design-matrix disclosure ---
                    "design_columns": ncol, "numeric_rank": rank,
                    "rank_deficiency": ncol - rank,
                    "cond": float(f"{diag['cond']:.6g}"),
                    "max_abs_pairwise_r": round(mcorr, 4),
                    "r2_covariates_on_programme": round(diag["r2"], 3),
                    # --- AUCs ---
                    "raw_auc": round(raw, 3),
                    "auc_in_sample": round(a_in, 3),
                    "auc_crossfit": round(a_cf, 3),
                    "ci_lo": round(lo, 3), "ci_hi": round(hi, 3),
                    "delta_vs_raw": round(a_in - raw, 3),
                    "passes_070": bool(a_in >= 0.70),
                    "passes_070_crossfit": bool(a_cf >= 0.70),
                })

    out = pd.DataFrame(rows)
    out.to_csv(f"{AN}/l2_sensitivity.csv", index=False)
    txt = "\n".join(log)
    open(f"{AN}/l2_sensitivity_summary.txt", "w", encoding="utf8").write(txt)
    print(txt)
    print(f"\n  -> l2_sensitivity.csv ({len(out)} rows) + l2_sensitivity_summary.txt")


if __name__ == "__main__":
    main()
