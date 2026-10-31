#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Verify a fresh run of `run_test.py` against the shipped reference results.

Three kinds of check, in order:

1. **Determinism** — two consecutive runs of every layer are identical.
2. **Agreement** — a fresh run matches the corresponding `expected_output_*.csv`
   to within 1e-6 on every numeric column and exactly on every other column,
   except the columns named in `MACHINE_DEPENDENT`, which are not reproducible
   across BLAS/LAPACK builds even for identical input.
3. **Behaviour** — the pipeline returns the verdicts the synthetic data were
   built to produce. This is the check that matters: the reference numbers
   alone would still pass if the code were silently broken in a way that was
   already baked into the reference. Every assertion below is written against
   the ground truth documented in `make_test_data.py`, not against whatever the
   code happens to output.

Exit code 0 = all checks passed.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_test as R                                              # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TOL = 1e-6
KEYS = ["L1", "L2", "L3", "L4", "L4_contrasts", "L5_projection",
        "L5_drug", "A1", "A2"]
# Columns that are not reproducible across machines *by construction* and are
# therefore excluded from the reference comparison.  The condition number of a
# rank-deficient design is singular up to floating-point rounding, so its value
# is decided by the rounding: the same fit reports 1e16 on one LAPACK build and
# 8e15 on another while the design is identically singular.  Only the order of
# magnitude and the exact numeric rank carry information, and the magnitude is
# asserted separately in `behaviour()` below.
MACHINE_DEPENDENT = ("cond",)

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"      {'OK  ' if ok else 'FAIL'}  {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def frames_equal(a: pd.DataFrame, b: pd.DataFrame) -> tuple[bool, str]:
    if list(a.columns) != list(b.columns) or len(a) != len(b):
        return False, "shape or column mismatch"
    # CSV round-trips change dtypes (a string subtype column comes back as
    # int64), so compare with the reference cast to the fresh frame's dtypes
    # rather than letting each side select its own dtypes.
    bb = b.copy()
    for c in a.columns:
        try:
            # `astype(object)` on an int64 column yields Python ints, not the
            # strings the fresh frame holds, so cast text columns explicitly.
            bb[c] = b[c].astype(str) if a[c].dtype == object else b[c].astype(a[c].dtype)
        except (TypeError, ValueError) as exc:
            return False, f"dtype cast failed on '{c}': {exc}"
    # Machine-dependent columns are excluded from the numeric comparison (see
    # MACHINE_DEPENDENT); their names still have to match, which the shape
    # check above already guarantees.
    drop = [c for c in MACHINE_DEPENDENT if c in a.columns]
    aa = a.drop(columns=drop)
    bbb = bb.drop(columns=drop)
    num = aa.select_dtypes("number")
    if not num.empty:
        d = (num - bbb.select_dtypes("number")).abs().to_numpy().max()
        if d > TOL:
            return False, f"max numeric difference {d:.3e}"
    other = aa.select_dtypes(exclude="number")
    if not other.empty and not other.equals(bbb.select_dtypes(exclude="number")):
        return False, "non-numeric columns differ"
    return True, ""


def sel(df: pd.DataFrame, **kw) -> pd.DataFrame:
    out = df.copy()
    for k, v in kw.items():
        out = out[out[k].astype(str) == str(v)]
    return out


# ---------------------------------------------------------------- behaviour
def behaviour(out: dict[str, pd.DataFrame]) -> None:
    L1, L2, L3 = out["L1"], out["L2"], out["L3"]
    L4, L4c = out["L4"], out["L4_contrasts"]
    L5p, L5d, A1, A2 = (out["L5_projection"], out["L5_drug"],
                        out["A1"], out["A2"])

    print("   L1 — planted programmes detected, control not")
    for co in ("A", "B"):
        g = sel(L1, cohort=co)
        npass = int(sel(g, passes_L1=True).programme.isin(R.SUBS).sum())
        ctrl = sel(g, programme="neg_control").iloc[0]
        check(f"L1/{co}: 4/4 planted programmes pass",
              npass == 4 and bool(sel(g, programme="1").passes_L1.iloc[0])
              and bool(sel(g, programme="2").passes_L1.iloc[0])
              and bool(sel(g, programme="3").passes_L1.iloc[0])
              and bool(sel(g, programme="4").passes_L1.iloc[0]))
        check(f"L1/{co}: negative control not flagged",
              not bool(ctrl.passes_L1), f"z = {ctrl.z:.3f}, p = {ctrl.emp_p:.3f}")
    big = sel(sel(L1, cohort="A"), programme="4").null_mean.iloc[0]
    small = sel(sel(L1, cohort="A"), programme="3").null_mean.iloc[0]
    check("L1: 315-gene null is elevated over the 9-gene null (size bias)",
          big - small > 0.15, f"{small:.3f} -> {big:.3f}")
    check('L1 verdict uses dispersion p only',
          bool((L1.passes_L1 == (L1.disp_emp_p <= 0.05)).all()))
    import importlib.util
    sp = importlib.util.spec_from_file_location('l1_checked', os.path.join(R.ANALYSIS, '44_dispersion_matched_null.py'))
    prod = importlib.util.module_from_spec(sp); sp.loader.exec_module(prod)
    rng = np.random.default_rng(431)
    latent = rng.normal(size=(10, 60))
    Z = np.repeat(latent, 20, axis=0) + rng.normal(scale=0.01, size=(200, 60))
    Z = (Z-Z.mean(axis=1, keepdims=True))/Z.std(axis=1, keepdims=True, ddof=1)
    pos = latent[0] > 0
    score = Z[:20].mean(axis=0)
    observed = R.auc(pos.astype(int), score)
    idx = np.array([rng.choice(200,20,replace=False) for _ in range(500)])
    from scipy.stats import rankdata
    uniform = prod.auc_from_ranks(rankdata(prod.set_scores(Z,idx),axis=1),pos)
    M, sd, ranks = prod.matched_reference(Z,20,score.std(ddof=1),np.random.default_rng(21),np.random.default_rng(22),n_probe=60,n_null=500)
    matched = prod.auc_from_ranks(ranks,pos)
    p_size=prod.l1_pvalue(uniform,observed); p_match=prod.l1_pvalue(matched,observed)
    check('L1 correlated-block fixture: size-only passes, matched reference does not',
          p_size<=0.05 and p_match>0.05,f'size p={p_size:.4f}; matched p={p_match:.4f}')
    check('L1 positive scaling preserves rank AUC',
          abs(R.auc(pos.astype(int),score)-R.auc(pos.astype(int),score*19+3))<1e-12)
    check('L1 threshold includes p=0.05 without an added z gate',
          prod.l1_pass(0.05) and not prod.l1_pass(0.050001) and not prod.l1_pass(float('nan')))
    check('L1 empirical floor is 1/(N+1)',
          prod.l1_pvalue(np.array([0.1,0.2,0.3]),0.9)==0.25)

    print("   L2 — composition adjustment separates composition from signal")
    for co in ("A", "B"):
        g = sel(L2, cohort=co)
        p1f = sel(g, subtype="1", adjustment="full").auc_in_sample.iloc[0]
        p1s = sel(g, subtype="1", adjustment="leaveown_strict").auc_in_sample.iloc[0]
        check(f"L2/{co}: programme 1 — leave-own-domain-out is not vacuous",
              p1s - p1f >= 0.10, f"full {p1f:.3f} -> strict {p1s:.3f}")
        for s in ("3", "4"):
            f = sel(g, subtype=s, adjustment="full").auc_in_sample.iloc[0]
            st = sel(g, subtype=s, adjustment="leaveown_strict").auc_in_sample.iloc[0]
            check(f"L2/{co}: programme {s} (label-driven) holds under both specs",
                  f >= R.L2_THRESHOLD and st >= R.L2_THRESHOLD,
                  f"full {f:.3f}, strict {st:.3f}")
        raw2 = sel(g, subtype="2", adjustment="full").raw_auc.iloc[0]
        f2 = sel(g, subtype="2", adjustment="full").auc_in_sample.iloc[0]
        s2 = sel(g, subtype="2", adjustment="leaveown_strict").auc_in_sample.iloc[0]
        check(f"L2/{co}: programme 2 (pure composition) fails both pre-registered specs",
              f2 < R.L2_THRESHOLD and s2 < R.L2_THRESHOLD,
              f"raw {raw2:.3f} -> full {f2:.3f}, strict {s2:.3f}")
        check(f"L2/{co}: programme 2 loses >= 0.10 AUC to adjustment",
              raw2 - s2 >= 0.10, f"delta {raw2 - s2:.3f}")
    full_rank = sel(sel(L2, cohort="A"), adjustment="full")
    check("L2: full specification is rank-deficient by exactly one column "
          "(ESTIMATE_score = Stromal + Immune)",
          bool(((full_rank.design_columns - full_rank.numeric_rank) == 1).all()))
    # The condition number itself is not reproducible between LAPACK builds for
    # a singular design, so assert the property it stands for rather than the
    # digits: rank-deficient designs must be numerically singular, and
    # full-rank designs must be comfortably well conditioned.
    deficient = L2[(L2.design_columns - L2.numeric_rank) > 0]
    fullrank = L2[(L2.design_columns - L2.numeric_rank) == 0]
    check("L2: rank-deficient designs are numerically singular (cond > 1e10) "
          "and full-rank designs are well conditioned (cond < 1e4)",
          bool((deficient.cond > 1e10).all()) and bool((fullrank.cond < 1e4).all()),
          f"{len(deficient)} deficient, min cond {deficient.cond.min():.2e}; "
          f"{len(fullrank)} full rank, max cond {fullrank.cond.max():.2f}")

    print("   L3 — bootstrap lower bound above 0.50, permutation p <= 0.05")
    for s in R.SUBS:
        r = sel(L3, programme=s).iloc[0]
        check(f"L3: programme {s} passes", bool(r.passes_L3),
              f"CI [{r.ci_lo:.3f}, {r.ci_hi:.3f}], perm p = {r.perm_p:.3f}")
    check("L3: negative control does not pass",
          not bool(sel(L3, programme="neg_control").iloc[0].passes_L3))

    print("   L4 — reporting only, no verdict")
    for co in ("A", "B"):
        g = sel(L4, cohort=co)
        top = g.loc[g.mean_estimate.idxmax(), "subtype"]
        check(f"L4/{co}: six pairwise contrasts reported, none carries a verdict",
              len(sel(L4c, cohort=co)) == 6
              and bool((sel(L4c, cohort=co).verdict == "reporting only").all()))
        check(f"L4/{co}: highest immune estimate is the immune-enriched subtype 2",
              top == "2", f"argmax = {top}")

    print("   L5 — projection and drug directions")
    check("L5: every projected group is non-empty and all 40 samples are called",
          bool((L5p.n_called > 0).all()) and int(L5p.n_called.sum()) == 40,
          " / ".join(f"{r.subtype}:{r.n_called}" for r in L5p.itertuples()))
    check("L5: score dispersion is compressed in the projection cohort",
          bool(L5p.sd_scores_C.mean() < L5p.sd_scores_A.mean()),
          f"{L5p.sd_scores_A.mean():.3f} -> {L5p.sd_scores_C.mean():.3f}")
    passing = L5d[L5d.passes_L5]
    check("L5: exactly one pre-specified direction survives per-drug BH-FDR",
          len(passing) == 1 and passing.iloc[0].drug == "abe"
          and passing.iloc[0].subtype == "4",
          f"{len(passing)} row(s): " + ", ".join(
              f"{r.drug}/{r.subtype} q={r.q_directional}" for r in passing.itertuples()))
    check("L5: the null drug and the undirected control give nothing",
          not bool(sel(L5d, drug="gam").passes_L5.any())
          and not bool(sel(L5d, drug="eve").passes_L5.any()))

    print("   A1 — confounder screen")
    flagged = set(sel(A1, flagged=True).subtype.astype(str))
    check("A1: only the histology-determined subtype 3 is flagged",
          flagged == {"3"}, f"flagged = {sorted(flagged)}")
    check("A1: subtype 3 histology-alone LOO AUC >= 0.80",
          float(sel(A1, subtype="3").auc_loo.iloc[0]) >= R.CONFOUND_THRESHOLD,
          f"LOO AUC = {sel(A1, subtype='3').auc_loo.iloc[0]:.3f}")

    print("   A2 — cross-cohort consistency")
    for screen in ("screen1_immune", "screen2_broad"):
        g = sel(A2, screen=screen)
        r1 = float(sel(g, subtype="1").rho.iloc[0])
        r4 = float(sel(g, subtype="4").rho.iloc[0])
        check(f"A2/{screen}: planted-inconsistent subtype 4 ranks below subtype 1",
              r4 < r1 - 0.30, f"rho 1 = {r1:.3f}, rho 4 = {r4:.3f}")
        check(f"A2/{screen}: the three planted-consistent subtypes pass",
              bool(sel(g, subtype="1").passes_A2.iloc[0])
              and bool(sel(g, subtype="2").passes_A2.iloc[0])
              and bool(sel(g, subtype="3").passes_A2.iloc[0]))
        check(f"A2/{screen}: subtype 4 does not pass",
              not bool(sel(g, subtype="4").passes_A2.iloc[0]))


def main() -> int:
    print("[1/3] determinism: two consecutive runs of every layer")
    a, b = R.run_all(), R.run_all()
    ok = True
    for k in KEYS:
        same, why = frames_equal(a[k], b[k])
        check(f"{k} identical across runs", same, why)
        ok = ok and same
    if not ok:
        return 1

    print("\n[2/3] agreement with the shipped expected_output_*.csv")
    for k in KEYS:
        path = os.path.join(HERE, f"expected_output_{k}.csv")
        if not os.path.exists(path):
            check(f"{k} reference present", False, f"missing {path}")
            continue
        same, why = frames_equal(a[k], pd.read_csv(path))
        check(f"{k} matches reference", same, why)

    print("\n[3/3] behaviour against the planted ground truth")
    behaviour(a)

    print()
    if FAILS:
        print(f"{len(FAILS)} CHECK(S) FAILED:")
        for f in FAILS:
            print(f"  - {f}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
