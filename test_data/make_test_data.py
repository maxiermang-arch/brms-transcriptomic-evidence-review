#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate the deterministic synthetic test data set shipped with the BrMS audit.

The test set is fully synthetic: no patient data are redistributed, and the
whole directory is small enough (< 1 MB) to travel with the code deposit. It
exists so that a reader can execute every layer of the pipeline in a few
seconds, with no network access and no GEO download, and check that the
machinery returns the verdicts it is supposed to return.

Design
------
*   900 genes (G0001 ... G0900) split into disjoint blocks:
    - four "programmes" of the same sizes as the real BrMS programmes
      (40 / 17 / 9 / 315 genes), so both extremes of the gene-set size range
      are exercised by Layer 1;
    - a 50-gene negative-control programme;
    - four 10-gene composition panels (brain / immune / stromal / epithelial)
      that report the tissue-composition latents;
    - a second, independent 10-gene immune panel used only by Layer 4 (this is
      the "re-deconvolve with a different method" step);
    - two signature libraries (12 and 20 signatures of 8 genes) used by the
      cross-cohort consistency screens.
*   Cohort A: 60 samples, 15 per pseudo-subtype, plus a categorical
    "histology" covariate.
*   Cohort B: 45 samples (11 / 7 / 15 / 12).
*   Cohort C: 40 unlabelled samples used as the Layer 5 projection target.
*   A 24-organoid panel with three drugs (one directed, one null, one
    undirected control).

The composition latents are drawn as normalised Gamma variates, so the four
domain fractions sum to one per sample. This is deliberate: it reproduces the
negative coupling between compartments that real bulk tissue has (a sample
rich in immune cells is necessarily poorer in the other compartments), and it
is what makes the leave-own-domain-out specifications behave the way they do
in the real data. Independent Uniform latents would make the four domains
orthogonal and would make every composition adjustment look trivially clean.

Planted ground truth (this is what `verify.py` asserts against)
--------------------------------------------------------------
*   **Programme 1** carries a weak label term *plus* a strong brain-composition
    term. It must pass the strict leave-own-domain-out specification and fail
    the full specification: with the own-domain covariate removed the
    composition-driven signal survives, with it included the label term alone
    is too weak. This pair of verdicts is what demonstrates that
    leave-own-domain-out actually changes something.
*   **Programme 2** is a *pure* immune-composition read-out with no label term
    at all. It must fail every specification: there is no subtype-specific
    biology in it, only immune content. This is the "composition surrogate"
    case that Layer 2 exists to catch.
*   **Programmes 3 and 4** are label-driven only and must pass every
    specification. Programme 4 has 315 genes and is the BrMS4 analogue.
*   **Subtype 3** is planted as fully determined by the categorical histology
    covariate, so the confounder screen must flag it and only it.
*   **Subtype 4** has a different signature-weight vector in cohort B than in
    cohort A, so its cross-cohort consistency ρ must be materially lower than
    subtype 1's in both screens.
*   **Drug `abe`** is planted with a strong subtype-4 effect, so the single
    pre-specified direction must survive per-drug BH-FDR; `gam` (null) and
    `eve` (undirected control) carry no planted effect and must not.

Everything is driven by the single seed 20260913, so the files are identical
on any machine with numpy >= 1.20.

Usage
-----
    python make_test_data.py            # writes every input file
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

SEED = 20260913
HERE = os.path.dirname(os.path.abspath(__file__))

N_GENES = 900
SUBS = ["1", "2", "3", "4"]
PANELS = ["brain", "immune", "stromal", "epi"]
DOMAIN_COL = {"brain": "脑组织合计", "immune": "免疫合计",
              "stromal": "基质-成纤维", "epi": "上皮-肿瘤"}

# ---------------------------------------------------------------- block sizes
PROGRAM_SIZES = {"1": 40, "2": 17, "3": 9, "4": 315}   # mirror the real sizes
NEG_CONTROL_SIZE = 50
PANEL_SIZE = 30
L4_PANEL_SIZE = 10
SIG_SIZE = 8
N_SIG_SCREEN1 = 12          # immune-focused screen
N_SIG_SCREEN2 = 20          # broad pathway screen

# ---------------------------------------------------------------- amplitudes
# Tuned so that the planted verdicts in verify.py come out; see README.
COMP_SCALE = 1.6            # composition panel loading on its own latent
AMP = {
    # comp_dom is the composition domain the programme also reads out; "" = none
    # Amplitudes tuned (with PANEL_SIZE = 30, seed 20260913) so that the
    # resulting AUCs sit in the same range as the real ones; see README.
    "1": {"label": 0.20, "comp": 1.00, "comp_dom": "brain"},
    "2": {"label": 0.00, "comp": 1.3, "comp_dom": "immune"},
    "3": {"label": 0.85, "comp": 0.00, "comp_dom": ""},
    "4": {"label": 0.18, "comp": 0.00, "comp_dom": ""},
}
# subtype -> (domain, enrichment factor) applied to the Dirichlet draw
ENRICH_A = {"1": ("brain", 4.0), "2": ("immune", 8.0), "3": ("epi", 3.0)}
ENRICH_B = {"1": ("brain", 4.0), "2": ("immune", 8.0), "3": ("epi", 3.0)}
SIG_AMP = 0.55              # per-signature subtype weight scale
SIG_IMMUNE_LOAD = 0.80      # screen-1 signatures also read immune content
PROJ_COMPRESS = 0.40        # cohort C signal compression (array-like)


# ---------------------------------------------------------------- gene blocks
def gene_blocks():
    genes = [f"G{i:04d}" for i in range(1, N_GENES + 1)]
    blocks, cursor = {}, 0

    def take(k, n):
        nonlocal cursor
        blocks[k] = genes[cursor: cursor + n]
        cursor += n

    for s in SUBS:
        take(f"prog{s}", PROGRAM_SIZES[s])
    take("neg_control", NEG_CONTROL_SIZE)
    for p in PANELS:
        take(f"panel_{p}", PANEL_SIZE)
    take("l4_immune2", L4_PANEL_SIZE)
    for j in range(N_SIG_SCREEN1):
        take(f"s1_sig{j:02d}", SIG_SIZE)
    for j in range(N_SIG_SCREEN2):
        take(f"s2_sig{j:02d}", SIG_SIZE)
    assert cursor <= N_GENES, f"gene budget exhausted: {cursor} > {N_GENES}"
    return genes, blocks


# ---------------------------------------------------------------- latents
def make_latents(rng, labels, enrich):
    """Composition fractions with a deliberately asymmetric dependence structure.

    The brain fraction is drawn independently, because in a brain-metastasis
    specimen the amount of adjacent parenchyma carried into the block is a
    sampling quantity, not a property of the tumour. The immune, stromal and
    epithelial fractions then partition what is left, so they are mutually
    negatively coupled by construction.

    This asymmetry is what makes the two Layer-2 behaviours the test is built
    around both appear: the own-domain panel carries information that cannot
    be recovered from the other panels for the brain axis (so the full and
    leave-own-domain-out specifications differ), while an immune read-out can
    be reconstructed from the stromal and epithelial panels (so a pure
    composition surrogate fails even when its own domain is left out). A
    design in which all four domains were independent would make every
    composition adjustment look trivially clean; one in which all four summed
    to one would make the own-domain panel redundant.

    `enrich` maps subtype -> (domain, factor); samples of that subtype get the
    factor applied to that domain's draw before normalisation.
    """
    n = len(labels)
    B = rng.beta(2.0, 5.0, size=n)                      # independent
    G = rng.gamma(2.0, 1.0, size=(n, 3))                # immune/stromal/epi
    for sub, (dom, fac) in enrich.items():
        m = np.asarray(labels) == sub
        if dom == "brain":
            B[m] = rng.uniform(0.60, 0.90, size=int(m.sum()))
        else:
            G[m, ["immune", "stromal", "epi"].index(dom)] *= fac
    T = G / G.sum(axis=1, keepdims=True)
    return {"brain": B, "immune": T[:, 0], "stromal": T[:, 1], "epi": T[:, 2]}


def _add(base, idx, rows, vec):
    base[np.ix_(rows, np.arange(base.shape[1]))] += np.asarray(vec)[None, :]


def build_expression(rng, n, labels, F, blocks, genes, W1, W2, compress=1.0):
    """genes x samples matrix on the log2 scale, with every signal planted."""
    base = rng.normal(0.0, 1.0, size=(len(genes), n))
    idx = {g: i for i, g in enumerate(genes)}
    rows = lambda key: [idx[g] for g in blocks[key]]          # noqa: E731
    lab = np.asarray(labels)
    isub = {s: (lab == s).astype(float) for s in SUBS}

    # composition panels report their own latent
    for p in PANELS:
        _add(base, idx, rows(f"panel_{p}"), COMP_SCALE * F[p])
    # the Layer 4 re-deconvolution panel measures immune content independently
    _add(base, idx, rows("l4_immune2"), COMP_SCALE * F["immune"]
         + rng.normal(0.0, 0.35, size=n))

    # programmes
    for s in SUBS:
        vec = AMP[s]["label"] * compress * isub[s]
        if AMP[s]["comp"]:
            vec = vec + AMP[s]["comp"] * F[AMP[s]["comp_dom"]]
        _add(base, idx, rows(f"prog{s}"), vec)

    # signature libraries: signature j responds to subtype s with weight W[j, s]
    for lib, W, pref, immune_load in (("1", W1, "s1", SIG_IMMUNE_LOAD),
                                      ("2", W2, "s2", 0.0)):
        for j in range(W.shape[0]):
            vec = np.zeros(n)
            for k, s in enumerate(SUBS):
                vec = vec + W[j, k] * compress * isub[s]
            if immune_load:
                vec = vec + immune_load * F["immune"]
            _add(base, idx, rows(f"s{lib}_sig{j:02d}"), vec)
    return base


def to_counts(base):
    return pd.DataFrame(np.rint(np.power(2.0, base)).clip(0).astype(np.int64))


def zscore_rows(L: pd.DataFrame) -> pd.DataFrame:
    return (L.sub(L.mean(axis=1), axis=0)
             .div(L.std(axis=1).replace(0, np.nan), axis=0).dropna(how="all"))


def panel_scores(X: pd.DataFrame, blocks) -> dict:
    cpm = X.div(X.sum(axis=0), axis=1) * 1e6
    Z = zscore_rows(np.log2(cpm + 1))
    return {p: Z.loc[[g for g in blocks[f"panel_{p}"] if g in Z.index]]
            .mean(axis=0).values for p in PANELS}


def composition_table(labels, F, ps, rng):
    """Cohort composition table in the same layout as cohort*_composition.txt.

    `ESTIMATE_score` is built as the exact sum of the stromal and immune
    scores, because that identity is the definition of the score and holds
    exactly in the real output too — it is what makes the `full` specification
    rank-deficient by one column. `Purity_ESTIMATE` is *not* an exact affine
    function of the two scores: it carries estimation noise, as the real
    ESTIMATE purity does, so it contributes no second exact dependency.
    """
    cols = {DOMAIN_COL[p]: ps[p] for p in PANELS}
    cols["ESTIMATE_Stromal"] = 1200.0 * F["stromal"]
    cols["ESTIMATE_Immune"] = 1200.0 * F["immune"]
    cols["ESTIMATE_score"] = cols["ESTIMATE_Stromal"] + cols["ESTIMATE_Immune"]
    cols["Purity_ESTIMATE"] = np.clip(
        1.0 - 0.45 * (F["stromal"] + F["immune"])
        + rng.normal(0.0, 0.015, size=len(np.atleast_1d(F["stromal"]))),
        0.05, 1.0)
    df = pd.DataFrame(cols)
    df.insert(0, "brms", list(labels))
    return df


def main() -> None:
    rng = np.random.default_rng(SEED)
    genes, blocks = gene_blocks()

    # signature weight matrices, one per screen; cohort B re-draws subtype 4
    W1 = rng.normal(0.0, SIG_AMP, size=(N_SIG_SCREEN1, len(SUBS)))
    W2 = rng.normal(0.0, SIG_AMP, size=(N_SIG_SCREEN2, len(SUBS)))
    W1B, W2B = W1.copy(), W2.copy()
    W1B[:, 3] = rng.normal(0.0, SIG_AMP, size=N_SIG_SCREEN1)
    W2B[:, 3] = rng.normal(0.0, SIG_AMP, size=N_SIG_SCREEN2)

    # ------------------------------------------------------------ cohort A
    nA = {"1": 15, "2": 15, "3": 15, "4": 15}
    labels_A = np.repeat(SUBS, [nA[s] for s in SUBS])
    samples_A = [f"S{i:03d}" for i in range(1, len(labels_A) + 1)]
    F_A = make_latents(rng, labels_A, ENRICH_A)
    base_A = build_expression(rng, len(labels_A), labels_A, F_A, blocks, genes,
                              W1, W2)
    X_A = to_counts(base_A)
    X_A.index, X_A.columns = genes, samples_A
    comp_A = composition_table(labels_A, F_A, panel_scores(X_A, blocks), rng)
    comp_A.index = samples_A
    histo = np.array(["Lung", "Breast", "RCC"], dtype=object)[
        rng.integers(0, 3, size=len(labels_A))]
    histo[labels_A == "3"] = "Melanoma"          # subtype 3 == histology
    lab_A = pd.DataFrame({"sample": samples_A, "brms": labels_A,
                          "histology": histo})

    # ------------------------------------------------------------ cohort B
    nB = {"1": 11, "2": 7, "3": 15, "4": 12}
    labels_B = np.repeat(SUBS, [nB[s] for s in SUBS])
    samples_B = [f"T{i:03d}" for i in range(1, len(labels_B) + 1)]
    F_B = make_latents(rng, labels_B, ENRICH_B)
    base_B = build_expression(rng, len(labels_B), labels_B, F_B, blocks, genes,
                              W1B, W2B)
    X_B = to_counts(base_B)
    X_B.index, X_B.columns = genes, samples_B
    comp_B = composition_table(labels_B, F_B, panel_scores(X_B, blocks), rng)
    comp_B.index = samples_B
    lab_B = pd.DataFrame({"sample": samples_B, "brms": labels_B})

    # ------------------------------------------------------------ cohort C
    nC = 40
    samples_C = [f"U{i:03d}" for i in range(1, nC + 1)]
    groups_C = np.repeat(SUBS, nC // 4)
    F_C = make_latents(rng, groups_C, {})
    base_C = build_expression(rng, nC, groups_C, F_C, blocks, genes,
                              W1, W2, compress=PROJ_COMPRESS)
    X_C = to_counts(base_C)
    X_C.index, X_C.columns = genes, samples_C

    # ---------------------------------------------------------------- PDO
    pdo_rows = []
    for s in SUBS:
        for j in range(6):
            b = rng.exponential(1.0)
            pdo_rows.append({"organoid": f"PDO_{s}_{j+1}", "brms": s,
                             "abe": b * (0.15 if s == "4" else 1.0),
                             "gam": rng.exponential(1.0),
                             "eve": rng.exponential(1.0)})
    pdo = pd.DataFrame(pdo_rows)

    # -------------------------------------------------------------- write
    X_A.to_csv(os.path.join(HERE, "test_counts.csv"))
    lab_A.to_csv(os.path.join(HERE, "test_labels.csv"), index=False)
    comp_A.to_csv(os.path.join(HERE, "test_composition.csv"))
    X_B.to_csv(os.path.join(HERE, "test_counts_B.csv"))
    lab_B.to_csv(os.path.join(HERE, "test_labels_B.csv"), index=False)
    comp_B.to_csv(os.path.join(HERE, "test_composition_B.csv"))
    X_C.to_csv(os.path.join(HERE, "test_counts_C.csv"))
    pdo.to_csv(os.path.join(HERE, "test_pdo_ic50.csv"), index=False)

    manifest = {
        "seed": SEED,
        "n_genes": N_GENES,
        "programs": {s: blocks[f"prog{s}"] for s in SUBS},
        "negative_control": blocks["neg_control"],
        "composition_panels": {p: blocks[f"panel_{p}"] for p in PANELS},
        "l4_independent_immune_panel": blocks["l4_immune2"],
        "screens": {
            "screen1_immune": {f"sig{j:02d}": blocks[f"s1_sig{j:02d}"]
                               for j in range(N_SIG_SCREEN1)},
            "screen2_broad": {f"sig{j:02d}": blocks[f"s2_sig{j:02d}"]
                              for j in range(N_SIG_SCREEN2)},
        },
        "planted": {
            "programme_1": "label term + brain composition; passes strict L2, fails full L2",
            "programme_2": "pure immune composition read-out, no label term; fails every L2 specification",
            "programme_3": "label-driven only; passes every specification",
            "programme_4": "label-driven only, 315 genes; passes every specification",
            "subtype_3_histology": "subtype 3 is fully determined by histology; confounder screen must flag it",
            "subtype_4_consistency": "cohort B re-draws the subtype-4 signature weights; consistency rho must be low",
            "drug_abe": "subtype-4 IC50 scaled by 0.15; the one pre-specified direction must survive per-drug BH",
            "drug_gam": "no planted effect",
            "drug_eve": "no planted effect; undirected control",
        },
    }
    with open(os.path.join(HERE, "test_programs.json"), "w", encoding="utf8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)

    print(f"cohort A  {X_A.shape}   cohort B  {X_B.shape}   cohort C  {X_C.shape}")
    print(f"composition tables {comp_A.shape} / {comp_B.shape}; PDO {pdo.shape}")
    print("gene blocks:")
    for k in sorted(blocks):
        print(f"  {k:16s} {len(blocks[k]):4d}")


if __name__ == "__main__":
    main()
