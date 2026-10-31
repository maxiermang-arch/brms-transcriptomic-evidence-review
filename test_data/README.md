# Test data set and expected results

This directory is the "test data set and results" required by the *Briefings in
Bioinformatics* Problem-solving protocol format. It is fully synthetic, about 600 KB in total, and runs without network access or a GEO download. Runtime depends on the host.

It covers **all five layers plus both added analyses**, not just layer 1. The
synthetic data carry planted ground truth for each layer, so the test asserts
that the pipeline returns the verdicts it is supposed to return — not merely
that it runs.

## Contents

| File | Role |
|---|---|
| `make_test_data.py` | Regenerates every input from seed `20260913` |
| `test_counts.csv` / `test_labels.csv` / `test_composition.csv` | Cohort A: 900 genes × 60 samples, 15 per subtype, with composition table and a histology covariate |
| `test_counts_B.csv` / `test_labels_B.csv` / `test_composition_B.csv` | Cohort B: 45 samples (11 / 7 / 15 / 12) |
| `test_counts_C.csv` | Cohort C: 40 unlabelled projection targets |
| `test_pdo_ic50.csv` | 24 organoids × 3 drugs |
| `test_programs.json` | Gene blocks, the two signature libraries, and a machine-readable statement of what was planted |
| `run_test.py` | Runs every layer, writes `test_output_*.csv` |
| `expected_output_*.csv` | Reference results under the pinned seed |
| `verify.py` | Determinism, agreement with the reference, and 43 behaviour and rule-regression assertions |
| `run_test.sh` | One-shot entry point: regenerate → run → verify |
| `_legacy_l1_only/` | The earlier layer-1-only test, kept for provenance; superseded |

Gene budget: four programmes of 40 / 17 / 9 / 315 genes (synthetic sizes; not all real-cohort sizes), a 50-gene negative control, four 30-gene composition panels,
an independent 10-gene immune panel for L4, and two signature libraries of 12
and 20 signatures × 8 genes for the consistency screens.

## How to run

```bash
bash run_test.sh          # or: PYTHON=/path/to/python bash run_test.sh
```

Requires Python ≥ 3.9 with `numpy`, `pandas`, `scipy`, `scikit-learn` and
`statsmodels` (the last is imported by the production module `analysis/_l2_common.py`,
which the test reuses rather than re-implements).
Exit code 0 means determinism, agreement with the reference and all behaviour
checks passed. `run_test.py` imports the production module
`analysis/_l2_common.py` rather than re-implementing layer 2, so the test also
guards selected L2 behaviours against regression; it is not proof that every production analysis is correct.

## What is planted

The composition latents are drawn with a deliberately asymmetric dependence
structure: the brain fraction is independent (in a brain-metastasis block the
amount of adjacent parenchyma is a sampling quantity), while the immune,
stromal and epithelial fractions partition the rest and are therefore
negatively coupled. Both extremes of that choice are wrong for the purpose —
four independent domains make every composition adjustment look trivially
clean, and four that sum to one make the own-domain panel redundant.

Ground truth, as recorded in `test_programs.json` under `planted`:

| Item | Planted | Expected verdict |
|---|---|---|
| Programme 1 | weak label term **plus** brain composition | passes strict L2, loses ≥ 0.10 AUC under full L2 — the pair shows leave-own-domain-out is not vacuous |
| Programme 2 | **pure** immune-composition read-out, no label term | fails both pre-registered L2 specifications, ≥ 0.10 AUC below raw — the composition-surrogate case |
| Programmes 3 and 4 | label-driven only | pass every L2 specification |
| Negative control | 50 genes, no signal | fails L1 and L3 |
| Subtype 3 | fully determined by the histology covariate | flagged by the confounder screen, and only it |
| Subtype 4 | signature weights re-drawn in cohort B | consistency ρ materially below subtype 1 in both screens |
| Drug `abe` | subtype-4 IC50 scaled by 0.15 | the one pre-specified direction survives per-drug BH-FDR |
| Drugs `gam`, `eve` | no planted effect (`eve` is the undirected control) | nothing passes |

## What the test asserts

`verify.py` runs three groups of checks:

1. **Determinism** — two consecutive runs of every layer are identical.
2. **Agreement** — every fresh result matches `expected_output_*.csv` to ≤ 1 × 10⁻⁶,
   with one documented exception: the `cond` column of `expected_output_L2.csv`
   is **not** compared. A rank-deficient design is singular up to floating-point
   rounding, so its condition number is decided by the rounding — the identical
   fit reports ≈1 × 10¹⁶ on one LAPACK build and ≈8 × 10¹⁵ on another while the
   design is identically singular. Only the order of magnitude is meaningful, and
   the property it stands for is asserted in group 3 instead: every
   rank-deficient design must have `cond` > 10¹⁰ and every full-rank design must
   have `cond` < 10⁴. Well-conditioned `cond` values *are* reproducible and are
   compared exactly.
3. **Behaviour** — 43 assertions written against the planted ground truth
   above, not against whatever the code happens to emit. A reference file
   alone would still pass if the code were silently broken in a way already
   baked into the reference; these checks detect the documented classes of error, not every possible defect.

## Current decision rules

L1 uses the production dispersion-matched reference builder with 60 probe sets and 500 reference draws for speed; real-data analysis uses 300 probes and 2,000 draws. Pass requires matched empirical p ≤ 0.05. Size-only z and p are diagnostics, not the current decision rule. A correlated-block regression fixture deliberately passes the old size-only rule but fails the matched rule. Positive score scaling leaves AUC unchanged.

L3 uses stratified bootstrap resampling; only a CI lower bound above 0.50 determines its verdict. The permutation p is reported separately. A2 uses BH correction across all eight tests as primary, retaining four-test within-screen results as secondary. Expected L1, L3 and A2 files were regenerated after these rule corrections; other expected files retain their historical values.

The recorded September 2026 verification passed 9 determinism comparisons, 9 expected-output comparisons and 43 behavioural/regression checks, 61 total. L1 and L2 share production code; the other layers use reference implementations. This does not constitute a rerun of the real-data or R analysis chain.

The shipped expected_output_*.csv files are the authoritative numerical test results. Install ../requirements-tested.txt for the Python versions used in the revision. Test outputs and legacy tests are archived in the organised delivery; run_test.sh regenerates current outputs locally.
