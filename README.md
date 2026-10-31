# Transcriptomic subtype evidence — reviewer materials

Public code and source-data snapshot for **Calibration and sensitivity of transcriptomic evidence for molecular subtypes in brain metastases**. This is a peer-review candidate, not an accepted publication. The fixed review version is **v0.1.0-review**.

## Start here

- **Numerical source data:** [`analysis/bmc_results/`](analysis/bmc_results/), including the full result registry and replicate-level data.
- **Six manuscript figures:** [`figures_revised/`](figures_revised/), as PDF, SVG and PNG.
- **Inputs and original sources:** [`data/README.md`](data/README.md), with frozen Parquet matrices in `analysis/`.
- **Reconstruction commands:** below and [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).
- **Attribution and upstream terms:** [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
- **Frozen snapshot and exclusions:** [`review_snapshot.json`](review_snapshot.json).

The manuscript and Supplementary Information are supplied through the journal submission. Their author, affiliation and declaration fields are maintained there; the hosting account is not asserted to be a manuscript author or scientific corresponding author.

## Quick verification

Use Python 3.12 with the locked dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-bmc-locked.txt
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
python tools/verify_review_snapshot.py
python tools/check_bmc_behavior.py
```

The first command verifies hashes, cohort alignment, frozen proxy definitions, all 16 probability AUCs and balanced accuracies, and retained simulation counts/conjunctions. It does not rerun every scientific analysis. The second checks revision-critical tail and concordance behavior. Rscript and the R `curl` package are additionally needed to rerun official MCPcounter scoring. Software versions used by the recorded run are retained in the results.

## Results map

| Evidence | Main files |
|---|---|
| 24 proxy-cohort analyses | `proxy_analysis_verified.csv`, `strict_adjustment_covariates.csv` |
| Cross-cohort concordance | `concordance.csv`, `concordance_aggregation.csv` |
| Frozen transfer | `transfer_integrated.csv`, `transfer_confusion.csv` |
| Official immune scoring | `mcp_official_A.csv`, `mcp_official_B.csv`, `mcp_coverage.csv`, `mcp_official_contrasts.csv` |
| 6,000 original module repeats | `benchmark_replicates.csv`, `benchmark_summary.csv` |
| 2,000 same-mechanism seed repeats | `same_mechanism_seed_replicates.csv`, `same_mechanism_seed_summary.csv` |
| 2,000 distinct normal-score checks | `standalone_label_permutation_replicates.csv`, `standalone_label_permutation_summary.csv` |
| 2,000 pure-noise selections | `selection_chain_replicates.csv` |
| 1,800 paired-cohort simulations | `joint_replicates.csv`, `joint_summary.csv` |

Paths in the table are relative to `analysis/bmc_results/`. Other retained supporting numerical results are in `analysis/` and `analysis/enhancement_results/`. Historical outputs in `analysis/legacy_results/` are provenance and must not be mixed with the canonical revision results. Historical script entry points are not all claimed to be portable from this curated snapshot.

## Scientific scope

The primary labelled cohorts are A: 52 (GSE245467) and B: 45 (GSE184869). Cohort C (E-MTAB-8659) is an unlabelled projection target. The reconstruction starts from retained public-source processed inputs; it does not reprocess sequencing reads or reconstruct the original atlas discovery/classifier. The 1e-10 tail tolerance preserves ties in rounded reference AUCs. Both-screen positive concordance is primary; any-screen aggregation is a sensitivity analysis.

Competitive enrichment, label independence and selection optimism have distinct targets. The original module-null batch, same-mechanism new-seed batch and standalone normal-score check remain separate. Transfer intervals condition on fitted models and omit training uncertainty. Expression-derived labels are not independent biological truth. Joint-rule fractions are conditional on the declared generators and are not biological validity verdicts.

Public viewing requires no login. This repository retains the hosting account identity and is not an identity-blinded review service. Third-party terms are documented separately; no blanket project-code licence or archival DOI is fabricated.
