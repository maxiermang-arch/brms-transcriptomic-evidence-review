# Reconstruction from retained processed inputs

Run from the repository root with the locked Python environment, OPENBLAS_NUM_THREADS=1 and OMP_NUM_THREADS=1. Preserve the published tag or work in a separate clone: reconstruction commands overwrite numerical outputs. Rscript and its `curl` package are needed for the official MCPcounter step.

```bash
python tools/verify_review_snapshot.py
python tools/check_bmc_behavior.py
python analysis/47_proxy_upgrade.py
python analysis/bmc_catalogue.py
python analysis/bmc_empirical.py
python analysis/bmc_mcp.py
python analysis/49_joint_simulation.py --reps 200
python analysis/bmc_benchmark.py --reps 2000
python analysis/bmc_precision_checks.py
python analysis/bmc_seed_replication.py
python analysis/bmc_technical_specification.py
python analysis/bmc_finalize_results.py
python analysis/bmc_figures.py
```

`47_proxy_upgrade.py` must finish all 24 combinations before empirical verification/finalization. Full Monte Carlo reconstruction is computationally expensive. All individual seeds and recorded budgets are supplied in the code, settings and source data. The small quick-verification commands are not substitutes for rerunning this sequence.

Canonical outputs are in `analysis/bmc_results/`. The figure script writes to `../submission/figures` relative to the repository directory and creates the directory. Frozen figure exports are already available inside this repository in `figures_revised/`.

The 50-pair inner-budget sensitivity covers one representative signal setting, not all nulls or cohort conditions. The paired-cohort generator and historical test fixtures have disclosed simplifications. Retained inputs include existing composition/ESTIMATE quantities; original upstream ESTIMATE processing and source-atlas discovery are not reconstructed by these commands. Some historical producers outside this entry list require original workbooks or inputs obtainable from the sources in `data/README.md`; their complete historical import is outside this review snapshot's reconstruction claim.

See `review_snapshot.json` for the exact inclusion/exclusion ledger and `SHA256SUMS.txt` for this review package's hashes. Inputs and scientific outputs copied from the verified BMC revision are byte-identical; only review documentation and packaging are added.
