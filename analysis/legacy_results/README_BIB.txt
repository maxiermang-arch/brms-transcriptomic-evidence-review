Local repository candidate for the brain-metastasis subtype audit
Prepared 2026-10-01

STATUS
This is a local candidate, not a public deposit or a DOI-bearing release.
The author must confirm authorship, the software licence, third-party data
redistribution permissions, and repository access before publication.
No new licence is assigned by this packaging operation.

CONTENTS AND PROVENANCE
Current analysis/, data/, and test_data/ are overlaid on retained historical
analysis inputs. This supplies historical derived inputs previously available
only in the local archive. Existing data files are not scientifically modified.
analysis/enhancement_results contains the September 2026 extension outputs.
figures/ retains historical generated figures; journal submission figures are
the separate Figure1–6 and FigureS1–S6 PDFs in the submission package.
Old manuscripts, review correspondence, unrelated backups and document-build
scripts are excluded. The repository candidate does not depend on a sibling
Chinese-named archive directory for its retained analysis inputs.

ENVIRONMENT
Use requirements-tested.txt and analysis/enhancement_results/python_environment.txt
for the recorded Python environment. R used estimate, MASS and survival; see
the retained R_sessionInfo.txt. R/Python installation itself is not bundled.
Do not run analysis scripts in the sole archival copy: extract a new workspace.
Some historical scripts contain historical paths and are retained for provenance;
this packaging step is not a claim that every historical script is portable.

SOFTWARE CHECK
From a fresh extracted root, with required Python dependencies installed:
  PYTHON=python bash test_data/run_test.sh
This checks synthetic software behaviour, not scientific validity.

EXTENSION COMMANDS
From the extracted root, set OPENBLAS_NUM_THREADS=1 and OMP_NUM_THREADS=1.
Run in this order if reproducing the extension (computationally intensive):
  python analysis/47_proxy_upgrade.py
  python analysis/48_transfer_histology.py
  python analysis/49_joint_simulation.py --reps 200
  Rscript analysis/13_estimate_abis.R
  Rscript analysis/17_abis_tpm.R
  Rscript analysis/50_survival_diagnostics.R
  python analysis/52_summarize_extension.py
  python tools/check_extension_release.py
Input verification script 51_verify_inputs.py additionally requires an independent
preserved-root copy; never point it at the working directory being modified.
All nine joint-simulation scenarios must finish, each with 200 paired replicates.

SCOPE
Selected real-data reconstructions, both R stages and synthetic checks were run
in September 2026. This package has been checked for archive integrity and content
coverage, not subjected to a new complete empirical or cross-platform rerun.
Stored cohort-B processing is reused. This is not a raw-read workflow.
Model-generated simulations are not additional human observations.
Before deposit, verify that any required external gene-set resources can be
obtained under their source licences; public availability is not blanket
permission to redistribute.
