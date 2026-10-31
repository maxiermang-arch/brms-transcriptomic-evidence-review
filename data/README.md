# Retained inputs and original data sources

The canonical reconstruction starts from the frozen Parquet matrices and label tables in `analysis/` and the two fixed composition tables in `data/purity/`. Exact file hashes are in `SHA256SUMS.txt` and `review_snapshot.json`. The expression matrices are public-source processed derivatives, not new patient recruitment or raw-read reprocessing.

- Cohort A: GSE245467, https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE245467
- Cohort B: GSE184869, https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE184869
- Cohort C (projection only; no subtype-labelled independent validation): E-MTAB-8659, https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-8659
- Source atlas: Yang Z, Wei S, Duan H, et al. A proteogenomic atlas of 1032 brain metastases identifies molecular subtypes, immune landscapes, and therapeutic vulnerabilities. Nature Communications. 2026;17:2038. https://doi.org/10.1038/s41467-026-68748-y
- Author Correction: https://doi.org/10.1038/s41467-026-71972-1

Whole publisher PDFs, complete supplementary workbooks, original download copies, duplicate expression exports, incomplete downloads and legacy pickle copies are not redistributed in this review snapshot. Original supplementary data can be obtained from the source article. Omitted source file names and hashes are recorded in `review_snapshot.json` for provenance.

The original article reports CC BY-NC-ND 4.0, https://creativecommons.org/licenses/by-nc-nd/4.0/. Original source materials retain their own terms; no blanket relicensing is made here. The retained labels, numerical derivatives and expression transformations must be interpreted with the Methods and source-data provenance. The complete historical import from original workbooks is outside the review reconstruction entry list.
