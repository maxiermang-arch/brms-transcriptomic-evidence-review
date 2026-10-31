# Third-party attribution and terms

This file records attribution and upstream terms; it does not apply one licence to the entire repository.

## MCPcounter

`analysis/genesets/MCPcounter.R` and `analysis/genesets/mcp_genes.txt` are retained upstream snapshots from https://github.com/ebecht/MCPcounter. Authors: Etienne Becht, Aurelien de Reynies and Florent Petitprez. The upstream DESCRIPTION identifies version 1.2.0 and GPL-3. The source R implementation is unmodified. The exact source URLs, retrieval date and hashes are in `analysis/genesets/mcp_provenance.json`. The upstream DESCRIPTION and a complete GPL v3 text are in `third_party/MCPcounter/`.

Reference: Becht E, Giraldo NA, Lacroix L, et al. Genome Biology. 2016;17:218. https://doi.org/10.1186/s13059-016-1070-5

## Hallmark / MSigDB

The retained Hallmark membership lists in `analysis/genesets/hallmark.json` derive from the Molecular Signatures Database, Broad Institute, Massachusetts Institute of Technology and Regents of the University of California. The upstream page states CC BY 4.0 for MSigDB v6.0-v7.5.1 and v2022.1 onward, with additional terms for some collections. This snapshot contains Hallmark gene sets, not the restricted KEGG subcollections. The original upstream release identifier was not recorded in the delivered material; reproducibility therefore uses the exact frozen membership and file hashes, without inventing a release number.

Terms: https://www.gsea-msigdb.org/gsea/msigdb_license_terms.jsp
Resource: https://www.gsea-msigdb.org/gsea/msigdb/
Reference: Liberzon A, Birger C, Thorvaldsdottir H, et al. Cell Systems. 2015;1:417-425. https://doi.org/10.1016/j.cels.2015.12.004

## Gene Ontology membership

`analysis/genesets/neural.json` and the GO-derived immune entries in `analysis/genesets/immune.json` retain Gene Ontology names, term identifiers and membership from the delivered analysis inputs. Gene Ontology Consortium data and data products are offered under CC BY 4.0, without warranty. The original upstream release was not recorded; full frozen gene membership and hashes are supplied instead. Lists have been serialized and subset for this analysis; this repository is not an official GO release.

Resource and terms: https://geneontology.org/docs/go-citation-policy/
License: https://creativecommons.org/licenses/by/4.0/
References: https://doi.org/10.1038/75556 and https://doi.org/10.1093/genetics/iyad031

## Public processed study data and subtype labels

Source accessions, source-atlas citation and correction, and omitted publisher materials are described in `data/README.md`. Original datasets, signatures and source materials remain attributed to their original creators and subject to their applicable terms. Processing, reanalysis and simulations do not transfer ownership of third-party materials to the hosting account.

## Project code

The rights holder has not yet selected a blanket licence for newly authored project code. Public access here supplies materials for scientific inspection; it is not a claim that all included material has one open-source licence. Existing upstream licences remain applicable to their respective components. Hosting and commit attribution do not determine manuscript authorship.
