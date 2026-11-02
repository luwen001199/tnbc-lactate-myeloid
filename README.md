# tnbc-lactate-myeloid

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23888671.svg)](https://doi.org/10.5281/zenodo.23888671)

Analysis code for **"Lactate-driven trained immunity defines prognostically distinct
myeloid states in triple-negative breast cancer: a multi-cohort transcriptomic and
single-cell study"**.

The pipeline builds two directional myeloid gene-programme scores — a pro-tumour
lactate-memory score (**LT-pro**) and an anti-tumour trained score (**LT-anti**) —
residualises each of them on an estimate of myeloid abundance, partitions tumours by a
deterministic quadrant rule, and validates the partition in three independent public
cohorts. No new experimental data were generated.

---

## Repository layout

```
├── README.md               this file
├── LICENSE                 MIT
├── CITATION.cff            metadata harvested by Zenodo when the release is archived
├── requirements.txt        pinned analysis environment
├── scripts/
│   ├── 00_genepanels.py            curated gene pools + rationale for every gene
│   ├── 01b_build_cohort.py         TCGA-BRCA cohort (expression + PanCan Atlas endpoints)
│   ├── 02_signature_scoring.py     signature scores, myeloid abundance, residualisation
│   ├── 03_subtypes.py              deterministic quadrant subtyping (primary)
│   ├── 03_consensus_subtypes.py    consensus-clustering comparison (reported as unstable)
│   ├── 04_metabric.py              METABRIC replication via cBioPortal
│   ├── 06_singlecell.py            GSE176078 myeloid comparison
│   ├── 06b_sc_traj_lr.py           monocyte→macrophage pseudotime + myeloid→CD8 wiring
│   ├── 08_epigenetics.py           promoter architecture / methylation latch layer
│   ├── 09_cox_multivariable.py     multivariable Cox models
│   ├── 12_sensitivity.py           PH test, landmark analysis, RMST
│   ├── 13_mcpcounter_sensitivity.py official-MCP-counter abundance sensitivity analysis
│   ├── 10_make_tables.py           assembles every manuscript table from results/
│   ├── 11_md2docx.py               Markdown → .docx
│   └── fig_bulk.py / fig_sc.py / fig_epi.py   figure rendering
├── panels/
│   ├── gene_panels.{json,tsv}      curated panels, one row per gene with rationale
│   └── mcpcounter_official_markers.json  official MCP-counter signatures (Becht 2016)
└── results/                        every numeric result quoted in the manuscript (CSV)
```

The intermediate binary caches (`*.pkl`, `*.h5ad`) are **not** included: they are
rebuilt by the scripts and would add ~200 MB to the archive.

---

## Environment

Python 3.12 (developed on 3.12.3). Install with:

```bash
python3 -m pip install -r requirements.txt
```

Everything except the single-cell step runs in a few minutes on a laptop. The
single-cell step needs ~16 GB RAM because GSE176078 contains 100,064 cells.

### Retargeting the paths

Every script resolves its input/output directories from a small constant block at the
top of the file, all rooted at `/workspace`. Point them at your own project root with:

```bash
sed -i 's#/workspace#/path/to/your/project#g' scripts/*.py
```

---

## Data

All inputs are public and de-identified. The bulk steps fetch their own data through
public REST APIs (UCSC Xena, GDC, cBioPortal); the single-cell and chromatin steps
expect files downloaded once into `data/`.

| Resource | Access | Used for |
|---|---|---|
| TCGA-BRCA RNA-seq (log2 RSEM+1, HiSeqV2) | UCSC Xena | discovery, n = 1,097 |
| TCGA Pan-Cancer Atlas clinical resource | cBioPortal REST | OS/DSS/DFS/PFS, PAM50, hypoxia scores |
| METABRIC (Illumina HT-12 v3) | cBioPortal REST | validation, n = 1,980 |
| GSE176078 (Wu *et al.* 2021) | GEO | single cell, 100,064 cells / 26 tumours |
| hg19/hg38 CpG islands, DNase clusters | UCSC | promoter annotation |
| ENCODE E029 (monocyte) / E124 (monocyte) H3K27ac, H3K4me3 | ENCODE | promoter chromatin state |
| TCGA-BRCA 450k methylation | GDC | promoter-methylation layer |

No controlled-access data were used. The TCGA-ATAC BAMs on GDC are dbGaP-controlled and
were deliberately *not* used; the chromatin layer is therefore built from ENCODE
monocyte marks plus TCGA 450k methylation, and the manuscript states this limitation.

---

## Reproducing the analysis

Run in this order:

```bash
python3 scripts/00_genepanels.py           # panels -> panels/gene_panels.{tsv,json}
python3 scripts/01b_build_cohort.py        # TCGA cohort + TNBC definition
python3 scripts/02_signature_scoring.py    # raw scores, abundance, residualisation
python3 scripts/03_subtypes.py             # quadrant subtyping (primary)
python3 scripts/03_consensus_subtypes.py   # clustering comparison (PAC diagnostic)
python3 scripts/04_metabric.py             # external validation cohort
python3 scripts/09_cox_multivariable.py    # multivariable Cox models
python3 scripts/12_sensitivity.py          # PH test, landmark, RMST
python3 scripts/13_mcpcounter_sensitivity.py  # official-MCP-counter sensitivity
python3 scripts/06_singlecell.py           # single-cell comparison
python3 scripts/06b_sc_traj_lr.py          # pseudotime + ligand-receptor wiring
python3 scripts/08_epigenetics.py          # chromatin / methylation layer
python3 scripts/10_make_tables.py          # manuscript tables
python3 scripts/fig_bulk.py scripts/fig_sc.py scripts/fig_epi.py
```

`03_subtypes.py` is the pipeline that produced the reported results.
`03_consensus_subtypes.py` is retained because its instability (PAC 0.43–0.61) is the
stated reason for preferring the deterministic rule.

---

## Determinism

Every stochastic component is seeded: `GaussianMixture(random_state=0, n_init=10)` for
the ESR1/PGR/ERBB2 cutoffs, `KMeans(random_state=…)` from a `np.random.default_rng(42)`
stream in the consensus-clustering comparison, and `sc.tl.score_genes(..., random_state=0)`
in the single-cell scoring. Re-running the pipeline from a clean `results/` directory
reproduces every number in `results/` exactly.

---

## Note on the myeloid-abundance estimator

The primary analysis estimates cell-type abundance from *curated* lineage-marker panels
(ten populations, scored as the mean log-expression of each panel). This is **not**
MCP-counter. Because abundance estimation is itself an analytic choice, the whole
analysis is repeated with the official MCP-counter signatures and estimator in
`scripts/13_mcpcounter_sensitivity.py`, and the comparison is reported as Section 3.9 and
Table S6 of the manuscript. The two marker sets are largely disjoint (Jaccard 0.04–0.43
per population), so this is a genuine test of robustness rather than a restatement.

---

## Licence and citation

MIT; see `LICENSE`. If you reuse this code, please cite the article and the archived
release:

> Lu W (2026). *tnbc-lactate-myeloid*: analysis pipeline for lactate-associated
> myeloid trained states in triple-negative breast cancer (Version 1.0.0) [Computer
> software]. Zenodo. https://doi.org/10.5281/zenodo.23888671
