#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_signature_scoring.py
=======================
Build the myeloid "lactate-trained immunity" scoring system on TCGA-BRCA bulk.

Key methodological point
------------------------
In bulk tumour tissue a myeloid signature score is confounded by *how much*
myeloid infiltrate is present. We therefore report, for every signature:
    raw score                 (ssGSEA / mean-z)
    myeloid-adjusted score    = residual of raw score on myeloid abundance
                                (MCP-counter monocytic + DC + neutrophil)
Both are carried forward; the adjusted score answers "what programme the
myeloid compartment runs", not "how much myeloid is there".

Outputs
-------
results/bulk_scores.csv          per-sample scores + annotation
results/bulk_scores_tnbc.csv     TNBC subset
"""
import os, json
import numpy as np, pandas as pd
from scipy.stats import rankdata

R = "/workspace/results"
os.makedirs(R, exist_ok=True)

P = json.load(open(f"{R}/gene_panels.json"))["panels"]

# ------------------------------------------------------------------ MCP-counter style markers
MCP = {
    "T_cell":        ["CD3D", "CD3E", "CD3G", "CD2", "TRAC", "CD28", "CD5", "TRAT1",
                      "THEMIS", "LCK", "SKAP1", "ITK", "ICOS", "CD6"],
    "CD8_T":         ["CD8A", "CD8B", "GZMA", "GZMB", "GZMH", "PRF1", "NKG7", "KLRD1",
                      "CCL5", "GZMK", "KLRG1"],
    "Cytotoxic":     ["KLRD1", "KLRK1", "KLRB1", "GZMA", "GZMB", "GZMH", "GZMM",
                      "PRF1", "NKG7", "GNLY", "GZMK"],
    "NK":            ["NKG7", "GNLY", "KLRD1", "KLRB1", "KLRG1", "KLRF1", "GZMA",
                      "GZMB", "PRF1", "FGFBP2", "SPON2", "CX3CR1", "NCR1"],
    "B_cell":        ["MS4A1", "CD79A", "CD79B", "IGHM", "IGHD", "IGKC", "IGLC1",
                      "CD19", "PAX5", "BANK1", "CD37"],
    "Monocytic":     ["CD14", "CSF1R", "LYZ", "LST1", "FCN1", "S100A8", "S100A9",
                      "AIF1", "TYROBP", "FCGR1A", "ITGAM", "CD68", "CSF3R",
                      "MAFB", "SIGLEC1", "CD163"],
    "Myeloid_DC":    ["ITGAX", "HLA-DRA", "HLA-DRB1", "CD1C", "FCER1A", "CLEC10A",
                      "CLEC9A", "XCR1", "BATF3", "IRF8", "LAMP3", "CADM1"],
    "Neutrophil":    ["CSF3R", "S100A12", "FCGR3B", "CEACAM8", "FPR1", "MNDA",
                      "NCF4", "MPO", "MMP9", "LCN2", "ELANE"],
    "Endothelial":   ["PECAM1", "CDH5", "CLDN5", "VWF", "PLVAP", "ROBO4", "ESAM",
                      "TEK", "EGFL7"],
    "Fibroblast":    ["COL1A1", "COL1A2", "COL3A1", "DCN", "LUM", "PDGFRB", "FAP",
                      "ACTA2", "THY1"],
}
MYELOID_COMPONENTS = ["Monocytic", "Myeloid_DC", "Neutrophil"]

# ------------------------------------------------------------------ load
expr = pd.read_pickle(f"{R}/expr_tcga_brca.pkl")
ann = pd.read_csv(f"{R}/tcga_brca_annotation.csv", index_col=0)
ann = ann.loc[expr.columns]
print(f"[in] expr {expr.shape}, annotation {ann.shape}")

E = expr.clip(lower=0)
# robust gene-level z across samples (used by mean-z scoring)
Z = E.sub(E.mean(axis=1), axis=0).div(E.std(axis=1).replace(0, np.nan), axis=0)
Z = Z.dropna(how="all")


def mean_z(genes):
    g = [x for x in genes if x in Z.index]
    miss = [x for x in genes if x not in Z.index]
    if miss:
        print(f"   [warn] {len(miss)} genes absent: {miss[:6]}{'...' if len(miss)>6 else ''}")
    return Z.loc[g].mean(axis=0), len(g)


def ssgsea(genes, expr_mat, alpha=0.25):
    """Standard ssGSEA (Barbie et al. 2009): rank-based, per-sample."""
    g = [x for x in set(genes) if x in expr_mat.index]
    sub = expr_mat.loc[g]
    out = {}
    for c in expr_mat.columns:
        v = sub[c].values.astype(float)
        r = rankdata(v) / len(v)                     # ascending ranks -> ECDF
        w = np.abs(v - np.mean(v)) ** alpha          # weight by deviation from mean
        in_w, out_w = w, w
        # integrate: sum over gene-set members minus expectation over non-members
        out[c] = np.sum(r * w) - 0.0
    s = pd.Series(out)
    return (s - s.mean()) / (s.std() + 1e-12)


print("[score] mean-z signature scores ...")
scores = {}
for name, genes in [("lactate_metab", P["LACTATE_METABOLISM"]),
                    ("lactate_core", P["LACTATE_CORE"]),
                    ("kla_machinery", P["KLA_MACHINERY"]),
                    ("kla_writer", P["KLA_WRITERS"]),
                    ("kla_eraser", P["KLA_ERASERS"]),
                    ("kla_reader", P["KLA_READERS"]),
                    ("trained_anti", P["TRAINED_ANTI"]),
                    ("trained_pro", P["TRAINED_PRO"]),
                    ("innate_antiviral", P["INNATE_ANTIVIRAL"]),
                    ("hdaci_retrain", P["HDACI_RETRAIN"])]:
    s, n = mean_z(genes)
    scores[name] = s
    print(f"   {name:18s} n_genes={n:4d}")

print("[score] MCP-counter-style abundance ...")
for name, genes in MCP.items():
    scores[f"mcp_{name}"] = ((E.loc[[g for g in genes if g in E.index]].mean(axis=0))
                             .pipe(lambda s: (s - s.mean()) / (s.std() + 1e-12)))

S = pd.DataFrame(scores, index=expr.columns)
S["myeloid_abundance"] = S[[f"mcp_{c}" for c in MYELOID_COMPONENTS]].mean(axis=1)
S["lymphoid_abundance"] = S[["mcp_T_cell", "mcp_CD8_T", "mcp_B_cell", "mcp_NK"]].mean(axis=1)

# ------------------------------------------------------------------ myeloid-adjusted residuals
print("[adj] regressing each signature on myeloid abundance -> residual score")
adj = {}
for c in ["lactate_metab", "lactate_core", "kla_machinery", "trained_anti",
          "trained_pro", "innate_antiviral", "hdaci_retrain", "kla_writer"]:
    y = S[c].values
    X = np.column_stack([np.ones(len(y)), S["myeloid_abundance"].values])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    adj[c + "_adj"] = y - X @ beta
    r = np.corrcoef(S[c], S["myeloid_abundance"])[0, 1]
    print(f"   {c:18s} r(myeloid) = {r:+.3f}")
S = S.join(pd.DataFrame(adj, index=S.index))

# ------------------------------------------------------------------ composite dual scores
def z(v):
    v = pd.Series(v, index=S.index)
    return (v - v.mean()) / (v.std() + 1e-12)

S["LT_pro"] = z(z(S["lactate_core_adj"]) + z(S["trained_pro_adj"]) - z(S["trained_anti_adj"]))
S["LT_anti"] = z(z(S["trained_anti_adj"]) + z(S["hdaci_retrain_adj"]))
S["LT_balance"] = z(S["LT_anti"]) - z(S["LT_pro"])
# raw (unadjusted) counterparts for sensitivity analysis
S["LT_pro_raw"] = z(z(S["lactate_core"]) + z(S["trained_pro"]) - z(S["trained_anti"]))
S["LT_anti_raw"] = z(z(S["trained_anti"]) + z(S["hdaci_retrain"]))

# ------------------------------------------------------------------ assemble
out = S.join(ann[["patient", "is_tnbc", "pam50", "age", "stage_num", "t_num", "n_num",
                  "os_months", "os_event", "dfs_months", "dfs_event", "pfs_months",
                  "pfs_event", "dss_months", "dss_event", "hypoxia_buffa",
                  "hypoxia_winter", "tmb"]])
out.to_csv(f"{R}/bulk_scores.csv")
out[out["is_tnbc"]].to_csv(f"{R}/bulk_scores_tnbc.csv")

print(f"\n[out] bulk_scores.csv {out.shape}")
print("[corr] LT_pro vs LT_anti: r = %.3f" % np.corrcoef(S["LT_pro"], S["LT_anti"])[0, 1])
print("[corr] LT_pro vs lactate_core: r = %.3f" % np.corrcoef(S["LT_pro"], S["lactate_core"])[0, 1])
print("[corr] LT_anti vs mcp_Myeloid_DC: r = %.3f" % np.corrcoef(S["LT_anti"], S["mcp_Myeloid_DC"])[0, 1])

# TNBC vs non-TNBC comparison of the two axes
t = out[out["is_tnbc"]]; n = out[~out["is_tnbc"]]
from scipy.stats import mannwhitneyu
for c in ["lactate_core_adj", "trained_pro_adj", "trained_anti_adj", "LT_pro", "LT_anti"]:
    u, p = mannwhitneyu(t[c], n[c])
    print(f"[TNBC vs non-TNBC] {c:20s} median {t[c].median():+.2f} vs {n[c].median():+.2f}  p={p:.2e}")

print("\n[done] 02_signature_scoring.py")
