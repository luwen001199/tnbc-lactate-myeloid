#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
13_mcpcounter_sensitivity.py
============================
Sensitivity analysis: replace the curated "MCP-counter-style" marker panel with
the *official* MCP-counter signatures (Becht et al. 2016, Genome Biol 17:218;
Signatures/genes.txt, 111 HUGO symbols, 10 populations) and the official
estimator (appendSignatures: arithmetic mean of the marker genes' expression,
no rank transform, no median).

Because the official estimator and the one used in the main analysis are both
linear (marker-gene mean followed by an affine z-scaling), residualisation on
either abundance estimate is invariant to the affine step; the only substantive
difference between the two pipelines is therefore the marker gene set itself.

Outputs
-------
results/mcpcounter_official_scores_tcga.csv
results/mcpcounter_official_scores_metabric.csv
results/mcpcounter_sensitivity_cox.csv      -> Supplementary Table S6
results/mcpcounter_marker_overlap.csv       -> marker-set comparison
"""
import os, json, time
import numpy as np, pandas as pd
import urllib.request
from scipy.stats import spearmanr
from lifelines import CoxPHFitter

R = "/workspace/results"
API = "https://www.cbioportal.org/api"
STUDY, PROF, SID = "brca_metabric", "brca_metabric_mrna", "brca_metabric_mrna"

OFFICIAL = json.load(open(f"{R}/mcpcounter_official_markers.json"))
# official population -> short key
KEY = {"T cells": "T_cell", "CD8 T cells": "CD8_T", "Cytotoxic lymphocytes": "Cytotoxic",
       "NK cells": "NK", "B lineage": "B_cell", "Monocytic lineage": "Monocytic",
       "Myeloid dendritic cells": "Myeloid_DC", "Neutrophils": "Neutrophil",
       "Endothelial cells": "Endothelial", "Fibroblasts": "Fibroblast"}
OFF = {KEY[k]: v for k, v in OFFICIAL.items()}
MYELOID = ["Monocytic", "Myeloid_DC", "Neutrophil"]


def z(s):
    s = pd.Series(s).astype(float)
    sd = s.std(ddof=0)
    return (s - s.mean()) / (sd if sd and sd > 0 else 1.0)


def official_scores(E):
    """Official MCP-counter estimator: mean expression of a population's markers."""
    out = {}
    for pop, genes in OFF.items():
        g = [x for x in genes if x in E.index]
        if not g:
            print(f"   [warn] no markers for {pop}")
            continue
        out[f"mcpoff_{pop}"] = E.loc[g].mean(axis=0)      # official: plain mean
        out[f"mcpoff_n_{pop}"] = np.nan
    S = pd.DataFrame(out)
    for pop in OFF:                                        # z-scaled per population
        if f"mcpoff_{pop}" in S:
            S[f"mcpoff_{pop}"] = z(S[f"mcpoff_{pop}"])
    S["myeloid_abundance_official"] = S[[f"mcpoff_{p}" for p in MYELOID
                                         if f"mcpoff_{p}" in S]].mean(axis=1)
    S["lymphoid_abundance_official"] = S[[f"mcpoff_{p}" for p in
                                          ["T_cell", "CD8_T", "B_cell", "NK"]
                                          if f"mcpoff_{p}" in S]].mean(axis=1)
    return S.drop(columns=[c for c in S.columns if c.startswith("mcpoff_n_")])


# ================================================================== TCGA
print("=" * 70)
print("TCGA-BRCA")
tc = pd.read_pickle(f"{R}/expr_tcga_brca.pkl")
S_t = official_scores(tc)
S_t.to_csv(f"{R}/mcpcounter_official_scores_tcga.csv")
print("  official populations:", [c for c in S_t.columns if c.startswith("mcpoff_")])

cur = pd.read_csv(f"{R}/bulk_scores.csv", index_col=0)
j = cur[["myeloid_abundance", "LT_pro", "LT_anti", "trained_anti", "trained_pro"]].join(
    S_t[["myeloid_abundance_official"]], how="inner")
rho = spearmanr(j.myeloid_abundance, j.myeloid_abundance_official).correlation
print(f"  Spearman(curated vs official myeloid abundance) = {rho:.3f}  (n={len(j)})")

# ================================================================== METABRIC
print("=" * 70)
print("METABRIC -- extend the expression matrix with official markers if needed")
me = pd.read_pickle(f"{R}/metabric_expr.pkl")
need = sorted(set(sum(OFF.values(), [])) - set(me.index))
print(f"  official markers already present: {len(set(sum(OFF.values(),[])) & set(me.index))}/111")
print(f"  to download: {len(need)}")

if need:
    gmap = json.load(open("/workspace/data/cbio_gene_map.json"))
    ent = {g: gmap[g] for g in need if g in gmap}
    print(f"  mapped to Entrez: {len(ent)}/{len(need)}")

    def api_post(path, payload):
        req = urllib.request.Request(f"{API}/{path}", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        for _ in range(4):
            try:
                with urllib.request.urlopen(req, timeout=240) as r:
                    return json.load(r)
            except Exception as e:
                print("   retry", e); time.sleep(4)
        raise RuntimeError(path)

    rows = {}
    ids = [int(v) for v in ent.values()]
    for i in range(0, len(ids), 100):
        d = api_post(f"molecular-profiles/{PROF}/molecular-data/fetch",
                     {"sampleListId": SID, "entrezGeneIds": ids[i:i + 100]})
        sym = {v: k for k, v in ent.items()}
        for r in d:
            g = sym.get(r.get("entrezGeneId"))
            if g:
                rows.setdefault(g, {})[r["sampleId"]] = r.get("value")
        print(f"   fetched {len(d)} records ({min(i+100,len(ids))}/{len(ids)})", flush=True)
    if rows:
        add = pd.DataFrame(rows).T
        add.index.name = me.index.name
        me = pd.concat([me, add])
        me = me[~me.index.duplicated(keep="first")]
        me.to_pickle(f"{R}/metabric_expr.pkl")
        print(f"  expression matrix now {me.shape}")

S_m = official_scores(me.astype(float))
S_m.to_csv(f"{R}/mcpcounter_official_scores_metabric.csv")
for p in MYELOID + ["T_cell", "CD8_T"]:
    n = len([g for g in OFF[p] if g in me.index])
    print(f"  METABRIC coverage {p:12s} {n}/{len(OFF[p])}")

# ================================================================== re-scoring
print("=" * 70)
print("Re-residualise the trained-immunity axes on the OFFICIAL abundance")
ms = pd.read_csv(f"{R}/metabric_scores.csv", index_col=0)
P = json.load(open(f"{R}/gene_panels.json"))["panels"]

# recompute raw panel scores for METABRIC from the (extended) expression matrix
E = me.astype(float).dropna(how="all", axis=1).dropna(how="all", axis=0).clip(-5, 15)
Z = E.sub(E.mean(axis=1), axis=0).div(E.std(axis=1).replace(0, np.nan), axis=0).dropna(how="all")


def mz(gs):
    g = [x for x in gs if x in Z.index]
    return Z.loc[g].mean(axis=0)


raw = pd.DataFrame(index=Z.columns)
raw["trained_anti"] = mz(P["TRAINED_ANTI"])
raw["trained_pro"] = mz(P["TRAINED_PRO"])
raw["lactate_core"] = mz(P["LACTATE_CORE"])
raw["hdaci_retrain"] = mz(P["HDACI_RETRAIN"])
raw["LT_pro_raw"] = raw["lactate_core"] + raw["trained_pro"] - raw["trained_anti"]
raw["LT_anti_raw"] = raw["trained_anti"] + raw["hdaci_retrain"] - raw["trained_pro"]


def resid(y, x):
    """Residual of y on x (both z-scaled); invariant to affine transforms of x."""
    d = pd.DataFrame({"y": z(y), "x": z(x)}).dropna()
    b = np.polyfit(d.x, d.y, 1)
    return z(pd.Series(d.y - np.polyval(b, d.x), index=d.index))


off_ab = S_m["myeloid_abundance_official"].reindex(raw.index)
adj = pd.DataFrame(index=raw.index)
for c in ["trained_anti", "trained_pro", "lactate_core", "hdaci_retrain",
          "LT_pro_raw", "LT_anti_raw"]:
    adj[c + "_off"] = resid(raw[c], off_ab)
adj["LT_pro_off"] = adj["LT_pro_raw_off"]
adj["LT_anti_off"] = adj["LT_anti_raw_off"]
adj["trained_anti_off"] = adj["trained_anti_off"]
adj.to_csv(f"{R}/metabric_scores_official_mcp.csv")

# compare with the curated-abundance version
cmp = ms[["LT_pro", "LT_anti", "trained_anti_adj"]].join(
    adj[["LT_pro_off", "LT_anti_off", "trained_anti_off"]], how="inner")
rho_p = spearmanr(cmp.LT_pro, cmp.LT_pro_off).correlation
rho_a = spearmanr(cmp.LT_anti, cmp.LT_anti_off).correlation
print(f"  Spearman(LT_pro curated vs official)  = {rho_p:.3f}  (n={len(cmp)})")
print(f"  Spearman(LT_anti curated vs official) = {rho_a:.3f}")

# subtype agreement under the same quadrant rule
def quadrant(d, pro, anti):
    return np.where((d[pro] > 0) & (d[anti] <= 0), "Pro",
           np.where((d[pro] <= 0) & (d[anti] > 0), "Anti", "Intermediate"))

sub_cur = quadrant(cmp, "LT_pro", "LT_anti")
sub_off = quadrant(cmp, "LT_pro_off", "LT_anti_off")
agree = (sub_cur == sub_off).mean()
print(f"  subtype agreement (curated vs official quadrant rule) = {agree*100:.1f}%")
print(pd.crosstab(sub_cur, sub_off))

# ================================================================== Cox
print("=" * 70)
print("Cox sensitivity: curated vs official abundance (METABRIC)")
clin = ms[["os_months", "os_event", "rfs_months", "rfs_event", "is_tnbc",
           "age", "LYMPH_NODES_EXAMINED_POSITIVE", "NPI", "CHEMOTHERAPY"]].reindex(cmp.index)
clin = clin.rename(columns={"LYMPH_NODES_EXAMINED_POSITIVE": "nodes"})
clin["nodes"] = pd.to_numeric(clin["nodes"], errors="coerce")
clin["chemo"] = (clin["CHEMOTHERAPY"].astype(str).str.upper() == "YES").astype(float)
clin.loc[clin["CHEMOTHERAPY"].isna(), "chemo"] = np.nan
VARS = [("LT_pro", "LT_pro_off"), ("LT_anti", "LT_anti_off"),
        ("trained_anti_adj", "trained_anti_off")]
rows = []
for ep in ["os", "rfs"]:
    for scope, mask in [("ALL", pd.Series(True, index=cmp.index)),
                        ("TNBC", clin.is_tnbc.fillna(False))]:
        for vc, vo in VARS:
            for lab, col in [("curated", vc), ("official", vo)]:
                d = pd.DataFrame({
                    "t": clin[f"{ep}_months"], "e": clin[f"{ep}_event"],
                    "v": z(cmp[col]), "age": clin.age, "nodes": clin.nodes,
                    "NPI": clin.NPI, "chemo": clin.chemo}).dropna()
                d = d[mask.reindex(d.index).fillna(False)]
                d = d[d.t > 0]
                if len(d) < 40 or d.e.sum() < 10:
                    continue
                try:
                    cph = CoxPHFitter()
                    cph.fit(d, "t", "e", robust=True)
                    s = cph.summary.loc["v"]
                    rows.append(dict(cohort="METABRIC", scope=scope, endpoint=ep,
                                     variable=vc, abundance=lab, n=len(d),
                                     events=int(d.e.sum()),
                                     hr=float(s["exp(coef)"]), lo=float(s["exp(coef) lower 95%"]),
                                     hi=float(s["exp(coef) upper 95%"]), p=float(s["p"])))
                except Exception as e:
                    print("   skip", vc, lab, e)
out = pd.DataFrame(rows)
out.to_csv(f"{R}/mcpcounter_sensitivity_cox.csv", index=False)
pd.set_option("display.width", 200)
print(out.round(3).to_string(index=False))

# ================================================================== marker overlap
cur_mcp = {
    "T_cell": ["CD3D", "CD3E", "CD3G", "CD2", "TRAC", "CD28", "CD5", "TRAT1", "THEMIS",
               "LCK", "SKAP1", "ITK", "ICOS", "CD6"],
    "CD8_T": ["CD8A", "CD8B", "GZMA", "GZMB", "GZMH", "PRF1", "NKG7", "KLRD1", "CCL5",
              "GZMK", "KLRG1"],
    "Cytotoxic": ["KLRD1", "KLRK1", "KLRB1", "GZMA", "GZMB", "GZMH", "GZMM", "PRF1",
                  "NKG7", "GNLY", "GZMK"],
    "NK": ["NKG7", "GNLY", "KLRD1", "KLRB1", "KLRG1", "KLRF1", "GZMA", "GZMB", "PRF1",
           "FGFBP2", "SPON2", "CX3CR1", "NCR1"],
    "B_cell": ["MS4A1", "CD79A", "CD79B", "IGHM", "IGHD", "IGKC", "IGLC1", "CD19",
               "PAX5", "BANK1", "CD37"],
    "Monocytic": ["CD14", "CSF1R", "LYZ", "LST1", "FCN1", "S100A8", "S100A9", "AIF1",
                  "TYROBP", "FCGR1A", "ITGAM", "CD68", "CSF3R", "MAFB", "SIGLEC1", "CD163"],
    "Myeloid_DC": ["ITGAX", "HLA-DRA", "HLA-DRB1", "CD1C", "FCER1A", "CLEC10A",
                   "CLEC9A", "XCR1", "BATF3", "IRF8", "LAMP3", "CADM1"],
    "Neutrophil": ["CSF3R", "S100A12", "FCGR3B", "CEACAM8", "FPR1", "MNDA", "NCF4",
                   "MPO", "MMP9", "LCN2", "ELANE"],
    "Endothelial": ["PECAM1", "CDH5", "CLDN5", "VWF", "PLVAP", "ROBO4", "ESAM", "TEK", "EGFL7"],
    "Fibroblast": ["COL1A1", "COL1A2", "COL3A1", "DCN", "LUM", "PDGFRB", "FAP", "ACTA2", "THY1"],
}
ov = []
for k in cur_mcp:
    a, b = set(cur_mcp[k]), set(OFF.get(k, []))
    ov.append(dict(population=k, n_curated=len(a), n_official=len(b),
                   shared=len(a & b),
                   jaccard=round(len(a & b) / len(a | b), 3) if (a | b) else np.nan,
                   metabric_official_covered=len([g for g in b if g in me.index])))
ovd = pd.DataFrame(ov)
ovd.to_csv(f"{R}/mcpcounter_marker_overlap.csv", index=False)
print("\nMarker-set overlap (curated vs official MCP-counter):")
print(ovd.to_string(index=False))
print("\nDone.")
