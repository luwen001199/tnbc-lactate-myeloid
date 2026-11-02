#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_metabric.py
==============
Independent external validation in METABRIC (Illumina HT-12, n~1.9-2.5k).
Expression is pulled from cBioPortal for the ~500 genes of interest only
(their Entrez IDs), clinical endpoints from the same portal.

The identical scoring / subtyping pipeline as in TCGA is applied.
"""
import os, json, sys, time
import numpy as np, pandas as pd
import urllib.request, urllib.parse
from lifelines import CoxPHFitter
from lifelines.statistics import multivariate_logrank_test
from sklearn.mixture import GaussianMixture

R, D = "/workspace/results", "/workspace/data"
API = "https://www.cbioportal.org/api"
STUDY = "brca_metabric"
PROF = "brca_metabric_mrna"

P = json.load(open(f"{R}/gene_panels.json"))["panels"]
MCP_FILE = None


def api(path, **kw):
    url = f"{API}/{path}"
    if kw:
        url += "?" + urllib.parse.urlencode(kw)
    for _ in range(4):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return json.load(r)
        except Exception as e:
            time.sleep(3)
    raise RuntimeError(url)


def api_post(path, payload):
    req = urllib.request.Request(f"{API}/{path}", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    for _ in range(4):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.load(r)
        except Exception:
            time.sleep(3)
    raise RuntimeError(path)


# ------------------------------------------------------------------ gene universe
genes = sorted(set(sum([P[k] for k in ["LACTATE_METABOLISM", "LACTATE_CORE", "KLA_MACHINERY",
                                       "TRAINED_ANTI", "TRAINED_PRO", "TRAINED_METABOLIC",
                                       "INNATE_ANTIVIRAL", "HDACI_RETRAIN"]], [])))
# MCP-counter markers (mirrors 02_signature_scoring.py)
MCP = {
    "T_cell": ["CD3D", "CD3E", "CD3G", "CD2", "TRAC", "CD28", "CD5", "TRAT1", "THEMIS", "LCK", "SKAP1", "ITK", "ICOS", "CD6"],
    "CD8_T": ["CD8A", "CD8B", "GZMA", "GZMB", "GZMH", "PRF1", "NKG7", "KLRD1", "CCL5", "GZMK", "KLRG1"],
    "Cytotoxic": ["KLRD1", "KLRK1", "KLRB1", "GZMA", "GZMB", "GZMH", "GZMM", "PRF1", "NKG7", "GNLY", "GZMK"],
    "NK": ["NKG7", "GNLY", "KLRD1", "KLRB1", "KLRG1", "KLRF1", "GZMA", "GZMB", "PRF1", "FGFBP2", "SPON2", "CX3CR1", "NCR1"],
    "B_cell": ["MS4A1", "CD79A", "CD79B", "IGHM", "IGHD", "IGKC", "IGLC1", "CD19", "PAX5", "BANK1", "CD37"],
    "Monocytic": ["CD14", "CSF1R", "LYZ", "LST1", "FCN1", "S100A8", "S100A9", "AIF1", "TYROBP", "FCGR1A", "ITGAM", "CD68", "CSF3R", "MAFB", "SIGLEC1", "CD163"],
    "Myeloid_DC": ["ITGAX", "HLA-DRA", "HLA-DRB1", "CD1C", "FCER1A", "CLEC10A", "CLEC9A", "XCR1", "BATF3", "IRF8", "LAMP3", "CADM1"],
    "Neutrophil": ["CSF3R", "S100A12", "FCGR3B", "CEACAM8", "FPR1", "MNDA", "NCF4", "MPO", "MMP9", "LCN2", "ELANE"],
    "Endothelial": ["PECAM1", "CDH5", "CLDN5", "VWF", "PLVAP", "ROBO4", "ESAM", "TEK", "EGFL7"],
    "Fibroblast": ["COL1A1", "COL1A2", "COL3A1", "DCN", "LUM", "PDGFRB", "FAP", "ACTA2", "THY1"],
}
genes = sorted(set(genes + sum(MCP.values(), []) + ["ESR1", "PGR", "ERBB2"]))
print(f"[genes] {len(genes)} query genes")

# ------------------------------------------------------------------ hugo -> entrez (local table)
_gmap = json.load(open(f"{D}/cbio_gene_map.json"))
ent = {g: _gmap[g] for g in genes if g in _gmap}
print(f"[genes] {len(ent)} mapped to Entrez")
missing = [g for g in genes if g not in ent]
print(f"[genes] unmapped: {len(missing)} {missing[:10]}")

# ------------------------------------------------------------------ sample list
sid = "brca_metabric_mrna"
print("[samples] using", sid)

# ------------------------------------------------------------------ expression (POST fetch)
eids = [ent[g] for g in genes if g in ent]
rows = {}
for i in range(0, len(eids), 100):
    chunk = [int(x) for x in eids[i:i + 100]]
    d = api_post(f"molecular-profiles/{PROF}/molecular-data/fetch",
                 {"sampleListId": sid, "entrezGeneIds": chunk})
    sym = {v: k for k, v in ent.items()}
    for r in d:
        g = sym.get(r.get("entrezGeneId"))
        if g:
            rows.setdefault(g, {})[r["sampleId"]] = r.get("value")
    print(f"   fetched {len(d)} records  ({min(i+100, len(eids))}/{len(eids)} genes)", flush=True)
X = pd.DataFrame(rows).T
X.index.name = "gene"
print(f"[expr] {X.shape[0]} genes x {X.shape[1]} samples")
X.to_pickle(f"{R}/metabric_expr.pkl")

# ------------------------------------------------------------------ clinical
clin = []
for pn in range(0, 6):
    d = api(f"studies/{STUDY}/clinical-data", clinicalDataType="PATIENT",
            projection="DETAILED", pageSize=10000, pageNumber=pn)
    if not d:
        break
    clin += d
    if len(d) < 10000:
        break
agg = {}
for r in clin:
    agg.setdefault(r["patientId"], {})[r["clinicalAttributeId"]] = r["value"]
C = pd.DataFrame(agg).T
print(f"[clin] {C.shape}")
C.to_csv(f"{R}/metabric_clinical.csv")

# ------------------------------------------------------------------ score
E = X.astype(float).dropna(how="all", axis=1).dropna(how="all", axis=0)
E = E.clip(lower=-5, upper=15)
Z = E.sub(E.mean(axis=1), axis=0).div(E.std(axis=1).replace(0, np.nan), axis=0).dropna(how="all")

def mz(gs):
    g = [x for x in gs if x in Z.index]
    return Z.loc[g].mean(axis=0)

S = pd.DataFrame(index=E.columns)
for name, gs in [("lactate_metab", P["LACTATE_METABOLISM"]), ("lactate_core", P["LACTATE_CORE"]),
                 ("kla_machinery", P["KLA_MACHINERY"]), ("trained_anti", P["TRAINED_ANTI"]),
                 ("trained_pro", P["TRAINED_PRO"]), ("innate_antiviral", P["INNATE_ANTIVIRAL"]),
                 ("hdaci_retrain", P["HDACI_RETRAIN"]), ("trained_metabolic", P["TRAINED_METABOLIC"])]:
    S[name] = mz(gs)
for k, v in MCP.items():
    S[f"mcp_{k}"] = ((E.loc[[g for g in v if g in E.index]].mean(axis=0))
                     .pipe(lambda s: (s - s.mean()) / (s.std() + 1e-12)))
S["myeloid_abundance"] = S[["mcp_Monocytic", "mcp_Myeloid_DC", "mcp_Neutrophil"]].mean(axis=1)
S["lymphoid_abundance"] = S[["mcp_T_cell", "mcp_CD8_T", "mcp_B_cell", "mcp_NK"]].mean(axis=1)

for c in ["lactate_metab", "lactate_core", "kla_machinery", "trained_anti", "trained_pro",
          "innate_antiviral", "hdaci_retrain"]:
    y = S[c].values
    Xm = np.column_stack([np.ones(len(y)), S["myeloid_abundance"].values])
    beta, *_ = np.linalg.lstsq(Xm, y, rcond=None)
    S[c + "_adj"] = y - Xm @ beta

def z(v):
    return (v - v.mean()) / (v.std() + 1e-12)
S["LT_pro"] = z(z(S["lactate_core_adj"]) + z(S["trained_pro_adj"]) - z(S["trained_anti_adj"]))
S["LT_anti"] = z(z(S["trained_anti_adj"]) + z(S["hdaci_retrain_adj"]))

# METABRIC: sampleId == patientId (MB-xxxx)
S["patient"] = list(S.index)
S = S.join(C, on="patient")
S["os_months"] = pd.to_numeric(S.get("OS_MONTHS"), errors="coerce")
S["os_event"] = S.get("OS_STATUS").astype(str).str.startswith("1").astype(float)
S["rfs_months"] = pd.to_numeric(S.get("RFS_MONTHS"), errors="coerce")
S["rfs_event"] = S.get("RFS_STATUS").astype(str).str.startswith("1").astype(float)
S["er_ihc"] = S.get("ER_IHC")
S["her2"] = S.get("HER2_SNP6")
S["age"] = pd.to_numeric(S.get("AGE_AT_DIAGNOSIS"), errors="coerce")
S["claudin_subtype"] = S.get("CLAUDIN_SUBTYPE")

# TNBC by mRNA (same GMM rule as TCGA)
def gmm_cut(v):
    x = np.asarray(v, float); x = x[np.isfinite(x)]
    gm = GaussianMixture(2, random_state=0, n_init=10).fit(x.reshape(-1, 1))
    hi = int(np.argmax(gm.means_.ravel()))
    grid = np.linspace(np.nanpercentile(x, 0.5), np.nanpercentile(x, 99.5), 2000)
    p = gm.predict_proba(grid.reshape(-1, 1))[:, hi]
    return float(grid[np.argmin(np.abs(p - 0.5))])

if all(g in E.index for g in ["ESR1", "PGR", "ERBB2"]):
    for g in ["ESR1", "PGR", "ERBB2"]:
        S[g] = E.loc[g]
    cuts = {g: gmm_cut(E.loc[g].values) for g in ["ESR1", "PGR", "ERBB2"]}
    print("[tnbc] METABRIC cutoffs:", {k: round(v, 3) for k, v in cuts.items()})
    S["is_tnbc"] = pd.DataFrame({g: E.loc[g] < c for g, c in cuts.items()}).all(axis=1).values
    S["is_tnbc_mrna"] = S["is_tnbc"]
    print(f"[tnbc] by mRNA: n={int(S['is_tnbc'].sum())}")
    # IHC-anchored definition (primary): ER_IHC negative & HER2 not gained
    er_neg = S["ER_IHC"].astype(str).str.lower().str.startswith("neg")
    her2_neg = ~S["HER2_SNP6"].astype(str).str.upper().eq("GAIN")
    S["is_tnbc_ihc"] = (er_neg & her2_neg).fillna(False)
    print(f"[tnbc] by IHC (ER- & HER2 non-gained): n={int(S['is_tnbc_ihc'].sum())}")
    ok = er_neg.notna() & (~er_neg.isna()) & S["is_tnbc"].notna()
    if ok.sum() > 50:
        sub = S[ok & er_neg.fillna(False)]
        print(f"[validate] among ER-IHC-negative (n={len(sub)}): "
              f"{(sub['ESR1'] < cuts['ESR1']).mean():.3f} are ESR1-mRNA-low")
        sub2 = S[ok & (~er_neg.fillna(False))]
        print(f"[validate] among ER-IHC-positive (n={len(sub2)}): "
              f"{(sub2['ESR1'] >= cuts['ESR1']).mean():.3f} are ESR1-mRNA-high")
    S["is_tnbc"] = S["is_tnbc_ihc"].fillna(S["is_tnbc_mrna"])

print(f"[cohort] n={len(S)}  TNBC={int(S['is_tnbc'].sum()) if 'is_tnbc' in S else 'NA'}  "
      f"OS events={int(S['os_event'].sum())}")

S["subtype"] = "Intermediate"
S.loc[(S["LT_anti"] > 0) & (S["LT_pro"] < 0), "subtype"] = "Anti-tumour-trained"
S.loc[(S["LT_pro"] > 0) & (S["LT_anti"] < 0), "subtype"] = "Pro-tumour-lactate-memory"
S.to_csv(f"{R}/metabric_scores.csv")

ORDER = ["Anti-tumour-trained", "Intermediate", "Pro-tumour-lactate-memory"]
print("\n===== METABRIC: Cox per 1 SD =====")
res = []
for scope, mask in [("ALL", pd.Series(True, index=S.index)),
                    ("TNBC", S.get("is_tnbc", pd.Series(False, index=S.index)).fillna(False))]:
    for ep in ["os", "rfs"]:
        if f"{ep}_months" not in S: continue
        for var in ["LT_pro", "LT_anti", "lactate_core_adj", "trained_pro_adj", "trained_anti_adj"]:
            d = S.loc[mask].dropna(subset=[f"{ep}_months", f"{ep}_event", var, "age"]).copy()
            d = d[d[f"{ep}_months"] > 0]
            if len(d) < 50 or d[f"{ep}_event"].sum() < 15: continue
            try:
                cph = CoxPHFitter().fit(d[[f"{ep}_months", f"{ep}_event", var, "age"]],
                                        f"{ep}_months", f"{ep}_event")
            except Exception:
                continue
            s = cph.summary.loc[var]; sd = d[var].std()
            hr = np.exp(s["coef"] * sd)
            lo, hi = np.exp((s["coef"] - 1.96 * s["se(coef)"]) * sd), np.exp((s["coef"] + 1.96 * s["se(coef)"]) * sd)
            res.append(dict(scope=scope, endpoint=ep, var=var, n=len(d),
                            events=int(d[f"{ep}_event"].sum()), hr_sd=hr, lo=lo, hi=hi, p=s["p"]))
            print(f"  {scope:5s} {ep.upper():4s} {var:18s} HR/SD={hr:.2f} [{lo:.2f}-{hi:.2f}] "
                  f"p={s['p']:.4g} n={len(d)} ev={int(d[f'{ep}_event'].sum())}")
pd.DataFrame(res).to_csv(f"{R}/metabric_cox.csv", index=False)

print("\n===== METABRIC: survival by subtype =====")
for scope, mask in [("ALL", pd.Series(True, index=S.index)),
                    ("TNBC", S.get("is_tnbc", pd.Series(False, index=S.index)).fillna(False))]:
    for ep in ["os", "rfs"]:
        if f"{ep}_months" not in S: continue
        d = S.loc[mask].dropna(subset=[f"{ep}_months", f"{ep}_event", "subtype"])
        d = d[d[f"{ep}_months"] > 0]
        if d["subtype"].nunique() < 2 or len(d) < 60: continue
        lr = multivariate_logrank_test(d[f"{ep}_months"], d["subtype"], d[f"{ep}_event"])
        print(f"  {scope:5s} {ep.upper():4s} logrank p={lr.p_value:.4g}  n={len(d)} ev={int(d[f'{ep}_event'].sum())}")
        print(d.groupby("subtype").agg(n=(f"{ep}_months", "size"),
                                       events=(f"{ep}_event", "sum"),
                                       med=(f"{ep}_months", "median")).reindex(ORDER).to_string())

print("\n[done] 04_metabric.py")
