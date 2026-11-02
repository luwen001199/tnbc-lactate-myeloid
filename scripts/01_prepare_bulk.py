#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_prepare_bulk.py
==================
Build the TCGA-BRCA bulk cohort:
  * expression (UCSC Xena HiSeqV2, log2 RSEM+1, genes x samples)
  * clinical (GDC API: stage, T/N/M, age, vital status, OS)
  * TNBC definition by mRNA (ESR1-/PGR-/ERBB2-) with 2-component GMM cutoffs,
    cross-checked against a basal-like transcriptional score.
Output: /workspace/results/tcga_bulk.{h5ad,pkl}, cohort tables in /workspace/results/
"""
import os, json, gzip, sys
import numpy as np, pandas as pd
from sklearn.mixture import GaussianMixture

D = "/workspace/data"
R = "/workspace/results"
os.makedirs(R, exist_ok=True)
rng = np.random.RandomState(42)

# ------------------------------------------------------------------ expression
print("[1] reading expression ...")
expr = pd.read_csv(f"{D}/TCGA.BRCA.HiSeqV2.gz", sep="\t", index_col=0)
expr.index.name = "gene"
print("   raw:", expr.shape)
expr = expr[~expr.index.duplicated(keep="first")]
expr = expr[~expr.index.isna()]
# keep tumour samples only (TCGA barcode position 14-15 == 01)
tum = [c for c in expr.columns if c.split("-")[3].startswith("01")]
expr = expr[tum]
print("   tumour samples:", expr.shape[1], "| genes:", expr.shape[0])
expr = expr.dropna(axis=1, how="all")
expr.to_pickle(f"{R}/expr_tcga_brca.pkl")

# ------------------------------------------------------------------ clinical
print("[2] parsing GDC clinical ...")
rows = json.load(open(f"{D}/GDC_BRCA_clinical.json"))
rec = []
for r in rows:
    dem = r.get("demographic") or {}
    dxs = r.get("diagnoses") or [{}]
    dx = dxs[0]
    rec.append(dict(
        submitter_id=r.get("submitter_id"),
        case_id=r.get("case_id"),
        gender=dem.get("gender"),
        vital_status=dem.get("vital_status"),
        days_to_death=dem.get("days_to_death"),
        age_at_index=dem.get("age_at_index"),
        ajcc_stage=(dx.get("ajcc_pathologic_stage") or "").replace("Stage ", ""),
        pathologic_t=dx.get("ajcc_pathologic_t"),
        pathologic_n=dx.get("ajcc_pathologic_n"),
        pathologic_m=dx.get("ajcc_pathologic_m"),
        days_to_last_follow_up=dx.get("days_to_last_follow_up"),
        morphology=dx.get("morphology"),
        primary_diagnosis=dx.get("primary_diagnosis"),
    ))
clin = pd.DataFrame(rec).drop_duplicates("submitter_id").set_index("submitter_id")

def stage_num(s):
    s = str(s).strip().upper()
    return {"I": 1, "IA": 1, "IB": 1, "II": 2, "IIA": 2, "IIB": 2,
            "III": 3, "IIIA": 3, "IIIB": 3, "IIIC": 3, "IV": 4, "X": np.nan,
            "": np.nan, "NAN": np.nan}.get(s, np.nan)

clin["stage_num"] = clin["ajcc_stage"].map(stage_num)
clin["t_num"] = clin["pathologic_t"].astype(str).str.extract(r"T(\d)")[0].astype(float)
clin["n_num"] = clin["pathologic_n"].astype(str).str.extract(r"N(\d)")[0].astype(float)
clin["m_num"] = clin["pathologic_m"].astype(str).str.extract(r"M(\d)")[0].astype(float)
clin["age"] = pd.to_numeric(clin["age_at_index"], errors="coerce")

# overall survival (days) -- TCGA recommended endpoints
clin["os_days"] = pd.to_numeric(clin["days_to_death"], errors="coerce")
fu = pd.to_numeric(clin["days_to_last_follow_up"], errors="coerce")
clin.loc[clin["os_days"].isna(), "os_days"] = fu[clin["os_days"].isna()]
clin["os_event"] = (clin["vital_status"].astype(str).str.lower() == "dead").astype(int)
clin.loc[clin["os_days"].isna(), "os_event"] = np.nan

# map to expression barcodes (TCGA-XX-YYYY-01)
bmap = pd.DataFrame({"barcode": list(expr.columns),
                     "submitter_id": [c[:12] for c in expr.columns]}).set_index("submitter_id")
clin = clin.join(bmap)
clin = clin[clin["barcode"].notna()].set_index("barcode")
print("   clinical matched to expression:", clin.shape[0])
clin.to_csv(f"{R}/tcga_brca_clinical.csv")

# ------------------------------------------------------------------ TNBC definition
print("[3] defining TNBC by mRNA (ESR1- / PGR- / ERBB2-) ...")

def gmm_cutoff(v):
    """2-component GMM on a 1-D vector; return midpoint-of-low-mode cutoff
       = upper bound of the low-expression component."""
    x = np.asarray(v, float)
    x = x[np.isfinite(x)].reshape(-1, 1)
    gm = GaussianMixture(2, covariance_type="full", random_state=0, n_init=10).fit(x)
    mu = gm.means_.ravel()
    lo = int(np.argmin(mu))
    sd = np.sqrt(gm.covariances_.ravel())[lo]
    return float(mu[lo] + 1.28 * sd)   # 90th pct of the low mode

e = expr.clip(lower=0)
cuts = {}
for g in ["ESR1", "PGR", "ERBB2"]:
    cuts[g] = gmm_cutoff(e.loc[g].values)
    print(f"   {g}: cutoff={cuts[g]:.3f}  (low-mode 90th pct)")

neg = pd.DataFrame({g: (e.loc[g] < cuts[g]) for g in cuts})
is_tnbc = neg.all(axis=1)

# basal-like cross-check (independent of the receptor cutoffs)
basal = ["KRT5", "KRT14", "KRT17", "EGFR", "MKI67", "ACTR3B", "ANXA8", "FABP7",
         "TRIM29", "SOX10", "MIA", "CDH2", "VIM", "FOXC1", "MYC"]
lum   = ["ESR1", "PGR", "FOXA1", "GATA3", "XBP1", "KRT18", "TFF1", "TFF3", "SLC39A6"]
avail_b = [g for g in basal if g in e.index]
avail_l = [g for g in lum if g in e.index]
zb = e.loc[avail_b].apply(lambda c: (c - c.mean()) / (c.std() + 1e-9)).mean()
zl = e.loc[avail_l].apply(lambda c: (c - c.mean()) / (c.std() + 1e-9)).mean()
basal_like = zb > zl

coh = pd.DataFrame({
    "sample": expr.columns,
    "ESR1": e.loc["ESR1"].values, "PGR": e.loc["PGR"].values, "ERBB2": e.loc["ERBB2"].values,
    "is_tnbc": is_tnbc.values, "basal_like": basal_like.values,
}).set_index("sample")
coh = coh.join(clin)
print(f"   TNBC n={int(coh['is_tnbc'].sum())} / {len(coh)}")
print(f"   TNBC & basal-like: {int((coh['is_tnbc'] & coh['basal_like']).sum())} "
      f"({100*(coh['is_tnbc'] & coh['basal_like']).sum()/max(1,coh['is_tnbc'].sum()):.1f}% of TNBC)")
print(f"   non-TNBC & basal-like: {int((~coh['is_tnbc'] & coh['basal_like']).sum())}")
coh.to_csv(f"{R}/tcga_cohort_annotation.csv")

print("[4] survival sanity check (all BRCA):")
d = coh.dropna(subset=["os_days", "os_event"])
print(f"   n with OS = {len(d)}, events = {int(d['os_event'].sum())}, "
      f"median f/u = {d['os_days'].median():.0f} d")

print("\n[done] 01_prepare_bulk.py")
