#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01b_build_cohort.py
===================
TCGA-BRCA cohort construction (definitive version).

Clinical endpoints : TCGA PanCan Atlas CDR (Liu et al. Cell 2018) via cBioPortal
                     -> OS / DSS / DFS / PFS months + status, PAM50 SUBTYPE,
                        hypoxia scores (Buffa / Ragnum / Winter), TMB.
Stage / demographics: GDC API (AJCC stage, T/N/M, age, sex).
Expression          : UCSC Xena HiSeqV2 (log2 RSEM+1), primary tumours only.
TNBC definition     : ESR1-/PGR-/ERBB2- by mRNA, cutoff = posterior-probability
                      crossing point of a 2-component GMM; validated against
                      PanCan Atlas PAM50 basal-like labels.
"""
import os, json
import numpy as np, pandas as pd
from sklearn.mixture import GaussianMixture

D, R = "/workspace/data", "/workspace/results"
os.makedirs(R, exist_ok=True)

# ---------------------------------------------------------------- expression
expr = pd.read_pickle(f"{R}/expr_tcga_brca.pkl")
print(f"[expr] {expr.shape[0]} genes x {expr.shape[1]} tumours")

# ---------------------------------------------------------------- cBioPortal CDR
def cbio_agg(fn, key):
    rec = json.load(open(f"{D}/{fn}"))
    out = {}
    for r in rec:
        out.setdefault(r[key], {})[r["clinicalAttributeId"]] = r["value"]
    return pd.DataFrame(out).T

pat = cbio_agg("cbio_pancan_PATIENT.json", "patientId")
smp = cbio_agg("cbio_pancan_SAMPLE.json", "sampleId")
print(f"[cdr] patients={pat.shape[0]} samples={smp.shape[0]}")

def num(x):
    return pd.to_numeric(x, errors="coerce")

cdr = pd.DataFrame(index=pat.index)
cdr["age"] = num(pat.get("AGE"))
cdr["sex"] = pat.get("SEX")
cdr["os_months"] = num(pat.get("OS_MONTHS"))
cdr["os_event"] = pat.get("OS_STATUS").astype(str).str.startswith("1").astype(float)
cdr["dss_months"] = num(pat.get("DSS_MONTHS"))
cdr["dss_event"] = pat.get("DSS_STATUS").astype(str).str.startswith("1").astype(float)
cdr["dfs_months"] = num(pat.get("DFS_MONTHS"))
cdr["dfs_event"] = pat.get("DFS_STATUS").astype(str).str.startswith("1").astype(float)
cdr["pfs_months"] = num(pat.get("PFS_MONTHS"))
cdr["pfs_event"] = pat.get("PFS_STATUS").astype(str).str.startswith("1").astype(float)
cdr["pam50"] = pat.get("SUBTYPE").astype(str).str.replace("BRCA_", "", regex=False)
cdr["hypoxia_buffa"] = num(pat.get("BUFFA_HYPOXIA_SCORE"))
cdr["hypoxia_ragnum"] = num(pat.get("RAGNUM_HYPOXIA_SCORE"))
cdr["hypoxia_winter"] = num(pat.get("WINTER_HYPOXIA_SCORE"))
cdr["tmb"] = num(smp.get("TMB_NONSYNONYMOUS")) if "TMB_NONSYNONYMOUS" in smp else np.nan
cdr["ajcc_stage"] = pat.get("AJCC_PATHOLOGIC_TUMOR_STAGE")
cdr["path_T"] = pat.get("PATH_T_STAGE")
cdr["path_N"] = pat.get("PATH_N_STAGE")
cdr["path_M"] = pat.get("PATH_M_STAGE")

# ---------------------------------------------------------------- GDC stage
gdc = pd.read_csv(f"{R}/tcga_brca_clinical.csv", index_col=0)
gdc["patient"] = [c[:12] for c in gdc.index]
g = gdc.drop_duplicates("patient").set_index("patient")
cdr["stage_num"] = cdr.index.map(g["stage_num"]) if "stage_num" in g else np.nan
cdr["t_num"] = cdr.index.map(g["t_num"]) if "t_num" in g else np.nan
cdr["n_num"] = cdr.index.map(g["n_num"]) if "n_num" in g else np.nan

# ---------------------------------------------------------------- TNBC by mRNA
def gmm_cut(v):
    """cutoff = value where P(high component)=0.5 (posterior crossing point)"""
    x = np.asarray(v, float); x = x[np.isfinite(x)]
    gm = GaussianMixture(2, random_state=0, n_init=10).fit(x.reshape(-1, 1))
    hi = int(np.argmax(gm.means_.ravel()))
    grid = np.linspace(np.nanpercentile(x, 0.5), np.nanpercentile(x, 99.5), 2000)
    p = gm.predict_proba(grid.reshape(-1, 1))[:, hi]
    return float(grid[np.argmin(np.abs(p - 0.5))]), gm

e = expr.clip(lower=0)
cuts = {}
for gname in ["ESR1", "PGR", "ERBB2"]:
    c, gm = gmm_cut(e.loc[gname].values)
    cuts[gname] = c
    print(f"[tnbc] {gname}: cutoff={c:.3f}  modes={np.sort(gm.means_.ravel()).round(2)}")

neg = pd.DataFrame({k: (e.loc[k] < c) for k, c in cuts.items()})
tnbc = neg.all(axis=1)

ann = pd.DataFrame({"sample": expr.columns,
                    "patient": [c[:12] for c in expr.columns],
                    "ESR1": e.loc["ESR1"].values, "PGR": e.loc["PGR"].values,
                    "ERBB2": e.loc["ERBB2"].values,
                    "is_tnbc": tnbc.values}).set_index("sample")
ann = ann.join(cdr, on="patient")
ann.to_csv(f"{R}/tcga_brca_annotation.csv")

print(f"[tnbc] n_TNBC = {int(ann['is_tnbc'].sum())} / {len(ann)} "
      f"({100*ann['is_tnbc'].mean():.1f}%)")
# cross-check vs PAM50
sub = ann.dropna(subset=["pam50"]).query("pam50 != 'nan'")
print(f"[validate] samples with PAM50 = {len(sub)}")
ct = pd.crosstab(sub["is_tnbc"], sub["pam50"])
print(ct)
if "Basal" in ct.columns and (True in ct.index):
    b = ct.loc[True, "Basal"]; n = ct.loc[True].sum()
    print(f"[validate] TNBC-mRNA & PAM50-Basal = {b}/{n} ({100*b/n:.1f}%)")

# survival availability
for ep in ["os", "dss", "dfs", "pfs"]:
    m, ev = f"{ep}_months", f"{ep}_event"
    d = ann.dropna(subset=[m, ev])
    d_t = d[d["is_tnbc"]]
    print(f"[surv] {ep.upper():4s}: all n={len(d):4d} events={int(d[ev].sum()):3d} | "
          f"TNBC n={len(d_t):3d} events={int(d_t[ev].sum()):3d}")

print("\n[done] 01b_build_cohort.py")
