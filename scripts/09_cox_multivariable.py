#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
09_cox_multivariable.py
=======================
Stage/age-adjusted and maximal-n Cox models for both cohorts.

The discovery Cox table (cox_continuous.csv) was fitted with age + stage as
covariates, which is the correct model but drops ~16% of TCGA-BRCA through
listwise deletion of missing stage.  Here we fit, for every variable and
endpoint:

  model U : univariable, maximal available n
  model A : adjusted for the covariates available in that cohort
            (TCGA: age + pathologic stage; METABRIC: age + node positivity +
             tumour size proxy NPI + chemotherapy)

and report both, so that no result depends on a complete-case artefact.
"""
import os, json
import numpy as np, pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test

R = "/workspace/results"
VARS = ["LT_pro", "LT_anti", "LT_balance", "lactate_core_adj",
        "trained_anti_adj", "trained_pro_adj", "hdaci_retrain_adj"]


def zscore(s):
    return (s - s.mean()) / s.std(ddof=0)


def fit(df, endpoint, var, covars, scope):
    if var not in df.columns:
        return None
    need = [f"{endpoint}_months", f"{endpoint}_event", var] + list(covars)
    d = df.dropna(subset=[c for c in need if c in df.columns]).copy()
    d = d[d[f"{endpoint}_months"] > 0]
    if len(d) < 40 or d[f"{endpoint}_event"].sum() < 10:
        return None
    use = [var] + [c for c in covars if c in d.columns and d[c].notna().all()]
    X = d[[f"{endpoint}_months", f"{endpoint}_event"] + use].copy()
    X[var] = zscore(X[var])
    try:
        cph = CoxPHFitter()
        cph.fit(X, f"{endpoint}_months", f"{endpoint}_event", robust=True)
    except Exception as e:
        print("   [skip]", var, endpoint, e)
        return None
    s = cph.summary.loc[var]
    return dict(cohort=df.attrs.get("name", ""), scope=scope, endpoint=endpoint, var=var,
                model=("adjusted" if covars else "univariable"),
                n=len(d), events=int(d[f"{endpoint}_event"].sum()),
                hr_sd=float(s["exp(coef)"]), lo=float(s["exp(coef) lower 95%"]),
                hi=float(s["exp(coef) upper 95%"]), p=float(s["p"]),
                covars=";".join([c for c in covars if c in d.columns]))


rows = []

# ---------------------------------------------------------------- TCGA
tcga = pd.read_csv(f"{R}/bulk_scores.csv", index_col=0)
asg = pd.read_csv(f"{R}/subtype_assignments.csv", index_col=0)
for c in [x for x in asg.columns if x.startswith("subtype_")]:
    tcga[c] = asg[c].reindex(tcga.index)
tcga.attrs["name"] = "TCGA-BRCA"
TCGA_COV = ["age", "stage_num"]
for scope, mask in [("ALL-BRCA", pd.Series(True, index=tcga.index)),
                    ("TNBC", tcga.is_tnbc.astype(bool))]:
    for ep in ["os", "dss", "pfs", "dfs"]:
        for v in VARS:
            for covs, _ in [(TCGA_COV, "A"), ([], "U")]:
                r = fit(tcga[mask.values], ep, v, covs, scope)
                if r:
                    rows.append(r)

# ---------------------------------------------------------------- METABRIC
met = pd.read_csv(f"{R}/metabric_scores.csv", index_col=0)
met.attrs["name"] = "METABRIC"
met["nodes"] = pd.to_numeric(met.get("LYMPH_NODES_EXAMINED_POSITIVE"), errors="coerce")
met["chemo"] = (met.get("CHEMOTHERAPY").astype(str).str.upper() == "YES").astype(float)
met.loc[met.get("CHEMOTHERAPY").isna(), "chemo"] = np.nan
MET_COV = ["age", "nodes", "NPI", "chemo"]
for scope, mask in [("ALL", pd.Series(True, index=met.index)),
                    ("TNBC", met.is_tnbc.fillna(False))]:
    for ep in ["os", "rfs"]:
        for v in VARS:
            for covs in [MET_COV, []]:
                r = fit(met[mask.values], ep, v, covs, scope)
                if r:
                    rows.append(r)

out = pd.DataFrame(rows)
out.to_csv(f"{R}/cox_multivariable.csv", index=False)

pd.set_option("display.width", 200)
print("\n=== METABRIC TNBC (adjusted: age + nodes + NPI + chemo) ===")
print(out[(out.cohort == "METABRIC") & (out.scope == "TNBC") & (out.model == "adjusted")]
      .round(4).to_string(index=False))
print("\n=== METABRIC ALL (adjusted) ===")
print(out[(out.cohort == "METABRIC") & (out.scope == "ALL") & (out.model == "adjusted")]
      .round(4).to_string(index=False))
print("\n=== TCGA (adjusted: age + stage) ===")
print(out[(out.cohort == "TCGA-BRCA") & (out.model == "adjusted")]
      .sort_values(["scope", "endpoint", "p"]).round(4).to_string(index=False))
print("\n=== TCGA (univariable, maximal n) ===")
print(out[(out.cohort == "TCGA-BRCA") & (out.model == "univariable")]
      .sort_values(["scope", "endpoint", "p"]).round(4).to_string(index=False))
print("\n[done] 09_cox_multivariable.py")
