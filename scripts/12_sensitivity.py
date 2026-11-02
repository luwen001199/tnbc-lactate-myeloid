#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
12_sensitivity.py
=================
Sensitivity analyses requested during manuscript revision:

  (i)   formal test of the proportional-hazards assumption
        (Schoenfeld-type test as implemented in lifelines
        `proportional_hazard_test`, rank-transformed time);
  (ii)  landmark analysis split at 5 years, for the axes where PH fails;
  (iii) restricted mean survival time (RMST) at a 10-year horizon by
        tertile of each axis - an assumption-free effect measure.

Outputs results/cox_sensitivity.csv  (PH)  and
        results/sensitivity_landmark_rmst.csv (landmark + RMST).
"""
import os
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import proportional_hazard_test
from lifelines.utils import restricted_mean_survival_time

R = "/workspace/results"
HORIZON = 120.0          # 10 years
LANDMARK = 60.0          # 5 years

AXES = ["LT_pro", "LT_anti", "lactate_core_adj", "lactate_metab_adj",
        "trained_anti_adj", "trained_pro_adj", "hdaci_retrain_adj"]

met = pd.read_csv(f"{R}/metabric_scores.csv", index_col=0)
tcga = pd.read_csv(f"{R}/bulk_scores.csv", index_col=0)
tcga_ann = pd.read_csv(f"{R}/tcga_brca_annotation.csv")
tcga = tcga.join(tcga_ann.set_index("sample")[["is_tnbc", "os_months", "os_event",
                                               "dss_months", "dss_event",
                                               "pfs_months", "pfs_event"]],
                 how="left", rsuffix="_ann")
for c in ["is_tnbc", "os_months", "os_event", "dss_months", "dss_event",
          "pfs_months", "pfs_event"]:
    if c + "_ann" in tcga.columns:
        tcga[c] = tcga[c].combine_first(tcga[c + "_ann"]) if tcga[c].dtype == float \
            else tcga[c + "_ann"]
        tcga = tcga.drop(columns=[c + "_ann"])

rows, rows2 = [], []


def ph_test(df, var, time_col, event_col):
    cols = [var, time_col, event_col]
    d = df.dropna(subset=cols)
    if len(d) < 50 or d[event_col].sum() < 10:
        return None
    cph = CoxPHFitter().fit(d[cols], time_col, event_col)
    r = proportional_hazard_test(cph, d[cols], time_transform="rank")
    return float(r.p_value[0]), len(d), int(d[event_col].sum())


# ---------------------------------------------------------------- PH assumption
for cohort, df, tcol, ecol in [("METABRIC", met, "os_months", "os_event"),
                               ("METABRIC", met, "rfs_months", "rfs_event"),
                               ("TCGA-BRCA", tcga, "os_months", "os_event"),
                               ("TCGA-BRCA", tcga, "dss_months", "dss_event"),
                               ("TCGA-BRCA", tcga, "pfs_months", "pfs_event")]:
    if tcol not in df.columns:
        continue
    for scope, mask in [("all", pd.Series(True, index=df.index))] + \
                       ([("TNBC", df["is_tnbc"] == True)] if "is_tnbc" in df.columns else []):
        for var in AXES:
            if var not in df.columns:
                continue
            out = ph_test(df[mask], var, tcol, ecol)
            if out is None:
                continue
            rows.append(dict(cohort=cohort, endpoint=tcol.split("_")[0], scope=scope,
                             variable=var, ph_p=out[0], n=out[1], events=out[2]))
ph = pd.DataFrame(rows)
ph.to_csv(f"{R}/cox_sensitivity_ph.csv", index=False)
print("=== proportional-hazards tests (rank-transformed Schoenfeld) ===")
print(ph.to_string(index=False))


# ------------------------------------------------- landmark + RMST, METABRIC
def landmark(d, var, tcol, ecol, cut=LANDMARK):
    out = []
    e = d.copy()
    e[ecol] = np.where((e[tcol] > cut) & (e[ecol] == 1), 0, e[ecol])
    e[tcol] = np.minimum(e[tcol], cut)
    l = d[d[tcol] > cut].copy()
    l[tcol] = l[tcol] - cut
    for lab, dd in [(f"0-{int(cut/12)}y", e), (f">{int(cut/12)}y landmark", l)]:
        if len(dd) < 30 or dd[ecol].sum() < 10:
            continue
        c = CoxPHFitter().fit(dd[[var, tcol, ecol]], tcol, ecol)
        s = c.summary.loc[var]
        out.append(dict(window=lab, hr=s["exp(coef)"], lo=s["exp(coef) lower 95%"],
                        hi=s["exp(coef) upper 95%"], p=s["p"], n=len(dd),
                        events=int(dd[ecol].sum())))
    return out


for scope, mask in [("METABRIC-TNBC", met["is_tnbc"] == True),
                    ("METABRIC-all", pd.Series(True, index=met.index))]:
    for var in ["LT_pro", "LT_anti", "lactate_core_adj", "trained_anti_adj"]:
        d = met[mask].dropna(subset=[var, "os_months", "os_event"])
        if len(d) < 60:
            continue
        for r in landmark(d, var, "os_months", "os_event"):
            rows2.append(dict(scope=scope, variable=var, analysis="landmark OS", **r))
        # RMST by tertile
        d = d.copy()
        d["grp"] = pd.qcut(d[var], 3, labels=["T1 low", "T2", "T3 high"])
        for g, gg in d.groupby("grp", observed=True):
            kmf = KaplanMeierFitter().fit(gg["os_months"], gg["os_event"])
            rows2.append(dict(scope=scope, variable=var,
                              analysis=f"RMST@{int(HORIZON/12)}y {g}",
                              window=g, hr=np.nan, lo=np.nan, hi=np.nan, p=np.nan,
                              n=len(gg), events=int(gg["os_event"].sum()),
                              rmst=restricted_mean_survival_time(kmf, t=HORIZON)))
sens = pd.DataFrame(rows2)
sens.to_csv(f"{R}/cox_sensitivity_landmark_rmst.csv", index=False)
print("\n=== landmark + RMST (METABRIC, OS) ===")
print(sens.to_string(index=False))
