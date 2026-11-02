#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_subtypes.py  (definitive)
=============================
Myeloid trained-immunity subtyping of TCGA-BRCA.

Rationale for a rule-based partition instead of unsupervised consensus clustering
--------------------------------------------------------------------------------
The two myeloid-adjusted axes (LT_pro / LT_anti) are continuous by construction;
forcing k-means on 3 continuous features gives unstable partitions (PAC 0.43-0.61,
see 03b_sensitivity_clustering.py). We therefore use an *a-priori, interpretable*
quadrant rule that (i) maps one-to-one onto the mechanistic hypothesis and
(ii) can be projected unchanged onto independent cohorts:

    Anti-tumour-trained     : LT_anti > 0  & LT_pro < 0
    Pro-tumour-lactate-mem  : LT_pro  > 0  & LT_anti < 0
    Intermediate            : everything else (both high / both low)

Unsupervised consensus clustering is retained as a sensitivity analysis.
"""
import os, json
import numpy as np, pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test, logrank_test
from scipy.stats import mannwhitneyu, chi2_contingency
from sklearn.metrics import adjusted_rand_score

R = "/workspace/results"
S = pd.read_csv(f"{R}/bulk_scores.csv", index_col=0)
print(f"[in] {S.shape[0]} samples | TNBC={int(S['is_tnbc'].sum())}")

# ------------------------------------------------------------------ quadrant rule
def assign(df, cut_pro=0.0, cut_anti=0.0):
    hi_pro = df["LT_pro"] > cut_pro
    hi_anti = df["LT_anti"] > cut_anti
    lab = pd.Series("Intermediate", index=df.index)
    lab[hi_anti & ~hi_pro] = "Anti-tumour-trained"
    lab[hi_pro & ~hi_anti] = "Pro-tumour-lactate-memory"
    return lab

ORDER = ["Anti-tumour-trained", "Intermediate", "Pro-tumour-lactate-memory"]

for scope, mask in [("TNBC", S["is_tnbc"]), ("ALL-BRCA", pd.Series(True, index=S.index))]:
    sub = S.loc[mask]
    lab = assign(sub)
    S.loc[mask, f"subtype_{scope}"] = lab
    print(f"\n===== {scope} (n={int(mask.sum())}) =====")
    print(lab.value_counts().reindex(ORDER).to_string())

S.to_csv(f"{R}/subtype_assignments.csv")
print("\n[out] subtype_assignments.csv")

# ------------------------------------------------------------------ immune / pathway contexture
print("\n===== contexture by subtype (TNBC) =====")
t = S[S["is_tnbc"]].dropna(subset=["subtype_TNBC"])
feat = [c for c in S.columns if c.startswith("mcp_")] + [
    "myeloid_abundance", "lymphoid_abundance", "LT_pro", "LT_anti",
    "lactate_core_adj", "lactate_metab_adj", "kla_machinery", "kla_writer",
    "kla_eraser", "kla_reader", "trained_anti_adj", "trained_pro_adj",
    "innate_antiviral_adj", "hdaci_retrain_adj", "hypoxia_buffa", "hypoxia_winter"]
feat = [c for c in feat if c in t.columns]
g = t.groupby("subtype_TNBC")[feat].mean().reindex(ORDER)
print(g.T.round(3).to_string())
g.T.round(4).to_csv(f"{R}/tnbc_subtype_contexture.csv")

# Kruskal-Wallis across the 3 subtypes
from scipy.stats import kruskal
print("\n===== subtype differences (Kruskal-Wallis, TNBC) =====")
rows = []
for c in feat:
    grps = [t.loc[t["subtype_TNBC"] == k, c].dropna().values for k in ORDER]
    if all(len(x) > 3 for x in grps):
        try:
            h, p = kruskal(*grps)
            rows.append([c, round(h, 2), f"{p:.2e}"])
        except Exception:
            pass
kw = pd.DataFrame(rows, columns=["feature", "H", "p"]).sort_values("p")
print(kw.head(25).to_string(index=False))
kw.to_csv(f"{R}/tnbc_subtype_kruskal.csv", index=False)

# ------------------------------------------------------------------ survival
def cox_cont(df, endpoint, var, covars=("age", "stage_num"), min_n=40, min_ev=10):
    cols = [f"{endpoint}_months", f"{endpoint}_event", var] + [c for c in covars if c in df.columns]
    d = df.dropna(subset=cols).copy()
    d = d[d[f"{endpoint}_months"] > 0]
    if len(d) < min_n or d[f"{endpoint}_event"].sum() < min_ev:
        return None
    cph = CoxPHFitter()
    try:
        cph.fit(d[cols], f"{endpoint}_months", f"{endpoint}_event")
    except Exception as e:
        return None
    s = cph.summary.loc[var]
    # per-SD HR
    sd = d[var].std()
    hr_sd = np.exp(s["coef"] * sd)
    lo, hi = np.exp((s["coef"] - 1.96 * s["se(coef)"]) * sd), np.exp((s["coef"] + 1.96 * s["se(coef)"]) * sd)
    return dict(var=var, endpoint=endpoint, n=len(d), events=int(d[f"{endpoint}_event"].sum()),
                hr_sd=hr_sd, lo=lo, hi=hi, p=s["p"])

def surv_tab(df, endpoint, sub_col):
    d = df.dropna(subset=[f"{endpoint}_months", f"{endpoint}_event", sub_col])
    d = d[d[f"{endpoint}_months"] > 0]
    if d[sub_col].nunique() < 2 or len(d) < 30:
        return None
    lr = multivariate_logrank_test(d[f"{endpoint}_months"], d[sub_col], d[f"{endpoint}_event"])
    rows = []
    for c in ORDER:
        gg = d[d[sub_col] == c]
        if len(gg) == 0:
            continue
        rows.append([c, len(gg), int(gg[f"{endpoint}_event"].sum()),
                     round(gg[f"{endpoint}_months"].median(), 1)])
    return pd.DataFrame(rows, columns=["subtype", "n", "events", "median_fu_mo"]), lr.p_value

print("\n===== Cox: continuous axes (per 1 SD) =====")
cox_rows = []
for scope, sub_col in [("TNBC", "subtype_TNBC"), ("ALL-BRCA", "subtype_ALL-BRCA")]:
    df = S[S["is_tnbc"]] if scope == "TNBC" else S
    for ep in ["os", "dfs", "pfs", "dss"]:
        for var in ["LT_pro", "LT_anti", "lactate_core_adj", "trained_anti_adj",
                    "trained_pro_adj", "LT_balance"]:
            if var not in df.columns:
                continue
            r = cox_cont(df, ep, var)
            if r:
                r["scope"] = scope
                cox_rows.append(r)
                print(f"  {scope:8s} {ep.upper():4s} {var:18s} HR/SD={r['hr_sd']:.2f} "
                      f"[{r['lo']:.2f}-{r['hi']:.2f}] p={r['p']:.4f} n={r['n']} ev={r['events']}")
coxdf = pd.DataFrame(cox_rows)
coxdf.to_csv(f"{R}/cox_continuous.csv", index=False)

print("\n===== survival by subtype =====")
surv_rows = []
for scope, sub_col in [("TNBC", "subtype_TNBC"), ("ALL-BRCA", "subtype_ALL-BRCA")]:
    df = S[S["is_tnbc"]] if scope == "TNBC" else S
    for ep in ["os", "dfs", "pfs", "dss"]:
        r = surv_tab(df, ep, sub_col)
        if r is None:
            print(f"  {scope:8s} {ep.upper():4s}: insufficient data")
            continue
        tab, p = r
        print(f"  {scope:8s} {ep.upper():4s} logrank p={p:.4f}")
        print(tab.to_string(index=False))
        tab["scope"], tab["endpoint"], tab["logrank_p"] = scope, ep, p
        surv_rows.append(tab)
if surv_rows:
    pd.concat(surv_rows).to_csv(f"{R}/survival_by_subtype.csv", index=False)

# pairwise Pro-tumour vs Anti-tumour
print("\n===== pairwise: Pro-tumour-lactate-memory vs Anti-tumour-trained (TNBC) =====")
for ep in ["os", "dfs", "pfs"]:
    d = S[S["is_tnbc"]].dropna(subset=[f"{ep}_months", f"{ep}_event", "subtype_TNBC"])
    d = d[d[f"{ep}_months"] > 0]
    a = d[d["subtype_TNBC"] == "Pro-tumour-lactate-memory"]
    b = d[d["subtype_TNBC"] == "Anti-tumour-trained"]
    if len(a) > 5 and len(b) > 5:
        lr = logrank_test(a[f"{ep}_months"], b[f"{ep}_months"],
                          a[f"{ep}_event"], b[f"{ep}_event"])
        # univariable Cox for the contrast
        dd = d[d["subtype_TNBC"].isin(["Pro-tumour-lactate-memory", "Anti-tumour-trained"])].copy()
        dd["grp"] = (dd["subtype_TNBC"] == "Pro-tumour-lactate-memory").astype(int)
        try:
            cph = CoxPHFitter().fit(dd[[f"{ep}_months", f"{ep}_event", "grp", "age", "stage_num"]].dropna(),
                                    f"{ep}_months", f"{ep}_event")
            s = cph.summary.loc["grp"]
            print(f"  {ep.upper():4s} nA={len(b)} nB={len(a)} logrank p={lr.p_value:.4f} | "
                  f"HR(Pro vs Anti)={s['exp(coef)']:.2f} "
                  f"[{np.exp(s['exp(coef) lower 95%']):.2f}-{np.exp(s['exp(coef) upper 95%']):.2f}] p={s['p']:.4f}")
        except Exception as e:
            print(f"  {ep.upper():4s} logrank p={lr.p_value:.4f} (Cox failed: {e})")

# ------------------------------------------------------------------ subtype vs clinical
print("\n===== subtype x clinical (TNBC) =====")
t = S[S["is_tnbc"]].dropna(subset=["subtype_TNBC"])
for c in ["stage_num", "t_num", "n_num", "age"]:
    if c in t.columns:
        grps = [t.loc[t["subtype_TNBC"] == k, c].dropna().values for k in ORDER]
        try:
            h, p = kruskal(*[g for g in grps if len(g) > 3])
            print(f"  {c:10s} Kruskal p={p:.4f}  means="
                  + ", ".join(f"{k[:12]}={np.nanmean(g):.2f}" for k, g in zip(ORDER, grps)))
        except Exception:
            pass
if "pam50" in t.columns:
    ct = pd.crosstab(t["subtype_TNBC"], t["pam50"])
    print("\n  subtype x PAM50:\n", ct.to_string())

print("\n[done] 03_subtypes.py")
