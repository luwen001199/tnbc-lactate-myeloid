#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_consensus_subtypes.py
========================
Consensus clustering of TCGA tumours on the two myeloid trained-immunity axes
(LT_pro, LT_anti) + lactate load, then prognostic / immune characterisation.

Deliberately NOT a black-box LASSO gene panel: the partition is defined by two
biologically interpretable, myeloid-adjusted scores, so each cluster maps onto a
mechanistic state (anti-tumour trained / intermediate / pro-tumour lactate memory).

Outputs: results/subtype_assignments.csv, results/subtype_stats.txt
"""
import os, json
import numpy as np, pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
from itertools import combinations

R = "/workspace/results"
os.makedirs(R, exist_ok=True)

S = pd.read_csv(f"{R}/bulk_scores.csv", index_col=0)
print(f"[in] {S.shape[0]} samples, TNBC={int(S['is_tnbc'].sum())}")

FEATURES = ["LT_pro", "LT_anti", "lactate_core_adj"]
POP = "TNBC"     # primary analysis population

def consensus_cluster(X, kmax=6, nrep=1000, frac_s=0.8, frac_f=0.8, seed=42):
    """Monti consensus clustering (ConsensusClusterPlus equivalent), vectorised."""
    rng = np.random.RandomState(seed)
    n = X.shape[0]
    Xz = (X - X.mean()) / (X.std() + 1e-12)
    Xv = np.asarray(Xz, float)
    res = {}
    for k in range(2, kmax + 1):
        CO = np.zeros((n, n))     # co-clustering counts
        M = np.zeros((n, n))      # times both were sampled
        ns = max(2, int(frac_s * n))
        nf = max(1, int(frac_f * X.shape[1]))
        for _ in range(nrep):
            si = rng.choice(n, ns, replace=False)
            fi = rng.choice(X.shape[1], nf, replace=False)
            sub = Xv[np.ix_(si, fi)]
            lab = KMeans(k, n_init=10, random_state=rng.randint(1e6)).fit_predict(sub)
            L = np.zeros((ns, k))
            L[np.arange(ns), lab] = 1.0
            idx = np.ix_(si, si)
            CO[idx] += L @ L.T
            M[idx] += 1.0
        C = np.zeros((n, n))
        keep = M > 0
        C[keep] = CO[keep] / M[keep]
        np.fill_diagonal(C, 1.0)
        Dm = np.clip(1.0 - C, 0, None); np.fill_diagonal(Dm, 0.0)
        D = squareform(Dm, checks=False)
        Z = linkage(D, method="average")
        lab = fcluster(Z, k, criterion="maxclust")
        pac = float(np.mean(((C > 0.1) & (C < 0.9))))
        res[k] = dict(labels=lab, C=C, pac=pac)
        print(f"   k={k}  PAC={pac:.4f}  cluster sizes={np.bincount(lab)[1:]}")
    return res

def run_population(S, pop_name, mask):
    print(f"\n===== population: {pop_name} (n={int(mask.sum())}) =====")
    X = S.loc[mask, FEATURES].dropna()
    print(f"[cluster] feature matrix {X.shape}")
    res = consensus_cluster(X.values, kmax=5, nrep=500, seed=42)
    best_k = min(res, key=lambda k: res[k]["pac"])
    print(f"[cluster] best k by PAC = {best_k}")
    lab = pd.Series(res[best_k]["labels"], index=X.index, name="cluster")
    # order clusters by LT_pro (low -> high) and name them
    m = S.loc[X.index].groupby(lab).mean(numeric_only=True)
    order = m["LT_pro"].sort_values().index.tolist()
    names = {}
    if best_k == 3:
        names = {order[0]: "Anti-tumour-trained", order[1]: "Intermediate",
                 order[2]: "Pro-tumour-lactate-memory"}
    else:
        for i, c in enumerate(order):
            names[c] = f"C{i+1}_LTpro-{m.loc[c, 'LT_pro']:+.2f}"
    print("[cluster] cluster means:")
    print(m[["LT_pro", "LT_anti", "lactate_core_adj", "trained_anti_adj",
             "trained_pro_adj", "myeloid_abundance"]].round(3))
    named = lab.map(names)
    return named, best_k, res

named, k, res = run_population(S, POP, S["is_tnbc"])

# also run on all BRCA for the pan-subtype comparison
named_all, k_all, res_all = run_population(S, "ALL-BRCA", pd.Series(True, index=S.index))

out = S.copy()
out["cluster_tnbc"] = named
out["cluster_all"] = named_all
out.to_csv(f"{R}/subtype_assignments.csv")

# ------------------------------------------------------------------ survival
def surv_table(S, sub_col, endpoint, subset=None):
    d = S if subset is None else S.loc[subset]
    d = d.dropna(subset=[f"{endpoint}_months", f"{endpoint}_event", sub_col])
    d = d[d[f"{endpoint}_months"] > 0]
    if d[sub_col].nunique() < 2 or len(d) < 30:
        return None
    lr = multivariate_logrank_test(d[f"{endpoint}_months"], d[sub_col], d[f"{endpoint}_event"])
    rows = []
    for c, g in d.groupby(sub_col):
        rows.append([c, len(g), int(g[f"{endpoint}_event"].sum()),
                     round(g[f"{endpoint}_months"].median(), 1)])
    return pd.DataFrame(rows, columns=["cluster", "n", "events", "median_fu_mo"]), lr.p_value

print("\n===== survival by cluster =====")
for ep in ["os", "dfs", "pfs", "dss"]:
    r = surv_table(S, "cluster_tnbc", ep, subset=S["is_tnbc"])
    if r is None:
        print(f"  {ep.upper()} (TNBC): insufficient data")
        continue
    tab, p = r
    print(f"  {ep.upper()} (TNBC) logrank p={p:.4f}")
    print(tab.to_string(index=False))

# Cox: cluster vs continuous axes, adjusted for stage/age
print("\n===== Cox models (TNBC) =====")
d = S[S["is_tnbc"]].dropna(subset=["os_months", "os_event"])
cols = ["LT_pro", "LT_anti", "lactate_core_adj", "cluster_tnbc", "age", "stage_num", "t_num", "n_num"]
d = d.dropna(subset=[c for c in cols if c in d.columns])
for ep in ["os", "dfs", "pfs"]:
    dd = d.dropna(subset=[f"{ep}_months", f"{ep}_event"]).copy()
    dd = dd[dd[f"{ep}_months"] > 0]
    if len(dd) < 40 or dd[f"{ep}_event"].sum() < 10:
        print(f"  {ep.upper()}: n={len(dd)} events={int(dd[f'{ep}_event'].sum())} -> skipped")
        continue
    for var in ["LT_pro", "LT_anti", "lactate_core_adj"]:
        df = dd[[f"{ep}_months", f"{ep}_event", var, "age", "stage_num"]].dropna()
        if len(df) < 40:
            continue
        cph = CoxPHFitter()
        try:
            cph.fit(df, f"{ep}_months", f"{ep}_event")
            s = cph.summary.loc[var]
            print(f"  {ep.upper():4s} {var:18s} HR={s['exp(coef)']:.3f} "
                  f"[{np.exp(s['exp(coef) lower 95%']):.3f}-{np.exp(s['exp(coef) upper 95%']):.3f}] p={s['p']:.4f}  n={len(df)}")
        except Exception as e:
            print(f"  {ep.upper()} {var}: failed ({e})")

# ------------------------------------------------------------------ immune contexture by cluster
print("\n===== immune contexture by cluster (TNBC) =====")
t = S[S["is_tnbc"]].dropna(subset=["cluster_tnbc"])
imm = [c for c in S.columns if c.startswith("mcp_")] + ["myeloid_abundance", "lymphoid_abundance",
       "trained_anti_adj", "trained_pro_adj", "lactate_core_adj", "kla_machinery",
       "innate_antiviral_adj", "hypoxia_buffa", "tmb"]
imm = [c for c in imm if c in t.columns]
g = t.groupby("cluster_tnbc")[imm].mean().round(3)
print(g.T.to_string())

with open(f"{R}/subtype_stats.txt", "w") as f:
    f.write(f"best_k_TNBC={k}\nbest_k_all={k_all}\n")
    f.write(g.T.to_string())

print("\n[done] 03_consensus_subtypes.py")
