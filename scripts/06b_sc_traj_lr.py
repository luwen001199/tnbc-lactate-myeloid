#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
06b_sc_traj_lr.py
=================
Memory-safe rerun of the two single-cell steps that did not finish in
06_singlecell.py: (i) monocyte -> macrophage pseudotime, (ii) myeloid -> CD8
ligand-receptor inference.

The 100k-cell h5ad is 4.3 GB, so here we read ONLY its .obs (backed mode),
then rebuild the myeloid and T-cell compartments directly from the sparse
count matrix -- a few hundred MB instead of several GB.
"""
import os
import numpy as np, pandas as pd, scipy.sparse as sp
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
from scipy.stats import spearmanr

D, R = "/workspace/data", "/workspace/results"
sc.settings.n_jobs = 8
sc.settings.verbosity = 1

# ------------------------------------------------------------------ obs only
print("[1] reading annotations (backed) ...")
A = sc.read_h5ad(f"{R}/sc_all_scored.h5ad", backed="r")
obs = A.obs.copy()
genes = A.var_names.tolist()
del A
print(f"   obs {obs.shape}, genes {len(genes)}")

M = sp.load_npz(f"{D}/sc176078_counts.npz")            # genes x cells
print(f"   counts {M.shape}")


def build(cell_mask, name):
    idx = np.where(cell_mask)[0]
    X = M[:, idx].T.tocsr().astype(np.float32)
    ad = sc.AnnData(X, obs=obs.iloc[idx].copy(),
                    var=pd.DataFrame(index=pd.Index(genes, name="gene")))
    ad.var_names_make_unique()
    ad.var["mt"] = ad.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(ad, qc_vars=["mt"], percent_top=None,
                               log1p=False, inplace=True)
    ad = ad[(ad.obs.n_genes_by_counts > 200) &
            (ad.obs.pct_counts_mt < 20)].copy()
    ad.layers["counts"] = ad.X.copy()
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)
    print(f"   [{name}] {ad.shape}")
    return ad


# ================================================================== 2. pseudotime
print("\n[2] monocyte -> macrophage trajectory")
myel = build((obs.celltype_major == "Myeloid").values, "myeloid")
print(myel.obs.celltype_minor.value_counts().to_string())

GE = ["LDHA", "SLC16A1", "SLC16A3", "EP300", "CREBBP", "IRF8", "CCL8", "CCR5",
      "HLA-DRA", "CD74", "TREM2", "MRC1", "CD163", "SPP1", "TGFB1", "CXCL9", "CXCL10"]
gex = pd.DataFrame({g: np.asarray(myel[:, g].X.todense()).ravel()
                    for g in GE if g in myel.var_names}, index=myel.obs_names)
print(f"   gene matrix {gex.shape}")

traj = myel[myel.obs.celltype_minor.isin(["Monocyte", "Macrophage"])].copy()
print(f"   trajectory cells: {traj.shape[0]}")
sc.pp.highly_variable_genes(traj, n_top_genes=1500)
traj = traj[:, traj.var.highly_variable].copy()
sc.pp.scale(traj, max_value=10)
sc.tl.pca(traj, n_comps=30, svd_solver="arpack")
sc.pp.neighbors(traj, n_neighbors=15)
sc.tl.diffmap(traj, n_comps=15)
dm = traj.obsm["X_diffmap"]
is_mono = (traj.obs.celltype_minor == "Monocyte").values
sep = np.abs(dm[is_mono].mean(0) - dm[~is_mono].mean(0))
k = int(np.argmax(sep))
sgn = 1.0 if dm[is_mono, k].mean() < dm[~is_mono, k].mean() else -1.0
root_idx = int(np.where(is_mono)[0][np.argmax(sgn * dm[is_mono, k])])
traj.uns["iroot"] = root_idx
print(f"   root: diffusion comp {k}, {traj.obs_names[root_idx]} "
      f"({traj.obs.celltype_minor.iloc[root_idx]})")
sc.tl.dpt(traj)
traj.obs["pseudotime"] = traj.obs["dpt_pseudotime"]

pt = traj.obs[["pseudotime", "celltype_minor", "subtype", "is_TNBC"]].copy()
for g in gex.columns:
    pt[g] = gex.loc[traj.obs_names, g].values
pt.to_csv(f"{R}/sc_pseudotime_gene_dynamics.csv")
print("\n   binned dynamics (deciles of pseudotime):")
show = [c for c in ["LDHA", "SLC16A1", "SLC16A3", "EP300", "IRF8", "CCL8",
                    "HLA-DRA", "TREM2"] if c in pt.columns]
print(pt.groupby(pd.qcut(pt.pseudotime, 10, duplicates="drop"),
                 observed=True)[show].mean().round(3).to_string())

print("\n   Spearman rho of pseudotime vs programme score:")
for s in ["lactyl", "trained_anti", "trained_pro", "apc_score", "m2_score"]:
    if s not in myel.obs.columns:
        continue
    v = myel.obs.loc[traj.obs_names, s]
    for tn in [True, False]:
        d = pt[pt.is_TNBC == tn].dropna(subset=["pseudotime"])
        vv = v.reindex(d.index)
        ok = np.isfinite(vv) & np.isfinite(d.pseudotime)
        if ok.sum() < 100:
            continue
        r, p = spearmanr(d.pseudotime[ok], vv[ok])
        print(f"      {s:14s} {'TNBC    ' if tn else 'non-TNBC'}: "
              f"rho={r:+.3f} p={p:.2e} (n={int(ok.sum())})")
del traj

# ================================================================== 3. ligand-receptor
print("\n[3] myeloid -> CD8 T ligand-receptor inference")
LR = [("CCL8", "CCR5"), ("CCL5", "CCR5"), ("CCL3", "CCR5"), ("CCL4", "CCR5"),
      ("CXCL9", "CXCR3"), ("CXCL10", "CXCR3"), ("CXCL11", "CXCR3"),
      ("TGFB1", "TGFBR1"), ("TGFB1", "TGFBR2"), ("TGFB2", "TGFBR1"),
      ("IL6", "IL6R"), ("TNF", "TNFRSF1A"), ("IL1B", "IL1R1"),
      ("LGALS9", "HAVCR2"), ("CD274", "PDCD1"), ("PDCD1LG2", "PDCD1"),
      ("SPP1", "CD44"), ("MIF", "CD74"), ("SLC16A1", "SLC16A1"),
      ("VEGFA", "FLT1"), ("IL10", "IL10RB"), ("CCL2", "CCR2"),
      ("TNFSF10", "TNFRSF10B"), ("IL15", "IL2RB"), ("CXCL12", "CXCR4")]
lr = [(l, r) for l, r in LR if l in genes and r in genes]
print(f"   {len(lr)} pairs evaluable")

T = build((obs.celltype_major == "T-cells").values, "T-cells")
sc.tl.score_genes(T, ["CD8A", "CD8B", "GZMA", "GZMB", "NKG7", "PRF1", "CCL5"],
                  score_name="cd8s", ctrl_size=50, random_state=0)
cd8 = T[T.obs.cd8s > T.obs.cd8s.median()].copy()
print(f"   CD8-like T cells: {cd8.shape[0]}")
cd8g = pd.DataFrame({r: np.asarray(cd8[:, r].X.todense()).ravel()
                     for _, r in sorted(set(lr)) if r in cd8.var_names},
                    index=cd8.obs_names)

rows = []
for tn in [True, False]:
    for minor in ["Monocyte", "Macrophage", "DCs"]:
        send_mask = (myel.obs.celltype_minor == minor) & (myel.obs.is_TNBC == tn)
        send = myel.obs_names[send_mask.values]
        recv = cd8.obs_names[(cd8.obs.is_TNBC == tn).values]
        if len(send) < 50 or len(recv) < 50:
            continue
        for l, rr in lr:
            if l not in myel.var_names or rr not in cd8g.columns:
                continue
            lv = np.asarray(myel[send, l].X.todense()).ravel()
            rv = cd8g.loc[recv, rr].values
            rows.append(dict(subtype="TNBC" if tn else "non-TNBC", sender=minor,
                             ligand=l, receptor=rr, n_sender=len(send),
                             n_receiver=len(recv),
                             lig_mean=float(lv.mean()), rec_mean=float(rv.mean()),
                             lr_score=float(lv.mean() * rv.mean()),
                             lig_frac=float((lv > 0).mean()),
                             rec_frac=float((rv > 0).mean())))
lrd = pd.DataFrame(rows)
lrd.to_csv(f"{R}/sc_lr_myeloid_cd8.csv", index=False)
piv = lrd.pivot_table(index=["sender", "ligand", "receptor"], columns="subtype",
                      values="lr_score").fillna(0)
piv["log2FC_TNBC_vs_other"] = np.log2((piv.get("TNBC", 0) + 1e-4) /
                                      (piv.get("non-TNBC", 0) + 1e-4))
print("\n   top INCREASED myeloid->CD8 axes in TNBC:")
print(piv.sort_values("log2FC_TNBC_vs_other", ascending=False).head(12).round(3).to_string())
print("\n   top DECREASED myeloid->CD8 axes in TNBC:")
print(piv.sort_values("log2FC_TNBC_vs_other").head(12).round(3).to_string())

print("\n[done] 06b_sc_traj_lr.py")
